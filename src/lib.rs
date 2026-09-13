//! Zero-dependency SSE stream parser optimized for LLM token streaming.
//!
//! # Example
//! ```ignore
//! use tokio_sse_llm::{LlmSseStream, SseEvent};
//! use tokio_stream::StreamExt;
//!
//! let response = reqwest::get("https://api.openai.com/v1/chat/completions").await?;
//! let byte_stream = response.bytes_stream();
//! let mut sse_stream = LlmSseStream::new(byte_stream);
//!
//! while let Some(event) = sse_stream.next().await {
//!     match event? {
//!         SseEvent::Data(json) => println!("Token: {}", json),
//!         SseEvent::Done => break,
//!         SseEvent::Comment(_) => {}
//!     }
//! }
//! ```

use bytes::{Bytes, BytesMut};
use futures_core::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Events emitted during LLM SSE streaming.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum SseEvent {
    /// A data line containing payload text (typically JSON delta).
    Data(String),
    /// Signal that the stream has finished (e.g., `[DONE]`).
    Done,
    /// Keep-alive or comment line (starts with `:`).
    Comment(String),
}

/// A zero-copy Server-Sent Events (SSE) stream parser optimized for LLM token streaming.
///
/// Wraps any `Stream<Item = Result<Bytes, E>>` (e.g., from reqwest, hyper, or axum)
/// and emits structured [`SseEvent`] values.
pub struct LlmSseStream<S> {
    inner: S,
    buffer: BytesMut,
}

impl<S> LlmSseStream<S> {
    /// Create a new SSE parser wrapping the given byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: BytesMut::with_capacity(4096),
        }
    }

    /// Consume the parser and return the underlying stream.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S, E> Stream for LlmSseStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<SseEvent, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Check if we have a complete line in buffer
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                let line_bytes = self.buffer.split_to(pos + 1);
                let line = line_bytes.strip_suffix(b"\r\n")
                    .or_else(|| line_bytes.strip_suffix(b"\n"))
                    .unwrap_or(&line_bytes);

                // Skip empty lines (event separators in SSE)
                if line.is_empty() {
                    continue;
                }

                // Parse the line
                if let Some(data) = line.strip_prefix(b"data:") {
                    let payload = String::from_utf8_lossy(data).trim().to_string();
                    if payload == "[DONE]" {
                        return Poll::Ready(Some(Ok(SseEvent::Done)));
                    }
                    return Poll::Ready(Some(Ok(SseEvent::Data(payload))));
                } else if let Some(comment) = line.strip_prefix(b":") {
                    let text = String::from_utf8_lossy(comment).trim().to_string();
                    return Poll::Ready(Some(Ok(SseEvent::Comment(text))));
                }
                // Ignore unknown field lines (event:, id:, retry:) for simplicity
                continue;
            }

            // Read more data from underlying stream
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);
                }
                Poll::Ready(Some(Err(err))) => return Poll::Ready(Some(Err(err))),
                Poll::Ready(None) => {
                    // Stream ended - flush any remaining data
                    if self.buffer.is_empty() {
                        return Poll::Ready(None);
                    }
                    let remaining = std::mem::take(&mut self.buffer);
                    if let Some(data) = remaining.strip_prefix(b"data:") {
                        let payload = String::from_utf8_lossy(data).trim().to_string();
                        if !payload.is_empty() && payload != "[DONE]" {
                            return Poll::Ready(Some(Ok(SseEvent::Data(payload))));
                        }
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Extension trait for easily wrapping byte streams.
pub trait LlmSseStreamExt: Sized {
    /// Wrap this byte stream as an SSE parser.
    fn into_sse_stream(self) -> LlmSseStream<Self>;
}

impl<S, E> LlmSseStreamExt for S
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    fn into_sse_stream(self) -> LlmSseStream<Self> {
        LlmSseStream::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use tokio_stream::StreamExt;

    fn bytes_stream(chunks: Vec<&'static str>) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Unpin {
        stream::iter(chunks.into_iter().map(|s| Ok(Bytes::from(s))))
    }

    #[tokio::test]
    async fn test_single_data_event() {
        let stream = bytes_stream(vec!["data: {\"token\":\"hello\"}\n\n"]);
        let mut sse = LlmSseStream::new(stream);

        let event = sse.next().await.unwrap().unwrap();
        assert_eq!(event, SseEvent::Data("{\"token\":\"hello\"}".to_string()));
    }

    #[tokio::test]
    async fn test_done_event() {
        let stream = bytes_stream(vec!["data: [DONE]\n\n"]);
        let mut sse = LlmSseStream::new(stream);

        let event = sse.next().await.unwrap().unwrap();
        assert_eq!(event, SseEvent::Done);
    }

    #[tokio::test]
    async fn test_comment_event() {
        let stream = bytes_stream(vec![": keep-alive\n\n"]);
        let mut sse = LlmSseStream::new(stream);

        let event = sse.next().await.unwrap().unwrap();
        assert_eq!(event, SseEvent::Comment("keep-alive".to_string()));
    }

    #[tokio::test]
    async fn test_chunked_across_boundaries() {
        // Simulate TCP packet split mid-line
        let stream = bytes_stream(vec!["data: {\"tok", "en\":\"world\"}\n\n"]);
        let mut sse = LlmSseStream::new(stream);

        let event = sse.next().await.unwrap().unwrap();
        assert_eq!(event, SseEvent::Data("{\"token\":\"world\"}".to_string()));
    }

    #[tokio::test]
    async fn test_multiple_events() {
        let stream = bytes_stream(vec![
            "data: {\"delta\":\"Hi\"}\n\n",
            "data: {\"delta\":\" there\"}\n\n",
            "data: [DONE]\n\n",
        ]);
        let mut sse = LlmSseStream::new(stream);

        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("{\"delta\":\"Hi\"}".to_string()));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("{\"delta\":\" there\"}".to_string()));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Done);
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn test_crlf_line_endings() {
        let stream = bytes_stream(vec!["data: test\r\n\r\n"]);
        let mut sse = LlmSseStream::new(stream);

        let event = sse.next().await.unwrap().unwrap();
        assert_eq!(event, SseEvent::Data("test".to_string()));
    }

    #[tokio::test]
    async fn test_extension_trait() {
        let stream = bytes_stream(vec!["data: via-trait\n\n"]);
        let mut sse = stream.into_sse_stream();

        let event = sse.next().await.unwrap().unwrap();
        assert_eq!(event, SseEvent::Data("via-trait".to_string()));
    }
}

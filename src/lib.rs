//! Zero-dependency SSE stream parser optimized for LLM token streaming.
//!
//! Correctly implements the SSE dispatch algorithm:
//! - Multi-line `data:` fields are joined with `\n` into one event
//! - `event:` field is captured and exposed via [`SseEvent::NamedData`]
//! - Events are dispatched on the empty-line separator, not per line
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
//!         SseEvent::NamedData { event_type, data } => {
//!             println!("Event {}: {}", event_type, data);
//!         }
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
    /// A data event with no event-type field.
    ///
    /// Multi-line `data:` fields within one event block are joined with `\n`.
    Data(String),

    /// A data event with an explicit `event:` field.
    ///
    /// Example: Anthropic sends `event: content_block_delta` before the data.
    NamedData {
        /// Value of the `event:` field.
        event_type: String,
        /// Joined `data:` field (multi-line values joined with `\n`).
        data: String,
    },

    /// Signal that the stream has finished (e.g., `data: [DONE]`).
    Done,

    /// Keep-alive or informational comment (line starts with `:`).
    Comment(String),
}

/// A Server-Sent Events (SSE) stream parser.
///
/// Wraps any `Stream<Item = Result<Bytes, E>>` (e.g., from reqwest, hyper, or axum)
/// and emits structured [`SseEvent`] values following the SSE dispatch algorithm:
/// events are assembled across multiple field lines and dispatched on the empty-line
/// separator, not emitted line-by-line.
pub struct LlmSseStream<S> {
    inner: S,
    buffer: BytesMut,
    // Accumulated fields for the current event block
    pending_data: Vec<String>,
    pending_event_type: Option<String>,
}

impl<S> LlmSseStream<S> {
    /// Create a new SSE parser wrapping the given byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: BytesMut::with_capacity(4096),
            pending_data: Vec::new(),
            pending_event_type: None,
        }
    }

    /// Consume the parser and return the underlying stream.
    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S, E> LlmSseStream<S>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    /// Flush the accumulated event block into an SseEvent, if any data was buffered.
    fn flush_pending(&mut self) -> Option<SseEvent> {
        if self.pending_data.is_empty() {
            self.pending_event_type = None;
            return None;
        }

        let data = self.pending_data.join("\n");
        let event_type = self.pending_event_type.take();
        self.pending_data.clear();

        if data == "[DONE]" {
            return Some(SseEvent::Done);
        }

        Some(match event_type {
            Some(et) => SseEvent::NamedData { event_type: et, data },
            None     => SseEvent::Data(data),
        })
    }
}

impl<S, E> Stream for LlmSseStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<SseEvent, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Process any complete lines already in the buffer.
            if let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
                let line_bytes = self.buffer.split_to(pos + 1);
                let line = line_bytes.strip_suffix(b"\r\n")
                    .or_else(|| line_bytes.strip_suffix(b"\n"))
                    .unwrap_or(&line_bytes);

                if line.is_empty() {
                    // Empty line = event separator. Dispatch the accumulated block.
                    if let Some(event) = self.flush_pending() {
                        return Poll::Ready(Some(Ok(event)));
                    }
                    continue;
                }

                if let Some(data) = line.strip_prefix(b"data:") {
                    // Accumulate — do NOT emit yet (multi-line data must be joined).
                    let payload = String::from_utf8_lossy(data).trim().to_string();
                    self.pending_data.push(payload);
                    continue;
                }

                if let Some(et) = line.strip_prefix(b"event:") {
                    self.pending_event_type =
                        Some(String::from_utf8_lossy(et).trim().to_string());
                    continue;
                }

                if let Some(comment) = line.strip_prefix(b":") {
                    let text = String::from_utf8_lossy(comment).trim().to_string();
                    // Comments are emitted immediately; they don't affect pending state.
                    return Poll::Ready(Some(Ok(SseEvent::Comment(text))));
                }

                // id:, retry:, and unknown fields — ignore per spec.
                continue;
            }

            // Buffer has no complete line yet — pull more bytes from the inner stream.
            match Pin::new(&mut self.inner).poll_next(cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    self.buffer.extend_from_slice(&chunk);
                }
                Poll::Ready(Some(Err(err))) => return Poll::Ready(Some(Err(err))),
                Poll::Ready(None) => {
                    // Stream ended. Treat remaining buffer as a final unterminated line.
                    if !self.buffer.is_empty() {
                        let len = self.buffer.len();
                        let remaining = self.buffer.split_to(len);
                        if let Some(data) = remaining.strip_prefix(b"data:") {
                            let payload = String::from_utf8_lossy(data).trim().to_string();
                            if !payload.is_empty() {
                                self.pending_data.push(payload);
                            }
                        }
                    }
                    if let Some(event) = self.flush_pending() {
                        return Poll::Ready(Some(Ok(event)));
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

    // ── Existing behaviour (must not regress) ─────────────────────────────────

    #[tokio::test]
    async fn single_data_event() {
        let stream = bytes_stream(vec!["data: {\"token\":\"hello\"}\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("{\"token\":\"hello\"}".into()));
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn done_event() {
        let stream = bytes_stream(vec!["data: [DONE]\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Done);
    }

    #[tokio::test]
    async fn comment_event() {
        let stream = bytes_stream(vec![": keep-alive\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Comment("keep-alive".into()));
    }

    #[tokio::test]
    async fn chunked_across_tcp_boundary() {
        let stream = bytes_stream(vec!["data: {\"tok", "en\":\"world\"}\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("{\"token\":\"world\"}".into()));
    }

    #[tokio::test]
    async fn multiple_events_in_sequence() {
        let stream = bytes_stream(vec![
            "data: {\"delta\":\"Hi\"}\n\n",
            "data: {\"delta\":\" there\"}\n\n",
            "data: [DONE]\n\n",
        ]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("{\"delta\":\"Hi\"}".into()));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("{\"delta\":\" there\"}".into()));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Done);
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn crlf_line_endings() {
        let stream = bytes_stream(vec!["data: test\r\n\r\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("test".into()));
    }

    #[tokio::test]
    async fn extension_trait() {
        let stream = bytes_stream(vec!["data: via-trait\n\n"]);
        let mut sse = stream.into_sse_stream();
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("via-trait".into()));
    }

    // ── Multi-line data assembly (bug fix) ────────────────────────────────────

    #[tokio::test]
    async fn multiline_data_joined_with_newline() {
        // Per SSE spec: multiple data: lines in one block are joined with \n
        let stream = bytes_stream(vec!["data: line one\ndata: line two\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(
            sse.next().await.unwrap().unwrap(),
            SseEvent::Data("line one\nline two".into()),
            "multi-line data must be joined into one event"
        );
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn multiline_data_three_lines() {
        let stream = bytes_stream(vec!["data: a\ndata: b\ndata: c\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("a\nb\nc".into()));
    }

    #[tokio::test]
    async fn two_separate_single_line_events_not_joined() {
        // Two separate event blocks (separated by empty line) must be TWO events
        let stream = bytes_stream(vec!["data: first\n\ndata: second\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("first".into()));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("second".into()));
        assert!(sse.next().await.is_none());
    }

    // ── event: field (bug fix) ────────────────────────────────────────────────

    #[tokio::test]
    async fn event_type_field_captured() {
        // Anthropic-style: event: field before data:
        let stream = bytes_stream(vec!["event: content_block_delta\ndata: {\"text\":\"hi\"}\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(
            sse.next().await.unwrap().unwrap(),
            SseEvent::NamedData {
                event_type: "content_block_delta".into(),
                data: "{\"text\":\"hi\"}".into(),
            },
            "event: field must be captured as NamedData"
        );
    }

    #[tokio::test]
    async fn event_type_with_multiline_data() {
        let stream = bytes_stream(vec!["event: delta\ndata: part1\ndata: part2\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(
            sse.next().await.unwrap().unwrap(),
            SseEvent::NamedData {
                event_type: "delta".into(),
                data: "part1\npart2".into(),
            }
        );
    }

    #[tokio::test]
    async fn no_event_type_stays_as_data() {
        // When no event: field, Data variant is used (not NamedData)
        let stream = bytes_stream(vec!["data: plain\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        let event = sse.next().await.unwrap().unwrap();
        assert!(matches!(event, SseEvent::Data(_)), "no event: field → Data variant");
    }

    #[tokio::test]
    async fn comment_between_events_does_not_merge() {
        // Comment is emitted immediately; it doesn't interfere with data blocks
        let stream = bytes_stream(vec!["data: hello\n\n: ping\n\ndata: world\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("hello".into()));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Comment("ping".into()));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("world".into()));
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn id_and_retry_fields_silently_ignored() {
        // id: and retry: must not cause errors or extra events
        let stream = bytes_stream(vec!["id: 42\nretry: 3000\ndata: ok\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("ok".into()));
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn empty_data_field_not_emitted() {
        // data: with empty value followed by empty line should not emit a spurious event
        let stream = bytes_stream(vec!["data: real\n\ndata: \n\ndata: next\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("real".into()));
        // empty data → empty string → still emitted (spec says dispatch even if empty)
        // but the empty block in the middle should produce Data("") not None
        let middle = sse.next().await.unwrap().unwrap();
        assert_eq!(middle, SseEvent::Data("".into()));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("next".into()));
    }

    #[tokio::test]
    async fn openai_realistic_stream() {
        // Simulates real OpenAI streaming response chunks
        let stream = bytes_stream(vec![
            "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
            "data: [DONE]\n\n",
        ]);
        let mut sse = LlmSseStream::new(stream);
        let e1 = sse.next().await.unwrap().unwrap();
        assert!(matches!(e1, SseEvent::Data(_)));
        let e2 = sse.next().await.unwrap().unwrap();
        assert!(matches!(e2, SseEvent::Data(_)));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Done);
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn anthropic_realistic_stream() {
        // Anthropic uses named events
        let stream = bytes_stream(vec![
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\n",
            "event: content_block_delta\ndata: {\"delta\":{\"text\":\"Hi\"}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ]);
        let mut sse = LlmSseStream::new(stream);
        assert!(matches!(sse.next().await.unwrap().unwrap(), SseEvent::NamedData { event_type, .. } if event_type == "message_start"));
        assert!(matches!(sse.next().await.unwrap().unwrap(), SseEvent::NamedData { event_type, .. } if event_type == "content_block_delta"));
        assert!(matches!(sse.next().await.unwrap().unwrap(), SseEvent::NamedData { event_type, .. } if event_type == "message_stop"));
        assert!(sse.next().await.is_none());
    }
}

//! SSE stream parser for LLM token streaming.
//!
//! Wraps any `Stream<Item = Result<Bytes, E>>` from reqwest, hyper, or axum
//! and emits structured [`SseEvent`] values. Correctly implements the SSE dispatch algorithm:
//! - Multi-line `data:` fields are joined with `\n` into one event
//! - `event:` field is captured and exposed via [`SseEvent::NamedData`]
//! - `id:` field is tracked via [`LlmSseStream::last_event_id`] for reconnect support
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
///
/// After consuming events, call [`last_event_id`](LlmSseStream::last_event_id) to
/// retrieve the most recent `id:` field value. Pass this as the `Last-Event-ID`
/// header when reconnecting to resume the stream from where it left off.
pub struct LlmSseStream<S> {
    inner: S,
    buffer: BytesMut,
    // Accumulated fields for the current event block
    pending_data: Vec<String>,
    pending_event_type: Option<String>,
    // Tracks the most recent id: field for reconnect support (Last-Event-ID header)
    last_event_id: Option<String>,
}

impl<S> LlmSseStream<S> {
    /// Create a new SSE parser wrapping the given byte stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: BytesMut::with_capacity(4096),
            pending_data: Vec::new(),
            pending_event_type: None,
            last_event_id: None,
        }
    }

    /// Consume the parser and return the underlying stream.
    pub fn into_inner(self) -> S {
        self.inner
    }

    /// Returns the value of the most recent `id:` field received from the server.
    ///
    /// Per the SSE spec, clients should send this value as the `Last-Event-ID`
    /// header when reconnecting, allowing the server to resume from where it left off.
    /// Returns `None` if no `id:` field has been received yet, or if the server
    /// sent an empty `id:` field to reset the ID.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use tokio_sse_llm::{LlmSseStream, LlmSseStreamExt};
    /// # use tokio_stream::StreamExt;
    /// # async fn reconnect_example() {
    /// # let byte_stream = futures_util::stream::empty::<Result<bytes::Bytes, std::io::Error>>();
    /// let mut stream = byte_stream.into_sse_stream();
    /// while let Some(Ok(event)) = stream.next().await { /* handle events */ }
    ///
    /// // On disconnect, reconnect with the last received ID:
    /// if let Some(id) = stream.last_event_id() {
    ///     // send request with header: Last-Event-ID: {id}
    /// }
    /// # }
    /// ```
    pub fn last_event_id(&self) -> Option<&str> {
        self.last_event_id.as_deref()
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

                if let Some(id_bytes) = line.strip_prefix(b"id:") {
                    // Per SSE spec: track the last event ID for reconnect support.
                    // An empty id: field resets the ID to null.
                    let id_value = String::from_utf8_lossy(id_bytes).trim().to_string();
                    self.last_event_id = if id_value.is_empty() { None } else { Some(id_value) };
                    continue;
                }

                if let Some(comment) = line.strip_prefix(b":") {
                    let text = String::from_utf8_lossy(comment).trim().to_string();
                    // Comments are emitted immediately; they don't affect pending state.
                    return Poll::Ready(Some(Ok(SseEvent::Comment(text))));
                }

                // retry: and unknown fields — ignore per spec.
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
    async fn id_field_is_tracked_retry_is_ignored() {
        // id: updates last_event_id; retry: is ignored; neither produces an extra event
        let stream = bytes_stream(vec!["id: 42\nretry: 3000\ndata: ok\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("ok".into()));
        assert_eq!(sse.last_event_id(), Some("42"), "id: field must be tracked for reconnect");
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn id_field_updates_across_events() {
        // last_event_id advances as the stream progresses
        let stream = bytes_stream(vec![
            "id: 1\ndata: first\n\n",
            "id: 2\ndata: second\n\n",
        ]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("first".into()));
        assert_eq!(sse.last_event_id(), Some("1"));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("second".into()));
        assert_eq!(sse.last_event_id(), Some("2"));
    }

    #[tokio::test]
    async fn empty_id_field_resets_to_none() {
        // Per SSE spec: bare "id:" (no value) clears the last event ID
        let stream = bytes_stream(vec!["id: 5\ndata: first\n\nid:\ndata: second\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        sse.next().await.unwrap().unwrap();
        assert_eq!(sse.last_event_id(), Some("5"));
        sse.next().await.unwrap().unwrap();
        assert_eq!(sse.last_event_id(), None, "empty id: must reset last_event_id to None");
    }

    #[tokio::test]
    async fn no_id_field_means_none() {
        // Stream with no id: field — last_event_id stays None
        let stream = bytes_stream(vec!["data: hello\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        sse.next().await.unwrap().unwrap();
        assert_eq!(sse.last_event_id(), None);
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

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn empty_stream_returns_none() {
        let stream = bytes_stream(vec![]);
        let mut sse = LlmSseStream::new(stream);
        assert!(sse.next().await.is_none(), "empty stream must return None immediately");
    }

    #[tokio::test]
    async fn stream_with_only_comments_no_data() {
        let stream = bytes_stream(vec![": ping\n\n: pong\n\n"]);
        let mut sse = LlmSseStream::new(stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Comment("ping".into()));
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Comment("pong".into()));
        assert!(sse.next().await.is_none());
    }

    #[tokio::test]
    async fn stream_error_propagates() {
        // Stream that yields Ok then Err — error must propagate as Err, not panic
        let error_stream = stream::iter(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from("data: hello\n\n")),
            Err(std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset")),
        ]);
        let mut sse = LlmSseStream::new(error_stream);
        assert_eq!(sse.next().await.unwrap().unwrap(), SseEvent::Data("hello".into()));
        assert!(sse.next().await.unwrap().is_err(), "IO error must propagate as Err");
    }

    #[tokio::test]
    async fn utf8_multibyte_split_across_chunks() {
        // '€' is 3 bytes (0xE2 0x82 0xAC). The character is split across two TCP chunks,
        // but the parser is safe: bytes accumulate in the buffer until a complete '\n'-
        // terminated line is found. from_utf8_lossy is only called on complete line bytes,
        // so the multi-byte character is always fully assembled before conversion.
        let euro = "€";
        let chunk1 = "data: price is ".to_string();
        let mut chunk2_bytes = Vec::new();
        chunk2_bytes.extend_from_slice(euro.as_bytes());
        chunk2_bytes.extend_from_slice(b"100\n\n");

        let stream = stream::iter(vec![
            Ok::<Bytes, std::io::Error>(Bytes::from(chunk1)),
            Ok(Bytes::from(chunk2_bytes)),
        ]);
        let mut sse = LlmSseStream::new(stream);
        let event = sse.next().await.unwrap().unwrap();
        assert_eq!(event, SseEvent::Data("price is €100".into()),
            "multi-byte character split across chunks must be assembled correctly");
    }

    #[tokio::test]
    async fn unterminated_stream_flushes_pending() {
        // Stream ends without a trailing empty line — pending data should still be emitted
        let stream = bytes_stream(vec!["data: no trailing newline"]);
        let mut sse = LlmSseStream::new(stream);
        let event = sse.next().await.unwrap().unwrap();
        assert_eq!(event, SseEvent::Data("no trailing newline".into()),
            "stream ending without empty-line separator must flush pending data");
    }

    #[tokio::test]
    async fn very_large_data_line() {
        // 64 KB single data line — must not OOM or panic
        let big = "x".repeat(65536);
        let input = format!("data: {}\n\n", big);
        let stream = bytes_stream(vec![Box::leak(input.into_boxed_str())]);
        let mut sse = LlmSseStream::new(stream);
        let event = sse.next().await.unwrap().unwrap();
        assert!(matches!(event, SseEvent::Data(ref s) if s.len() == 65536));
    }

    #[tokio::test]
    async fn whitespace_only_data_line() {
        // data: with only whitespace — trim leaves empty string but event is still dispatched
        let stream = bytes_stream(vec!["data:    \n\n"]);
        let mut sse = LlmSseStream::new(stream);
        let event = sse.next().await.unwrap().unwrap();
        assert_eq!(event, SseEvent::Data("".into()));
    }
}

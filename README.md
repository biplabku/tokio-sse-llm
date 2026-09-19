# tokio-sse-llm

[![crates.io](https://img.shields.io/crates/v/tokio-sse-llm.svg)](https://crates.io/crates/tokio-sse-llm)

Correct, zero-dependency SSE stream parser for LLM token streaming.

Correctly implements the SSE dispatch algorithm: multi-line `data:` fields are
assembled into one event, `event:` field is captured, events are dispatched on
the empty-line separator. Works with OpenAI, Anthropic, Ollama, and any
standards-compliant SSE server.

## Installation

```toml
[dependencies]
tokio-sse-llm = "0.2"
tokio-stream = "0.1"   # for StreamExt
```

## Usage

### OpenAI / Ollama (unnamed events)

```rust
use tokio_sse_llm::{LlmSseStream, SseEvent, LlmSseStreamExt};
use tokio_stream::StreamExt;

let response = client
    .post("https://api.openai.com/v1/chat/completions")
    .json(&body)
    .send()
    .await?;

let mut stream = response.bytes_stream().into_sse_stream();

while let Some(event) = stream.next().await {
    match event? {
        SseEvent::Data(json) => println!("{}", json),
        SseEvent::Done => break,
        SseEvent::Comment(_) => {}  // keep-alive
        SseEvent::NamedData { .. } => {}  // won't happen with OpenAI
    }
}
```

### Anthropic (named events)

Anthropic uses the `event:` field (`message_start`, `content_block_delta`, etc.):

```rust
while let Some(event) = stream.next().await {
    match event? {
        SseEvent::NamedData { event_type, data } => {
            match event_type.as_str() {
                "content_block_delta" => println!("delta: {}", data),
                "message_stop" => break,
                _ => {}
            }
        }
        SseEvent::Data(json) => println!("{}", json),
        SseEvent::Done => break,
        SseEvent::Comment(_) => {}
    }
}
```

## SseEvent variants

| Variant | When emitted |
|---------|-------------|
| `Data(String)` | Event block with `data:` but no `event:` field |
| `NamedData { event_type, data }` | Event block with both `data:` and `event:` field |
| `Done` | `data: [DONE]` received |
| `Comment(String)` | Line starting with `:` (keep-alive) |

Multi-line `data:` fields within one event block are joined with `\n`.

## SSE spec compliance

- ✅ Multi-line data joined with `\n`
- ✅ `event:` field captured in `NamedData`
- ✅ Empty line is the event separator (not each data line)
- ✅ `id:` and `retry:` fields silently ignored
- ✅ CRLF and LF line endings
- ✅ Chunks split across TCP packets reassembled correctly
- ✅ Stream errors propagate as `Err`

## CLI

```bash
cargo install tokio-sse-llm --features cli
```

Pipe from curl:
```bash
curl -sN https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}' \
  | sse
```

Options: `--extract openai|anthropic|ollama|auto`, `--raw`, `--verbose`

## License

MIT OR Apache-2.0

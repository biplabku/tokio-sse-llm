# tokio-sse-llm

Zero-dependency SSE stream parser optimized for LLM token streaming.

When building microservices or CLI tools that call OpenAI, Anthropic, or Ollama, you need to stream Server-Sent Events (SSE). This crate provides an efficient stream adapter that wraps any HTTP response body and emits structured events.

## Features

- **Minimal dependencies**: Only `bytes` and `futures-core`
- **Zero-copy parsing**: Efficient buffer management
- **Handles chunked transfers**: Correctly reassembles SSE events split across TCP packets
- **Works with any HTTP client**: reqwest, hyper, axum, etc.

## Usage

```rust
use tokio_sse_llm::{LlmSseStream, SseEvent, LlmSseStreamExt};
use tokio_stream::StreamExt;

// With reqwest
let response = client.post("https://api.openai.com/v1/chat/completions")
    .json(&body)
    .send()
    .await?;

let mut stream = response.bytes_stream().into_sse_stream();

while let Some(event) = stream.next().await {
    match event? {
        SseEvent::Data(json) => {
            // Parse JSON delta, extract token
            println!("{}", json);
        }
        SseEvent::Done => break,
        SseEvent::Comment(_) => {} // Keep-alive
    }
}
```

## CLI

Install with the `cli` feature:

```bash
cargo install tokio-sse-llm --features cli
```

### Pipe from curl

```bash
curl -sN https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}' \
  | sse
```

### Direct request

```bash
sse --url https://api.openai.com/v1/chat/completions \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d '{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}'
```

### Options

- `--extract openai|anthropic|ollama|auto` — extract text from provider-specific JSON
- `--raw` — output raw data lines (no extraction)
- `--verbose` — show event types and comments

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.

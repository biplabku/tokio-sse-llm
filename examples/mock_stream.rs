//! Example: Parsing a mock SSE stream (no API key needed)
//!
//! ```bash
//! cargo run --example mock_stream
//! ```

use bytes::Bytes;
use futures_util::stream;
use tokio_sse_llm::{LlmSseStreamExt, SseEvent};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() {
    // Simulate an SSE stream with chunks split across "packets"
    let chunks = vec![
        "data: {\"delta\":\"Hello\"}\n\n",
        "data: {\"del",  // Split mid-JSON
        "ta\":\" world\"}\n\n",
        ": keepalive\n\n",  // Comment/ping
        "data: [DONE]\n\n",
    ];

    let byte_stream = stream::iter(
        chunks.into_iter().map(|s| Ok::<_, std::io::Error>(Bytes::from(s)))
    );

    let mut sse = byte_stream.into_sse_stream();

    println!("Parsing SSE stream:");
    while let Some(event) = sse.next().await {
        match event.unwrap() {
            SseEvent::Data(json) => println!("  DATA: {json}"),
            SseEvent::Done => {
                println!("  DONE");
                break;
            }
            SseEvent::Comment(c) => println!("  COMMENT: {c}"),
            SseEvent::NamedData { event_type, data } => println!("  EVENT[{event_type}]: {data}"),
        }
    }
}

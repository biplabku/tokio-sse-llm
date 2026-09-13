//! Example: Streaming OpenAI completions with reqwest
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run --example reqwest_openai
//! ```

use tokio_sse_llm::{LlmSseStream, SseEvent};
use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("Set OPENAI_API_KEY environment variable");

    let client = reqwest::Client::new();
    let response = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .body(r#"{
            "model": "gpt-4o-mini",
            "stream": true,
            "messages": [{"role": "user", "content": "Say hello in 3 words"}]
        }"#)
        .send()
        .await?;

    if !response.status().is_success() {
        eprintln!("Error: {}", response.text().await?);
        return Ok(());
    }

    let mut stream = LlmSseStream::new(response.bytes_stream());

    print!("Response: ");
    while let Some(event) = stream.next().await {
        match event? {
            SseEvent::Data(json) => {
                // Parse the delta content from OpenAI's response
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
                        print!("{content}");
                    }
                }
            }
            SseEvent::Done => {
                println!();
                break;
            }
            SseEvent::Comment(_) => {}
        }
    }

    Ok(())
}

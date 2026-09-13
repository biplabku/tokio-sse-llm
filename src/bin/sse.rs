//! CLI for parsing LLM SSE streams.
//!
//! # Examples
//!
//! Parse from stdin (pipe curl):
//! ```bash
//! curl -sN https://api.openai.com/v1/chat/completions \
//!   -H "Authorization: Bearer $OPENAI_API_KEY" \
//!   -d '{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"hi"}]}' \
//!   | sse
//! ```
//!
//! Or make request directly:
//! ```bash
//! sse --url https://api.openai.com/v1/chat/completions \
//!     --header "Authorization: Bearer $OPENAI_API_KEY" \
//!     --data '{"model":"gpt-4o-mini","stream":true,"messages":[...]}'
//! ```

use bytes::Bytes;
use std::io::{self, BufRead};
use tokio_sse_llm::{LlmSseStream, SseEvent};
use tokio_stream::StreamExt;

fn print_usage() {
    eprintln!("sse - Parse LLM SSE streams");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  curl ... | sse [OPTIONS]           Parse SSE from stdin");
    eprintln!("  sse --url URL [OPTIONS]            Make streaming request");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  --url URL         Make POST request to URL");
    eprintln!("  --header HEADER   Add header (repeatable), e.g. 'Authorization: Bearer sk-...'");
    eprintln!("  --data JSON       Request body (implies POST)");
    eprintln!("  --extract FORMAT  Extract text only: openai, anthropic, ollama, or auto (default)");
    eprintln!("  --raw             Output raw data: lines (no extraction)");
    eprintln!("  --verbose         Show event types and comments");
    eprintln!("  --help            Show this help");
}

#[derive(Default)]
struct Args {
    url: Option<String>,
    headers: Vec<(String, String)>,
    data: Option<String>,
    extract: ExtractMode,
    raw: bool,
    verbose: bool,
}

#[derive(Default, Clone, Copy)]
enum ExtractMode {
    #[default]
    Auto,
    OpenAI,
    Anthropic,
    Ollama,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut argv: Vec<String> = std::env::args().skip(1).collect();

    while !argv.is_empty() {
        let arg = argv.remove(0);
        match arg.as_str() {
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            "--url" => {
                args.url = Some(argv.remove(0));
            }
            "--header" | "-H" => {
                let h = argv.remove(0);
                if let Some((k, v)) = h.split_once(':') {
                    args.headers.push((k.trim().to_string(), v.trim().to_string()));
                }
            }
            "--data" | "-d" => {
                args.data = Some(argv.remove(0));
            }
            "--extract" => {
                let mode = argv.remove(0);
                args.extract = match mode.as_str() {
                    "openai" => ExtractMode::OpenAI,
                    "anthropic" => ExtractMode::Anthropic,
                    "ollama" => ExtractMode::Ollama,
                    "auto" => ExtractMode::Auto,
                    _ => return Err(format!("Unknown extract mode: {mode}")),
                };
            }
            "--raw" => args.raw = true,
            "--verbose" | "-v" => args.verbose = true,
            other => return Err(format!("Unknown argument: {other}")),
        }
    }
    Ok(args)
}

fn extract_text(json: &str, mode: ExtractMode) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;

    match mode {
        ExtractMode::OpenAI => {
            v["choices"][0]["delta"]["content"].as_str().map(|s| s.to_string())
        }
        ExtractMode::Anthropic => {
            // Anthropic uses content_block_delta with delta.text
            v["delta"]["text"].as_str().map(|s| s.to_string())
        }
        ExtractMode::Ollama => {
            v["message"]["content"].as_str().map(|s| s.to_string())
        }
        ExtractMode::Auto => {
            // Try each format
            extract_text(json, ExtractMode::OpenAI)
                .or_else(|| extract_text(json, ExtractMode::Anthropic))
                .or_else(|| extract_text(json, ExtractMode::Ollama))
        }
    }
}

async fn process_stream<S, E>(stream: S, args: &Args) -> Result<(), Box<dyn std::error::Error>>
where
    S: futures_core::Stream<Item = Result<Bytes, E>> + Unpin,
    E: std::error::Error + 'static,
{
    let mut sse = LlmSseStream::new(stream);

    while let Some(event) = sse.next().await {
        match event? {
            SseEvent::Data(json) => {
                if args.raw {
                    println!("{json}");
                } else if let Some(text) = extract_text(&json, args.extract) {
                    print!("{text}");
                } else if args.verbose {
                    eprintln!("[data] {json}");
                }
            }
            SseEvent::Done => {
                if args.verbose {
                    eprintln!("\n[done]");
                } else if !args.raw {
                    println!();
                }
                break;
            }
            SseEvent::Comment(c) => {
                if args.verbose {
                    eprintln!("[comment] {c}");
                }
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        print_usage();
        std::process::exit(1);
    });

    if let Some(url) = &args.url {
        // Make HTTP request
        let client = reqwest::Client::new();
        let mut builder = if args.data.is_some() {
            client.post(url)
        } else {
            client.get(url)
        };

        for (k, v) in &args.headers {
            builder = builder.header(k, v);
        }

        if let Some(data) = &args.data {
            builder = builder.header("Content-Type", "application/json").body(data.clone());
        }

        let response = builder.send().await?;

        if !response.status().is_success() {
            eprintln!("HTTP {}: {}", response.status(), response.text().await?);
            std::process::exit(1);
        }

        process_stream(response.bytes_stream(), &args).await?;
    } else {
        // Read from stdin
        let stdin = io::stdin();
        let lines: Vec<String> = stdin.lock().lines().map_while(Result::ok).collect();
        let data = lines.join("\n");

        let stream = futures_util::stream::iter(vec![Ok::<_, io::Error>(Bytes::from(data))]);

        process_stream(stream, &args).await?;
    }

    Ok(())
}

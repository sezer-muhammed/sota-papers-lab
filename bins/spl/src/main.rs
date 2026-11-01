//! `spl`: one binary for every front door.
//!
//!   spl serve                       universal HTTP API (+ web shell)
//!   spl ops                         list every registered op
//!   spl run <op> [--input JSON]     run an op in-process and record it
//!   spl put <file>                  store a file as an artifact, print its ref

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use spl_core::ArtifactKind;
use spl_host::{Config, Host};

#[derive(Parser)]
#[command(name = "spl", version, about = "SOTA Papers Lab")]
struct Cli {
    /// TOML config; `SPL_*` environment variables override it.
    #[arg(long, global = true, env = "SPL_CONFIG")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the universal API and, if configured, the web shell.
    Serve,
    /// List registered ops with their summaries.
    Ops {
        /// Print full descriptors, including JSON Schemas.
        #[arg(long)]
        json: bool,
    },
    /// Run one op and print its run record.
    Run {
        op: String,
        /// Op input as JSON.
        #[arg(long, default_value = "{}")]
        input: String,
    },
    /// Store a file as a `blob` artifact, recording its file name, and print
    /// the artifact reference (its `id` is what op inputs take).
    Put {
        file: PathBuf,
        /// Media type; guessed from the extension when omitted.
        #[arg(long)]
        media_type: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let config = Config::load(cli.config.as_deref())?;
    let registry = spl_catalog::registry();

    match cli.command {
        Command::Ops { json } => {
            let ops = registry.describe();
            if json {
                println!("{}", serde_json::to_string_pretty(&ops)?);
            } else {
                for op in ops {
                    println!("{:<32} {}", op.id, op.summary);
                }
            }
        }
        Command::Run { op, input } => {
            let host = Host::open(&config, registry).await?;
            let run = host.run(&op, serde_json::from_str(&input)?).await?;
            println!("{}", serde_json::to_string_pretty(&run)?);
        }
        Command::Put { file, media_type } => {
            let bytes = std::fs::read(&file).with_context(|| format!("read {}", file.display()))?;
            let media_type = media_type.unwrap_or_else(|| guess_media_type(&file).to_owned());
            let meta = file
                .file_name()
                .map(|name| serde_json::json!({ "filename": name.to_string_lossy() }));
            let host = Host::open(&config, registry).await?;
            let artifact = host
                .put_artifact(bytes, ArtifactKind::Blob, &media_type, meta)
                .await?;
            println!("{}", serde_json::to_string_pretty(&artifact)?);
        }
        Command::Serve => {
            if config.auth_token.is_none() && !is_private_bind(&config.bind) {
                tracing::warn!(bind = %config.bind, "serving without SPL_AUTH_TOKEN on a non-loopback, non-tailnet address");
            }
            let host = Host::open(&config, registry).await?;
            let interrupted = host.fail_interrupted_runs().await?;
            if interrupted > 0 {
                tracing::warn!(
                    "marked {interrupted} unfinished runs of a previous process as failed"
                );
            }
            let app = spl_host::router(host, &config);
            let listener = tokio::net::TcpListener::bind(&config.bind).await?;
            tracing::info!(
                "SOTA Papers Lab listening on http://{}",
                listener.local_addr()?
            );
            axum_serve(listener, app).await?;
        }
    }
    Ok(())
}

/// Media types for the files people feed this lab; anything else is a
/// plain byte stream.
fn guess_media_type(path: &Path) -> &'static str {
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("arw") => "image/x-sony-arw",
        Some("dng") => "image/x-adobe-dng",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("tif" | "tiff") => "image/tiff",
        Some("webp") => "image/webp",
        Some("xmp") => "application/rdf+xml",
        Some("json") => "application/json",
        Some("txt") => "text/plain",
        _ => "application/octet-stream",
    }
}

/// Loopback, or Tailscale's 100.64.0.0/10 range: reachable only from this
/// machine or from devices on the tailnet.
fn is_private_bind(bind: &str) -> bool {
    match bind.parse::<std::net::SocketAddr>() {
        Ok(std::net::SocketAddr::V4(addr)) => {
            let ip = addr.ip();
            let [a, b, ..] = ip.octets();
            ip.is_loopback() || (a == 100 && (64..128).contains(&b))
        }
        Ok(std::net::SocketAddr::V6(addr)) => addr.ip().is_loopback(),
        Err(_) => false,
    }
}

async fn axum_serve(listener: tokio::net::TcpListener, app: axum::Router) -> std::io::Result<()> {
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
}

/// Ctrl-C, or SIGTERM from `docker stop`: as PID 1 in the container, `spl`
/// would otherwise ignore SIGTERM and be killed after Docker's timeout.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut stream) => {
                stream.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        () = ctrl_c => {}
        () = terminate => {}
    }
    tracing::info!("shutting down");
}

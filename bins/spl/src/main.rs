//! `spl`: one binary for every front door.
//!
//!   spl serve                       universal HTTP API (+ web shell)
//!   spl ops                         list every registered op
//!   spl run <op> [--input JSON]     run an op in-process and record it

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
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
        Command::Serve => {
            if config.auth_token.is_none() && !config.bind.starts_with("127.") {
                tracing::warn!(bind = %config.bind, "serving without SPL_AUTH_TOKEN on a non-loopback address");
            }
            let host = Host::open(&config, registry).await?;
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

async fn axum_serve(listener: tokio::net::TcpListener, app: axum::Router) -> std::io::Result<()> {
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
}

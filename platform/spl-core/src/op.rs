use std::future::Future;
use std::pin::Pin;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use ts_rs::TS;

use crate::{Result, RunCx};

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One typed unit of work, addressed by a stable ID such as `imgcodec.encode`.
///
/// Inputs and outputs are plain serde types. Large data travels as
/// [`ArtifactRef`](crate::ArtifactRef)s, never inline, so the same op runs
/// in-process, over HTTP, inside a pipeline or under the benchmark harness.
pub trait Op: Send + Sync + 'static {
    /// `<app>.<verb>`, lowercase, dot-separated. Never reuse a retired ID.
    const ID: &'static str;
    /// One line for listings and auto-generated UI.
    const SUMMARY: &'static str;

    type Input: DeserializeOwned + JsonSchema + Send;
    type Output: Serialize + JsonSchema + Send;

    fn run(
        &self,
        cx: &RunCx,
        input: Self::Input,
    ) -> impl Future<Output = Result<Self::Output>> + Send;
}

/// What a host needs to list an op and render a form for it.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(export)]
pub struct OpDescriptor {
    pub id: String,
    pub summary: String,
    #[ts(type = "Record<string, unknown>")]
    pub input_schema: serde_json::Value,
    #[ts(type = "Record<string, unknown>")]
    pub output_schema: serde_json::Value,
}

/// Type-erased [`Op`] so a registry can hold every app's ops side by side.
pub trait DynOp: Send + Sync {
    fn descriptor(&self) -> OpDescriptor;
    fn run_json<'a>(
        &'a self,
        cx: &'a RunCx,
        input: serde_json::Value,
    ) -> BoxFuture<'a, Result<serde_json::Value>>;
}

impl<T: Op> DynOp for T {
    fn descriptor(&self) -> OpDescriptor {
        OpDescriptor {
            id: T::ID.to_owned(),
            summary: T::SUMMARY.to_owned(),
            input_schema: schemars::schema_for!(T::Input).to_value(),
            output_schema: schemars::schema_for!(T::Output).to_value(),
        }
    }

    fn run_json<'a>(
        &'a self,
        cx: &'a RunCx,
        input: serde_json::Value,
    ) -> BoxFuture<'a, Result<serde_json::Value>> {
        Box::pin(async move {
            let input: T::Input = serde_json::from_value(input)
                .map_err(|e| anyhow::anyhow!("invalid input for {}: {e}", T::ID))?;
            let output = self.run(cx, input).await?;
            Ok(serde_json::to_value(output)?)
        })
    }
}

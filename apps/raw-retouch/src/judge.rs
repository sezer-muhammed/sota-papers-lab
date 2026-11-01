//! Claude as analyst, look planner and pairwise judge, through the
//! subscription-backed `claude -p` CLI. Never the numeric optimizer.
//! Owner: judge.
//!
//! Every request runs a fresh CLI process in an empty temporary directory
//! with no tools, settings, MCP servers, skills or saved session, and comes
//! back as an object the CLI has validated against a JSON schema. Judging
//! shows images under neutral labels only, never slider values or how a
//! candidate was made, and asks twice with the order reversed so position
//! bias cancels out. Verdicts are cached on disk, so a re-run costs nothing.

use std::collections::HashMap;
use std::ffi::OsString;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use base64::Engine as _;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use spl_core::{Context, Result, anyhow, bail};
use tokio::io::AsyncWriteExt as _;
use tokio::process::Command;
use tokio::sync::Semaphore;
use ts_rs::TS;

use crate::encode;
use crate::types::{Params, RawInfo, Rgb8Image, Stats};
use crate::xmp;

#[derive(Debug, Clone)]
pub struct JudgeOptions {
    /// `claude --model` alias or full name ("opus", "sonnet", ...).
    pub model: String,
    /// `claude --effort` (low, medium, high, xhigh, max); None = CLI default.
    pub effort: Option<String>,
    /// Per call, including retries.
    pub timeout_s: u64,
    /// Verdicts are cached here by a hash of model, prompt, schema and images.
    pub cache_dir: Option<PathBuf>,
}

/// One image shown to Claude. `label` is the caller's name for it, used in
/// logs and errors only: prompts show images under neutral letters by
/// position ("A", "B", ...), so a label such as "vivid" cannot sway a verdict.
pub struct JudgeImage {
    pub label: String,
    pub jpeg: Vec<u8>,
}

impl JudgeImage {
    /// Long edge, in pixels, of every image Claude judges.
    pub const EDGE: usize = 1280;
    /// JPEG quality of every image Claude judges.
    pub const QUALITY: u8 = 88;

    /// Prepares an upright render the way every judged image must be:
    /// candidates differ only in their pixels, never in size or quality.
    pub fn from_rgb(label: impl Into<String>, upright: &Rgb8Image) -> Result<Self> {
        let small = encode::resize(upright, Self::EDGE);
        Ok(Self {
            label: label.into(),
            jpeg: encode::jpeg(&small, Self::QUALITY)?,
        })
    }
}

/// What Claude is told about the photo besides the pixels.
#[derive(Debug, Clone, Serialize)]
pub struct PhotoBrief {
    /// "Sony ILCE-7M3, FE 16-35mm F2.8 GM"
    pub camera: String,
    /// "1/250 s, f/4, ISO 400, 35 mm"
    pub exposure: String,
    /// Histogram percentiles and clipping of the neutral render.
    pub stats: serde_json::Value,
}

impl PhotoBrief {
    /// The brief for one photo: body and lens, shooting settings, and the
    /// neutral render's measurements with the as-shot white balance, which
    /// the planner needs to place its temperature and tint.
    pub fn new(info: &RawInfo, neutral: &Stats) -> Self {
        let mut stats = serde_json::to_value(neutral).unwrap_or(Value::Null);
        if let Value::Object(map) = &mut stats {
            map.insert(
                "as_shot_temperature".into(),
                json!(info.as_shot_temperature.round()),
            );
            map.insert("as_shot_tint".into(), json!(info.as_shot_tint.round()));
            map.insert(
                "raw_clipped_fraction".into(),
                json!(info.raw_clipped_fraction),
            );
        }
        Self {
            camera: camera_line(info),
            exposure: exposure_line(info),
            stats,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, TS)]
#[ts(export, export_to = "raw-retouch/")]
pub struct PlannedLook {
    pub name: String,
    pub rationale: String,
    /// Absolute Lightroom values, clamped to the bounds given to `plan`.
    pub params: Params,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanVerdict {
    /// One or two sentences on the scene and what it needs.
    pub scene: String,
    pub looks: Vec<PlannedLook>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankVerdict {
    /// Candidate indices, best first (Borda count over the order-swapped calls).
    pub order: Vec<usize>,
    pub winner: usize,
    /// Share of calls whose first choice was the winner (0 to 1).
    pub agreement: f32,
    pub reasons: String,
    pub scene: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineVerdict {
    /// Absolute Lightroom values, clamped to the bounds given to `refine`.
    pub params: Params,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareVerdict {
    /// True only when B won in both orders.
    pub prefer_b: bool,
    pub votes_for_b: u32,
    pub votes: u32,
    pub reasons: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JudgeUsage {
    pub calls: u32,
    pub cache_hits: u32,
    /// What the CLI reports as `total_cost_usd` (subscription usage, not a bill).
    pub cost_usd: f64,
    pub seconds: f64,
}

const SYSTEM_PROMPT: &str = include_str!("../prompts/system.txt");
const PLAN_PROMPT: &str = include_str!("../prompts/plan.txt");
const RANK_PROMPT: &str = include_str!("../prompts/rank.txt");
const REFINE_PROMPT: &str = include_str!("../prompts/refine.txt");
const COMPARE_PROMPT: &str = include_str!("../prompts/compare.txt");

/// CLI processes running at once across the whole process: enough for both
/// orders of a comparison to overlap, few enough for the subscription's
/// rate limits when several photos are judged together.
static CLI_SLOTS: Semaphore = Semaphore::const_new(3);

/// Changes whenever a cached verdict would no longer answer the same request.
const CACHE_VERSION: &str = "spl-judge-1";

const DEFAULT_TIMEOUT_S: u64 = 300;

const LABELS: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];

pub struct Claude {
    bin: PathBuf,
    opts: JudgeOptions,
    usage: Mutex<JudgeUsage>,
}

impl Claude {
    /// Finds the CLI ($SPL_CLAUDE_BIN, then PATH, then ~/.local/bin/claude).
    pub fn locate(opts: JudgeOptions) -> Result<Self> {
        let bin = find_bin(
            std::env::var_os("SPL_CLAUDE_BIN"),
            std::env::var_os("PATH"),
            std::env::var_os("HOME").map(PathBuf::from),
        )?;
        Ok(Self::with_bin(bin, opts))
    }

    fn with_bin(bin: PathBuf, opts: JudgeOptions) -> Self {
        Self {
            bin,
            opts,
            usage: Mutex::new(JudgeUsage::default()),
        }
    }

    /// Reads the neutral and auto renders plus stats; proposes up to three
    /// visibly different looks inside `lo`..`hi`.
    pub async fn plan(
        &self,
        photo: &PhotoBrief,
        neutral: &JudgeImage,
        auto: &JudgeImage,
        auto_params: &Params,
        lo: &Params,
        hi: &Params,
    ) -> Result<PlanVerdict> {
        check_images(&[neutral, auto])?;
        let ranges = Ranges::new(lo, hi);
        let stats = compact(&photo.stats).to_string();
        let text = fill(
            PLAN_PROMPT,
            &[
                ("camera", &photo.camera),
                ("shot", &photo.exposure),
                ("auto", &describe(auto_params)),
                ("stats", &stats),
                ("bounds", &ranges.describe()),
            ],
        );
        let images = [
            ("Image \"Neutral\":".to_owned(), neutral.jpeg.as_slice()),
            ("Image \"Auto\":".to_owned(), auto.jpeg.as_slice()),
        ];
        let blocks = with_images(&text, &images);
        let reply: PlanReply = self
            .ask("plan", &plan_schema(&ranges), &blocks, |r: &PlanReply| {
                if r.looks.is_empty() {
                    bail!("no looks proposed");
                }
                Ok(())
            })
            .await?;
        let looks = reply
            .looks
            .into_iter()
            .take(3)
            .enumerate()
            .map(|(i, look)| PlannedLook {
                name: non_empty(look.name, || format!("Look {}", i + 1)),
                rationale: look.rationale.trim().to_owned(),
                params: ranges.fit(look.settings.values()),
            })
            .collect();
        Ok(PlanVerdict {
            scene: reply.analysis.trim().to_owned(),
            looks,
        })
    }

    /// Ranks 2 to 8 candidates: two calls with the order reversed, Borda count.
    ///
    /// Ties go to the lower index, so order `candidates` by your own
    /// preference. If one of the two calls fails, the other decides alone.
    pub async fn rank(&self, photo: &PhotoBrief, candidates: &[JudgeImage]) -> Result<RankVerdict> {
        let n = candidates.len();
        match n {
            0 => bail!("rank needs at least one candidate"),
            1 => {
                return Ok(RankVerdict {
                    order: vec![0],
                    winner: 0,
                    agreement: 1.0,
                    reasons: "Only one candidate.".into(),
                    scene: None,
                });
            }
            _ if n > LABELS.len() => {
                bail!("rank takes at most {} candidates, got {n}", LABELS.len())
            }
            _ => {}
        }
        check_images(&candidates.iter().collect::<Vec<_>>())?;
        let forward: Vec<usize> = (0..n).collect();
        let backward: Vec<usize> = (0..n).rev().collect();
        let (first, second) = tokio::join!(
            self.rank_once(photo, candidates, &forward),
            self.rank_once(photo, candidates, &backward)
        );
        let calls = settle("rank", [first, second])?;
        let rankings: Vec<Vec<usize>> = calls.iter().map(|c| c.order.clone()).collect();
        let (order, agreement) = borda(n, &rankings);
        let winner = order[0];
        let reasons = match calls.iter().find(|c| c.order[0] == winner) {
            Some(call) => call.reasons.clone(),
            None => {
                let notes: Vec<&str> = calls
                    .iter()
                    .filter_map(|c| c.notes.get(&winner).map(String::as_str))
                    .collect();
                format!(
                    "Best on aggregate across both orders, though first in neither. {}",
                    notes.join(" ")
                )
                .trim()
                .to_owned()
            }
        };
        let scene = calls
            .iter()
            .map(|c| c.scene.as_str())
            .find(|s| !s.is_empty())
            .map(str::to_owned);
        Ok(RankVerdict {
            order,
            winner,
            agreement,
            reasons,
            scene,
        })
    }

    /// One targeted improvement of the current best, inside `lo`..`hi`.
    ///
    /// Sliders Claude leaves alone keep their exact incoming values, so
    /// `params` comes back unchanged when Claude finds nothing to improve.
    pub async fn refine(
        &self,
        photo: &PhotoBrief,
        current: &JudgeImage,
        params: &Params,
        lo: &Params,
        hi: &Params,
    ) -> Result<RefineVerdict> {
        check_images(&[current])?;
        let ranges = Ranges::new(lo, hi);
        let text = fill(
            REFINE_PROMPT,
            &[
                ("params", &describe(params)),
                ("camera", &photo.camera),
                ("shot", &photo.exposure),
                ("bounds", &ranges.describe()),
            ],
        );
        let blocks = with_images(
            &text,
            &[("Current edit:".to_owned(), current.jpeg.as_slice())],
        );
        let reply: RefineReply = self
            .ask(
                "refine",
                &refine_schema(&ranges),
                &blocks,
                |_: &RefineReply| Ok(()),
            )
            .await?;
        let rationale = [reply.assessment.trim(), reply.change.trim()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(RefineVerdict {
            params: ranges.adjust(params, reply.settings.values()),
            rationale,
        })
    }

    /// Blind A/B in both orders.
    pub async fn compare(
        &self,
        photo: &PhotoBrief,
        a: &JudgeImage,
        b: &JudgeImage,
    ) -> Result<CompareVerdict> {
        check_images(&[a, b])?;
        let (ab, ba) = tokio::join!(
            self.compare_once(photo, a, b),
            self.compare_once(photo, b, a)
        );
        if let (Err(e), Err(_)) = (&ab, &ba) {
            return Err(anyhow!("{e:#}").context("claude compare: both orders failed"));
        }
        let choice = |r: &Result<CompareReply>| r.as_ref().ok().map(|r| r.better.clone());
        let (votes, votes_for_b) = tally(choice(&ab).as_deref(), choice(&ba).as_deref());
        // The assessments describe the images by appearance, so they read
        // the same whichever order they came from.
        let mut reasons = vec![format!("b won {votes_for_b} of {votes} orders.")];
        for (reply, order) in [(&ab, "a then b"), (&ba, "b then a")] {
            match reply {
                Ok(r) => {
                    if !r.assessment.trim().is_empty() {
                        reasons.push(r.assessment.trim().to_owned());
                    }
                }
                Err(e) => {
                    tracing::warn!(order, error = %format!("{e:#}"), "one compare order failed");
                    reasons.push(format!("Shown {order}: no verdict ({e:#})."));
                }
            }
        }
        Ok(CompareVerdict {
            prefer_b: votes == 2 && votes_for_b == 2,
            votes_for_b,
            votes,
            reasons: reasons.join(" "),
        })
    }

    pub fn usage(&self) -> JudgeUsage {
        self.lock_usage().clone()
    }

    async fn rank_once(
        &self,
        photo: &PhotoBrief,
        candidates: &[JudgeImage],
        order: &[usize],
    ) -> Result<RankCall> {
        let labels = &LABELS[..order.len()];
        let (schema, blocks) = rank_request(photo, candidates, order);
        let reply: RankReply = self
            .ask("rank", &schema, &blocks, |r: &RankReply| {
                full_ranking(&r.ranking, labels).map(|_| ())
            })
            .await?;
        let shown = full_ranking(&reply.ranking, labels)?;
        let mut notes = HashMap::new();
        for note in &reply.notes {
            if let Some(pos) = label_position(labels, &note.label) {
                notes
                    .entry(order[pos])
                    .or_insert_with(|| note.note.trim().to_owned());
            }
        }
        Ok(RankCall {
            order: shown.into_iter().map(|p| order[p]).collect(),
            reasons: reply.reasons.trim().to_owned(),
            scene: reply.scene.trim().to_owned(),
            notes,
        })
    }

    async fn compare_once(
        &self,
        photo: &PhotoBrief,
        first: &JudgeImage,
        second: &JudgeImage,
    ) -> Result<CompareReply> {
        let text = fill(
            COMPARE_PROMPT,
            &[("camera", &photo.camera), ("shot", &photo.exposure)],
        );
        let images = [
            ("Image A:".to_owned(), first.jpeg.as_slice()),
            ("Image B:".to_owned(), second.jpeg.as_slice()),
        ];
        let blocks = with_images(&text, &images);
        self.ask("compare", &compare_schema(), &blocks, |r: &CompareReply| {
            let b = r.better.trim();
            if ["A", "B", "tie"].iter().any(|c| b.eq_ignore_ascii_case(c)) {
                Ok(())
            } else {
                bail!("unexpected choice {b:?}")
            }
        })
        .await
    }

    /// One request: from the cache when possible, otherwise through the CLI
    /// with one retry, all within the timeout (counted from when a CLI slot
    /// frees up, so queueing behind other photos does not use it up).
    async fn ask<T: DeserializeOwned>(
        &self,
        what: &str,
        schema: &Value,
        blocks: &[Block<'_>],
        check: impl Fn(&T) -> Result<()>,
    ) -> Result<T> {
        let schema = schema.to_string();
        let key = self.cache_key(&schema, blocks);
        if let Some(output) = self.cache_get(&key)
            && let Ok(reply) = serde_json::from_value::<T>(output)
            && check(&reply).is_ok()
        {
            self.lock_usage().cache_hits += 1;
            tracing::debug!(what, key, "claude verdict from cache");
            return Ok(reply);
        }
        let input = input_line(blocks);
        let _slot = CLI_SLOTS
            .acquire()
            .await
            .context("claude CLI slots closed")?;
        let timeout = Duration::from_secs(match self.opts.timeout_s {
            0 => DEFAULT_TIMEOUT_S,
            s => s,
        });
        let deadline = Instant::now() + timeout;
        let mut failure = anyhow!("no time left to call claude");
        for attempt in 1..=2 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                break;
            }
            let started = Instant::now();
            let run = tokio::time::timeout(left, self.run_cli(&schema, &input)).await;
            let seconds = started.elapsed().as_secs_f64();
            let error = match run {
                Err(_) => {
                    // The child is killed on drop; the budget is spent, so no retry.
                    self.record_call(seconds, 0.0);
                    failure = anyhow!("timed out after {} s", timeout.as_secs());
                    tracing::warn!(what, attempt, "claude call timed out");
                    break;
                }
                Ok(Err(e)) => {
                    self.record_call(seconds, 0.0);
                    e
                }
                Ok(Ok(reply)) => {
                    self.record_call(seconds, reply.cost_usd);
                    let parsed = reply.output().and_then(|output| {
                        let parsed = serde_json::from_value::<T>(output.clone())
                            .context("reply does not match the schema")?;
                        check(&parsed)?;
                        Ok((output, parsed))
                    });
                    match parsed {
                        Ok((output, parsed)) => {
                            tracing::info!(
                                what,
                                model = %self.opts.model,
                                seconds,
                                cost_usd = reply.cost_usd,
                                "claude verdict"
                            );
                            self.cache_put(&key, &output, reply.cost_usd, seconds);
                            return Ok(parsed);
                        }
                        Err(e) => e,
                    }
                }
            };
            tracing::warn!(what, attempt, error = %format!("{error:#}"), "claude call failed");
            failure = error;
            if attempt == 1 {
                let pause =
                    Duration::from_secs(1).min(deadline.saturating_duration_since(Instant::now()));
                tokio::time::sleep(pause).await;
            }
        }
        Err(failure.context(format!("claude {what} ({})", self.opts.model)))
    }

    async fn run_cli(&self, schema: &str, input: &str) -> Result<CliReply> {
        let dir = tempfile::Builder::new()
            .prefix("spl-judge-")
            .tempdir()
            .context("create an empty working directory for claude")?;
        let mut child = self
            .command(schema, dir.path())
            .spawn()
            .with_context(|| format!("start {}", self.bin.display()))?;
        let mut stdin = child.stdin.take().context("claude stdin")?;
        let feed = async move {
            let written = stdin.write_all(input.as_bytes()).await;
            // Closing stdin is what tells the CLI the request is complete.
            drop(stdin);
            written
        };
        let (fed, output) = tokio::join!(feed, child.wait_with_output());
        let output = output.context("wait for claude")?;
        if let Some(reply) = parse_result(&String::from_utf8_lossy(&output.stdout)) {
            return Ok(reply);
        }
        let mut msg = format!("claude exited ({}) without a result", output.status);
        if let Err(e) = fed {
            msg.push_str(&format!("; sending the request failed: {e}"));
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let detail = if stderr.trim().is_empty() {
            stdout
        } else {
            stderr
        };
        let detail = tail(detail.trim(), 800);
        if !detail.is_empty() {
            msg.push_str(": ");
            msg.push_str(detail);
        }
        Err(anyhow!(msg))
    }

    fn command(&self, schema: &str, cwd: &Path) -> Command {
        let mut cmd = Command::new(&self.bin);
        cmd.args([
            "-p",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
        ])
        .arg("--verbose")
        // No tools, MCP servers, settings files, skills or saved session:
        // the answer depends on this request alone.
        .args([
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
        ])
        .args(["--no-session-persistence", "--disable-slash-commands"]);
        let model = self.opts.model.trim();
        if !model.is_empty() {
            cmd.args(["--model", model]);
        }
        if let Some(effort) = self.opts.effort.as_deref().map(str::trim)
            && !effort.is_empty()
        {
            cmd.args(["--effort", effort]);
        }
        cmd.args([
            "--system-prompt",
            SYSTEM_PROMPT.trim(),
            "--json-schema",
            schema,
        ])
        .current_dir(cwd)
        // An API key would move the CLI from the subscription to paid API billing.
        .env_remove("ANTHROPIC_API_KEY")
        // Marks a parent Claude Code session (a server started from one), under
        // which the CLI may refuse to start.
        .env_remove("CLAUDECODE")
        .env("DISABLE_AUTOUPDATER", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
        cmd
    }

    fn lock_usage(&self) -> MutexGuard<'_, JudgeUsage> {
        self.usage
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn record_call(&self, seconds: f64, cost_usd: f64) {
        let mut usage = self.lock_usage();
        usage.calls += 1;
        usage.seconds += seconds;
        usage.cost_usd += cost_usd;
    }

    fn cache_key(&self, schema: &str, blocks: &[Block<'_>]) -> String {
        let mut hasher = blake3::Hasher::new();
        let mut field = |tag: &str, bytes: &[u8]| {
            hasher.update(tag.as_bytes());
            hasher.update(&(bytes.len() as u64).to_le_bytes());
            hasher.update(bytes);
        };
        field("version", CACHE_VERSION.as_bytes());
        field("model", self.opts.model.trim().as_bytes());
        field(
            "effort",
            self.opts.effort.as_deref().unwrap_or("").trim().as_bytes(),
        );
        field("system", SYSTEM_PROMPT.as_bytes());
        field("schema", schema.as_bytes());
        for block in blocks {
            match block {
                Block::Text(text) => field("text", text.as_bytes()),
                Block::Image(jpeg) => field("image", jpeg),
            }
        }
        hasher.finalize().to_hex().to_string()
    }

    fn cache_path(&self, key: &str) -> Option<PathBuf> {
        Some(self.opts.cache_dir.as_ref()?.join(format!("{key}.json")))
    }

    fn cache_get(&self, key: &str) -> Option<Value> {
        let bytes = std::fs::read(self.cache_path(key)?).ok()?;
        let mut entry: Value = serde_json::from_slice(&bytes).ok()?;
        Some(entry.get_mut("output")?.take())
    }

    fn cache_put(&self, key: &str, output: &Value, cost_usd: f64, seconds: f64) {
        let Some(path) = self.cache_path(key) else {
            return;
        };
        let entry = json!({
            "model": self.opts.model,
            "effort": self.opts.effort,
            "cost_usd": cost_usd,
            "seconds": seconds,
            "output": output,
        });
        if let Err(e) = write_atomic(&path, entry.to_string().as_bytes()) {
            tracing::warn!(path = %path.display(), error = %e, "cannot cache a claude verdict");
        }
    }
}

/// Resolves the CLI: an explicit `SPL_CLAUDE_BIN` must exist; otherwise the
/// first `claude` on PATH, then the installer's default location.
fn find_bin(
    explicit: Option<OsString>,
    path: Option<OsString>,
    home: Option<PathBuf>,
) -> Result<PathBuf> {
    if let Some(bin) = explicit.filter(|b| !b.is_empty()) {
        let bin = PathBuf::from(bin);
        if is_executable(&bin) {
            return Ok(bin);
        }
        bail!(
            "SPL_CLAUDE_BIN is {} but that is not an executable file",
            bin.display()
        );
    }
    let on_path = path
        .iter()
        .flat_map(std::env::split_paths)
        .filter(|dir| !dir.as_os_str().is_empty())
        .map(|dir| dir.join("claude"));
    let installed = home.map(|h| h.join(".local/bin/claude"));
    on_path
        .chain(installed)
        .find(|bin| is_executable(bin))
        .context(
            "claude CLI not found on PATH or in ~/.local/bin: install Claude Code, \
             log in with `claude`, or set SPL_CLAUDE_BIN",
        )
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        path.metadata()
            .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir)?;
    tmp.write_all(bytes)?;
    tmp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

/// The `"type":"result"` line of the CLI's stream-json output.
#[derive(Debug, Default)]
struct CliReply {
    subtype: String,
    is_error: bool,
    result: String,
    structured: Option<Value>,
    cost_usd: f64,
}

impl CliReply {
    /// The schema-validated object, or why there is none.
    fn output(&self) -> Result<Value> {
        if self.is_error || self.subtype != "success" {
            bail!(
                "claude reported {}{}: {}",
                self.subtype,
                if self.is_error { " (error)" } else { "" },
                head(self.result.trim(), 800)
            );
        }
        match &self.structured {
            Some(output) if output.is_object() => Ok(output.clone()),
            _ => bail!(
                "claude returned no structured output: {}",
                head(self.result.trim(), 800)
            ),
        }
    }
}

fn parse_result(stdout: &str) -> Option<CliReply> {
    stdout.lines().rev().find_map(|line| {
        let line = line.trim();
        if !line.starts_with('{') {
            return None;
        }
        let v: Value = serde_json::from_str(line).ok()?;
        if v.get("type")?.as_str()? != "result" {
            return None;
        }
        Some(CliReply {
            subtype: v["subtype"].as_str().unwrap_or_default().to_owned(),
            is_error: v["is_error"].as_bool().unwrap_or(false),
            result: v["result"].as_str().unwrap_or_default().to_owned(),
            structured: v.get("structured_output").filter(|s| !s.is_null()).cloned(),
            cost_usd: v["total_cost_usd"].as_f64().unwrap_or(0.0),
        })
    })
}

/// A piece of the user message: text, or a JPEG.
enum Block<'a> {
    Text(String),
    Image(&'a [u8]),
}

/// The prompt with each image, under its caption, where it says `{images}`.
fn with_images<'a>(text: &str, images: &[(String, &'a [u8])]) -> Vec<Block<'a>> {
    let (before, after) = text.split_once("{images}").unwrap_or((text, ""));
    let mut blocks = Vec::new();
    if !before.trim().is_empty() {
        blocks.push(Block::Text(before.trim().to_owned()));
    }
    for (caption, jpeg) in images {
        blocks.push(Block::Text(caption.clone()));
        blocks.push(Block::Image(jpeg));
    }
    if !after.trim().is_empty() {
        blocks.push(Block::Text(after.trim().to_owned()));
    }
    blocks
}

/// The one stream-json line the CLI reads from stdin.
fn input_line(blocks: &[Block<'_>]) -> String {
    let content: Vec<Value> = blocks
        .iter()
        .map(|block| match block {
            Block::Text(text) => json!({"type": "text", "text": text}),
            Block::Image(jpeg) => json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64::engine::general_purpose::STANDARD.encode(jpeg),
                },
            }),
        })
        .collect();
    let mut line = json!({
        "type": "user",
        "message": {"role": "user", "content": content},
    })
    .to_string();
    line.push('\n');
    line
}

fn fill(template: &str, vars: &[(&str, &str)]) -> String {
    vars.iter().fold(template.to_owned(), |text, (key, value)| {
        text.replace(&format!("{{{key}}}"), value)
    })
}

fn check_images(images: &[&JudgeImage]) -> Result<()> {
    for img in images {
        if !img.jpeg.starts_with(&[0xFF, 0xD8, 0xFF]) {
            bail!("judge image {:?} is not a JPEG", img.label);
        }
    }
    let sizes: Vec<_> = images.iter().map(|img| jpeg_size(&img.jpeg)).collect();
    if sizes.windows(2).any(|w| w[0] != w[1]) {
        tracing::warn!(
            ?sizes,
            "judged images differ in size; build them with JudgeImage::from_rgb"
        );
    }
    Ok(())
}

/// Width and height from a JPEG's start-of-frame segment.
fn jpeg_size(jpeg: &[u8]) -> Option<(u16, u16)> {
    let mut i = 2;
    while i + 9 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            return None;
        }
        let marker = jpeg[i + 1];
        if marker == 0xFF {
            i += 1;
            continue;
        }
        let len = usize::from(u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]));
        if (0xC0..=0xCF).contains(&marker) && ![0xC4, 0xC8, 0xCC].contains(&marker) {
            let h = u16::from_be_bytes([jpeg[i + 5], jpeg[i + 6]]);
            let w = u16::from_be_bytes([jpeg[i + 7], jpeg[i + 8]]);
            return Some((w, h));
        }
        i += 2 + len;
    }
    None
}

/// The outcomes of independent calls that answer the same question:
/// whichever succeeded, or the first error when none did.
fn settle<T, const N: usize>(what: &str, results: [Result<T>; N]) -> Result<Vec<T>> {
    let mut ok = Vec::new();
    let mut first_error = None;
    for result in results {
        match result {
            Ok(value) => ok.push(value),
            Err(e) => {
                tracing::warn!(what, error = %format!("{e:#}"), "one of the order-swapped calls failed");
                first_error.get_or_insert(e);
            }
        }
    }
    match first_error {
        Some(e) if ok.is_empty() => Err(e.context(format!("claude {what}: every order failed"))),
        _ => Ok(ok),
    }
}

/// One rank call, translated back to candidate indices.
struct RankCall {
    order: Vec<usize>,
    reasons: String,
    scene: String,
    notes: HashMap<usize, String>,
}

/// The rank request for one presentation order: `order[k]` is the
/// candidate shown k-th, under label `LABELS[k]`. Only pixels and the
/// camera reach Claude: no slider values, sources or caller labels.
fn rank_request<'a>(
    photo: &PhotoBrief,
    candidates: &'a [JudgeImage],
    order: &[usize],
) -> (Value, Vec<Block<'a>>) {
    let labels = &LABELS[..order.len()];
    let text = fill(
        RANK_PROMPT,
        &[
            ("n", &order.len().to_string()),
            ("labels", &join_labels(labels)),
            ("camera", &photo.camera),
            ("shot", &photo.exposure),
        ],
    );
    let images: Vec<(String, &[u8])> = order
        .iter()
        .zip(labels)
        .map(|(&c, label)| (format!("Image {label}:"), candidates[c].jpeg.as_slice()))
        .collect();
    (rank_schema(labels), with_images(&text, &images))
}

fn join_labels(labels: &[&str]) -> String {
    match labels {
        [] => String::new(),
        [one] => (*one).to_owned(),
        [init @ .., last] => format!("{} and {last}", init.join(", ")),
    }
}

fn label_position(labels: &[&str], label: &str) -> Option<usize> {
    labels
        .iter()
        .position(|l| l.eq_ignore_ascii_case(label.trim()))
}

/// Positions in the shown order, best first, if `ranking` names every label exactly once.
fn full_ranking(ranking: &[String], labels: &[&str]) -> Result<Vec<usize>> {
    let mut seen = vec![false; labels.len()];
    let mut order = Vec::with_capacity(labels.len());
    for label in ranking {
        let pos = label_position(labels, label)
            .with_context(|| format!("ranking names an unknown label {label:?}"))?;
        if std::mem::replace(&mut seen[pos], true) {
            bail!("ranking names {label:?} twice");
        }
        order.push(pos);
    }
    if order.len() != labels.len() {
        bail!("ranking has {} of {} labels", order.len(), labels.len());
    }
    Ok(order)
}

/// Borda count over rankings of `n` candidates, each best first: a first
/// place is worth n-1 points and a last place none. Ties go to the lower
/// index, the caller's own preference. Returns the aggregate order and the
/// share of rankings whose first pick is the winner.
fn borda(n: usize, rankings: &[Vec<usize>]) -> (Vec<usize>, f32) {
    let mut points = vec![0usize; n];
    for ranking in rankings {
        for (pos, &c) in ranking.iter().enumerate() {
            points[c] += n - 1 - pos;
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&c| (std::cmp::Reverse(points[c]), c));
    let winner = order[0];
    let agreeing = rankings
        .iter()
        .filter(|r| r.first() == Some(&winner))
        .count();
    (order, agreeing as f32 / rankings.len().max(1) as f32)
}

/// Votes and votes for b from the two compare orders' answers ("A", "B" or
/// "tie"; None when that call failed). b is B in the first order and A in
/// the second.
fn tally(ab: Option<&str>, ba: Option<&str>) -> (u32, u32) {
    let mut votes = 0;
    let mut for_b = 0;
    for (answer, b_label) in [(ab, "B"), (ba, "A")] {
        if let Some(answer) = answer {
            votes += 1;
            if answer.trim().eq_ignore_ascii_case(b_label) {
                for_b += 1;
            }
        }
    }
    (votes, for_b)
}

#[derive(Deserialize)]
struct RankReply {
    #[serde(default)]
    notes: Vec<RankNote>,
    ranking: Vec<String>,
    #[serde(default)]
    reasons: String,
    #[serde(default)]
    scene: String,
}

#[derive(Deserialize)]
struct RankNote {
    label: String,
    #[serde(default)]
    note: String,
}

#[derive(Deserialize)]
struct CompareReply {
    #[serde(default)]
    assessment: String,
    better: String,
}

#[derive(Deserialize)]
struct PlanReply {
    #[serde(default)]
    analysis: String,
    looks: Vec<LookReply>,
}

#[derive(Deserialize)]
struct LookReply {
    #[serde(default)]
    name: String,
    #[serde(default)]
    rationale: String,
    settings: Settings,
}

#[derive(Deserialize)]
struct RefineReply {
    #[serde(default)]
    assessment: String,
    #[serde(default)]
    change: String,
    settings: Settings,
}

/// Slider values as Claude writes them, in Lightroom units.
#[derive(Deserialize)]
struct Settings {
    temperature: f64,
    tint: f64,
    exposure: f64,
    contrast: f64,
    highlights: f64,
    shadows: f64,
    whites: f64,
    blacks: f64,
    vibrance: f64,
    saturation: f64,
}

impl Settings {
    /// In [`Params::NAMES`] order.
    fn values(&self) -> [f64; Params::COUNT] {
        [
            self.temperature,
            self.tint,
            self.exposure,
            self.contrast,
            self.highlights,
            self.shadows,
            self.whites,
            self.blacks,
            self.vibrance,
            self.saturation,
        ]
    }
}

const EXPOSURE: usize = 2;

/// Lightroom's legal range per slider, in [`Params::NAMES`] order.
const LEGAL: [(f64, f64); Params::COUNT] = [
    (2000.0, 50000.0),
    (-150.0, 150.0),
    (-5.0, 5.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
    (-100.0, 100.0),
];

/// Lightroom's steps per unit: whole kelvin and slider units, 0.01 EV.
const STEPS_PER_UNIT: [f64; Params::COUNT] = [1.0, 1.0, 100.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// The box Claude may move in: per slider, inclusive bounds on Lightroom's
/// grid, rounded inward so every value on the grid inside them is legal.
struct Ranges {
    lo: [f64; Params::COUNT],
    hi: [f64; Params::COUNT],
}

impl Ranges {
    fn new(lo: &Params, hi: &Params) -> Self {
        let (lo, hi) = (lo.to_array(), hi.to_array());
        let bounds: [(f64, f64); Params::COUNT] =
            std::array::from_fn(|i| grid_bounds(i, lo[i], hi[i]));
        Self {
            lo: bounds.map(|b| b.0),
            hi: bounds.map(|b| b.1),
        }
    }

    /// Claude's values on Lightroom's grid, clamped into the box.
    fn fit(&self, values: [f64; Params::COUNT]) -> Params {
        Params::from_array(std::array::from_fn(|i| self.fit_one(i, values[i])))
    }

    fn fit_one(&self, i: usize, value: f64) -> f32 {
        let per = STEPS_PER_UNIT[i];
        let v = if value.is_finite() {
            (value * per).round() / per
        } else {
            (self.lo[i] + self.hi[i]) / 2.0
        };
        v.clamp(self.lo[i], self.hi[i]) as f32
    }

    /// Like [`Ranges::fit`], except that sliders Claude kept (equal to the
    /// current value on Lightroom's grid) keep their exact current value.
    fn adjust(&self, current: &Params, values: [f64; Params::COUNT]) -> Params {
        let now = current.to_array();
        Params::from_array(std::array::from_fn(|i| {
            let per = STEPS_PER_UNIT[i];
            let kept = (f64::from(now[i]) * per).round() == (values[i] * per).round();
            if kept {
                (f64::from(now[i]).clamp(self.lo[i], self.hi[i])) as f32
            } else {
                self.fit_one(i, values[i])
            }
        }))
    }

    fn settings_schema(&self) -> Value {
        let mut properties = Map::new();
        for (i, name) in Params::NAMES.iter().enumerate() {
            let property = if i == EXPOSURE {
                json!({"type": "number", "minimum": self.lo[i], "maximum": self.hi[i]})
            } else {
                json!({"type": "integer", "minimum": self.lo[i] as i64, "maximum": self.hi[i] as i64})
            };
            properties.insert((*name).to_owned(), property);
        }
        json!({
            "type": "object",
            "properties": properties,
            "required": Params::NAMES,
            "additionalProperties": false,
        })
    }

    fn describe(&self) -> String {
        Params::NAMES
            .iter()
            .enumerate()
            .map(|(i, name)| {
                format!(
                    "- {name}: {} to {}",
                    slider_value(i, self.lo[i]),
                    slider_value(i, self.hi[i])
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Slider `i`'s bounds within Lightroom's legal range, rounded inward to
/// its grid. A box narrower than one step collapses to its rounded middle.
fn grid_bounds(i: usize, lo: f32, hi: f32) -> (f64, f64) {
    let (legal_lo, legal_hi) = LEGAL[i];
    let bound = |v: f32, fallback: f64| {
        if v.is_finite() {
            f64::from(v).clamp(legal_lo, legal_hi)
        } else {
            fallback
        }
    };
    let (a, b) = (bound(lo, legal_lo), bound(hi, legal_hi));
    let (a, b) = if a <= b { (a, b) } else { (b, a) };
    let per = STEPS_PER_UNIT[i];
    // Snap away f32 noise (0.35 is 0.34999999) before rounding inward.
    let snap = |v: f64| (v * per * 1e4).round() / 1e4;
    let (grid_lo, grid_hi) = (snap(a).ceil() / per, snap(b).floor() / per);
    if grid_lo <= grid_hi {
        (grid_lo, grid_hi)
    } else {
        let mid = ((a + b) / 2.0 * per).round() / per;
        (mid, mid)
    }
}

/// One slider value as Lightroom shows it: "5400 K", "+0.35 EV", "-20".
fn slider_value(i: usize, v: f64) -> String {
    match i {
        0 => format!("{} K", v.round() as i64),
        EXPOSURE => format!("{} EV", xmp::exposure(v as f32)),
        _ => xmp::signed(v as f32),
    }
}

/// "temperature 5400 K, tint +5, exposure +0.35 EV, contrast +10, ..."
fn describe(params: &Params) -> String {
    let values = params.quantized().to_array();
    Params::NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| format!("{name} {}", slider_value(i, f64::from(values[i]))))
        .collect::<Vec<_>>()
        .join(", ")
}

fn rank_schema(labels: &[&str]) -> Value {
    let n = labels.len();
    json!({
        "type": "object",
        "properties": {
            "notes": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": labels},
                        "note": {"type": "string"},
                    },
                    "required": ["label", "note"],
                    "additionalProperties": false,
                },
            },
            "ranking": {
                "type": "array",
                "minItems": n,
                "maxItems": n,
                "uniqueItems": true,
                "items": {"type": "string", "enum": labels},
            },
            "reasons": {"type": "string"},
            "scene": {"type": "string"},
        },
        "required": ["notes", "ranking", "reasons", "scene"],
        "additionalProperties": false,
    })
}

fn compare_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "assessment": {"type": "string"},
            "better": {"type": "string", "enum": ["A", "B", "tie"]},
        },
        "required": ["assessment", "better"],
        "additionalProperties": false,
    })
}

fn plan_schema(ranges: &Ranges) -> Value {
    json!({
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "looks": {
                "type": "array",
                "minItems": 1,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "rationale": {"type": "string"},
                        "settings": ranges.settings_schema(),
                    },
                    "required": ["name", "rationale", "settings"],
                    "additionalProperties": false,
                },
            },
        },
        "required": ["analysis", "looks"],
        "additionalProperties": false,
    })
}

fn refine_schema(ranges: &Ranges) -> Value {
    json!({
        "type": "object",
        "properties": {
            "assessment": {"type": "string"},
            "change": {"type": "string"},
            "settings": ranges.settings_schema(),
        },
        "required": ["assessment", "change", "settings"],
        "additionalProperties": false,
    })
}

/// Stats as a short JSON for the prompt: three significant digits, and long
/// histograms summed down to 16 bins.
fn compact(value: &Value) -> Value {
    const BINS: usize = 16;
    match value {
        Value::Number(n) if n.is_f64() => n.as_f64().map_or(Value::Null, |x| json!(round_sig(x))),
        Value::Array(items) if items.len() > BINS && items.iter().all(Value::is_number) => {
            let per_bin = items.len().div_ceil(BINS);
            let pooled: Vec<Value> = items
                .chunks(per_bin)
                .map(|chunk| json!(round_sig(chunk.iter().filter_map(Value::as_f64).sum())))
                .collect();
            Value::Array(pooled)
        }
        Value::Array(items) => Value::Array(items.iter().map(compact).collect()),
        Value::Object(map) => {
            Value::Object(map.iter().map(|(k, v)| (k.clone(), compact(v))).collect())
        }
        other => other.clone(),
    }
}

fn round_sig(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let scale = 10f64.powi(2 - x.abs().log10().floor() as i32);
    (x * scale).round() / scale
}

fn camera_line(info: &RawInfo) -> String {
    let make = info.make.trim();
    let model = info.model.trim();
    let mut line = if make.is_empty() || model.to_lowercase().starts_with(&make.to_lowercase()) {
        model.to_owned()
    } else {
        format!("{make} {model}")
    };
    if let Some(lens) = info.lens.as_deref().map(str::trim)
        && !lens.is_empty()
    {
        if !line.is_empty() {
            line.push_str(", ");
        }
        line.push_str(lens);
    }
    non_empty(line, || "unknown".into())
}

fn exposure_line(info: &RawInfo) -> String {
    let positive = |v: Option<f32>| v.filter(|v| v.is_finite() && *v > 0.0);
    let mut parts = Vec::new();
    if let Some(t) = positive(info.exposure_time_s) {
        parts.push(if t < 0.75 {
            format!("1/{} s", (1.0 / t).round())
        } else {
            format!("{} s", one_decimal(t))
        });
    }
    if let Some(f) = positive(info.f_number) {
        parts.push(format!("f/{}", one_decimal(f)));
    }
    if let Some(iso) = info.iso {
        parts.push(format!("ISO {iso}"));
    }
    if let Some(mm) = positive(info.focal_length_mm) {
        parts.push(format!("{} mm", mm.round()));
    }
    non_empty(parts.join(", "), || "unknown settings".into())
}

/// "4", "2.8", "1.3": one decimal, without a trailing ".0".
fn one_decimal(v: f32) -> String {
    let s = format!("{v:.1}");
    s.strip_suffix(".0").map(str::to_owned).unwrap_or(s)
}

fn non_empty(s: String, fallback: impl FnOnce() -> String) -> String {
    if s.trim().is_empty() {
        fallback()
    } else {
        s.trim().to_owned()
    }
}

/// At most the first `max` characters of `s`.
fn head(s: &str, max: usize) -> &str {
    s.char_indices().nth(max).map_or(s, |(i, _)| &s[..i])
}

/// At most the last `max` characters of `s`.
fn tail(s: &str, max: usize) -> &str {
    let skip = s.chars().count().saturating_sub(max);
    s.char_indices().nth(skip).map_or(s, |(i, _)| &s[i..])
}

#[cfg(test)]
mod tests;

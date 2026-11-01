use std::path::PathBuf;

use super::*;

/// What `claude -p --input-format stream-json --output-format stream-json
/// --verbose --json-schema ...` printed for a tiny request (claude 2.1.299,
/// sonnet, on the GPU host). Only the init line is shortened.
const RECORDED: &str = r#"{"type":"system","subtype":"init","cwd":"/tmp/tmp.8CFDs532U2","session_id":"0edec4a9-f038-4eaf-8371-fac568c9cfdc","tools":["StructuredOutput"],"mcp_servers":[],"model":"claude-sonnet-5-5","permissionMode":"default","apiKeySource":"none","claude_code_version":"2.1.299"}
{"type":"assistant","message":{"model":"claude-sonnet-5-5","id":"msg_011CgcNGK62Y9MSnGtKMnwPy","type":"message","role":"assistant","content":[{"type":"tool_use","id":"toolu_01Av2CcmEW3xqvvh1K6Vt7Rd","name":"StructuredOutput","input":{"word":"ok","n":3},"caller":{"type":"direct"}}],"container":null,"stop_reason":null,"stop_sequence":null,"stop_details":null},"parent_tool_use_id":null,"session_id":"0edec4a9-f038-4eaf-8371-fac568c9cfdc"}
{"type":"user","message":{"role":"user","content":[{"tool_use_id":"toolu_01Av2CcmEW3xqvvh1K6Vt7Rd","type":"tool_result","content":"Structured output provided successfully"}]},"parent_tool_use_id":null,"session_id":"0edec4a9-f038-4eaf-8371-fac568c9cfdc","uuid":"6c75680a-7b4f-4ed8-a9a8-1e3a5f8cd415","timestamp":"2026-11-01T14:29:33.244Z","tool_use_result":"Structured output provided successfully"}
{"type":"rate_limit_event","rate_limit_info":{"status":"allowed","resetsAt":1793559600,"rateLimitType":"five_hour","overageStatus":"rejected","overageDisabledReason":"org_level_disabled","isUsingOverage":false,"unifiedWindows":{"five_hour":{"utilization":0.06,"resetsAt":1793559600}}},"uuid":"c8114e92-a44e-4ddf-a0f7-439cb379f647","session_id":"0edec4a9-f038-4eaf-8371-fac568c9cfdc"}
{"duration_api_ms":1422,"stop_reason":"tool_use","session_id":"0edec4a9-f038-4eaf-8371-fac568c9cfdc","total_cost_usd":0.0020532,"usage":{"input_tokens":2,"cache_creation_input_tokens":311,"cache_read_input_tokens":752,"output_tokens":73,"output_tokens_details":{"thinking_tokens":0},"server_tool_use":{"web_search_requests":0,"web_fetch_requests":0},"service_tier":"standard","cache_creation":{"ephemeral_1h_input_tokens":311,"ephemeral_5m_input_tokens":0},"inference_geo":"global","iterations":[{"input_tokens":2,"output_tokens":73,"cache_read_input_tokens":752,"cache_creation_input_tokens":311,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":311},"type":"message"}],"speed":"standard","fallback_credit":null},"modelUsage":{"claude-sonnet-5-5":{"inputTokens":2,"outputTokens":73,"cacheReadInputTokens":752,"cacheCreationInputTokens":311,"webSearchRequests":0,"costUSD":0.0020532,"contextWindow":1000000,"maxOutputTokens":128000,"thinkingTokens":0,"canonicalModel":"claude-sonnet-5-5","provider":"firstParty","costBasis":"list"}},"permission_denials":[],"terminal_reason":"completed","fast_mode_state":"off","fast_mode_disabled_reason":"sdk_opt_in_required","subagent_stats":{"spawned":0,"requested":{"background":0,"foreground":0,"unset":0},"started_in_background":0,"max_depth":0,"spawned_by_subagents":0,"completed":0,"failed":0,"killed":{"parent":0,"user":0,"system":0},"refused":{"depth_limit":0,"concurrency_limit":0,"budget":0},"by_type":{}},"safety_stops":0,"is_error":false,"num_turns":2,"subtype":"success","api_error_status":null,"result":"{\"word\":\"ok\",\"n\":3}","structured_output":{"word":"ok","n":3},"ttft_ms":1648,"type":"result","duration_ms":1667,"uuid":"d917f2a6-a2ca-4654-a141-c60c03975209","ttft_stream_ms":988,"time_to_request_ms":246,"first_content_frame_ms":989,"queued_turn_count":0,"result_index":0}
"#;

fn options(cache_dir: Option<PathBuf>, timeout_s: u64) -> JudgeOptions {
    JudgeOptions {
        model: "sonnet".into(),
        effort: None,
        timeout_s,
        cache_dir,
    }
}

fn brief() -> PhotoBrief {
    PhotoBrief {
        camera: "SONY ILCE-7M3, FE 16-35mm F2.8 GM".into(),
        exposure: "1/250 s, f/4, ISO 400, 35 mm".into(),
        stats: json!({"luma_p50": 0.41}),
    }
}

fn flat_image(shade: u8) -> JudgeImage {
    let img = Rgb8Image {
        width: 16,
        height: 8,
        data: vec![shade; 16 * 8 * 3],
    };
    JudgeImage::from_rgb(format!("flat-{shade}"), &img).unwrap()
}

fn bounds() -> (Params, Params) {
    let lo = Params {
        temperature: 3800.0,
        tint: -25.0,
        exposure: -1.5,
        contrast: -40.0,
        highlights: -100.0,
        shadows: -40.0,
        whites: -50.0,
        blacks: -60.0,
        vibrance: -30.0,
        saturation: -40.0,
        ..Params::neutral(5200.0, 0.0)
    };
    let hi = Params {
        temperature: 8000.0,
        tint: 30.0,
        exposure: 2.0,
        contrast: 50.0,
        highlights: 30.0,
        shadows: 80.0,
        whites: 50.0,
        blacks: 30.0,
        vibrance: 60.0,
        saturation: 50.0,
        ..Params::neutral(5200.0, 0.0)
    };
    (lo, hi)
}

#[cfg(unix)]
fn set_mode(path: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt as _;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

#[test]
fn parses_recorded_stream_json() {
    let reply = parse_result(RECORDED).unwrap();
    assert_eq!(reply.subtype, "success");
    assert!(!reply.is_error);
    assert_eq!(reply.output().unwrap(), json!({"word": "ok", "n": 3}));
    assert!((reply.cost_usd - 0.0020532).abs() < 1e-12);
}

#[test]
fn cli_failures_are_explained() {
    let limit = parse_result(
        r#"{"type":"result","subtype":"success","is_error":true,"result":"Claude AI usage limit reached|1793559600","total_cost_usd":0}"#,
    )
    .unwrap();
    let msg = limit.output().unwrap_err().to_string();
    assert!(msg.contains("usage limit reached"), "{msg}");

    let turns =
        parse_result(r#"{"type":"result","subtype":"error_max_turns","is_error":true}"#).unwrap();
    assert!(
        turns
            .output()
            .unwrap_err()
            .to_string()
            .contains("error_max_turns")
    );

    let missing = parse_result(
        r#"{"type":"result","subtype":"success","is_error":false,"result":"I prefer B."}"#,
    )
    .unwrap();
    let msg = missing.output().unwrap_err().to_string();
    assert!(
        msg.contains("no structured output") && msg.contains("I prefer B."),
        "{msg}"
    );

    let progress_only: Vec<&str> = RECORDED.lines().take(4).collect();
    assert!(parse_result(&progress_only.join("\n")).is_none());
    assert!(parse_result("").is_none());
    assert!(parse_result("Error: something\n{\"type\":\"system\"}\n").is_none());
}

#[test]
fn borda_counts_both_orders() {
    // Both orders put candidate 1 first; 0 and 2 tie on points and the lower index wins.
    let (order, agreement) = borda(3, &[vec![1, 0, 2], vec![1, 2, 0]]);
    assert_eq!(order, vec![1, 0, 2]);
    assert_eq!(agreement, 1.0);
    // Position bias: each order prefers whatever came first.
    let (order, agreement) = borda(2, &[vec![0, 1], vec![1, 0]]);
    assert_eq!(order, vec![0, 1]);
    assert_eq!(agreement, 0.5);
    // A consistent second place beats two split first places.
    let (order, agreement) = borda(4, &[vec![0, 2, 3, 1], vec![1, 2, 3, 0]]);
    assert_eq!(order[0], 2);
    assert_eq!(agreement, 0.0);
    // A single surviving call decides alone.
    assert_eq!(borda(3, &[vec![2, 0, 1]]), (vec![2, 0, 1], 1.0));
}

#[test]
fn rankings_must_name_every_label_once() {
    let labels = ["A", "B", "C"];
    let ranking = |v: &[&str]| v.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    assert_eq!(
        full_ranking(&ranking(&["C", "a", " B "]), &labels).unwrap(),
        vec![2, 0, 1]
    );
    assert!(full_ranking(&ranking(&["A", "B"]), &labels).is_err());
    assert!(full_ranking(&ranking(&["A", "A", "B"]), &labels).is_err());
    assert!(full_ranking(&ranking(&["A", "B", "D"]), &labels).is_err());
}

#[test]
fn compare_needs_b_in_both_orders() {
    assert_eq!(tally(Some("B"), Some("A")), (2, 2));
    // Always picking the second image is position bias, not a preference.
    assert_eq!(tally(Some("B"), Some("B")), (2, 1));
    assert_eq!(tally(Some("tie"), Some("A")), (2, 1));
    assert_eq!(tally(Some("b"), None), (1, 1));
    assert_eq!(tally(None, None), (0, 0));
}

#[test]
fn ranges_round_inward_and_clamp() {
    let (lo, hi) = bounds();
    let ranges = Ranges::new(
        &Params {
            temperature: 3800.4,
            exposure: -1.234,
            saturation: 30.0,
            ..lo
        },
        &Params {
            exposure: 0.35,
            saturation: -40.0,
            ..hi
        },
    );
    assert_eq!((ranges.lo[0], ranges.hi[0]), (3801.0, 8000.0));
    assert_eq!((ranges.lo[2], ranges.hi[2]), (-1.23, 0.35));
    // Swapped bounds are put right.
    assert_eq!((ranges.lo[9], ranges.hi[9]), (-40.0, 30.0));

    let p = ranges.fit([
        9000.0,
        0.4,
        0.351,
        12.6,
        -120.0,
        f64::NAN,
        0.0,
        0.0,
        0.0,
        0.0,
    ]);
    assert_eq!(p.temperature, 8000.0);
    assert_eq!(p.tint, 0.0);
    assert_eq!(p.exposure, 0.35);
    assert_eq!(p.contrast, 13.0);
    assert_eq!(p.highlights, -100.0);
    // Not a number: the middle of the box.
    assert_eq!(p.shadows, 20.0);

    let narrow = Ranges::new(
        &Params {
            exposure: 0.331,
            ..lo
        },
        &Params {
            exposure: 0.334,
            ..hi
        },
    );
    assert_eq!((narrow.lo[2], narrow.hi[2]), (0.33, 0.33));
    let wide = Ranges::new(
        &Params {
            temperature: 100.0,
            ..lo
        },
        &Params {
            temperature: f32::INFINITY,
            ..hi
        },
    );
    assert_eq!((wide.lo[0], wide.hi[0]), (2000.0, 50000.0));
}

#[test]
fn refine_keeps_untouched_sliders_exactly() {
    let (lo, hi) = bounds();
    let current = Params {
        temperature: 5600.4,
        tint: 3.7,
        exposure: 0.456,
        contrast: 35.2,
        highlights: -10.0,
        shadows: 10.0,
        whites: 20.0,
        blacks: 0.0,
        vibrance: 45.0,
        saturation: 40.0,
        ..Params::neutral(5200.0, 0.0)
    };
    let reply = [5600.0, 4.0, 0.46, 35.0, -10.0, 10.0, 20.0, 0.0, 45.0, 25.0];
    let p = Ranges::new(&lo, &hi).adjust(&current, reply);
    assert_eq!(
        p,
        Params {
            saturation: 25.0,
            ..current
        }
    );
}

#[test]
fn schemas_carry_bounds_and_labels() {
    let (lo, hi) = bounds();
    let ranges = Ranges::new(&lo, &hi);
    let plan = plan_schema(&ranges);
    let settings = &plan["properties"]["looks"]["items"]["properties"]["settings"];
    assert_eq!(
        settings["properties"]["temperature"],
        json!({"type": "integer", "minimum": 3800, "maximum": 8000})
    );
    assert_eq!(
        settings["properties"]["exposure"],
        json!({"type": "number", "minimum": -1.5, "maximum": 2.0})
    );
    assert_eq!(
        settings["properties"]["highlights"],
        json!({"type": "integer", "minimum": -100, "maximum": 30})
    );
    assert_eq!(settings["required"], json!(&Params::NAMES[..Params::BASIC]));
    assert_eq!(plan["properties"]["looks"]["maxItems"], 3);
    assert_eq!(
        refine_schema(&ranges)["properties"]["settings"],
        ranges.settings_schema()
    );

    let rank = rank_schema(&["A", "B", "C"]);
    assert_eq!(
        rank["properties"]["ranking"]["items"]["enum"],
        json!(["A", "B", "C"])
    );
    assert_eq!(rank["properties"]["ranking"]["minItems"], 3);
    assert_eq!(rank["properties"]["ranking"]["maxItems"], 3);
    assert_eq!(
        compare_schema()["properties"]["better"]["enum"],
        json!(["A", "B", "tie"])
    );
    // Claude fills properties in the order it sees them: critique before verdict.
    let keys = |v: &Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
    assert_eq!(
        keys(&compare_schema()["properties"]),
        ["assessment", "better"]
    );
    assert_eq!(
        keys(&rank["properties"]),
        ["notes", "ranking", "reasons", "scene"]
    );
    assert_eq!(keys(&plan["properties"]), ["analysis", "looks"]);
    assert_eq!(
        keys(&refine_schema(&ranges)["properties"]),
        ["assessment", "change", "settings"]
    );
}

#[test]
fn rank_requests_show_only_pixels_under_neutral_labels() {
    let candidates: Vec<JudgeImage> = [10, 120, 240]
        .into_iter()
        .map(|shade| JudgeImage {
            label: format!("auto-vivid-{shade}"),
            ..flat_image(shade)
        })
        .collect();
    let photo = PhotoBrief {
        stats: json!({"secret_temperature": 4321}),
        ..brief()
    };
    for order in [vec![0, 1, 2], vec![2, 1, 0]] {
        let (_, blocks) = rank_request(&photo, &candidates, &order);
        let mut images = Vec::new();
        let mut text = String::new();
        for block in &blocks {
            match block {
                Block::Text(t) => {
                    text.push_str(t);
                    text.push('\n');
                }
                Block::Image(jpeg) => images.push(*jpeg),
            }
        }
        assert_eq!(images.len(), 3);
        for (k, &c) in order.iter().enumerate() {
            assert_eq!(images[k], candidates[c].jpeg.as_slice());
            assert!(text.contains(&format!("Image {}:", LABELS[k])));
        }
        let lower = text.to_lowercase();
        for leak in ["auto", "vivid", "temperature", "secret", "4321", "neutral"] {
            assert!(!lower.contains(leak), "{leak:?} leaked into:\n{text}");
        }
        assert!(text.contains("labelled A, B and C"));
        assert!(text.contains("Camera: SONY ILCE-7M3, FE 16-35mm F2.8 GM."));
        assert!(text.contains("Shot at 1/250 s, f/4, ISO 400, 35 mm."));
        assert!(!text.contains('{'), "unfilled placeholder in:\n{text}");
    }
}

#[test]
fn plan_and_refine_prompts_are_filled() {
    let (lo, hi) = bounds();
    let ranges = Ranges::new(&lo, &hi);
    let auto = Params::neutral(5600.0, 4.0);
    for (template, extra) in [
        (PLAN_PROMPT, ("auto", describe(&auto))),
        (REFINE_PROMPT, ("params", describe(&auto))),
    ] {
        let text = fill(
            template,
            &[
                ("camera", "SONY ILCE-7M3"),
                ("shot", "1/250 s"),
                ("stats", "{}"),
                ("bounds", &ranges.describe()),
                (extra.0, &extra.1),
            ],
        );
        assert!(text.contains("{images}"));
        for key in [
            "camera", "shot", "auto", "params", "stats", "bounds", "n", "labels",
        ] {
            assert!(
                !text.contains(&format!("{{{key}}}")),
                "{{{key}}} unfilled:\n{text}"
            );
        }
        assert!(text.contains("temperature 5600 K, tint +4, exposure 0.00 EV, contrast 0"));
        assert!(text.contains("- temperature: 3800 K to 8000 K"));
        assert!(text.contains("- exposure: -1.50 EV to +2.00 EV"));
        assert!(text.contains("- highlights: -100 to +30"));
    }
}

#[test]
fn compact_rounds_and_pools_histograms() {
    let stats = Stats {
        luma_hist: (0..64)
            .map(|i| if i < 32 { 1.0 / 32.0 } else { 0.0 })
            .collect(),
        luma_p50: 0.123_456,
        clip_hi: 0.000_432,
        ..Stats::default()
    };
    let v = compact(&serde_json::to_value(&stats).unwrap());
    let pooled = v["luma_hist"].as_array().unwrap();
    assert_eq!(pooled.len(), 16);
    assert_eq!(pooled[0], json!(0.125));
    assert_eq!(pooled[15], json!(0.0));
    assert_eq!(v["luma_p50"], json!(0.123));
    assert_eq!(v["clip_hi"], json!(0.000432));
    assert_eq!(
        compact(&json!({"n": [1, 2], "s": "x"})),
        json!({"n": [1, 2], "s": "x"})
    );
}

#[test]
fn brief_describes_camera_and_exposure() {
    let mut info = RawInfo {
        make: "SONY".into(),
        model: "ILCE-7M3".into(),
        lens: Some("FE 16-35mm F2.8 GM".into()),
        width: 6000,
        height: 4000,
        orientation: 1,
        iso: Some(400),
        exposure_time_s: Some(1.0 / 250.0),
        f_number: Some(4.0),
        focal_length_mm: Some(35.0),
        captured_at: None,
        as_shot_temperature: 5212.6,
        as_shot_tint: 3.2,
        raw_clipped_fraction: 0.01,
    };
    let b = PhotoBrief::new(&info, &Stats::default());
    assert_eq!(b.camera, "SONY ILCE-7M3, FE 16-35mm F2.8 GM");
    assert_eq!(b.exposure, "1/250 s, f/4, ISO 400, 35 mm");
    assert_eq!(b.stats["as_shot_temperature"], json!(5213.0));
    assert_eq!(b.stats["as_shot_tint"], json!(3.0));
    assert!(b.stats["luma_hist"].is_array());

    info.make = "Canon".into();
    info.model = "Canon EOS R5".into();
    info.lens = None;
    info.exposure_time_s = Some(2.5);
    info.f_number = Some(2.8);
    info.iso = None;
    info.focal_length_mm = None;
    let b = PhotoBrief::new(&info, &Stats::default());
    assert_eq!(b.camera, "Canon EOS R5");
    assert_eq!(b.exposure, "2.5 s, f/2.8");
}

#[test]
fn judge_images_share_size_and_quality() {
    let img = Rgb8Image {
        width: 1536,
        height: 1024,
        data: vec![128; 1536 * 1024 * 3],
    };
    let j = JudgeImage::from_rgb("A", &img).unwrap();
    assert_eq!(jpeg_size(&j.jpeg), Some((1280, 853)));
    assert!(check_images(&[&j]).is_ok());
    let not_jpeg = JudgeImage {
        label: "png".into(),
        jpeg: b"\x89PNG\r\n".to_vec(),
    };
    assert!(check_images(&[&j, &not_jpeg]).is_err());
}

fn text_and_image<'a>(text: &str, jpeg: &'a [u8]) -> Vec<Block<'a>> {
    vec![Block::Text(text.into()), Block::Image(jpeg)]
}

#[test]
fn cache_key_covers_every_input() {
    let (one, two) = (flat_image(10), flat_image(200));
    let claude = Claude::with_bin("/bin/false".into(), options(None, 30));
    let base = claude.cache_key("{}", &text_and_image("rank", &one.jpeg));
    assert_eq!(
        base,
        claude.cache_key("{}", &text_and_image("rank", &one.jpeg))
    );
    assert_ne!(
        base,
        claude.cache_key("{}", &text_and_image("rank", &two.jpeg))
    );
    assert_ne!(
        base,
        claude.cache_key("{}", &text_and_image("rank!", &one.jpeg))
    );
    assert_ne!(
        base,
        claude.cache_key("{\"a\":1}", &text_and_image("rank", &one.jpeg))
    );
    let opus = Claude::with_bin(
        "/bin/false".into(),
        JudgeOptions {
            model: "opus".into(),
            ..options(None, 30)
        },
    );
    assert_ne!(
        base,
        opus.cache_key("{}", &text_and_image("rank", &one.jpeg))
    );
    let high = Claude::with_bin(
        "/bin/false".into(),
        JudgeOptions {
            effort: Some("high".into()),
            ..options(None, 30)
        },
    );
    assert_ne!(
        base,
        high.cache_key("{}", &text_and_image("rank", &one.jpeg))
    );
    // Field boundaries are unambiguous: text cannot pose as an image.
    assert_ne!(
        claude.cache_key("{}", &[Block::Text("ab".into())]),
        claude.cache_key("{}", &[Block::Image(b"ab")])
    );
}

#[test]
fn command_runs_isolated_on_the_subscription() {
    let claude = Claude::with_bin(
        "/opt/claude".into(),
        JudgeOptions {
            effort: Some("high".into()),
            ..options(None, 30)
        },
    );
    let cmd = claude.command("{\"type\":\"object\"}", Path::new("/tmp/judge-x"));
    let std = cmd.as_std();
    assert_eq!(std.get_program(), "/opt/claude");
    let args: Vec<String> = std
        .get_args()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let value = |flag: &str| {
        let i = args.iter().position(|a| a == flag)?;
        args.get(i + 1).map(String::as_str)
    };
    assert_eq!(args[0], "-p");
    assert_eq!(value("--input-format"), Some("stream-json"));
    assert_eq!(value("--output-format"), Some("stream-json"));
    assert_eq!(value("--tools"), Some(""));
    assert_eq!(value("--setting-sources"), Some(""));
    assert_eq!(value("--model"), Some("sonnet"));
    assert_eq!(value("--effort"), Some("high"));
    assert_eq!(value("--json-schema"), Some("{\"type\":\"object\"}"));
    assert!(
        value("--system-prompt")
            .unwrap()
            .starts_with("You are a senior photo editor and print judge.")
    );
    for flag in [
        "--verbose",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--disable-slash-commands",
    ] {
        assert!(args.iter().any(|a| a == flag), "missing {flag}");
    }
    let envs: HashMap<String, Option<String>> = std
        .get_envs()
        .map(|(k, v)| {
            (
                k.to_string_lossy().into_owned(),
                v.map(|v| v.to_string_lossy().into_owned()),
            )
        })
        .collect();
    assert_eq!(envs.get("ANTHROPIC_API_KEY"), Some(&None));
    assert_eq!(envs.get("CLAUDECODE"), Some(&None));
    assert_eq!(envs.get("DISABLE_AUTOUPDATER"), Some(&Some("1".to_owned())));
    assert_eq!(std.get_current_dir(), Some(Path::new("/tmp/judge-x")));

    let plain = Claude::with_bin("/opt/claude".into(), options(None, 30));
    let plain = plain.command("{}", Path::new("/"));
    assert!(!plain.as_std().get_args().any(|a| a == "--effort"));
}

#[cfg(unix)]
#[test]
fn finds_the_cli_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let make = |rel: &str, mode: u32| {
        let path = dir.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "#!/bin/sh\n").unwrap();
        set_mode(&path, mode);
        path
    };
    let explicit = make("custom/claude-dev", 0o755);
    let on_path = make("bin/claude", 0o755);
    let installed = make("home/.local/bin/claude", 0o755);
    make("plain/claude", 0o644);
    let home = Some(dir.path().join("home"));
    let path = std::env::join_paths([
        dir.path().join("missing"),
        dir.path().join("plain"),
        dir.path().join("bin"),
    ])
    .unwrap();

    let found = find_bin(
        Some(explicit.clone().into()),
        Some(path.clone()),
        home.clone(),
    );
    assert_eq!(found.unwrap(), explicit);
    assert_eq!(
        find_bin(None, Some(path.clone()), home.clone()).unwrap(),
        on_path
    );
    assert_eq!(
        find_bin(Some("".into()), None, home.clone()).unwrap(),
        installed
    );
    // A wrong explicit setting is an error, not a silent fallback.
    let wrong = find_bin(Some(dir.path().join("nope").into()), Some(path), home);
    assert!(wrong.unwrap_err().to_string().contains("SPL_CLAUDE_BIN"));
    assert!(find_bin(None, Some(dir.path().join("plain").into()), None).is_err());
}

#[test]
fn judge_futures_are_send() {
    fn send<T: Send>(_: &T) {}
    let claude = Claude::with_bin("/bin/false".into(), options(None, 30));
    let (photo, img, p) = (brief(), flat_image(1), Params::neutral(5000.0, 0.0));
    send(&claude.plan(&photo, &img, &img, &p, &p, &p));
    send(&claude.rank(&photo, std::slice::from_ref(&img)));
    send(&claude.refine(&photo, &img, &p, &p, &p));
    send(&claude.compare(&photo, &img, &img));
}

/// The whole subprocess path against a stand-in for the CLI.
#[cfg(unix)]
mod fake_cli {
    use super::*;

    /// An executable stand-in for `claude` that records each call
    /// (arguments, working directory and its contents, environment, stdin)
    /// in `dir`, then runs `behaviour` with `$n` set to the call number.
    fn fake(dir: &Path, behaviour: &str) -> PathBuf {
        let header = [
            "#!/bin/sh".to_owned(),
            format!("d='{}'", dir.display()),
            r#"n=$(cat "$d/count" 2>/dev/null || echo 0)"#.to_owned(),
            r#"echo $((n + 1)) > "$d/count""#.to_owned(),
            r#"printf '%s\n' "$@" > "$d/args.$n""#.to_owned(),
            r#"pwd > "$d/cwd.$n""#.to_owned(),
            r#"ls -A | wc -l | tr -d ' ' > "$d/files.$n""#.to_owned(),
            r#"env > "$d/env.$n""#.to_owned(),
            r#"cat > "$d/stdin.$n""#.to_owned(),
        ];
        let path = dir.join("claude");
        std::fs::write(&path, format!("{}\n{behaviour}\n", header.join("\n"))).unwrap();
        set_mode(&path, 0o755);
        path
    }

    fn result_line(output: &Value, cost_usd: f64) -> String {
        json!({
            "type": "result",
            "subtype": "success",
            "is_error": false,
            "result": output.to_string(),
            "structured_output": output,
            "total_cost_usd": cost_usd,
            "duration_ms": 900,
        })
        .to_string()
    }

    #[tokio::test]
    async fn retries_caches_and_counts() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let output = json!({
            "assessment": "Colour is a little loud.",
            "change": "Saturation from +40 to +25.",
            "settings": {
                "temperature": 5600, "tint": 4, "exposure": 0.46, "contrast": 35,
                "highlights": -10, "shadows": 10, "whites": 20, "blacks": 0,
                "vibrance": 45, "saturation": 25,
            },
        });
        // The first answer lacks the structured output, the retry has it.
        let behaviour = [
            r#"if [ "$n" -eq 0 ]; then"#,
            r#"  echo '{"type":"result","subtype":"success","is_error":false,"result":"Sure.","total_cost_usd":0.1}'"#,
            "  exit 0",
            "fi",
            "cat <<'EOF'",
            r#"{"type":"system","subtype":"init"}"#,
            &result_line(&output, 0.25),
            "EOF",
        ]
        .join("\n");
        let claude = Claude::with_bin(
            fake(dir.path(), &behaviour),
            options(Some(cache.clone()), 30),
        );
        let (lo, hi) = bounds();
        let current = Params {
            temperature: 5600.4,
            tint: 3.7,
            exposure: 0.456,
            contrast: 35.2,
            highlights: -10.0,
            shadows: 10.0,
            whites: 20.0,
            blacks: 0.0,
            vibrance: 45.0,
            saturation: 40.0,
            ..Params::neutral(5200.0, 0.0)
        };
        let img = flat_image(90);

        let verdict = claude
            .refine(&brief(), &img, &current, &lo, &hi)
            .await
            .unwrap();
        assert_eq!(
            verdict.params,
            Params {
                saturation: 25.0,
                ..current
            }
        );
        assert_eq!(
            verdict.rationale,
            "Colour is a little loud. Saturation from +40 to +25."
        );
        let usage = claude.usage();
        assert_eq!((usage.calls, usage.cache_hits), (2, 0));
        assert!((usage.cost_usd - 0.35).abs() < 1e-9, "{usage:?}");

        let read = |name: &str| std::fs::read_to_string(dir.path().join(name)).unwrap();
        let args: Vec<String> = read("args.1").lines().map(str::to_owned).collect();
        assert!(args.iter().any(|a| a == "--no-session-persistence"));
        let tools = args.iter().position(|a| a == "--tools").unwrap();
        assert_eq!(args[tools + 1], "");
        assert_eq!(
            read("files.1").trim(),
            "0",
            "the CLI must start in an empty directory"
        );
        let cwd = PathBuf::from(read("cwd.1").trim());
        assert!(!cwd.exists(), "the working directory is removed afterwards");
        let env = read("env.1");
        assert!(env.lines().any(|l| l == "DISABLE_AUTOUPDATER=1"));
        assert!(!env.lines().any(|l| l.starts_with("ANTHROPIC_API_KEY=")));

        // The request: one stream-json user message with the image inline.
        let stdin = read("stdin.1");
        assert_eq!(stdin.lines().count(), 1);
        let message: Value = serde_json::from_str(stdin.trim()).unwrap();
        assert_eq!(message["type"], "user");
        assert_eq!(message["message"]["role"], "user");
        let content = message["message"]["content"].as_array().unwrap();
        let image = content.iter().find(|b| b["type"] == "image").unwrap();
        assert_eq!(image["source"]["media_type"], "image/jpeg");
        let data = base64::engine::general_purpose::STANDARD
            .decode(image["source"]["data"].as_str().unwrap())
            .unwrap();
        assert_eq!(data, img.jpeg);
        let text: String = content.iter().filter_map(|b| b["text"].as_str()).collect();
        assert!(text.contains("saturation +40"), "{text}");
        assert!(text.contains("- exposure: -1.50 EV to +2.00 EV"), "{text}");

        // The same request again comes from the cache without starting the CLI.
        let again = claude
            .refine(&brief(), &img, &current, &lo, &hi)
            .await
            .unwrap();
        assert_eq!(again.params, verdict.params);
        let usage = claude.usage();
        assert_eq!((usage.calls, usage.cache_hits), (2, 1));
        assert_eq!(read("count").trim(), "2");
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1);
    }

    /// A stand-in that answers `output` only once `calls` CLI processes are
    /// running at the same time (each marks its start with a file of its
    /// own), and fails after 5 s otherwise.
    fn together(dir: &Path, calls: usize, output: &Value) -> PathBuf {
        let behaviour = [
            r#"touch "$d/started.$$""#.to_owned(),
            "i=0".to_owned(),
            format!(
                r#"while [ "$(ls "$d" | grep -c '^started\.')" -lt {calls} ] && [ "$i" -lt 50 ]; do sleep 0.1; i=$((i + 1)); done"#
            ),
            format!(
                r#"if [ "$(ls "$d" | grep -c '^started\.')" -lt {calls} ]; then echo 'ran alone' >&2; exit 1; fi"#
            ),
            "cat <<'EOF'".to_owned(),
            result_line(output, 0.1),
            "EOF".to_owned(),
        ]
        .join("\n");
        fake(dir, &behaviour)
    }

    #[tokio::test]
    async fn rank_asks_both_orders_at_once_and_keeps_notes() {
        let dir = tempfile::tempdir().unwrap();
        let output = json!({
            "notes": [
                {"label": "A", "note": "note A"},
                {"label": "B", "note": "note B"},
                {"label": "C", "note": ""},
            ],
            "ranking": ["B", "A", "C"],
            "reasons": "The middle one.",
            "scene": "A test card.",
        });
        let claude = Claude::with_bin(together(dir.path(), 2, &output), options(None, 30));
        let images = [flat_image(10), flat_image(20), flat_image(30)];
        let verdict = claude.rank(&brief(), &images).await.unwrap();
        // Shown A, B, C = 0, 1, 2 and then 2, 1, 0: B (1) wins both orders.
        assert_eq!(verdict.order, [1, 0, 2]);
        assert_eq!((verdict.winner, verdict.agreement), (1, 1.0));
        // An empty note gives way to the other order's note on that candidate.
        assert_eq!(
            verdict.notes,
            [
                Some("note A".to_owned()),
                Some("note B".to_owned()),
                Some("note A".to_owned())
            ]
        );
        assert_eq!(verdict.scene.as_deref(), Some("A test card."));
        assert_eq!(claude.usage().calls, 2);
    }

    #[tokio::test]
    async fn compare_asks_both_orders_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let output = json!({"assessment": "The brighter version.", "better": "B"});
        let claude = Claude::with_bin(together(dir.path(), 2, &output), options(None, 30));
        let verdict = claude
            .compare(&brief(), &flat_image(1), &flat_image(2))
            .await
            .unwrap();
        // "B" in both orders: b once, a once.
        assert_eq!((verdict.votes, verdict.votes_for_b), (2, 1));
        assert!(!verdict.prefer_b);
        assert_eq!(claude.usage().calls, 2);
    }

    #[tokio::test]
    async fn a_hung_cli_is_killed_at_the_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let claude = Claude::with_bin(fake(dir.path(), "exec sleep 30"), options(None, 1));
        let (lo, hi) = bounds();
        let started = Instant::now();
        let current = Params::neutral(5600.0, 0.0);
        let err = claude
            .refine(&brief(), &flat_image(9), &current, &lo, &hi)
            .await
            .unwrap_err();
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
        assert!(
            format!("{err:#}").contains("timed out after 1 s"),
            "{err:#}"
        );
        assert_eq!(claude.usage().calls, 1);
    }

    #[tokio::test]
    async fn cli_errors_reach_the_caller() {
        let dir = tempfile::tempdir().unwrap();
        let behaviour = "echo 'Error: not logged in' >&2\nexit 1";
        let claude = Claude::with_bin(fake(dir.path(), behaviour), options(None, 30));
        let err = claude
            .compare(&brief(), &flat_image(1), &flat_image(2))
            .await
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("not logged in") && msg.contains("both orders failed"),
            "{msg}"
        );
        // Both orders, each with its retry.
        assert_eq!(claude.usage().calls, 4);
    }
}

/// Real calls through the CLI on the user's subscription (about six). Run
/// natively on the GPU host, where `claude` is logged in:
/// `cargo test --release -p spl-app-raw-retouch judge -- --ignored --nocapture --test-threads=1`
/// Set SPL_TEST_OUT to keep the images Claude saw.
mod live {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Look {
        Good,
        Dark,
        Oversaturated,
        Flat,
    }

    fn hash(x: usize, y: usize) -> f32 {
        let mut h = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA77);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2C1B_3C6D);
        h ^= h >> 12;
        h = h.wrapping_mul(0x297A_2D39);
        h ^= h >> 15;
        (h >> 8) as f32 / (1u32 << 24) as f32
    }

    fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
        std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
    }

    /// Scene radiance in linear light (mid grey about 0.18) of a small
    /// landscape: sky with sun and clouds, hazy hills, a red barn, a white
    /// cottage and a meadow with flowers.
    fn radiance(u: f32, v: f32, grain: f32, aspect: f32) -> [f32; 3] {
        let ridge = 0.50 + 0.045 * (6.0 * u + 1.0).sin() + 0.02 * (15.0 * u + 0.3).sin();
        let meadow = 0.63 + 0.025 * (4.0 * u + 2.0).sin();
        let roof = 0.42 + (u - 0.24).abs() * 0.6;
        if (0.16..0.32).contains(&u) && v > roof && v < meadow + 0.02 {
            if v < 0.50 {
                return [0.05, 0.045, 0.05];
            }
            if (0.225..0.255).contains(&u) && v > 0.565 {
                return [0.05, 0.03, 0.02];
            }
            if (0.215..0.265).contains(&u) && v > 0.555 {
                return [0.55, 0.55, 0.52];
            }
            let shade = if u > 0.29 { 0.55 } else { 1.0 };
            return [0.32 * shade, 0.04 * shade, 0.03 * shade];
        }
        if (0.58..0.70).contains(&u) && v > 0.555 && v < meadow + 0.03 {
            let window = ((0.60..0.625).contains(&u) || (0.655..0.68).contains(&u))
                && (0.575..0.61).contains(&v);
            return if window {
                [0.03, 0.035, 0.04]
            } else {
                [0.62, 0.60, 0.56]
            };
        }
        if v < ridge {
            let t = (v / ridge).clamp(0.0, 1.0).powf(1.4);
            let mut c = mix([0.05, 0.11, 0.34], [0.42, 0.52, 0.68], t);
            let (du, dv) = ((u - 0.80) * aspect, v - 0.17);
            let d = (du * du + dv * dv).sqrt();
            if d < 0.022 {
                return [9.0, 8.5, 7.0];
            }
            let glow = 0.55 * (-d / 0.07).exp();
            c = [c[0] + glow, c[1] + glow * 0.85, c[2] + glow * 0.55];
            for (cu, cv, r) in [(0.30, 0.16, 0.10), (0.42, 0.22, 0.08), (0.58, 0.12, 0.07)] {
                let (du, dv) = ((u - cu) * aspect / 1.8, v - cv);
                let cover = (-2.0 * (du * du + dv * dv) / (r * r)).exp() * 0.75;
                c = mix(c, [0.75, 0.74, 0.72], cover);
            }
            return c;
        }
        if v < meadow {
            let t = ((v - ridge) / (meadow - ridge)).clamp(0.0, 1.0);
            return mix([0.16, 0.19, 0.25], [0.07, 0.10, 0.08], t).map(|x| x * (0.9 + 0.2 * grain));
        }
        if grain > 0.985 {
            return [0.55, 0.45, 0.04];
        }
        let depth = ((v - meadow) / (1.0 - meadow)).clamp(0.0, 1.0);
        mix([0.10, 0.16, 0.04], [0.035, 0.07, 0.015], depth).map(|x| x * (0.75 + 0.5 * grain))
    }

    /// The scene developed four ways, as a 1536 x 1024 sRGB render.
    fn photo(look: Look) -> Rgb8Image {
        let (w, h) = (1536, 1024);
        let (gain, saturation, contrast) = match look {
            Look::Good => (1.0, 1.08, 1.05),
            Look::Dark => (0.15, 1.0, 1.0),
            Look::Oversaturated => (1.1, 2.8, 1.3),
            Look::Flat => (0.7, 0.8, 0.8),
        };
        let aspect = w as f32 / h as f32;
        let mut data = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
                let c = radiance(u, v, hash(x / 2, y / 2), aspect).map(|c| c * gain);
                let luma = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                for c in c {
                    let c = (luma + (c - luma) * saturation).max(0.0);
                    let toned = 1.0 - (-1.8 * c).exp();
                    let encoded = if toned <= 0.003_130_8 {
                        12.92 * toned
                    } else {
                        1.055 * toned.powf(1.0 / 2.4) - 0.055
                    };
                    let graded = (0.5 + (encoded - 0.5) * contrast).clamp(0.0, 1.0);
                    data.push((graded * 255.0).round() as u8);
                }
            }
        }
        Rgb8Image {
            width: w,
            height: h,
            data,
        }
    }

    fn stats_of(img: &Rgb8Image) -> Stats {
        let mut luma: Vec<f32> = img
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| {
                (0.2126 * f32::from(p[0]) + 0.7152 * f32::from(p[1]) + 0.0722 * f32::from(p[2]))
                    / 255.0
            })
            .collect();
        let n = luma.len() as f32;
        let mut hist = vec![0f32; 64];
        for &l in &luma {
            hist[((l * 64.0) as usize).min(63)] += 1.0 / n;
        }
        let clip_hi = img
            .data
            .as_chunks::<3>()
            .0
            .iter()
            .filter(|p| p.iter().any(|&c| c >= 254))
            .count() as f32
            / n;
        let clip_lo = luma.iter().filter(|&&l| l <= 2.0 / 255.0).count() as f32 / n;
        let mean = luma.iter().sum::<f32>() / n;
        luma.sort_by(f32::total_cmp);
        let pct = |p: f32| luma[(p * (luma.len() - 1) as f32) as usize];
        Stats {
            luma_hist: hist,
            luma_mean: mean,
            luma_p01: pct(0.01),
            luma_p05: pct(0.05),
            luma_p50: pct(0.5),
            luma_p95: pct(0.95),
            luma_p99: pct(0.99),
            clip_hi,
            clip_hi_new: clip_hi,
            clip_lo,
            ..Stats::default()
        }
    }

    fn judged(look: Look) -> JudgeImage {
        JudgeImage::from_rgb(format!("{look:?}"), &photo(look)).unwrap()
    }

    /// Writes the images Claude sees to $SPL_TEST_OUT/judge, if set.
    fn save_for_review() {
        let Some(dir) = std::env::var_os("SPL_TEST_OUT").map(|d| PathBuf::from(d).join("judge"))
        else {
            return;
        };
        std::fs::create_dir_all(&dir).unwrap();
        for look in [Look::Good, Look::Dark, Look::Oversaturated, Look::Flat] {
            let path = dir.join(format!("{look:?}.jpg").to_lowercase());
            std::fs::write(&path, judged(look).jpeg).unwrap();
        }
        println!("judged images in {}", dir.display());
    }

    #[test]
    #[ignore = "writes the synthetic photos for review; no Claude calls"]
    fn synthetic_photos_for_review() {
        save_for_review();
    }

    fn assert_inside(p: &Params, lo: &Params, hi: &Params) {
        let (p, lo, hi) = (p.to_array(), lo.to_array(), hi.to_array());
        for (i, name) in Params::NAMES.iter().enumerate() {
            assert!(
                lo[i] - 1e-3 <= p[i] && p[i] <= hi[i] + 1e-3,
                "{name} = {} outside {}..{}",
                p[i],
                lo[i],
                hi[i]
            );
        }
    }

    /// A logged-in CLI on sonnet with a fresh verdict cache, and the brief
    /// of the synthetic photo.
    struct Live {
        claude: Claude,
        brief: PhotoBrief,
        _cache: tempfile::TempDir,
    }

    impl Live {
        fn new() -> Self {
            let cache = tempfile::tempdir().unwrap();
            let claude = Claude::locate(JudgeOptions {
                model: "sonnet".into(),
                effort: None,
                timeout_s: 300,
                cache_dir: Some(cache.path().to_owned()),
            })
            .expect("claude CLI");
            save_for_review();
            Self {
                claude,
                brief: PhotoBrief::new(&live_info(), &stats_of(&photo(Look::Flat))),
                _cache: cache,
            }
        }

        /// Prints what one step took and cost.
        fn report(&self, step: &str, started: Instant, before: &JudgeUsage, verdict: &str) {
            let now = self.claude.usage();
            println!(
                "\n{step}: {:.1} s, {} calls, {} cache hits, ${:.4}\n{verdict}",
                started.elapsed().as_secs_f64(),
                now.calls - before.calls,
                now.cache_hits - before.cache_hits,
                now.cost_usd - before.cost_usd
            );
        }
    }

    fn live_info() -> RawInfo {
        RawInfo {
            make: "SONY".into(),
            model: "ILCE-7M3".into(),
            lens: Some("FE 24-70mm F2.8 GM".into()),
            width: 6000,
            height: 4000,
            orientation: 1,
            iso: Some(100),
            exposure_time_s: Some(1.0 / 250.0),
            f_number: Some(8.0),
            focal_length_mm: Some(35.0),
            captured_at: None,
            as_shot_temperature: 5600.0,
            as_shot_tint: 4.0,
            raw_clipped_fraction: 0.004,
        }
    }

    #[tokio::test]
    #[ignore = "calls the real claude CLI on the subscription; run natively on the GPU host"]
    async fn live_plan() {
        let live = Live::new();
        let (lo, hi) = bounds();
        let auto = Params {
            temperature: 5600.0,
            tint: 4.0,
            exposure: 0.45,
            contrast: 12.0,
            highlights: -30.0,
            shadows: 25.0,
            whites: 8.0,
            blacks: -6.0,
            vibrance: 15.0,
            saturation: 0.0,
            ..Params::neutral(5200.0, 0.0)
        };
        let (neutral, auto_image) = (judged(Look::Flat), judged(Look::Good));
        let (t, u) = (Instant::now(), live.claude.usage());
        let plan = live
            .claude
            .plan(&live.brief, &neutral, &auto_image, &auto, &lo, &hi)
            .await
            .unwrap();
        let verdict = serde_json::to_string_pretty(&plan).unwrap();
        live.report("plan", t, &u, &verdict);
        assert!((1..=3).contains(&plan.looks.len()));
        for look in &plan.looks {
            assert_inside(&look.params, &lo, &hi);
        }
    }

    #[tokio::test]
    #[ignore = "calls the real claude CLI on the subscription; run natively on the GPU host"]
    async fn live_rank() {
        let live = Live::new();
        // The well-exposed render sits in the middle, so neither position wins it.
        let candidates = [
            judged(Look::Oversaturated),
            judged(Look::Good),
            judged(Look::Dark),
        ];
        let (t, u) = (Instant::now(), live.claude.usage());
        let rank = live.claude.rank(&live.brief, &candidates).await.unwrap();
        let verdict = serde_json::to_string_pretty(&rank).unwrap();
        live.report("rank of [oversaturated, good, dark]", t, &u, &verdict);
        assert_ne!(rank.winner, 2, "the dark render won");
    }

    #[tokio::test]
    #[ignore = "calls the real claude CLI on the subscription; run natively on the GPU host"]
    async fn live_refine() {
        let live = Live::new();
        let (lo, hi) = bounds();
        let current = Params {
            temperature: 5600.0,
            tint: 4.0,
            exposure: 0.6,
            contrast: 35.0,
            highlights: -10.0,
            shadows: 10.0,
            whites: 20.0,
            blacks: 0.0,
            vibrance: 45.0,
            saturation: 40.0,
            ..Params::neutral(5200.0, 0.0)
        };
        let image = judged(Look::Oversaturated);
        let (t, u) = (Instant::now(), live.claude.usage());
        let refine = live
            .claude
            .refine(&live.brief, &image, &current, &lo, &hi)
            .await
            .unwrap();
        let verdict = format!(
            "from {}\nto   {}\n{}",
            describe(&current),
            describe(&refine.params),
            refine.rationale
        );
        live.report("refine of the oversaturated render", t, &u, &verdict);
        assert_inside(&refine.params, &lo, &hi);
    }

    #[tokio::test]
    #[ignore = "calls the real claude CLI on the subscription; run natively on the GPU host"]
    async fn live_compare_then_cache() {
        let live = Live::new();
        let (dark, good) = (judged(Look::Dark), judged(Look::Good));
        let (t, u) = (Instant::now(), live.claude.usage());
        let compare = live
            .claude
            .compare(&live.brief, &dark, &good)
            .await
            .unwrap();
        let verdict = serde_json::to_string_pretty(&compare).unwrap();
        live.report("compare a = dark, b = good", t, &u, &verdict);
        assert!(
            compare.prefer_b,
            "the well-exposed render should beat the dark one"
        );

        let (t, u) = (Instant::now(), live.claude.usage());
        let again = live
            .claude
            .compare(&live.brief, &dark, &good)
            .await
            .unwrap();
        live.report("the same compare again", t, &u, "");
        assert_eq!(again.prefer_b, compare.prefer_b);
        let usage = live.claude.usage();
        assert_eq!(usage.calls, u.calls);
        assert_eq!(usage.cache_hits, u.cache_hits + 2);
    }
}

//! Fixed coverage choices and the smoke run's numeric-before-state order.
use super::*;

pub(super) const SMOKE_IDS: [&str; 6] = [
    "prose-512",
    "conversation-512",
    "json-2048",
    "rust-2048",
    "retrieval-4096",
    "multilingual-4096",
];
pub(super) const SHORT_STATE_IDS: [&str; 1] = ["conversation-512"];
pub(super) const STATE_IDS: [&str; 3] = ["conversation-512", "json-2048", "retrieval-4096"];
const EXTENDED_IDS: [&str; 24] = [
    "prose-512",
    "prose-2048",
    "prose-4096",
    "rust-512",
    "rust-2048",
    "rust-4096",
    "arithmetic-512",
    "arithmetic-2048",
    "arithmetic-4096",
    "json-512",
    "json-2048",
    "json-4096",
    "extraction-512",
    "extraction-2048",
    "extraction-4096",
    "multilingual-512",
    "multilingual-2048",
    "multilingual-4096",
    "conversation-512",
    "conversation-2048",
    "conversation-4096",
    "retrieval-512",
    "retrieval-2048",
    "retrieval-4096",
];

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
pub(super) enum Profile {
    #[default]
    #[serde(rename = "extended-24")]
    Extended24,
    #[serde(rename = "smoke-6")]
    Smoke6,
    #[serde(rename = "smoke-6-short-state")]
    Smoke6ShortState,
}

impl Profile {
    pub(super) fn is_smoke(self) -> bool {
        matches!(self, Self::Smoke6 | Self::Smoke6ShortState)
    }

    pub(super) fn state_ids(self) -> &'static [&'static str] {
        match self {
            Self::Smoke6ShortState => &SHORT_STATE_IDS,
            Self::Extended24 | Self::Smoke6 => &STATE_IDS,
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Extended24 => "extended-24",
            Self::Smoke6 => "smoke-6",
            Self::Smoke6ShortState => "smoke-6-short-state",
        }
    }
}

pub(super) fn selected_indices(
    profile: Profile,
    fail_fast: bool,
    selected: &[&str],
    prepared: &[&str],
    internal: &[&str],
) -> Result<Vec<usize>> {
    match profile {
        Profile::Extended24 => {
            ensure!(!fail_fast, "fail_fast requires a smoke profile");
            // Keep the explicit legacy --pilot path. The normal runner and
            // independent verifier still require 24 cases unless pilot is set.
            let unique: HashSet<_> = selected.iter().copied().collect();
            let states: HashSet<_> = internal.iter().copied().collect();
            ensure!(
                !selected.is_empty()
                    && selected == prepared
                    && unique.len() == selected.len()
                    && selected.iter().all(|id| valid_name(id)),
                "legacy suite must cover the complete prepared fixture once in order"
            );
            ensure!(
                !internal.is_empty()
                    && states.len() == internal.len()
                    && internal.iter().all(|id| unique.contains(id)),
                "legacy internal selection differs from prepared cases"
            );
        }
        Profile::Smoke6 | Profile::Smoke6ShortState => {
            ensure!(
                internal == profile.state_ids(),
                "smoke profile requires its fixed state cases"
            );
            ensure!(
                selected == SMOKE_IDS,
                "smoke-6 requires the exact ordered six cases"
            );
            ensure!(
                prepared == SMOKE_IDS || prepared == EXTENDED_IDS,
                "smoke-6 requires the exact six-case or original 24-case fixture"
            );
        }
    }
    selected
        .iter()
        .map(|id| {
            prepared
                .iter()
                .position(|candidate| candidate == id)
                .context("selected case absent")
        })
        .collect()
}

pub(super) fn smoke_noise<'a>(
    prepared: &'a [PreparedCase],
    case: &PreparedCase,
) -> Result<&'a PreparedCase> {
    let noise = prepared
        .iter()
        .filter(|other| {
            other.prompt_tokens != case.prompt_tokens
                && other
                    .prompt_tokens
                    .len()
                    .checked_add(8)
                    .is_some_and(|n| n <= case.context_tokens)
        })
        .min_by_key(|other| other.prompt_tokens.len())
        .context("no eligible state noise prompt")?;
    ensure!(
        noise.id == "prose-512",
        "smoke state noise must remain prose-512"
    );
    Ok(noise)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Numeric,
    State,
}

fn initial_case(case: &PreparedCase, required: bool) -> Value {
    let mut internal = initial_internal_report(required);
    if required {
        internal["valid"] = Value::Null;
        internal["passed"] = Value::Null;
    }
    json!({"id":case.id,"domain":case.domain,"context_tokens":case.context_tokens,
        "prompt_tokens":case.prompt_tokens.len(),"rows":ROWS,"attempted":false,
        "valid":null,"passed":null,"status":"not_run","seconds":0.0,
        "reference":{"valid":null,"passed":null,"completed_rows":0},"internal":internal})
}

// The closure owns real execution and cleanup. Tests can drive this same order
// with fixed outcomes without a model or backend.
fn execute(
    profile: Profile,
    selected: &[&PreparedCase],
    fail_fast: bool,
    report: &mut Value,
    mut step: impl FnMut(Phase, usize, &mut Value) -> Result<()>,
) -> Vec<Value> {
    let mut results: Vec<_> = selected
        .iter()
        .map(|case| initial_case(case, profile.state_ids().contains(&case.id.as_str())))
        .collect();
    let mut stopped: Option<(&str, String)> = None;
    report["early_stop"] = Value::Null;
    for (index, result) in results.iter_mut().enumerate() {
        if let Some((reason, _)) = &stopped {
            result["status"] = json!(format!("not_run_after_{reason}"));
            continue;
        }
        let started = Instant::now();
        result["attempted"] = json!(true);
        if let Err(error) = step(Phase::Numeric, index, result) {
            result["execution_error"] = json!(format!("{error:#}"));
        }
        let valid = result["reference"]["valid"] == true
            && result["reference"]["passed"].is_boolean()
            && result["reference"]["completed_rows"] == ROWS
            && result["execution_error"].is_null();
        result["valid"] = json!(valid);
        result["seconds"] = json!(started.elapsed().as_secs_f64());
        if !valid {
            result["passed"] = json!(false);
            result["status"] = json!("execution_error");
            stopped = Some(("execution_error", selected[index].id.clone()));
        } else if result["reference"]["passed"] != true {
            result["passed"] = json!(false);
            result["status"] = json!("numeric_failure");
            if fail_fast {
                stopped = Some(("numeric_failure", selected[index].id.clone()));
                report["early_stop"] = json!({"phase":"numeric","case_id":selected[index].id,
                    "reason":"numeric_failure"});
            }
        } else {
            result["status"] = json!("numeric_complete");
            if result["internal"]["required"] != true {
                result["passed"] = json!(true);
            }
        }
    }
    let all_numeric_passed = results
        .iter()
        .all(|r| r["attempted"] == true && r["valid"] == true && r["reference"]["passed"] == true);
    if all_numeric_passed {
        let mut state_error = false;
        for (index, result) in results.iter_mut().enumerate() {
            if result["internal"]["required"] != true {
                continue;
            }
            if state_error {
                result["internal"]["status"] = json!("not_run_after_execution_error");
                continue;
            }
            let started = Instant::now();
            result["internal"]["attempted"] = json!(true);
            if let Err(error) = step(Phase::State, index, result) {
                result["execution_error"] = json!(format!("{error:#}"));
            }
            let valid = result["internal"]["valid"] == true && result["execution_error"].is_null();
            result["valid"] = json!(valid);
            result["passed"] = json!(valid && result["internal"]["passed"] == true);
            result["internal"]["status"] = json!(if valid { "complete" } else { "error" });
            result["status"] = json!(if valid { "complete" } else { "execution_error" });
            result["seconds"] =
                json!(result["seconds"].as_f64().unwrap_or(0.0) + started.elapsed().as_secs_f64());
            state_error |= !valid;
        }
        report["state_phase_status"] = json!(if state_error { "error" } else { "complete" });
    } else {
        let reason = if stopped
            .as_ref()
            .is_some_and(|(reason, _)| *reason == "execution_error")
        {
            "not_run_after_execution_error"
        } else {
            "not_run_after_numeric_failure"
        };
        for result in &mut results {
            if result["internal"]["required"] == true {
                result["internal"]["status"] = json!(reason);
            }
        }
        report["state_phase_status"] = json!(reason);
    }
    report["cases"] = json!(results);
    results
}

pub(super) fn summarize(profile: Profile, results: &[Value], fail_fast: bool, report: &mut Value) {
    let state_count = profile.state_ids().len();
    let attempted = results.iter().filter(|r| r["attempted"] == true).count();
    let numeric_passed = results
        .iter()
        .filter(|r| r["reference"]["passed"] == true)
        .count();
    let numeric_failed = results
        .iter()
        .filter(|r| r["attempted"] == true && r["reference"]["passed"] == false)
        .count();
    let numeric_valid = results.iter().filter(|r| r["attempted"] == true).all(|r| {
        r["reference"]["valid"] == true
            && r["reference"]["passed"].is_boolean()
            && r["reference"]["completed_rows"] == ROWS
            && r["execution_error"].is_null()
            && r["cleanup_error"].is_null()
    });
    let unrun = results.len() - attempted;
    let justified_stop = if unrun == 0 {
        true
    } else {
        fail_fast
            && attempted > 0
            && results[..attempted - 1]
                .iter()
                .all(|r| r["reference"]["passed"] == true)
            && results[attempted - 1]["reference"]["passed"] == false
            && results[attempted..].iter().all(|r| {
                r["attempted"] == false
                    && r["valid"].is_null()
                    && r["passed"].is_null()
                    && r["status"] == "not_run_after_numeric_failure"
            })
            && report["early_stop"]
                == json!({"phase":"numeric",
                "case_id":results[attempted - 1]["id"],"reason":"numeric_failure"})
    };
    let internal: Vec<_> = results
        .iter()
        .filter(|r| r["internal"]["required"] == true)
        .collect();
    let state_attempted = internal
        .iter()
        .filter(|r| r["internal"]["attempted"] == true)
        .count();
    let state_passed = internal
        .iter()
        .filter(|r| r["internal"]["passed"] == true)
        .count();
    let state_failed = internal
        .iter()
        .filter(|r| r["internal"]["passed"] == false)
        .count();
    let state_valid = if numeric_passed == SMOKE_IDS.len() {
        state_attempted == state_count && internal.iter().all(|r| r["internal"]["valid"] == true)
    } else {
        numeric_failed > 0
            && state_attempted == 0
            && internal
                .iter()
                .all(|r| r["internal"]["valid"].is_null() && r["internal"]["passed"].is_null())
    };
    report["valid"] = json!(
        results.len() == SMOKE_IDS.len()
            && attempted > 0
            && numeric_valid
            && justified_stop
            && state_valid
    );
    report["passed"] = json!(
        report["valid"] == true && numeric_passed == SMOKE_IDS.len() && state_passed == state_count
    );
    report["numeric_counts"] = json!({"planned":SMOKE_IDS.len(),"attempted":attempted,
        "passed":numeric_passed,"failed":numeric_failed,"unrun":unrun});
    report["counts"] = json!({"planned":SMOKE_IDS.len(),"attempted":attempted,
        "passed":results.iter().filter(|r| r["passed"] == true).count(),
        "failed":results.iter().filter(|r| r["passed"] == false).count(),"unrun":unrun,
        "incomplete":results.iter().filter(|r| r["attempted"] == true && r["passed"].is_null()).count()});
    report["internal_counts"] = json!({"planned":state_count,"attempted":state_attempted,
        "passed":state_passed,"failed":state_failed,"unrun":state_count - state_attempted});
}

pub(super) fn run_smoke(
    stage: &mut PipelineStage,
    suite: &Suite,
    prepared: &Prepared,
    selected: &[&PreparedCase],
    ready: &[ReadyCase],
    out: &Path,
    report: &mut Value,
) -> Result<Vec<Value>> {
    let results = execute(
        suite.coverage_profile,
        selected,
        suite.fail_fast,
        report,
        |phase, index, result| {
            let case = selected[index];
            stage.set_max_kv_tokens(case.context_tokens);
            eprintln!(
                "Checking {} {:?}: {} prompt tokens, context {}",
                case.id,
                phase,
                case.prompt_tokens.len(),
                case.context_tokens
            );
            let outcome = match phase {
                Phase::Numeric => reference_case(
                    stage,
                    case,
                    &ready[index],
                    suite.internal.prefill_chunk,
                    out,
                    &mut result["reference"],
                ),
                Phase::State => smoke_noise(&prepared.cases, case).and_then(|noise| {
                    result["internal"]["noise_case_id"] = json!(noise.id);
                    session_checks::qualify(
                        stage,
                        &case.prompt_tokens,
                        &noise.prompt_tokens,
                        &suite.internal,
                        &mut result["internal"],
                    )
                }),
            };
            if let Err(error) = &outcome {
                let section = if phase == Phase::Numeric {
                    "reference"
                } else {
                    "internal"
                };
                result[section]["error"] = json!(format!("{error:#}"));
            }
            // Cleanup must run on both success and failure before deciding to stop.
            let cleanup = session_checks::clean(stage);
            if let Err(error) = &cleanup {
                result["cleanup_error"] = json!(format!("{error:#}"));
            }
            if phase == Phase::Numeric {
                let raw = out.join(format!("{}.actual.f32", case.id));
                if raw.exists() {
                    result["reference"]["actual_logits_sha256"] = json!(sha256(&raw)?);
                    result["reference"]["actual_logits_bytes"] = json!(raw.metadata()?.len());
                }
                // Keep completed numeric evidence if a later state phase is stopped.
                // This file is progress only, never a full-case or suite verdict.
                write_json_new(
                    &out.join(format!("{}.numeric-progress.json", case.id)),
                    &json!({"scope":"numeric-only progress; state checks not included",
                    "case_id":case.id,"reference":result["reference"],
                    "numeric_execution_completed":outcome.is_ok() && cleanup.is_ok(),
                    "cleanup_error":result["cleanup_error"],"full_case_verdict":null}),
                )?;
            }
            outcome.and(cleanup)
        },
    );
    for result in &results {
        let id = result["id"].as_str().context("missing result case ID")?;
        write_json_new(&out.join(format!("{id}.result.json")), result)?;
    }
    Ok(results)
}

/// Separate diagnostic entry point. It proves state behavior only; its report
/// cannot stand in for a numeric matrix run. Recovery must bind numeric evidence
/// from its original run in a separate, explicit record.
pub(super) fn run_state_only(
    stage: &mut PipelineStage,
    suite: &Suite,
    prepared: &Prepared,
    selected: &[&PreparedCase],
    out: &Path,
) -> Result<Vec<Value>> {
    ensure!(
        suite.coverage_profile == Profile::Smoke6ShortState,
        "state-only requires smoke-6-short-state"
    );
    let case = selected
        .iter()
        .find(|case| case.id == SHORT_STATE_IDS[0])
        .context("missing short state case")?;
    let noise = smoke_noise(&prepared.cases, case)?;
    let mut result = initial_case(case, true);
    result["rows"] = json!(0);
    result["attempted"] = json!(true);
    result["internal"]["attempted"] = json!(true);
    result["internal"]["noise_case_id"] = json!(noise.id);
    let started = Instant::now();
    stage.set_max_kv_tokens(case.context_tokens);
    eprintln!(
        "Checking {} StateOnly: {} prompt tokens, context {}",
        case.id,
        case.prompt_tokens.len(),
        case.context_tokens
    );
    let outcome = session_checks::qualify(
        stage,
        &case.prompt_tokens,
        &noise.prompt_tokens,
        &suite.internal,
        &mut result["internal"],
    );
    if let Err(error) = &outcome {
        result["execution_error"] = json!(format!("{error:#}"));
    }
    let cleanup = session_checks::clean(stage);
    if let Err(error) = &cleanup {
        result["cleanup_error"] = json!(format!("{error:#}"));
    }
    let valid = outcome.is_ok() && cleanup.is_ok() && result["internal"]["valid"] == true;
    result["valid"] = json!(valid);
    result["passed"] = json!(valid && result["internal"]["passed"] == true);
    result["internal"]["status"] = json!(if valid { "complete" } else { "error" });
    result["status"] = json!(if valid {
        "state_only_complete"
    } else {
        "execution_error"
    });
    result["seconds"] = json!(started.elapsed().as_secs_f64());
    write_json_new(&out.join(format!("{}.result.json", case.id)), &result)?;
    Ok(vec![result])
}

pub(super) fn summarize_state_only(results: &[Value], report: &mut Value) {
    let valid = results.len() == 1
        && results[0]["id"] == SHORT_STATE_IDS[0]
        && results[0]["valid"] == true
        && results[0]["rows"] == 0
        && results[0]["reference"]["valid"].is_null()
        && results[0]["reference"]["passed"].is_null()
        && results[0]["reference"]["completed_rows"] == 0
        && results[0]["internal"]["attempted"] == true
        && results[0]["internal"]["valid"] == true
        && results[0]["internal"]["passed"].is_boolean();
    let passed = valid && results[0]["internal"]["passed"] == true;
    report["valid"] = json!(valid);
    report["passed"] = json!(passed);
    report["numeric_phase_status"] = json!("not_requested");
    report["numeric_counts"] = json!({"planned":0,"attempted":0,"passed":0,"failed":0,"unrun":0});
    report["state_phase_status"] = json!(if valid { "complete" } else { "error" });
    report["internal_counts"] = json!({"planned":1,"attempted":results.len(),
        "passed":usize::from(passed),"failed":usize::from(valid && !passed),"unrun":0});
    report["counts"] = report["internal_counts"].clone();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixtures() -> Vec<PreparedCase> {
        SMOKE_IDS
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let (domain, context) = id.rsplit_once('-').unwrap();
                let prompt = vec![i as u32; i + 2];
                PreparedCase {
                    id: (*id).into(),
                    domain: domain.into(),
                    context_tokens: context.parse().unwrap(),
                    rendered_prompt_utf8: format!("complete prompt {id}"),
                    continuation_utf8: "continuation".into(),
                    prompt_original_ids: prompt.clone(),
                    prompt_tokens: prompt,
                    continuation_original_ids: vec![1; ROWS],
                    teacher_forcing_tokens: vec![1; ROWS],
                    prompt_add_special: true,
                    continuation_add_special: false,
                    parse_special: true,
                }
            })
            .collect()
    }

    fn complete(phase: Phase, passed: bool, result: &mut Value) {
        match phase {
            Phase::Numeric => {
                result["reference"] = json!({"valid":true,"passed":passed,"completed_rows":ROWS})
            }
            Phase::State => {
                result["internal"]["valid"] = json!(true);
                result["internal"]["passed"] = json!(passed);
            }
        }
    }

    #[test]
    fn fixed_smoke_selection_accepts_fresh_six_and_unchanged_full_fixture() -> Result<()> {
        assert_eq!(
            selected_indices(Profile::Smoke6, true, &SMOKE_IDS, &SMOKE_IDS, &STATE_IDS)?,
            (0..6).collect::<Vec<_>>()
        );
        let full = selected_indices(Profile::Smoke6, true, &SMOKE_IDS, &EXTENDED_IDS, &STATE_IDS)?;
        assert_eq!(
            full.iter().map(|&i| EXTENDED_IDS[i]).collect::<Vec<_>>(),
            SMOKE_IDS
        );
        let mut reversed = SMOKE_IDS;
        reversed.swap(0, 1);
        assert!(
            selected_indices(Profile::Smoke6, true, &reversed, &EXTENDED_IDS, &STATE_IDS).is_err()
        );
        let mut duplicate = SMOKE_IDS;
        duplicate[5] = duplicate[0];
        assert!(
            selected_indices(Profile::Smoke6, true, &duplicate, &EXTENDED_IDS, &STATE_IDS).is_err()
        );
        assert!(
            selected_indices(Profile::Smoke6, true, &SMOKE_IDS, &reversed, &STATE_IDS).is_err()
        );
        assert!(
            selected_indices(
                Profile::Smoke6,
                true,
                &SMOKE_IDS,
                &SMOKE_IDS,
                &STATE_IDS[..2]
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn omitted_fields_keep_extended_default_and_unknown_profiles_fail() -> Result<()> {
        let pin = json!({"path":"unused","sha256":"0".repeat(64)});
        let value = json!({"schema_version":1,"suite_id":"suite","backend":"cpu",
            "prepared":pin,"corpus":pin,"expected_binary_sha256":"0".repeat(64),
            "expected_token_identity_sha256":"0".repeat(64),"source_pins":[],"cases":[],
            "internal_cases":[],"internal":{"decode_tokens":8,"prefill_chunk":128,
                "comparison_prefill_chunk":32,"cache_max_bytes":4294967296u64}});
        let suite: Suite = serde_json::from_value(value.clone())?;
        assert_eq!(suite.coverage_profile, Profile::Extended24);
        assert!(!suite.fail_fast);
        let mut wrong = value;
        wrong["coverage_profile"] = json!("arbitrary-subset");
        assert!(serde_json::from_value::<Suite>(wrong).is_err());
        assert!(
            selected_indices(
                Profile::Extended24,
                true,
                &EXTENDED_IDS,
                &EXTENDED_IDS,
                &STATE_IDS
            )
            .is_err()
        );
        selected_indices(
            Profile::Extended24,
            false,
            &SMOKE_IDS,
            &SMOKE_IDS,
            &STATE_IDS,
        )?;
        assert_eq!(
            selected_indices(
                Profile::Extended24,
                false,
                &["prose-512"],
                &["prose-512"],
                &["prose-512"]
            )?,
            vec![0]
        );
        assert!(
            selected_indices(
                Profile::Extended24,
                false,
                &["prose-512"],
                &SMOKE_IDS,
                &["prose-512"]
            )
            .is_err()
        );
        assert!(
            selected_indices(
                Profile::Extended24,
                false,
                &["prose-512", "prose-512"],
                &["prose-512", "prose-512"],
                &["prose-512"]
            )
            .is_err()
        );
        selected_indices(
            Profile::Extended24,
            false,
            &EXTENDED_IDS,
            &EXTENDED_IDS,
            &STATE_IDS,
        )?;
        Ok(())
    }

    #[test]
    fn noise_must_remain_the_original_prose_prompt() -> Result<()> {
        let mut cases = fixtures();
        assert_eq!(smoke_noise(&cases, &cases[1])?.id, "prose-512");
        cases[3].prompt_tokens = vec![99];
        assert!(smoke_noise(&cases, &cases[1]).is_err());
        Ok(())
    }

    #[test]
    fn short_state_profile_keeps_six_numeric_cases_and_exact_one_state_case() -> Result<()> {
        for prepared in [&SMOKE_IDS[..], &EXTENDED_IDS[..]] {
            let indices = selected_indices(
                Profile::Smoke6ShortState,
                true,
                &SMOKE_IDS,
                prepared,
                &SHORT_STATE_IDS,
            )?;
            assert_eq!(
                indices.iter().map(|&i| prepared[i]).collect::<Vec<_>>(),
                SMOKE_IDS
            );
        }
        assert!(
            selected_indices(
                Profile::Smoke6ShortState,
                true,
                &SMOKE_IDS,
                &SMOKE_IDS,
                &STATE_IDS
            )
            .is_err()
        );
        assert!(
            selected_indices(
                Profile::Smoke6,
                true,
                &SMOKE_IDS,
                &SMOKE_IDS,
                &SHORT_STATE_IDS
            )
            .is_err()
        );
        assert!(
            selected_indices(
                Profile::Smoke6ShortState,
                true,
                &SMOKE_IDS[..5],
                &SMOKE_IDS,
                &SHORT_STATE_IDS
            )
            .is_err()
        );
        assert_eq!(Profile::Smoke6ShortState.name(), "smoke-6-short-state");
        Ok(())
    }

    #[test]
    fn short_state_runs_only_conversation_after_all_six_numeric_passes() {
        let cases = fixtures();
        let selected: Vec<_> = cases.iter().collect();
        let mut report = json!({});
        let mut calls = Vec::new();
        let results = execute(
            Profile::Smoke6ShortState,
            &selected,
            true,
            &mut report,
            |phase, index, result| {
                calls.push((phase, index));
                complete(phase, true, result);
                Ok(())
            },
        );
        summarize(Profile::Smoke6ShortState, &results, true, &mut report);
        let mut expected: Vec<_> = (0..6).map(|i| (Phase::Numeric, i)).collect();
        expected.push((Phase::State, 1));
        assert_eq!(calls, expected);
        assert_eq!(report["valid"], true);
        assert_eq!(report["passed"], true);
        assert_eq!(report["numeric_counts"]["passed"], 6);
        assert_eq!(
            report["internal_counts"],
            json!({"planned":1,"attempted":1,"passed":1,"failed":0,"unrun":0})
        );
        assert_eq!(results[2]["internal"]["required"], false);
        assert_eq!(results[4]["internal"]["required"], false);
    }

    #[test]
    fn short_state_failure_and_skipped_state_never_become_passes() {
        let cases = fixtures();
        let selected: Vec<_> = cases.iter().collect();
        for failed_phase in [Phase::Numeric, Phase::State] {
            let mut report = json!({});
            let mut calls = Vec::new();
            let results = execute(
                Profile::Smoke6ShortState,
                &selected,
                true,
                &mut report,
                |phase, index, result| {
                    calls.push((phase, index));
                    complete(phase, phase != failed_phase || index != 1, result);
                    Ok(())
                },
            );
            summarize(Profile::Smoke6ShortState, &results, true, &mut report);
            assert_eq!(report["valid"], true);
            assert_eq!(report["passed"], false);
            if failed_phase == Phase::Numeric {
                assert_eq!(calls, vec![(Phase::Numeric, 0), (Phase::Numeric, 1)]);
                assert!(results[1]["internal"]["passed"].is_null());
                assert_eq!(report["internal_counts"]["unrun"], 1);
                assert_eq!(report["numeric_counts"]["unrun"], 4);
            } else {
                assert_eq!(report["internal_counts"]["failed"], 1);
                assert_eq!(report["numeric_counts"]["passed"], 6);
            }
        }
    }

    #[test]
    fn state_only_report_never_claims_numeric_work() {
        let cases = fixtures();
        let mut result = initial_case(&cases[1], true);
        result["rows"] = json!(0);
        result["valid"] = json!(true);
        result["internal"]["attempted"] = json!(true);
        complete(Phase::State, true, &mut result);
        let mut report = json!({});
        summarize_state_only(&[result.clone()], &mut report);
        assert_eq!(report["valid"], true);
        assert_eq!(report["passed"], true);
        assert_eq!(report["numeric_phase_status"], "not_requested");
        assert_eq!(report["numeric_counts"]["attempted"], 0);
        result["reference"]["passed"] = json!(true);
        summarize_state_only(&[result], &mut report);
        assert_eq!(report["valid"], false);
        assert_eq!(report["passed"], false);
    }

    #[test]
    fn numeric_rejection_stops_before_later_numeric_or_any_state_work() {
        let cases = fixtures();
        let selected: Vec<_> = cases.iter().collect();
        let mut calls = Vec::new();
        let mut report = json!({});
        let results = execute(
            Profile::Smoke6,
            &selected,
            true,
            &mut report,
            |phase, index, result| {
                calls.push((phase, index));
                complete(phase, index != 2, result);
                Ok(())
            },
        );
        summarize(Profile::Smoke6, &results, true, &mut report);
        assert_eq!(
            calls,
            vec![
                (Phase::Numeric, 0),
                (Phase::Numeric, 1),
                (Phase::Numeric, 2)
            ]
        );
        assert_eq!(report["valid"], true);
        assert_eq!(report["passed"], false);
        assert_eq!(
            report["numeric_counts"],
            json!({"planned":6,"attempted":3,"passed":2,"failed":1,"unrun":3})
        );
        assert_eq!(
            report["counts"],
            json!({"planned":6,"attempted":3,"passed":1,"failed":1,"incomplete":1,"unrun":3})
        );
        assert_eq!(
            report["internal_counts"],
            json!({"planned":3,"attempted":0,"passed":0,"failed":0,"unrun":3})
        );
        assert!(results[1]["passed"].is_null());
        for result in &results[3..] {
            assert_eq!(result["attempted"], false);
            assert!(result["valid"].is_null() && result["passed"].is_null());
            assert_eq!(
                result["reference"],
                json!({"valid":null,"passed":null,"completed_rows":0})
            );
        }
        assert_eq!(
            results[4]["internal"]["status"],
            "not_run_after_numeric_failure"
        );
    }

    #[test]
    fn last_numeric_failure_still_skips_all_states() {
        let cases = fixtures();
        let selected: Vec<_> = cases.iter().collect();
        let mut report = json!({});
        let results = execute(
            Profile::Smoke6,
            &selected,
            true,
            &mut report,
            |phase, index, result| {
                assert_eq!(phase, Phase::Numeric);
                complete(phase, index != 5, result);
                Ok(())
            },
        );
        summarize(Profile::Smoke6, &results, true, &mut report);
        assert_eq!(report["valid"], true);
        assert_eq!(report["passed"], false);
        assert_eq!(report["numeric_counts"]["unrun"], 0);
        assert_eq!(report["internal_counts"]["unrun"], 3);
        assert_eq!(report["early_stop"]["case_id"], "multilingual-4096");
    }

    #[test]
    fn disabling_fail_fast_keeps_all_numeric_work_but_no_states_after_rejection() {
        let cases = fixtures();
        let selected: Vec<_> = cases.iter().collect();
        let mut report = json!({});
        let mut count = 0;
        let results = execute(
            Profile::Smoke6,
            &selected,
            false,
            &mut report,
            |phase, index, result| {
                assert_eq!(phase, Phase::Numeric);
                count += 1;
                complete(phase, index != 0, result);
                Ok(())
            },
        );
        summarize(Profile::Smoke6, &results, false, &mut report);
        assert_eq!(count, 6);
        assert_eq!(report["valid"], true);
        assert_eq!(report["passed"], false);
        assert!(report["early_stop"].is_null());
        assert_eq!(report["internal_counts"]["unrun"], 3);
    }

    #[test]
    fn all_numeric_work_precedes_states_and_state_rejection_does_not_skip_later_checks() {
        let cases = fixtures();
        let selected: Vec<_> = cases.iter().collect();
        let mut report = json!({});
        let mut calls = Vec::new();
        let results = execute(
            Profile::Smoke6,
            &selected,
            true,
            &mut report,
            |phase, index, result| {
                calls.push((phase, index));
                complete(phase, phase == Phase::Numeric || index != 1, result);
                Ok(())
            },
        );
        summarize(Profile::Smoke6, &results, true, &mut report);
        let mut expected: Vec<_> = (0..6).map(|i| (Phase::Numeric, i)).collect();
        expected.extend([(Phase::State, 1), (Phase::State, 2), (Phase::State, 4)]);
        assert_eq!(calls, expected);
        assert_eq!(report["valid"], true);
        assert_eq!(report["passed"], false);
        assert_eq!(
            report["internal_counts"],
            json!({"planned":3,"attempted":3,"passed":2,"failed":1,"unrun":0})
        );
        assert_eq!(report["state_phase_status"], "complete");
    }

    #[test]
    fn full_smoke_pass_requires_all_numeric_and_state_work() {
        let cases = fixtures();
        let selected: Vec<_> = cases.iter().collect();
        let mut report = json!({});
        let results = execute(
            Profile::Smoke6,
            &selected,
            true,
            &mut report,
            |phase, _, result| {
                complete(phase, true, result);
                Ok(())
            },
        );
        summarize(Profile::Smoke6, &results, true, &mut report);
        assert_eq!(report["valid"], true);
        assert_eq!(report["passed"], true);
        assert_eq!(report["counts"]["passed"], 6);
        assert_eq!(report["internal_counts"]["passed"], 3);
        let mut forged = results;
        forged[0]["attempted"] = json!(false);
        forged[0]["valid"] = Value::Null;
        summarize(Profile::Smoke6, &forged, true, &mut report);
        assert_eq!(report["valid"], false);
    }

    #[test]
    fn cleanup_or_execution_errors_are_never_valid_numeric_rejections() {
        let cases = fixtures();
        let selected: Vec<_> = cases.iter().collect();
        for failed_phase in [Phase::Numeric, Phase::State] {
            let mut report = json!({});
            let mut calls = Vec::new();
            let results = execute(
                Profile::Smoke6,
                &selected,
                true,
                &mut report,
                |phase, index, result| {
                    calls.push((phase, index));
                    complete(phase, true, result);
                    if phase == failed_phase && index == 1 {
                        result["cleanup_error"] = json!("retained cache");
                        anyhow::bail!("cleanup failed");
                    }
                    Ok(())
                },
            );
            summarize(Profile::Smoke6, &results, true, &mut report);
            assert_eq!(report["valid"], false);
            assert_eq!(report["passed"], false);
            assert!(!calls.contains(&(failed_phase, 2)));
        }
    }
}

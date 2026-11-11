"""Tiny synthetic rows exercise smoke rejection, reuse and archive coverage."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

import verify_results as v
import archive_rows as archive
from test_verify_results import fake_job, pin, write


def smoke_job(root, fail_at=None, state_failure=False, reuse=False, profile="smoke-6"):
    internal_ids = v.SMOKE_INTERNALS[profile]
    fake_job(root)
    prepared = v.load(root / "prepared.json")
    corpus = v.load(root / "corpus.json")
    suite = v.load(root / "suite.json")
    oracle_suite = v.load(root / "oracle/suite.json")
    report = v.load(root / "engine/report.json")
    execution = v.load(root / "execution.json")
    old = None
    if reuse:
        old = root.parent / (root.name + "-old")
        shutil.copytree(root, old)
        for label in ["prepare", "oracle"]:
            command = v.load(old / f"{label}.command.json")
            command["timed_out"] = False
            command["command"][-2:] = list(map(str, [old / "corpus.json", old / "prepared.json"] if label == "prepare"
                                               else [old / "prepared.json", old / "oracle"]))
            write(old / f"{label}.command.json", command)
    else:
        prepared["cases"] = [{c["id"]: c for c in prepared["cases"]}[label] for label in v.SMOKE_CASES]
        corpus["cases"] = [{c["id"]: c for c in corpus["cases"]}[label] for label in v.SMOKE_CASES]
        write(root / "corpus.json", corpus)
        prepared["corpus_sha256"] = v.digest(root / "corpus.json")
        prepared["token_identity_sha256"] = hashlib.sha256(json.dumps(prepared["cases"], ensure_ascii=False,
                                                    separators=(",", ":")).encode()).hexdigest()
        write(root / "prepared.json", prepared)
        oracle_suite.update(prepared_sha256=v.digest(root / "prepared.json"),
                            corpus_sha256=prepared["corpus_sha256"], completed_cases=6, expected_cases=6)
    prepared_sha = v.digest(root / "prepared.json")
    selected = [{c["id"]: c for c in prepared["cases"]}[label] for label in v.SMOKE_CASES]
    write(root / "shared-identity.json", {"model_id": "model", "corpus_sha256": prepared["corpus_sha256"],
          "token_identity_sha256": prepared["token_identity_sha256"], "cases": prepared["cases"]})
    write(root / "smoke-identity.json", {"model_id": "model", "coverage_profile": "smoke-6", "cases": selected})
    entry_by_id = {e["id"]: e for e in suite["cases"]}
    result_by_id = {r["id"]: r for r in report["cases"]}
    oracle_by_id = {c["id"]: c for c in oracle_suite["cases"]}
    results, refs, entries = [], [], []
    n = 6 if fail_at is None else fail_at + 1
    for i, label in enumerate(v.SMOKE_CASES):
        oracle_path = root / "oracle" / f"{label}.json"
        raw_path = root / "oracle" / f"{label}.f32"
        oracle = v.load(oracle_path)
        if not reuse:
            oracle["prepared_sha256"] = prepared_sha
            write(oracle_path, oracle)
            oracle_by_id[label]["oracle_json_sha256"] = v.digest(oracle_path)
        entry = entry_by_id[label]
        entry["reference"] = pin(oracle_path)
        entries.append(entry)
        refs.append({"json": pin(oracle_path, True), "raw": pin(raw_path, True)})
        result = result_by_id[label]
        if label not in internal_ids:
            result["internal"] = {"required": False, "attempted": False, "status": "not_selected",
                                  "valid": None, "passed": None, "modes": []}
        if label in internal_ids and fail_at is None:
            result["internal"]["noise_case_id"] = "prose-512"
        actual = root / "engine" / f"{label}.actual.f32"
        if i >= n:
            actual.unlink()
            result.update(attempted=False, valid=None, passed=None, status="not_run_after_numeric_failure",
                          reference={"valid": None, "passed": None, "completed_rows": 0})
        elif i == fail_at:
            data = np.tile(np.array([-1, 0, 3], dtype="<f4"), (128, 1))
            actual.write_bytes(data.tobytes())
            reference = np.array([1, 2, -1], dtype="<f4")
            rows = [v.metric(reference, data[j], 1, j) for j in range(128)]
            result["reference"].update(passed=False, comparisons=rows, matching_argmax=0,
                actual_logits_sha256=v.digest(actual), mean_kl=sum(r["kl"] for r in rows)/128,
                max_kl=max(r["kl"] for r in rows), actual_mean_nll=sum(r["actual_nll"] for r in rows)/128,
                mean_nll_delta=sum(r["nll_delta"] for r in rows)/128)
            result["passed"] = False
        if fail_at is not None:
            required = label in internal_ids
            result["internal"] = {"required": required, "attempted": False, "valid": None, "passed": None,
                                  "modes": [], "status": "not_run_after_numeric_failure" if required else "not_selected"}
            if i < n and i != fail_at and required:
                result["passed"] = None
        elif state_failure and label == internal_ids[0]:
            check = result["internal"]["modes"][0]["checks"][0]
            check.update(passed=False, first_mismatch=0, actual_tokens=[2] + [1]*7)
            result["internal"]["passed"] = False
            result["passed"] = False
        write(root / "engine" / f"{label}.result.json", result)
        results.append(result)
    for directory in ("oracle", "engine"):
        for path in (root / directory).iterdir():
            if path.name not in ("suite.json", "report.json") and not any(path.name.startswith(label + ".") for label in v.SMOKE_CASES):
                path.unlink()
    if not reuse:
        oracle_suite["cases"] = [oracle_by_id[label] for label in v.SMOKE_CASES]
        write(root / "oracle/suite.json", oracle_suite)
    suite.update(coverage_profile=profile, fail_fast=True, cases=entries, internal_cases=internal_ids,
                 expected_token_identity_sha256=prepared["token_identity_sha256"],
                 prepared=pin(root / "prepared.json"), corpus=pin(root / "corpus.json"))
    suite["internal"]["cache_max_bytes"] = 4 * 1024**3
    write(root / "suite.json", suite)
    passed = fail_at is None and not state_failure
    report.update(coverage_profile=profile, fail_fast=True, execution_policy="numeric_then_state", cases=results,
        internal_case_ids=internal_ids,
        selected_case_ids=v.SMOKE_CASES,
        passed=passed, token_identity_sha256=prepared["token_identity_sha256"], internal_options=suite["internal"],
        state_phase_status="complete" if fail_at is None else "not_run_after_numeric_failure",
        early_stop=None if fail_at is None else {"phase": "numeric", "case_id": v.SMOKE_CASES[fail_at], "reason": "numeric_failure"},
        numeric_counts={"planned": 6, "attempted": n, "passed": n-(fail_at is not None), "failed": int(fail_at is not None), "unrun": 6-n},
        internal_counts={"planned": len(internal_ids), "attempted": len(internal_ids) if fail_at is None else 0,
                         "passed": (len(internal_ids)-int(state_failure)) if fail_at is None else 0,
                         "failed": int(state_failure) if fail_at is None else 0, "unrun": 0 if fail_at is None else len(internal_ids)},
        counts={"planned": 6, "attempted": n, "passed": sum(r["passed"] is True for r in results),
                "failed": sum(r["passed"] is False for r in results), "unrun": 6-n,
                "incomplete": sum(r["attempted"] and r["passed"] is None for r in results)})
    execution.update(coverage_profile=profile, engine_passed=passed, engine_exit_code=0 if passed else 1,
                     identity=pin(root / "shared-identity.json"), smoke_identity=pin(root / "smoke-identity.json"))
    if reuse:
        reference_build = root / "reference-build.json"
        write(reference_build, {"binary": pin(root / "engine-bin")})
        execution["source_pins"][str(reference_build)] = v.digest(reference_build)
        write(root / "pins.json", execution["source_pins"])
        suite["source_pins"] = [{"path": p, "sha256": h} for p, h in execution["source_pins"].items()]
        write(root / "suite.json", suite)
        destinations = ["prepared.json", "corpus.json", "oracle/suite.json"] + [f"oracle/{label}{s}" for label in v.SMOKE_CASES for s in (".json", ".f32")]
        write(root / "oracle-reuse.json", {"schema_version": 1, "source_job": str(old),
            "original_commands": {label: pin(old / f"{label}.command.json") for label in ("prepare", "oracle")},
            "original_pins": pin(old / "pins.json"), "reference_build": pin(reference_build),
            "preserved_files": [{"source": pin(old / dest), "destination": dest} for dest in destinations]})
        execution["oracle_reuse"] = pin(root / "oracle-reuse.json")
        execution["commands"] = execution["commands"][-1:]
    report["input_pins"].update(suite=pin(root / "suite.json", True), prepared=pin(root / "prepared.json", True),
        corpus=pin(root / "corpus.json", True), references=refs,
        sources=[pin(Path(p), True) for p in execution["source_pins"]])
    execution["commands"][-1]["exit_code"] = execution["engine_exit_code"]
    write(root / "engine.command.json", execution["commands"][-1])
    write(root / "execution.json", execution)
    write(root / "engine/report.json", report)


class SmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name).resolve()
        self.root = self.parent / "job"
        self.root.mkdir()

    def review(self):
        return v.review(self.root, profile="smoke-6")

    def mutate(self, callback):
        report = v.load(self.root / "engine/report.json")
        callback(report)
        write(self.root / "engine/report.json", report)
        for result in report["cases"]:
            write(self.root / "engine" / f"{result['id']}.result.json", result)

    def test_six_passes_with_original_36_state_checks(self):
        smoke_job(self.root)
        r = self.review()
        self.assertTrue(r["passed"])
        self.assertEqual((r["cases_checked"], r["rows_checked"], r["internal_checks_checked"], len(r["raw_files"])), (6,768,36,12))

    def test_first_and_last_numeric_rejections_are_valid(self):
        for index in (0, 5):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)/"job";root.mkdir();smoke_job(root, fail_at=index)
                r = v.review(root, profile="smoke-6")
                self.assertTrue(r["valid_evidence"]);self.assertFalse(r["passed"])
                self.assertEqual((r["rows_checked"],r["oracle_rows_checked"],r["internal_checks_checked"]),((index+1)*128,768,0))
                self.assertEqual(len(r["raw_files"]),6+index+1)

    def test_state_rejection_keeps_all_state_checks(self):
        smoke_job(self.root,state_failure=True)
        r=self.review();self.assertFalse(r["passed"]);self.assertEqual(r["internal_checks_checked"],36)

    def test_full24_oracle_reuse_keeps_bytes_and_selected_scope(self):
        smoke_job(self.root,reuse=True,fail_at=2)
        before=(self.root/'prepared.json').read_bytes()
        r=self.review();self.assertFalse(r['passed']);self.assertEqual(r['rows_checked'],384)
        self.assertEqual((self.root/'prepared.json').read_bytes(),before)
        self.assertEqual(len(v.load(self.root/'prepared.json')['cases']),24)

    def test_reuse_changed_copy_is_rejected(self):
        smoke_job(self.root,reuse=True)
        p=self.root/'oracle-reuse.json';r=v.load(p);r['preserved_files'][0]['source']['sha256']='0'*64;write(p,r)
        e=v.load(self.root/'execution.json');e['oracle_reuse']=pin(p);write(self.root/'execution.json',e)
        with self.assertRaises(ValueError):self.review()

    def test_unrun_cannot_claim_pass(self):
        smoke_job(self.root,fail_at=0)
        self.mutate(lambda r:r['cases'][1].update(passed=True))
        with self.assertRaisesRegex(ValueError,'unrun numeric'):self.review()

    def test_changed_noise_fixture_is_rejected(self):
        smoke_job(self.root)
        self.mutate(lambda r:r['cases'][1]['internal'].update(noise_case_id='rust-2048'))
        with self.assertRaisesRegex(ValueError,'state noise'):self.review()

    def test_reuse_optional_admission_is_hash_checked(self):
        smoke_job(self.root,reuse=True)
        admission=self.root/'admission.json';write(admission,{'prior_check':True})
        p=self.root/'oracle-reuse.json';r=v.load(p);r['admission']=pin(admission);write(p,r)
        e=v.load(self.root/'execution.json');e['oracle_reuse']=pin(p);write(self.root/'execution.json',e)
        self.assertTrue(self.review()['passed'])
        write(admission,{'prior_check':False})
        with self.assertRaisesRegex(ValueError,'pin hash'):self.review()

    def test_unrun_cannot_have_actual_raw(self):
        smoke_job(self.root,fail_at=0)
        (self.root/'engine/conversation-512.actual.f32').write_bytes(b'\0'*1536)
        with self.assertRaisesRegex(ValueError,'unrun case has actual'):self.review()

    def test_skip_without_true_numeric_reject_is_rejected(self):
        smoke_job(self.root,fail_at=0)
        self.mutate(lambda r:r.update(early_stop={'phase':'numeric','case_id':'json-2048','reason':'numeric_failure'}))
        with self.assertRaisesRegex(ValueError,'early-stop'):self.review()

    def test_state_run_after_numeric_failure_is_rejected(self):
        smoke_job(self.root,fail_at=0)
        self.mutate(lambda r:r['cases'][1]['internal'].update(attempted=True,passed=True))
        with self.assertRaisesRegex(ValueError,'unrun state'):self.review()

    def test_unrun_oracle_still_requires_finite_rows(self):
        smoke_job(self.root,fail_at=0)
        p=self.root/'oracle/multilingual-4096.f32';p.write_bytes(b'bad')
        with self.assertRaises(ValueError):self.review()

    def test_smoke_requires_explicit_profile(self):
        smoke_job(self.root)
        with self.assertRaisesRegex(ValueError,'profile'):v.review(self.root)

    def test_archive_smoke_rejection_binds_only_produced_raws(self):
        smoke_job(self.root,fail_at=1)
        r=self.review()
        # Production suite ID is irrelevant to the metric fixture; set only
        # these archive scope fields after review for the small archive test.
        prepared=v.load(self.root/'prepared.json');prepared['suite_id']=archive.SUITE_ID;write(self.root/'prepared.json',prepared)
        r.update(suite_id=archive.SUITE_ID,prepared_sha256=v.digest(self.root/'prepared.json'))
        write(self.root/'independent-review.json',r)
        _,_,files,_=archive.validate_job(self.parent,self.root)
        self.assertEqual(len(files),8)
        zstd=shutil.which('zstd')
        if zstd:
            receipt,removal=archive.archive(self.parent,self.root,Path(zstd))
            self.assertFalse(receipt['independent_review']['passed'])
            self.assertEqual(receipt['coverage_profile'],'smoke-6')
            self.assertEqual(len(removal['removed_raw_paths']),8)

    def test_archive_rejects_omitted_unrun_reference(self):
        smoke_job(self.root,fail_at=0)
        r=self.review();r['raw_files']=r['raw_files'][:-1]
        prepared=v.load(self.root/'prepared.json');prepared['suite_id']=archive.SUITE_ID;write(self.root/'prepared.json',prepared)
        r.update(suite_id=archive.SUITE_ID,prepared_sha256=v.digest(self.root/'prepared.json'))
        write(self.root/'independent-review.json',r)
        with self.assertRaisesRegex(ValueError,'smoke raw coverage'):archive.validate_job(self.parent,self.root)


class ShortStateTests(unittest.TestCase):
    setUp = SmokeTests.setUp
    mutate = SmokeTests.mutate
    review = SmokeTests.review

    def short_review(self):
        return v.review(self.root, profile='smoke-6-short-state')

    def test_short_profile_has_twelve_checks(self):
        smoke_job(self.root, profile='smoke-6-short-state')
        r = self.short_review()
        self.assertTrue(r['passed'])
        self.assertEqual((r['cases_checked'], r['rows_checked'], r['internal_modes_checked'],
                          r['internal_checks_checked']), (6, 768, 2, 12))
        self.assertEqual(r['internal_case_ids'], ['conversation-512'])
        self.assertEqual(v.load(self.root/'smoke-identity.json')['coverage_profile'], 'smoke-6')

    def test_short_numeric_rejection_has_zero_states(self):
        smoke_job(self.root, fail_at=2, profile='smoke-6-short-state')
        r = self.short_review()
        self.assertTrue(r['valid_evidence']); self.assertFalse(r['passed'])
        self.assertEqual((r['rows_checked'], r['internal_checks_checked']), (384, 0))
        self.assertEqual(r['cases'][1]['passed'], None)
        self.assertEqual(r['cases'][2]['passed'], False)

    def test_short_state_failure_is_valid_with_all_twelve_checks(self):
        smoke_job(self.root, state_failure=True, profile='smoke-6-short-state')
        r = self.short_review()
        self.assertTrue(r['valid_evidence']); self.assertFalse(r['passed'])
        self.assertEqual(r['internal_checks_checked'], 12)

    def test_profiles_are_not_interchangeable(self):
        smoke_job(self.root, profile='smoke-6-short-state')
        with self.assertRaisesRegex(ValueError, 'profile'):
            self.review()

    def test_old_state_count_cannot_claim_short_pass(self):
        smoke_job(self.root, profile='smoke-6-short-state')
        self.mutate(lambda r: r['internal_counts'].update(planned=3, attempted=3, passed=3))
        with self.assertRaisesRegex(ValueError, 'internal counts'):
            self.short_review()

    def test_short_reuse_preserves_full24_prepared_metadata(self):
        smoke_job(self.root, reuse=True, profile='smoke-6-short-state')
        before = (self.root/'prepared.json').read_bytes()
        r = self.short_review()
        self.assertTrue(r['passed']); self.assertEqual(r['internal_checks_checked'], 12)
        self.assertEqual(before, (self.root/'prepared.json').read_bytes())

    def test_state_only_report_is_not_a_complete_numeric_cell(self):
        smoke_job(self.root, profile='smoke-6-short-state')
        self.mutate(lambda r: r.update(execution_policy='state_only'))
        with self.assertRaisesRegex(ValueError, 'execution policy'):
            self.short_review()

    def test_archive_short_profile_keeps_exact_twelve_raws(self):
        smoke_job(self.root, profile='smoke-6-short-state')
        r = self.short_review()
        prepared = v.load(self.root/'prepared.json'); prepared['suite_id'] = archive.SUITE_ID
        write(self.root/'prepared.json', prepared)
        r.update(suite_id=archive.SUITE_ID, prepared_sha256=v.digest(self.root/'prepared.json'))
        write(self.root/'independent-review.json', r)
        _, reviewed, files, _ = archive.validate_job(self.parent, self.root)
        self.assertEqual(reviewed['coverage_profile'], 'smoke-6-short-state')
        self.assertEqual(len(files), 12)
        r['internal_checks_checked'] = 36
        write(self.root/'independent-review.json', r)
        with self.assertRaisesRegex(ValueError, 'state coverage'):
            archive.validate_job(self.parent, self.root)


if __name__ == '__main__':
    unittest.main()

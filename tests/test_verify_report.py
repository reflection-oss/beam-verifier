"""Offline tests for verify-report parsing and per-suite aggregation (no network)."""

import json

from beam_verifier import cli

FUNCTIONAL_TESTS = "beam_verifier/functional_tests"


def _write_report(tmp_path, tests):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"tests": [{"nodeid": n, "outcome": o} for n, o in tests]}))
    return path


def test_parse_report_extracts_suite_and_exact_node_id(tmp_path):
    path = _write_report(
        tmp_path,
        [
            (
                f"{FUNCTIONAL_TESTS}/e2e/test_request_contract.py::test_canonical_param_accepted[temperature=0.6]",
                "passed",
            ),
            (
                f"{FUNCTIONAL_TESTS}/e2e/subdir/test_nested.py::test_case[streaming-earth]",
                "failed",
            ),
        ],
    )
    counts, cases = cli.parse_pytest_report(path)
    assert counts == {"passed": 1, "failed": 1}
    assert cases[0]["suite"] == "e2e"
    assert cases[0]["name"] == "e2e/test_request_contract.py::test_canonical_param_accepted[temperature=0.6]"
    # A nested directory still belongs to its top-level suite.
    assert cases[1]["suite"] == "e2e"
    assert cases[1]["name"] == "e2e/subdir/test_nested.py::test_case[streaming-earth]"


def test_parse_report_missing_file_is_empty(tmp_path):
    counts, cases = cli.parse_pytest_report(tmp_path / "absent.json")
    assert counts == {} and cases == []


def test_summarize_suites_fail_with_exact_failing_ids():
    cases = [
        {"suite": "e2e", "name": "e2e/a.py::t_ok", "outcome": "passed"},
        {"suite": "e2e", "name": "e2e/a.py::t_bad[x]", "outcome": "failed"},
        {"suite": "e2e", "name": "e2e/a.py::t_informational", "outcome": "xfailed"},
    ]
    by_suite = {s.suite: s for s in cli.summarize_suites(cases)}
    e2e = by_suite["e2e"]
    assert e2e.status == "FAIL" and e2e.required
    assert e2e.failing_ids == ["e2e/a.py::t_bad[x]"]
    assert e2e.counts == {"passed": 1, "failed": 1, "xfailed": 1}


def test_summarize_suites_pass_and_optional_skips():
    cases = [{"suite": "e2e", "name": "e2e/a.py::t_ok", "outcome": "passed"}]
    by_suite = {s.suite: s for s in cli.summarize_suites(cases)}
    assert by_suite["e2e"].status == "PASS"
    parser = by_suite["parser"]
    assert parser.status == "SKIP" and not parser.required
    assert parser.skip_reason == "no --parse-url"
    assert by_suite["formatter"].status == "SKIP"
    assert by_suite["raw_completions"].skip_reason == "no --completions-url"


def test_summarize_suites_all_skipped_is_skip():
    cases = [{"suite": "e2e", "name": "e2e/a.py::t", "outcome": "skipped"}]
    by_suite = {s.suite: s for s in cli.summarize_suites(cases)}
    assert by_suite["e2e"].status == "SKIP"
    assert by_suite["e2e"].skip_reason == "all cases skipped"


def test_summarize_suites_reports_unregistered_directories():
    cases = [{"suite": "surprise", "name": "surprise/a.py::t", "outcome": "passed"}]
    by_suite = {s.suite: s for s in cli.summarize_suites(cases)}
    assert by_suite["surprise"].status == "PASS" and not by_suite["surprise"].required


def test_parser_only_run_requires_parser_but_not_e2e():
    cases = [{"suite": "parser", "name": "parser/test_parser.py::test_case", "outcome": "passed"}]
    by_suite = {s.suite: s for s in cli.summarize_suites(cases, required_suites=("parser",))}
    assert by_suite["parser"].status == "PASS" and by_suite["parser"].required
    assert by_suite["e2e"].status == "SKIP" and not by_suite["e2e"].required
    assert by_suite["e2e"].skip_reason == "not selected"
    assert by_suite["formatter"].status == "SKIP" and not by_suite["formatter"].required


def test_api_suite_preserves_pass_with_skipped_cases():
    cases = [
        {"suite": "e2e", "name": "e2e/a.py::passed", "outcome": "passed"},
        {"suite": "e2e", "name": "e2e/a.py::skipped", "outcome": "skipped"},
    ]
    by_suite = {s.suite: s for s in cli.summarize_suites(cases, required_suites=("e2e",))}
    assert by_suite["e2e"].status == "PASS"
    assert by_suite["e2e"].counts == {"passed": 1, "skipped": 1}


def test_partial_parser_coverage_cannot_pass():
    cases = [
        {"suite": "parser", "name": "parser/test_parser.py::tools", "outcome": "passed"},
        {
            "suite": "parser",
            "name": "parser/test_parser.py::reasoning",
            "outcome": "skipped",
            "metadata": {"skip_reason": "UNSUPPORTED: /separate_reasoning returned HTTP 404"},
        },
    ]
    by_suite = {s.suite: s for s in cli.summarize_suites(cases, required_suites=("parser",))}
    assert by_suite["parser"].status == "FAIL"
    assert by_suite["parser"].required
    assert by_suite["parser"].skip_reason == "UNSUPPORTED: /separate_reasoning returned HTTP 404"


def test_report_preserves_parser_evidence(tmp_path):
    path = tmp_path / "report.json"
    metadata = {"profile": "harmony-xml-rfl-v1", "request": {"text": "wire"}, "actual": {"tool_calls": []}}
    path.write_text(
        json.dumps(
            {
                "tests": [
                    {
                        "nodeid": f"{FUNCTIONAL_TESTS}/parser/test_parser.py::test_case",
                        "outcome": "failed",
                        "metadata": metadata,
                    }
                ]
            }
        )
    )
    _, cases = cli.parse_pytest_report(path)
    assert cases[0]["metadata"] == metadata


def test_raw_completions_selection_requires_complete_coverage():
    cases = [
        {"suite": "raw_completions", "name": "raw_completions/test_raw_completions.py::final", "outcome": "passed"},
        {"suite": "raw_completions", "name": "raw_completions/test_raw_completions.py::tool", "outcome": "skipped"},
    ]
    by_suite = {s.suite: s for s in cli.summarize_suites(cases, required_suites=("raw_completions",))}
    assert by_suite["raw_completions"].required
    assert by_suite["raw_completions"].status == "FAIL"
    assert not by_suite["e2e"].required

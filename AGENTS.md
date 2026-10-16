# Agent Guide

beam-verifier validates model deployments behind OpenAI-compatible
endpoints: `beam-verifier verify` runs functional conformance tests and
`beam-verifier eval` reproduces benchmark scores. The only required
integration point is a `/v1/chat/completions` base URL.

## Commands

```bash
uv sync --extra dev                          # environment
uv run pytest tests                          # offline unit tests (no network)
uv run ruff format . && uv run ruff check .  # format + lint
uv run mypy                                  # type check

# Functional tests
uv run beam-verifier verify --base-url <url>/v1 --model-name <model-id> -k <filter>
```

To run pytest directly, point it at the `functional_tests` directory inside
the package with `--base-url=<url>/v1 --model=<model-id>`.

`uv` is the only supported toolchain: `ruff` and `mypy` are pinned in
`pyproject.toml` and `ci.sh` reproduces the full check in an isolated venv.
Do not add a second lint or type checker configuration. Run the offline
tests, ruff, and mypy before handing work back.

## Adapter design

Keep `benchmarks/` as wiring around the downstream libraries. Forward native
option mappings, leave unset values to library defaults, and reuse public
preflight/configuration APIs instead of maintaining duplicate validators or
CLI parsers. Convert downstream failures into visible dataset errors.
Keep local logic only where it owns a real integration requirement: result
translation, endpoint wiring, credential isolation, or resource cleanup.
Document those requirements in docstrings. Do not replace behavior with a
library helper when it changes scoring, cancellation, or request replay semantics.

## How datasets and runners fit together

The sole goal of `eval` is to reliably reproduce benchmark numbers, in
pre-defined settings, from an endpoint URL. The pre-defined settings are the
registry rows; the endpoint is the only thing a caller brings. There is one
dataset table, and each dataset carries the runner that executes it — so nothing
outside a runner knows which eval library is involved.

- **A dataset belongs to exactly one runner.** The runner instance carries that
dataset's identity and configuration — its id and its task — so running it is
`DATASETS[dataset_id].runner(request)`: no kind enum, no lookup table, no options
union, and no way to express a dataset paired with a runner that cannot execute
it.
- **`benchmarks/registry.py` is the source of truth.** It holds the `Dataset`
record — the runner plus the metadata reports need, taking its `dataset_id` from
the runner so the id is written once — and the `DATASETS` table. Adding a dataset
is a row in the table, not new code.
- **A runner is a configured instance.** `scaffolds/base.py` defines
`DatasetRunner` — the `__call__(request) -> RunOutcome` contract — plus `RunRequest`
(base URL plus the run's `EvalSettings`), `RunOutcome` and `RunnerError`. Each
eval library gets a scaffold beside it: `scaffolds/inspect_ai/` holds `runner.py`
and `log.py` (the parser for the logs it reads). Runners never print; they raise
`RunnerError`.
- **Record invocation settings honestly.** The manifest keeps caller settings
in `config` and resolved per-dataset snapshots in `dataset_settings`. Snapshot
the whole plan before evaluation; resume uses those snapshots and passes explicit
overrides to the harness. The harness owns resume behavior, including which
overrides it accepts. Native logs retain the prompt factory and concrete template.
- **Source dataset defaults.** Standard eval rows select dataset/family YAML files
in `benchmarks/configs/`; optional rows keep inline registry settings and native
prompts. Explicit settings win, including zero, false and null.
Use Beam sampling and repetitions where sourced; leave other generation values
null. Never cap explicit caller settings.
`client_timeout` bounds the endpoint client's wait.
- **Share one prompt factory.** `benchmarks/prompts.py` owns `BeamPromptFactory`;
YAML owns its name and text. Runner adapters use the resolved factory with native
evaluation behavior.
Instruction text is data, never a shell command. Public code cannot depend on
private prompt implementations.
- **Option channels are declared once.** `options.py` owns every shared CLI
option and the decoding behind it: `endpoint_arguments`, the `kwargs_option`
factory each `--*-kwargs` channel is built from (so the repeat-and-decode
contract is written once and a channel's help names only what it forwards to),
`parse_kwargs`, `parse_model_kwargs`, and the `merge_options` merge.
Adding a channel is a `kwargs_option` call, never a fresh `click.option` block;
a channel more than one command offers gets a module-level decorator there, as
`model_kwargs_option` does.
- **Any library option stays reachable.** `--harness-kwargs KEY=VALUE` (repeatable,
YAML-decoded) lands in `EvalSettings.harness_kwargs` and is forwarded verbatim by the
dataset's scaffold to its library's API — InspectAI's `eval()` — winning over the dedicated flags on
a name collision. Never add a dedicated flag for a knob this already reaches.
- **Model request options are one channel.** `--model-kwargs KEY=VALUE` is
offered by every command that issues model requests — `eval` and `verify`, through the shared `cli.model_kwargs_option` — and lands in
`EvalSettings.model_kwargs`: options sent with each model request, whichever
component makes the request. Every layer with tunable model requests forwards it through
`options.merge_options` (nested mappings merge per key, any other
value replaces) to the layer its library sends with each request — InspectAI's
generate config in its runner,
the request body in `functional_tests.completions` and the raw-completions
suite. A new dataset, scaffold or suite honors the flag by calling the merge
where its model request options are built. There is one spelling for each option:
a field the OpenAI schema defines is named directly, and anything else is an
engine option carried in `extra_body`, which every engine forwards into the
request body. Nothing routes between the two — an option on the wrong side is
an error naming the spelling to use, so the same flag means the same thing on
every command.
- **Scaffold modules import light.** The dataset table imports every scaffold
module, so an eval library (InspectAI) is imported inside `__call__`,
when the dataset actually runs — `list-datasets` must stay instant.
- **Adding a scaffold** is one module or package under `scaffolds/`
plus its rows in `DATASETS`; never a second registry, a dispatch layer, or a
branch in `cli.py`. Inside a scaffold package, the runner module is `runner.py`.
- **No code in `__init__.py`.** Package `__init__` files carry a docstring and
nothing else; import the module you mean.
- **Scoring belongs to the eval library.** This package runs datasets and records
what came back; it does not grade. A dataset whose library cannot score it is not
a dataset this package can run.
## How the functional tests are built

- **Fixtures are the source of truth.** Each test module under
`functional_tests/e2e/` pairs with a same-name YAML file in
`e2e/fixtures/`. Adding a case is a YAML edit; expectations (JSON schemas,
accepted/rejected parameter values, expected answers) live in the fixture,
never hardcoded in the test.
- **One test function per behavior**, parametrized over the fixture cases.
A test that should run in both transport modes takes the `stream` fixture.
- **Every generation request goes through** `completions.generate_completion` (bounded
`max_tokens`, bounded client timeout). Cross-cutting invariants belong in
`completions.assert_response_contract`, which runs on every completion —
not in per-test assertions.
- **Parser tests do not generate.** `functional_tests/parser/` sends fixed
token IDs through `parser_api.py` to the exact `--parse-url`, using the single
contract documented in README.md. Do not infer providers or companion routes
from the URL; native API adaptation belongs on the serving side. Reuse the YAML loader and reporting;
do not import engine parsers into the verifier.
Fixture expectations are independent of the deployed parser. Keep partial
or unavailable coverage visible, and do not label parser results as formatter
or end-to-end coverage.
- **Formatter tests do not generate.** `functional_tests/formatter/` compares
the `--format-url` output with pinned gold token IDs and independent prompts.
`formatter_api.py` preserves fixture requests; `native_api.py` shares HTTP
transport with parser tests. Never render expected prompts using the deployed
template. Report unsupported APIs as missing coverage, never a pass.
- **Rejections are strict.** Invalid request input must yield HTTP 400
(`openai.BadRequestError`). Do not loosen a rejection to "any error": a
5xx for bad input is a serving bug the suite must report.
- **Failures must be attributable on their own.** Write assertion messages
that name the violated property and the observed value, e.g.
`Expected at least 2 tool calls in one turn, got 0`.
- **A red test against an in-progress deployment is a finding, not a suite
bug.** Never weaken a contract to make an endpoint pass; if a value must
wait for a fix, park it in the fixture with a comment saying why.
- **Offline unit tests live in** `tests/` at the repo root and must run with
no network. Pure logic (the dataset table, contract assertions, the InspectAI log
parser, report aggregation) gets covered there. Anything that needs a live
endpoint belongs in `functional_tests/`, never in `tests/`.

## Style

These are the conventions review keeps flagging. Match the surrounding
code when a case is not covered here.

- Use plain dataclasses without `frozen=True`, and omit `@override` decorators.
- Python 3.12. Fully annotate signatures and parameterize every generic
(`list[dict[str, object]]`, never `list[dict]`). Prefer `object` over `Any`
when the value's type is genuinely irrelevant to the code handling it.
- Import modules, not symbols (`from x import y` where `y` is a module);
`typing` and `collections.abc` are exempt.
- Give tests and fixtures descriptive names; no shorthand category codes.
- Call the harness library **InspectAI** in prose, comments, docstrings and
messages — never bare "Inspect" or "inspect". Lowercase `inspect` is reserved
for the literal executable, and `inspect_ai` / `inspect_evals` / `inspect-ai`
for module paths and the package name.
- Docstrings are Google-style, three-double-quoted, opening with a one-line
summary. Document parameters, results, and exceptions in `Args:`,
`Returns:` (or `Yields:`), and `Raises:` sections with a hanging indent —
not Sphinx `:param:` fields:

```python
def grade_rows(rows: list[Row], strict: bool = False) -> dict[str, float]:
    """Grades each row and aggregates the per-row metrics.

    Args:
        rows: Input rows, in dataset order.
        strict: Whether an ungradable row raises instead of scoring zero.

    Returns:
        Metric name to aggregate value.

    Raises:
        ValueError: If `strict` is set and a row cannot be graded.
    """
```

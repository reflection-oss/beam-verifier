![logo](src/beam_verifier/assets/beam-logo.png)

# Beam Verifier

Verify a live Beam endpoint along 2 dimensions:

- **Functional correctness** `beam-verifier verify` runs functional tests —
parameter acceptance, request validation, chat, tool calling, structured
output, `tool_choice`, multi-turn, and streaming.
- **Performance** `beam-verifier eval` reproduces benchmark numbers
(GPQA-Diamond, AIME) in their
pre-defined settings, so you can check a deployment against a reference score.

`beam-verifier` integrates [InspectAI](https://github.com/UKGovernmentBEIS/inspect_ai) for GPQA-Diamond and AIME.

The only requirement is an OpenAI-compatible `/v1/chat/completions` endpoint.

For walkthroughs, endpoint contracts, configuration, and troubleshooting, see
the [Beam Verifier developer docs](https://developers.reflection.ai/beam-verifier).

## Table of Contents

- [Install](#install)
- [Verify](#verify)
- [Eval](#eval)
- [Results](#results)



## Install

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install beam-verifier
beam-verifier list-datasets   # confirms the install; no endpoint needed
```

If your endpoint requires authentication, set `REFLECTION_API_KEY` in your
environment, or pass `--api-key`.

## Verify

Runs the functional suites against `/v1/chat/completions`: parameter
acceptance, request validation, chat, tool calling, structured output,
`tool_choice`, multi-turn, and streaming. Every case is expected to pass on a
sane Beam endpoint: the expected result is 100%. The `harness_tool_calls`
cases replay coding-agent requests whose tool arguments Beam often breaks
unless the endpoint constrains them to each tool's schema.

```bash
beam-verifier verify --base-url http://localhost:8000/v1 --model-name my-model

# Narrow to one suite and stop at the first failure.
beam-verifier verify --base-url http://localhost:8000/v1 --model-name my-model \
  -k tool_protocol -x
# --model-kwargs forwards request options to the endpoint.
beam-verifier verify --base-url http://localhost:8000/v1 --model-name my-model \
  --model-kwargs temperature=0
```

> [!NOTE]
> Setting `temperature=0` can help you ensure correctness in a repeatable way

`--parse-url`, `--format-url`, and `--completions-url` add optional suites
that check raw Harmony parsing, chat-template formatting, and raw completions
directly, without going through chat. See the
[developer docs](https://developers.reflection.ai/beam-verifier#optional-tests)

## Eval

Reproduces benchmark numbers — GPQA-Diamond, AIME — in their pre-defined settings, so a run is comparable to the
published score. All presets are overridable at will.

Here are the target scores on each dataset:


| Model | GPQA-Diamond | AIME 2026 |
| ----- | ------------ | --------- |
| Beam  | 90.6%        | 97.7%     |


```bash
beam-verifier eval --base-url http://localhost:8000/v1 --model-name my-model \
  --parallel-tasks 8
```

A run can be resumed from where it left off; the dataset list and
`--base-url` / `--model-name` are fixed to the saved run and the API key must
be supplied again:

```bash
beam-verifier eval --resume .runs/<run-directory> \
  --base-url http://localhost:8000/v1 --model-name my-model
```



## Results

Runs are saved under `.runs/` in the current directory. Inspect the latest run
or write a self-contained HTML report:

```bash
beam-verifier view .runs/<run>
beam-verifier report .runs/<run> -o report.html
```

Use `beam-verifier --help` for the
command list and the [developer docs](https://developers.reflection.ai/beam-verifier)
for full walkthroughs, every flag, and troubleshooting.
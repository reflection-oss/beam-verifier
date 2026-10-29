![logo](beam_verifier/assets/beam-logo.png)

# Beam Verifier

Verify a live Beam endpoint along 2 dimensions:

- **Functional correctness** `beam-verifier verify` runs functional tests —
parameter acceptance, request validation, chat, tool calling, structured
output, `tool_choice`, multi-turn, and streaming.
- **Performance** `beam-verifier eval` reproduces benchmark numbers
(GPQA-Diamond, AIME, SWE-bench Pro) in their
pre-defined settings, using [Mirror](https://github.com/reflection-oss/mirror) as the agent for coding benchmarks.

`beam-verifier` integrates [InspectAI](https://github.com/UKGovernmentBEIS/inspect_ai) for GPQA-Diamond and AIME, and [Harbor](https://github.com/harbor-framework/harbor) for SWE-bench Pro.

Mirror 0.1.8 is bundled with its own Linux x86-64/ARM64 runtime
for installation and execution inside task sandboxes.

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

Coding benchmarks use local Docker sandboxes by default. For remote sandboxes,
install the Daytona extra: `uv tool install 'beam-verifier[daytona]'`.

If your endpoint requires authentication, set `REFLECTION_API_KEY` in your
environment, or pass `--api-key`.

## Verify

Runs the functional suites against `/v1/chat/completions`: parameter
acceptance, request validation, chat, tool calling, structured output,
`tool_choice`, multi-turn, and streaming. Every case is expected to pass on a
sane Beam endpoint: the expected result is 100%.

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

Reproduces benchmark numbers — GPQA-Diamond, AIME,
SWE-bench Pro — in their pre-defined settings, so a run is comparable to the
published score. All presets are overridable.

Here are the target scores on each dataset
([model card](https://huggingface.co/reflection/Beam-1.0-501B-A23B#benchmark-results)):


| Model | GPQA-Diamond | AIME 2026 | SWE-bench Pro v1 |
| ----- | ------------ | --------- | --------------- |
| Beam  | 90.6%        | 97.7%     | 66.1%           |


```bash
beam-verifier eval --base-url https://model.example.com/v1 --model-name my-model \
  --parallel-tasks 8
```

Without a dataset name, `eval` runs all three benchmarks. SWE-bench Pro
score reproduction is still **unverified**.

SWE-bench Pro (`swebench-pro`)
runs through [Harbor](https://github.com/harbor-framework/harbor) with
[Mirror](https://github.com/reflection-oss/mirror) as the agent. It runs
against local Docker by default, which is fine to get started, but
[Daytona](https://www.daytona.io) is recommended for scale: it runs task
sandboxes remotely, so concurrency is not limited by your host machine's
resources.

```bash
# Local Docker sandboxes (default).
beam-verifier eval swebench-pro \
  --base-url https://model.example.com/v1 --model-name my-model

# use Daytona, recommended for scale.
uv tool install 'beam-verifier[daytona]'
export DAYTONA_API_KEY=...
beam-verifier eval swebench-pro --sandbox daytona \
  --base-url https://model.example.com/v1 --model-name my-model \
  --parallel-tasks 8

# --host-model-proxy spins a model proxy on the host, to allow connectivity for a protected endpoint from the remote sandboxes.
export REFLECTION_API_KEY=...
beam-verifier eval swebench-pro --sandbox daytona --host-model-proxy \
  --base-url https://model.example.com/v1 --model-name my-model \
  --parallel-tasks 8
```

Mirror is operating inside the task container, sending requests to the
endpoint from there. The endpoint must be reachable from there. Direct
connections require an unauthenticated endpoint. Use `--host-model-proxy`
with `--sandbox daytona` for authenticated endpoints, or when the endpoint
is reachable only from the host — it relays model requests through a local
proxy, so the sandbox never sees the credential. Local Docker does not
support `--host-model-proxy`. See the
[developer docs](https://developers.reflection.ai/beam-verifier#evaluate-your-endpoint) for how to
set it up, along with the full detail on tuning generation settings, Mirror,
and sandbox/Daytona options.

A run can be resumed from where it left off; the dataset list and
`--base-url` / `--model-name` are fixed to the saved run and the API key must
be supplied again:

```bash
beam-verifier eval --resume .runs/<run-directory> \
  --base-url https://model.example.com/v1 --model-name my-model
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

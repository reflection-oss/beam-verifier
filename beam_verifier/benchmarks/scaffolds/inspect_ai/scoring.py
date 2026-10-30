"""Adapt Beam's boxed-answer protocol to Math-Verify grading."""

import re
import resource

import anyio
import math_verify
from anyio import to_process
from inspect_ai import scorer, solver


def _last_boxed_answer(text: str) -> str | None:
    """Extract the final box, rejecting unmatched braces instead of falling back.

    Math-Verify and latex2sympy can recover earlier answers from malformed
    output. The Beam protocol selects the last box command, including one
    nested in another box; surrounding prose and earlier boxes are ignored.
    """
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    escaped = False
    for position in range(start, len(text)):
        character = text[position]
        if not escaped:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start:position]
        escaped = character == "\\" and not escaped
    return None


def _compare(answer: str, target: str) -> bool:
    """Delegate parsing and equivalence to Math-Verify in a worker process."""
    extraction = [
        math_verify.LatexExtractionConfig(
            try_extract_without_anchor=True,
            boxed_match_priority=0,
            normalization_config=math_verify.LatexNormalizationConfig(
                basic_latex=True,
                units=False,
                malformed_operators=False,
                nits=False,
                boxed="last",
                equations=False,
            ),
        ),
        math_verify.ExprExtractionConfig(try_extract_without_anchor=True),
    ]
    prediction = math_verify.parse(f"\\boxed{{{answer}}}", extraction_config=extraction, parsing_timeout=5)
    gold = math_verify.parse(f"${target}$", extraction_config=extraction, parsing_timeout=10)
    return math_verify.verify(
        gold,
        prediction,
        float_rounding=6,
        numeric_precision=15,
        strict=True,
        allow_set_relation_comp=True,
        timeout_seconds=10,
    )


def _verify(answer: str, target: str) -> bool:
    """Bound native allocations that Math-Verify's signal timeouts cannot stop."""
    limits = resource.getrlimit(resource.RLIMIT_AS)
    limit = min(value for value in (2 * 1024**3, *limits) if value != resource.RLIM_INFINITY)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limits[1]))
    try:
        return _compare(answer, target)
    finally:
        # AnyIO can reuse this worker for another caller after grading.
        resource.setrlimit(resource.RLIMIT_AS, limits)


@scorer.scorer(metrics=[scorer.accuracy(), scorer.stderr()])
def beam_boxed_math() -> scorer.Scorer:
    """Grade the final boxed answer without modifying the model's response.

    Math-Verify uses signal-based timeouts, so it runs in cancellable worker
    processes rather than blocking the evaluation loop. Worker failures remain
    evaluation errors; only missing answers or failed comparisons earn zero.

    Returns:
      A native InspectAI scorer for the Beam boxed-answer protocol.
    """
    limiter = anyio.CapacityLimiter(4)

    async def score(state: solver.TaskState, target: scorer.Target) -> scorer.Score:
        output = state.output
        submitted = not output.empty and output.error is None and not output.message.tool_calls
        answer = _last_boxed_answer(output.completion) if submitted else None
        correct = False
        if answer is not None:
            async with limiter:
                with anyio.fail_after(120):
                    correct = await to_process.run_sync(_verify, answer, target.text, cancellable=True)
        return scorer.Score(
            value=scorer.CORRECT if correct else scorer.INCORRECT,
            answer=answer,
            explanation=state.output.completion,
        )

    return score

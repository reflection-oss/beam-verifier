"""Check boxed-answer extraction and native mathematical comparison together."""

import pytest
from inspect_ai import model, scorer, solver

from beam_verifier.benchmarks.scaffolds.inspect_ai import scoring


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("completion", "expected_answer", "expected_score"),
    [
        ("\\[\n\\boxed{42}\n\\]", "42", "C"),
        (r"\boxed{042.0}", "042.0", "C"),
        (r"\fbox {\frac{84}{2}}", r"\frac{84}{2}", "C"),
        (r"\boxed{41} then \boxed{42}. Done.", "42", "C"),
        (r"\boxed{42} then \boxed{41}. The target was 42.", "41", "I"),
        (r"The answer is 42.", None, "I"),
        (r"\boxed{42} then \boxed{41", None, "I"),
        (r"\boxed{\{42\}}", r"\{42\}", "C"),
        (r"\boxed{1+\boxed{42}}", "42", "C"),
        (r"\boxed{\frac{1}{\boxed{42}}}", "42", "C"),
        (r"\boxed{}", "", "I"),
    ],
)
async def test_beam_boxed_math_grades_only_final_box(
    completion: str, expected_answer: str | None, expected_score: str
) -> None:
    state = solver.TaskState(
        model=model.ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="What is 6 times 7?",
        messages=[],
    )
    state.output = model.ModelOutput.from_content("mockllm", completion)

    result = await scoring.beam_boxed_math()(state, scorer.Target("42"))

    assert result is not None
    assert result.answer == expected_answer
    assert result.value == expected_score
    assert state.output.completion == completion


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        model.ModelOutput.from_content("mockllm", r"\boxed{42}", error="Invalid model output"),
        model.ModelOutput.for_tool_call("mockllm", "python", {}, content=r"\boxed{42}"),
    ],
)
async def test_beam_boxed_math_requires_final_submission(output: model.ModelOutput) -> None:
    state = solver.TaskState(
        model=model.ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="What is 6 times 7?",
        messages=[],
    )
    state.output = output

    result = await scoring.beam_boxed_math()(state, scorer.Target("42"))

    assert result is not None
    assert result.value == "I"
    assert result.answer is None

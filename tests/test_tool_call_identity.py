"""Offline tests for tool-call identity (no network).

Parallel-call tests count distinct calls, so two calls that differ only in
how their arguments are spelled must collapse to one identity.
"""

from beam_verifier.functional_tests import completions


def _call(name="get_weather", arguments='{"location": "Paris"}'):
    return {"id": "call_x", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_whitespace_variants_are_one_call():
    calls = [_call(arguments='{"location": "Paris"}'), _call(arguments='{  "location":"Paris"  }')]
    assert len({completions.canonical_tool_call(call) for call in calls}) == 1


def test_key_order_variants_are_one_call():
    calls = [
        _call(name="add_product", arguments='{"name": "Widget", "price": 19.99}'),
        _call(name="add_product", arguments='{"price": 19.99, "name": "Widget"}'),
    ]
    assert len({completions.canonical_tool_call(call) for call in calls}) == 1


def test_nested_key_order_variants_are_one_call():
    calls = [
        _call(arguments='{"where": {"city": "Paris", "country": "France"}}'),
        _call(arguments='{"where": {"country": "France", "city": "Paris"}}'),
    ]
    assert len({completions.canonical_tool_call(call) for call in calls}) == 1


def test_repeated_task_does_not_satisfy_two_distinct_calls():
    # The same task asked twice, spelled differently, must not read as parallel work.
    calls = [_call(arguments='{"location": "Paris"}'), _call(arguments='{"location":"Paris"}')]
    assert len({completions.canonical_tool_call(call) for call in calls}) < 2


def test_different_arguments_stay_distinct():
    calls = [_call(arguments='{"location": "Paris"}'), _call(arguments='{"location": "Tokyo"}')]
    assert len({completions.canonical_tool_call(call) for call in calls}) == 2


def test_same_arguments_on_different_tools_stay_distinct():
    calls = [_call(name="get_weather"), _call(name="get_local_time")]
    assert len({completions.canonical_tool_call(call) for call in calls}) == 2


def test_list_order_is_semantic():
    calls = [_call(arguments='{"tags": ["new", "sale"]}'), _call(arguments='{"tags": ["sale", "new"]}')]
    assert len({completions.canonical_tool_call(call) for call in calls}) == 2

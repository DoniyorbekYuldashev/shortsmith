"""Checks for the Phase 6 exercise. Hand-made states; no LangGraph, no API.
Run from the repo root:  uv run pytest exercises/phase-6
"""

import operator

import pytest

from graph_runner import END, merge_update, route_after_critic, run_graph  # pytest puts this folder on sys.path


def add_usage(old: dict, new: dict) -> dict:
    """The custom reducer from agents/state.py, simplified: tokens summed per node."""
    merged = dict(old)
    for node, tokens in new.items():
        merged[node] = merged.get(node, 0) + tokens
    return merged


REDUCERS = {"critic_rounds": operator.add, "usage": add_usage}


# --- merge_update ------------------------------------------------------------------------------


def test_a_field_without_a_reducer_is_replaced():
    assert merge_update({"plan": "old", "topic": "plov"}, {"plan": "new"}, REDUCERS) == {"plan": "new", "topic": "plov"}


def test_a_list_with_operator_add_is_appended():
    state = {"critic_rounds": [{"score": 5}]}
    merged = merge_update(state, {"critic_rounds": [{"score": 8}]}, REDUCERS)
    assert merged["critic_rounds"] == [{"score": 5}, {"score": 8}]


def test_a_custom_reducer_sums_per_node():
    state = {"usage": {"writer": 300, "critic": 100}}
    merged = merge_update(state, {"usage": {"critic": 50, "revise": 20}}, REDUCERS)
    assert merged["usage"] == {"writer": 300, "critic": 150, "revise": 20}


def test_a_new_key_with_a_reducer_starts_from_the_update():
    assert merge_update({}, {"critic_rounds": [{"score": 5}]}, REDUCERS) == {"critic_rounds": [{"score": 5}]}


def test_the_old_state_is_not_changed():
    state = {"plan": "old", "critic_rounds": [1]}
    merge_update(state, {"plan": "new", "critic_rounds": [2]}, REDUCERS)
    assert state == {"plan": "old", "critic_rounds": [1]}  # a checkpoint of the old state must stay valid


# --- route_after_critic ----------------------------------------------------------------------------


def reports(*rounds: tuple[float, list[int]]) -> dict:
    return {"critic_rounds": [{"score": score, "weak": weak} for score, weak in rounds]}


def test_weak_scenes_and_rounds_left_mean_revise():
    assert route_after_critic(reports((5, [3]))) == "revise"


def test_a_good_score_ends():
    assert route_after_critic(reports((7.5, [3]))) == END  # good enough, even with a nitpick
    assert route_after_critic(reports((7.0, []))) == END


def test_nothing_flagged_ends_even_with_a_low_score():
    assert route_after_critic(reports((4, []))) == END


def test_at_most_max_rounds_revisions():
    assert route_after_critic(reports((5, [3]), (6, [3]))) == "revise"  # 1 revision made, 1 left
    assert route_after_critic(reports((5, [3]), (6, [3]), (6, [2]))) == END  # 2 made: stop
    assert route_after_critic(reports((5, [3])), max_rounds=0) == END


def test_the_pass_score_is_a_setting():
    assert route_after_critic(reports((8, [2])), pass_score=9) == "revise"


# --- run_graph --------------------------------------------------------------------------------------


def test_a_straight_line_runs_every_node_once():
    nodes = {"write": lambda s: {"script": "s1"}, "edit": lambda s: {"video": s["script"] + ".mp4"}}
    state, path = run_graph(nodes, {"write": "edit", "edit": lambda s: END}, "write", {})
    assert path == ["write", "edit"] and state == {"script": "s1", "video": "s1.mp4"}


def test_a_conditional_edge_loops_until_the_critic_is_happy():
    scores = iter([5, 6, 8])
    nodes = {
        "produce": lambda s: {"made": s.get("made", 0) + 1},
        "critic": lambda s: {"critic_rounds": [{"score": next(scores), "weak": [2]}]},
        "revise": lambda s: {"usage": {"revise": 10}},
    }
    edges = {"produce": "critic", "critic": route_after_critic, "revise": "produce"}
    state, path = run_graph(nodes, edges, "produce", {}, REDUCERS)
    assert path == ["produce", "critic", "revise", "produce", "critic", "revise", "produce", "critic"]
    assert [r["score"] for r in state["critic_rounds"]] == [5, 6, 8] and state["usage"] == {"revise": 20}


def test_the_loop_stops_at_max_rounds_when_the_critic_is_never_happy():
    nodes = {"critic": lambda s: {"critic_rounds": [{"score": 3, "weak": [1]}]}, "revise": lambda s: {}}
    state, path = run_graph(nodes, {"critic": route_after_critic, "revise": "critic"}, "critic", {}, REDUCERS)
    assert path == ["critic", "revise", "critic", "revise", "critic"] and len(state["critic_rounds"]) == 3


def test_a_loop_without_an_exit_hits_the_recursion_limit():
    nodes = {"a": lambda s: {}, "b": lambda s: {}}
    with pytest.raises(RuntimeError) as stopped:
        run_graph(nodes, {"a": "b", "b": "a"}, "a", {}, max_steps=10)
    assert not isinstance(stopped.value, NotImplementedError)  # (NotImplementedError is a RuntimeError too)

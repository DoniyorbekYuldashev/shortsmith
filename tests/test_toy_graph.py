"""Step 6.1's toy graph: state, nodes, a conditional edge, compile() and invoke(), with no LLM."""

from shortsmith.agents import toy_graph


def test_invoke_runs_the_loop_until_the_hook_fits():
    final = toy_graph.build().compile().invoke({"topic": "plov"})

    assert isinstance(final, dict)  # invoke() returns the state's values as a dict, even for a Pydantic state
    assert final["hook"] == "you needed to know about plov today" and final["words"] == 7
    assert final["checks"] == 4  # 13 -> 11 -> 9 -> 7 words: the conditional edge looped three times


def test_the_reducer_appends_and_plain_fields_are_replaced():
    final = toy_graph.build().compile().invoke({"topic": "plov"})
    assert final["history"][:3] == ["write: 13 words", "check: 13 words",
                                    "shorten: -> 'everything you never knew you needed to know about plov today'"]
    assert len(final["history"]) == 8  # every node's line was kept (operator.add), while `hook` holds only the last value


def test_updates_show_one_node_at_a_time():
    updates = list(toy_graph.build().compile().stream({"topic": "plov"}, stream_mode="updates"))
    assert [next(iter(u)) for u in updates] == ["write", "check", "shorten", "check", "shorten", "check", "shorten", "check"]
    assert updates[1] == {"check": {"words": 13, "checks": 1, "history": ["check: 13 words"]}}  # only what the node returned


def test_the_route_reads_the_state_and_changes_nothing():
    state = toy_graph.ToyState(topic="x", hook="a b c", words=3)
    assert toy_graph.route(state) == toy_graph.END
    assert toy_graph.route(state.model_copy(update={"words": 9})) == "shorten"

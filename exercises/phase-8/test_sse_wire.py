"""Checks for the optional Phase 8 exercise. Plain text; no server, no network.
Run from the repo root:  uv run pytest exercises/phase-8
"""

import pytest
from fastapi.sse import format_sse_event

from sse_wire import Message, events_after, parse_sse, sse_message  # pytest puts this folder on sys.path

# --- sse_message -------------------------------------------------------------------------------------------


def test_one_event_with_its_name_and_id():
    assert sse_message('{"id": 1}', event="queued", id="1") == 'event: queued\ndata: {"id": 1}\nid: 1\n\n'


def test_a_value_with_line_breaks_is_several_data_lines():
    assert sse_message("line one\nline two") == "data: line one\ndata: line two\n\n"


def test_a_comment_and_a_retry():
    assert sse_message(comment="run 7", retry=3000) == ": run 7\nretry: 3000\n\n"


@pytest.mark.parametrize("fields", [
    {"data": '{"type": "done"}', "event": "done", "id": "42"},
    {"comment": "keep-alive"},
    {"comment": "run x", "retry": 3000},
    {"data": "a\nb\nc", "event": "x"},
    {"data": "", "id": "1"},
])
def test_the_same_bytes_as_fastapi(fields):
    expected = format_sse_event(data_str=fields.get("data"), event=fields.get("event"), id=fields.get("id"),
                                retry=fields.get("retry"), comment=fields.get("comment")).decode()
    assert sse_message(**fields) == expected


# --- parse_sse -----------------------------------------------------------------------------------------------


def test_messages_with_and_without_a_name():
    assert parse_sse("event: a\ndata: 1\nid: 7\n\ndata: 2\n\n") == [Message("a", "1", "7"), Message("message", "2", "7")]


def test_comments_and_a_block_without_data_send_nothing():
    assert parse_sse(": run 1\nretry: 3000\n\n: keep-alive\n\n") == []


def test_data_lines_are_joined_and_one_space_is_dropped():
    assert parse_sse("data:no space\ndata:  two spaces\n\n") == [Message("message", "no space\n two spaces", "")]


def test_the_id_carries_over_and_retry_is_read():
    found = parse_sse("id: 3\ndata: x\nretry: 1500\n\ndata: y\n\n")
    assert [(m.id, m.retry) for m in found] == [("3", 1500), ("3", None)]


def test_round_trip_with_your_own_writer():
    text = sse_message(comment="run", retry=3000) + sse_message('{"id": 1}', event="queued", id="1") \
        + sse_message('{"id": 2}', event="done", id="2")
    assert [(m.event, m.data, m.id) for m in parse_sse(text)] == [("queued", '{"id": 1}', "1"), ("done", '{"id": 2}', "2")]


# --- events_after ----------------------------------------------------------------------------------------------

EVENTS = [{"id": 1, "type": "queued"}, {"id": 2, "type": "run_started"}, {"id": 3, "type": "done"}]


def test_a_reconnect_gets_only_what_came_after():
    assert events_after(EVENTS, "1") == EVENTS[1:]


@pytest.mark.parametrize("header", [None, "", "abc", "-1", "1.5"])
def test_no_usable_id_means_everything(header):
    assert events_after(EVENTS, header) == EVENTS


def test_at_or_past_the_end_there_is_nothing():
    assert events_after(EVENTS, "3") == [] and events_after(EVENTS, "99") == []

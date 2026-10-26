"""Optional exercise, Step 8.1: Server-Sent Events on the wire, and replay after a reconnect.

Phase 8 has no 🧠 step; this one is for practice. The live run page gets its events as a
text/event-stream: plain text, one message per block, a blank line between messages:

    : run 2026-10-25_141809          <- a comment (ignored by the browser; also our keep-alive)
    retry: 3000                      <- "if the connection drops, wait 3 s, then reconnect"

    event: node_started              <- the message's name: addEventListener("node_started", ...)
    data: {"id": 3, "node": "writer"}
    id: 3                            <- remembered by the browser, sent back as Last-Event-ID

Three small functions are the core of it:
1. sse_message: one message as text, exactly as FastAPI writes it (api.py uses fastapi.sse.format_sse_event).
2. parse_sse: the browser's side: text -> messages (what EventSource does for you).
3. events_after: the server's side of a reconnect: only what came after Last-Event-ID (events.EventLog.after).

Write these first, then compare with the real ones.
Check them from the repo root:  uv run pytest exercises/phase-8
The spec (short, worth reading): https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream
"""

from dataclasses import dataclass


def sse_message(data: str | None = None, event: str | None = None, id: str | None = None, retry: int | None = None,
                comment: str | None = None) -> str:
    """One SSE message, in this field order: comment lines, event, data lines, id, retry, then a blank line.

    Example: sse_message('{"a": 1}', event="queued", id="1")
             -> 'event: queued\\ndata: {"a": 1}\\nid: 1\\n\\n'
             sse_message("line one\\nline two") -> 'data: line one\\ndata: line two\\n\\n'
             sse_message(comment="keep-alive") -> ': keep-alive\\n\\n'

    Hints:
    1. Each field is one line "name: value". A value with line breaks becomes several lines of the SAME
       field (data: ... / data: ...): a line break inside a value would otherwise end the field.
    2. A comment line starts with ":" (here ": " + the text), one per line of the comment.
    3. The message ends with an empty line, so the whole thing ends with "\\n\\n".
    4. Skip fields that are None; `retry` is a number of milliseconds.
    """
    raise NotImplementedError("your turn: follow the hints above")


@dataclass
class Message:
    event: str  # "message" when the block had no event: line (the browser's default name)
    data: str  # every data: line of the block, joined with "\n"
    id: str  # the last id seen so far in the stream ("" if none): it carries over to later messages
    retry: int | None = None  # a retry: line in this block, if any


def parse_sse(text: str) -> list[Message]:
    """Parse a stream into messages, the way EventSource does.

    Rules (a subset of the spec, enough for our server):
    - Blocks are separated by a blank line. A block with no data: line dispatches nothing
      (our first block, the comment + retry, is one of those).
    - A line starting with ":" is a comment: ignore it.
    - Otherwise "name: value" or "name:value" (ONE space after the colon is removed, if there is one).
    - data lines are joined with "\\n"; event names the message ("message" if missing);
      id sets the stream's last id, and it stays set for the following messages too.

    Example: parse_sse('event: a\\ndata: 1\\nid: 7\\n\\ndata: 2\\n\\n')
             -> [Message("a", "1", "7"), Message("message", "2", "7")]
    """
    raise NotImplementedError("your turn")


def events_after(events: list[dict], last_event_id: str | None) -> list[dict]:
    """What a reconnecting browser should get: the events whose "id" is above Last-Event-ID, in order.

    - None, "" or anything that isn't a whole number (a header can hold any text): everything.
    - An id at or past the last event: nothing.

    Example: events [{"id": 1}, {"id": 2}, {"id": 3}], "1" -> [{"id": 2}, {"id": 3}]
    Hint: str.isdigit() tells you whether the header is a number.
    """
    raise NotImplementedError("your turn")

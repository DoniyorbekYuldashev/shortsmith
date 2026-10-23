"""MLflow tracing (Step 6.7): every graph run becomes one trace you can open in the MLflow UI.

    trace: run 2026-10-23_221500                 (one per graph invocation; a resume is a new trace, same run_id tag)
      |- writer      span: what the node read and returned (summaries), tokens, cost, seconds
      |    |- Completions   span from autolog: the exact request, the answer, token usage
      |    `- Completions
      |- director ...

Two sources of spans:
- mlflow.openai.autolog() patches the OpenAI SDK, so every chat.completions.create that llm.py
  sends (to Gemini, through its OpenAI-compatible endpoint) becomes a span with the full messages,
  the answer and the usage. llm.py doesn't change.
- graph.py's node wrapper opens one span per node, with summaries (not the whole state: a
  trace that copies 3 scripts into every span is unreadable and big).
The critic's base64 frames are stored by MLflow as trace attachments (mlruns/<experiment>/traces/
<id>/artifacts), so the UI shows the frames the critic saw, and the span text stays small.

Local only: the store is sqlite:///mlflow.db in the repo root (artifacts in mlruns/), both
gitignored; no server is needed to write. The UI reads the same file:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
Telemetry: MLflow 3 sends anonymous usage events unless MLFLOW_DISABLE_TELEMETRY=true (or
DO_NOT_TRACK=true); setup() sets both before mlflow is imported.

Optional, and it must never break a run: without mlflow, with tracing turned off, or when
anything in here fails, a warning goes to stderr and the run goes on untraced. Every function
here is safe to call when tracing is off.
"""

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

TRACKING_URI = "sqlite:///mlflow.db"  # relative to the repo root, where every command runs
EXPERIMENT = "shortsmith"

log = logging.getLogger("shortsmith.tracing")  # stderr: the MCP server's stdout is its channel
_mlflow: Any = None  # the mlflow module, once setup() succeeded


def setup(experiment: str = EXPERIMENT, uri: str = TRACKING_URI) -> bool:
    """Turn tracing on. Returns False (and the run goes on) when mlflow can't be used."""
    global _mlflow
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("DO_NOT_TRACK", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # a "load this skill" hint MLflow 3.17 prints on import
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        mlflow.openai.autolog(silent=True)  # base64 frames in a request are saved as trace attachments (viewable)
        _mlflow = mlflow
        return True
    except Exception as e:  # ImportError, a locked database, a broken install: never fatal
        log.warning("tracing off: %s: %s", type(e).__name__, e)
        _mlflow = None
        return False


def disable() -> None:
    global _mlflow
    if _mlflow is not None:
        try:
            _mlflow.openai.autolog(disable=True)
        except Exception:
            pass
    _mlflow = None


def enabled() -> bool:
    return _mlflow is not None


class _Span:
    """What a node's code talks to. Wraps MLflow's live span, or nothing when tracing is off;
    every call swallows its own errors (tracing must not break a run)."""

    def __init__(self, live: Any = None):
        self.live = live

    def set_outputs(self, outputs: Any) -> None:
        self._call("set_outputs", outputs)

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self._call("set_attributes", attributes)

    @property
    def trace_id(self) -> str | None:
        return getattr(self.live, "trace_id", None)

    def _call(self, method: str, value: Any) -> None:
        if self.live is None:
            return
        try:
            getattr(self.live, method)(value)
        except Exception as e:
            log.warning("tracing: %s failed: %s", method, e)


@contextmanager
def span(name: str, span_type: str = "CHAIN", inputs: Any = None, tags: dict[str, str] | None = None) -> Iterator[_Span]:
    """A span when tracing is on, a do-nothing one when it's off. The body's own exceptions pass
    through untouched (and mark the span as failed); tracing's exceptions never escape."""
    if _mlflow is None:
        yield _Span()
        return
    try:
        manager = _mlflow.start_span(name=name, span_type=span_type)
        live = manager.__enter__()
        if inputs is not None:
            live.set_inputs(inputs)
        if tags:
            _mlflow.update_current_trace(tags=tags)
    except Exception as e:
        log.warning("tracing: could not start span %r: %s", name, e)
        yield _Span()
        return
    try:
        yield _Span(live)
    except BaseException as error:
        try:
            manager.__exit__(type(error), error, error.__traceback__)
        except Exception as e:
            log.warning("tracing: could not close span %r: %s", name, e)
        raise
    try:
        manager.__exit__(None, None, None)
    except Exception as e:
        log.warning("tracing: could not close span %r: %s", name, e)


@contextmanager
def mlflow_run(name: str, params: dict[str, Any] | None = None, tags: dict[str, str] | None = None,
               nested: bool = False) -> Iterator[Any]:
    """An MLflow *run* (params + metrics, for comparing experiments; 6.8) or None when tracing is off."""
    if _mlflow is None:
        yield None
        return
    try:
        manager = _mlflow.start_run(run_name=name, nested=nested, tags=tags)
        active = manager.__enter__()
        if params:
            _mlflow.log_params(params)
    except Exception as e:
        log.warning("tracing: could not start MLflow run %r: %s", name, e)
        yield None
        return
    try:
        yield active
    finally:
        try:
            manager.__exit__(None, None, None)
        except Exception as e:
            log.warning("tracing: could not end MLflow run %r: %s", name, e)


def log_metrics(metrics: dict[str, float]) -> None:
    if _mlflow is None:
        return
    try:
        _mlflow.log_metrics(metrics)
    except Exception as e:
        log.warning("tracing: could not log metrics: %s", e)

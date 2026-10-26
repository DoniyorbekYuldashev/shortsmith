"""Step 6.9: what context trimming saves, per node, measured on a real finished run.

Each node gets only what it needs (the Director never sees research notes, the Critic never sees
trend data, the pick never sees image prompts). The naive alternative is common: hand every agent
the whole state. This script measures the difference on the SAME inputs:

1. From the run's checkpoints (one per node), take the state each LLM node received, and rebuild
   the exact request it sent (the same builder functions the nodes use).
2. Build the "whole state" version: the same request plus everything else the run knew (the
   whole RunState as JSON and research.json: trends, retrieved hooks, brand rules).
3. --send: send each whole-state request once (ledger step "ctx-full-<node>") and compare the
   provider's prompt_tokens with the trimmed call's own ledger line from the run.

    uv run --env-file .env python -m shortsmith.agents.context_eval <run_id> --send
Without --send it only estimates with tiktoken (free). Results: runs/experiments/phase6_context_<run_id>.json
"""

import argparse
import json
from pathlib import Path

from pydantic import BaseModel

from shortsmith import ledger, llm
from shortsmith.agents import critic, director, graph, reviser, writer
from shortsmith.agents.state import RunState
from shortsmith.llm import Message
from shortsmith.models import CriticVerdict

class Measured(BaseModel):
    step: str  # ledger step name
    trimmed_estimate: int  # tiktoken, text only
    full_estimate: int
    trimmed_real: int | None  # the run's own ledger line (provider's count, images included)
    full_real: int | None  # the ctx-full call (with --send)


def text_of(messages: list[Message]) -> str:
    parts = []
    for m in messages:
        content = m["content"]
        parts += [content] if isinstance(content, str) else [p["text"] for p in content if p["type"] == "text"]
    return "\n".join(parts)


def with_everything(messages: list[Message], s: RunState) -> list[Message]:
    """The naive version: the node's request plus the whole state and the raw research."""
    research = (s.folder / "research.json").read_text(encoding="utf-8") if (s.folder / "research.json").exists() else "{}"
    extra = ("\n\nEverything this run knows (the whole pipeline state, JSON):\n" + s.model_dump_json()
             + "\n\nresearch.json:\n" + research)
    user = dict(messages[1])
    if isinstance(user["content"], str):
        user["content"] += extra
    else:
        user["content"] = [*user["content"], {"type": "text", "text": extra}]
    return [messages[0], user]


def requests(app, run_id: str) -> dict[str, tuple[list[Message], type[BaseModel], str | None, RunState]]:
    """step -> (the request the node sent, its schema, its model, the state it read): first occurrence of each."""
    history = list(reversed(list(app.get_state_history(graph.config(run_id)))))
    before = {}
    for snapshot in history:
        if snapshot.next and snapshot.next[0] in graph.NODES and snapshot.next[0] not in before:
            before[snapshot.next[0]] = RunState.model_validate(snapshot.values)
    built = {}
    if s := before.get("writer"):
        built["writer"] = (writer.options_messages(s.topic, s.language, s.research, s.prompt_version), writer.ScriptOptions, None, s)
    if s := before.get("director"):  # the writer's output is in the director's input state
        if len(s.script_options) > 1:
            built["writer-pick"] = (writer.pick_messages(s.topic, s.script_options, writer.hook_sources(s.research)),
                                    writer.WriterPick, None, s)
        built["director"] = (director.director_messages(s.chosen, s.research.visual_style if s.research else ""),
                             director.DirectorPlan, None, s)
    if s := before.get("critic"):
        folder = s.folder / "critic" / "round_0"
        frames = [(int(p.stem.split("_")[1]), 0.0, p) for p in sorted(folder.glob("scene_*.jpg"))]
        built["critic"] = (critic.critic_messages(s.topic, s.plan, critic.checks(s.plan, s.research), frames),
                           CriticVerdict, critic.critic_model(), s)
    if s := before.get("revise"):
        built["revise"] = (reviser.revise_messages(s.plan, s.critic_rounds[-1].weak_scenes), reviser.SceneFixes, None, s)
    return built


def main() -> None:
    parser = argparse.ArgumentParser(description="Context trimming: trimmed vs whole-state requests, per node.")
    parser.add_argument("run_id")
    parser.add_argument("--send", action="store_true", help="send each whole-state request once (free tier) for real counts")
    parser.add_argument("--steps", nargs="+", help="measure only these steps; earlier results in the output file are kept")
    args = parser.parse_args()
    run_lines = ledger.entries(ledger.DEFAULT_PATH, args.run_id)
    out = Path("runs/experiments") / f"phase6_context_{args.run_id}.json"
    kept = {r["step"]: Measured(**r) for r in json.loads(out.read_text(encoding="utf-8"))} if out.exists() else {}
    rows: list[Measured] = []
    ledger.CURRENT_RUN.set(args.run_id)  # the ctx-full lines are tagged with the run they measure
    with graph.open_graph() as app:
        for step, (messages, schema, model, s) in requests(app, args.run_id).items():
            if args.steps and step not in args.steps:
                if step in kept:
                    rows.append(kept[step])
                continue
            full = with_everything(messages, s)
            real = next((e.prompt_tokens for e in run_lines if e.step == step), None)
            sent = None
            if args.send:
                result = llm.paced(lambda: llm.chat_schema(full, schema, model=model, step=f"ctx-full-{step}"))
                sent = result.prompt_tokens
            rows.append(Measured(step=step, trimmed_estimate=llm.estimate_tokens(text_of(messages)),
                                 full_estimate=llm.estimate_tokens(text_of(full)), trimmed_real=real, full_real=sent))
            print(f"{step:<12} trimmed {rows[-1].trimmed_real or '-':>6} real ({rows[-1].trimmed_estimate:>6} est.) | "
                  f"whole state {rows[-1].full_real or '-':>6} real ({rows[-1].full_estimate:>6} est.)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.model_dump() for r in rows], indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

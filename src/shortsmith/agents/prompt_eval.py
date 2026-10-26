"""Step 6.8: is writer prompt v2 better than v1? 5 topics, judged by the Critic, logged to MLflow.

Per topic: ONE research step, shared by both versions (so the writer prompt is the only thing that
differs), then the Writer node as the graph runs it (3 options + checks + pick) with v1 and with
v2, then the Critic in script mode (no frames: a full render per version would add ~2 minutes and
no information about the words) on each chosen script. The Critic is gemini-3.8-flash, a
stronger model than the writer (gemini-3.5-flash-lite), and it never learns which version wrote what.

Measured per version:
- the critic's score (0-10) and weak scenes for the chosen script (5 samples per version: noisy);
- code facts on EVERY option (up to 15 per version: more samples): words inside the target
  range, hook copied word for word from the notes or the hook library;
- the writer's tokens and cost.
Which version runs first alternates per topic: the free tier's queueing changes over minutes.

LLM calls: 5 research + 2 versions x 5 topics x (1 options + 0-1 pick + 1 critic) = 35 at most.

    uv run --env-file .env python -m shortsmith.agents.prompt_eval
Results: runs/experiments/phase6_prompts_<time>/results.json and report.txt, and the MLflow
experiment "shortsmith-prompt-eval" (a parent run, one nested run per topic and version).
"""

import json
import time
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from shortsmith import ledger, llm, tracing
from shortsmith.agents import critic, researcher, writer
from shortsmith.agents.state import Usage
from shortsmith.models import NARRATION_WORDS, CriticReport, ResearchNotes, Script

TOPICS: list[tuple[str, str]] = [
    ("coffee shop in Tashkent, cozy autumn vibe", "en"),  # Phase 5's loss: narration described the pictures
    ("how to save money as a university student", "en"),
    ("a day trip from Tashkent to the Chimgan mountains", "en"),
    ("Samarqand Registoni kechqurun", "uz"),
    ("uyda palov tayyorlash sirlari", "uz"),
]
EXPERIMENT = "shortsmith-prompt-eval"


class OptionFacts(BaseModel):
    title: str
    hook: str
    words: int
    in_range: bool
    hook_copied_from: str | None


class VersionResult(BaseModel):
    version: str
    options: list[OptionFacts]  # every option that passed the Script rules
    rejected: list[str]  # options that didn't (too few or too many words, ...)
    chosen: Script
    pick_reason: str | None
    review: CriticReport
    writer_usage: Usage
    critic_usage: Usage
    seconds: float


class TopicResult(BaseModel):
    topic: str
    language: str
    notes: ResearchNotes
    results: dict[str, VersionResult]


def facts(script: Script, notes: ResearchNotes) -> OptionFacts:
    low, high = NARRATION_WORDS[script.language]
    n = writer.words(script)
    return OptionFacts(title=script.title, hook=script.hook, words=n, in_range=low <= n <= high,
                       hook_copied_from=writer.copied_from(script.hook, writer.hook_sources(notes)))


def run_version(topic: str, language: str, notes: ResearchNotes, version: str) -> VersionResult:
    start = time.perf_counter()
    written = writer.write(topic, language, notes, version)
    review, result = critic.review(topic, written.chosen, notes)  # no frames: script mode
    return VersionResult(version=version, options=[facts(o, notes) for o in written.options], rejected=written.rejected,
                         chosen=written.chosen, pick_reason=written.pick.reason if written.pick else None,
                         review=review, writer_usage=Usage.of(written.results), critic_usage=Usage.of([result]),
                         seconds=round(time.perf_counter() - start, 1))


def metrics(r: VersionResult) -> dict[str, float]:
    return {
        "critic_score": r.review.score,
        "weak_scenes": len(r.review.weak_scenes),
        "chosen_words": r.review.checks.total_words,
        "chosen_in_range": float(r.review.checks.in_range),
        "chosen_hook_copied": float(r.review.checks.hook_copied_from is not None),
        "options_valid": len(r.options),
        "options_rejected": len(r.rejected),
        "options_in_range": sum(o.in_range for o in r.options),
        "options_hook_copied": sum(o.hook_copied_from is not None for o in r.options),
        "writer_prompt_tokens": r.writer_usage.prompt_tokens,
        "writer_completion_tokens": r.writer_usage.completion_tokens,
        "writer_cost_usd": r.writer_usage.cost_usd,
        "critic_prompt_tokens": r.critic_usage.prompt_tokens,
        "seconds": r.seconds,
    }


def report(results: list[TopicResult]) -> str:
    rows = [f"{'topic':<46}{'ver':<5}{'score':>6}{'weak':>5}{'words':>7}{'range':>7}{'copied':>8}"
            f"{'opts in range':>15}{'opts copied':>13}{'rejected':>10}  hook"]
    totals: dict[str, dict[str, float]] = {}
    for t in results:
        for version, r in sorted(t.results.items()):
            m = metrics(r)
            for key, value in m.items():
                totals.setdefault(version, {}).setdefault(key, 0.0)
                totals[version][key] += value
            rows.append(f"{t.topic[:44]:<46}{version:<5}{r.review.score:>6g}{len(r.review.weak_scenes):>5}"
                        f"{r.review.checks.total_words:>7}{'yes' if r.review.checks.in_range else 'no':>7}"
                        f"{'yes' if r.review.checks.hook_copied_from else 'no':>8}"
                        f"{int(m['options_in_range']):>9}/{len(r.options):<5}{int(m['options_hook_copied']):>7}/{len(r.options):<5}"
                        f"{len(r.rejected):>10}  {r.chosen.hook!r}")
    rows.append("")
    for version, m in sorted(totals.items()):
        n = len(results)
        rows.append(f"{version}: mean critic score {m['critic_score'] / n:.2f} | chosen in range {int(m['chosen_in_range'])}/{n} | "
                    f"chosen hook copied {int(m['chosen_hook_copied'])}/{n} | options in range {int(m['options_in_range'])}/"
                    f"{int(m['options_valid'])} valid (+{int(m['options_rejected'])} rejected) | options copied "
                    f"{int(m['options_hook_copied'])}/{int(m['options_valid'])} | writer {int(m['writer_prompt_tokens'])} in + "
                    f"{int(m['writer_completion_tokens'])} out tokens, ${m['writer_cost_usd']:.4f} list price")
    return "\n".join(rows)


def main() -> None:
    out = Path("runs/experiments") / f"phase6_prompts_{datetime.now():%Y-%m-%d_%H%M%S}"
    out.mkdir(parents=True)
    ledger.CURRENT_RUN.set(out.name)  # every ledger line of the evaluation carries its folder's name
    traced = tracing.setup(EXPERIMENT)
    print(f"prompt evaluation | {len(TOPICS)} topics | writer {llm.model_for("writer")}, critic {critic.critic_model()} | "
          f"MLflow: {'on' if traced else 'off'}")
    results: list[TopicResult] = []
    with tracing.mlflow_run(f"prompt-eval {out.name}", params={"topics": len(TOPICS), "versions": "v1,v2",
                                                                "writer_model": llm.model_for("writer"),
                                                                "critic_model": critic.critic_model()}):
        for n, (topic, language) in enumerate(TOPICS):
            print(f"\n== {topic} ({language})")
            found = llm.paced(lambda: researcher.research(topic, language))
            print(researcher.summary(found))
            versions = ["v1", "v2"] if n % 2 == 0 else ["v2", "v1"]
            done: dict[str, VersionResult] = {}
            for version in versions:
                with tracing.mlflow_run(f"{version} | {topic[:40]}", nested=True,
                                        params={"topic": topic, "language": language, "prompt_version": version}):
                    with tracing.span(f"eval {version}", "CHAIN", inputs={"topic": topic, "prompt_version": version}):
                        r = run_version(topic, language, found.notes, version)
                    tracing.log_metrics(metrics(r))
                done[version] = r
                print(f"  {version}: score {r.review.score:g} | {r.review.checks.total_words} words "
                      f"({'in range' if r.review.checks.in_range else 'OUT of range'}) | hook {r.chosen.hook!r}"
                      f"{' (COPIED: ' + r.review.checks.hook_copied_from + ')' if r.review.checks.hook_copied_from else ''}"
                      f" | options {len(r.options)} valid, {len(r.rejected)} rejected | {r.seconds:.0f} s")
            results.append(TopicResult(topic=topic, language=language, notes=found.notes, results=done))
            (out / "results.json").write_text(json.dumps([t.model_dump(mode="json") for t in results], indent=2,
                                                         ensure_ascii=False), encoding="utf-8")  # after every topic
        table = report(results)
        for version in ("v1", "v2"):
            scores = [t.results[version].review.score for t in results]
            tracing.log_metrics({f"{version}_mean_critic_score": sum(scores) / len(scores)})
    (out / "report.txt").write_text(table, encoding="utf-8")
    print("\n" + table + f"\n\nsaved: {out}")


if __name__ == "__main__":
    main()

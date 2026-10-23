"""Step 5.5: does research make better scripts? A small, fair side-by-side.

For each topic the same writer (same model, same settings, same system prompt) writes twice:
once from the topic alone, once with the Researcher's notes. Then:
- a judge model scores both scripts blind (labelled A and B, never "with research") on a fixed
  rubric, twice, with A and B swapped the second time: a judge that prefers whatever comes first
  shows up as a disagreement between the two orders;
- each script's hook is embedded and matched to its nearest hook pattern in the knowledge base,
  so "did the writer use a retrieved pattern?" is measured, not guessed, for both arms (a script
  written without research can follow a known pattern by itself).

Run from the repo root (about 20 LLM calls, paced for the free tier; YouTube is cached for 24 h):
    uv run --env-file .env python -m shortsmith.research.experiment
Results: runs/experiments/phase5_<time>/results.json, plus a table on the console.
"""

import json
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

import openai
from pydantic import BaseModel, ValidationError

from shortsmith import llm
from shortsmith.agents import researcher
from shortsmith.models import ResearchNotes, Script
from shortsmith.research import rag

TOPICS: list[tuple[str, str]] = [
    ("coffee shop in Tashkent, cozy autumn vibe", "en"),
    ("how to save money as a university student", "en"),
    ("Samarqand Registoni kechqurun", "uz"),
    ("uyda palov tayyorlash sirlari", "uz"),
]
JUDGE_MODEL = "gemini-3.8-flash"  # a stronger model than the writer, and not the writer itself
PAUSE_S = 4.5  # the free tier allows ~15 requests a minute
WRITER_ATTEMPTS = 2  # a script that breaks a rule (e.g. too few words) is written once more

JUDGE_SYSTEM_PROMPT = """You judge scripts for ~30-second vertical short videos (YouTube Shorts, Reels, TikTok).
You get a topic and two scripts, A and B, written for the same topic and language. Score each from 1 to 10 on:
- hook: would the first spoken line (the first 2 seconds) stop a viewer from scrolling?
- specificity: concrete details, names, numbers and images instead of generic phrases.
- relevance: does it fit what viewers of this topic want to watch today, and does it deliver what the hook promises?
- flow: a clear path from hook to payoff to call to action, in natural spoken language for the script's language.
Then name the overall winner (A, B or tie) and say why in one sentence.
The order of the scripts says nothing about their quality. Be strict: 5 is average, 10 is exceptional."""


class Scores(BaseModel):
    hook: int
    specificity: int
    relevance: int
    flow: int

    @property
    def total(self) -> int:
        return self.hook + self.specificity + self.relevance + self.flow


class Verdict(BaseModel):
    a: Scores
    b: Scores
    winner: Literal["A", "B", "tie"]
    reason: str


class Arm(BaseModel):
    """One script and what we measured about it."""

    script: Script
    words: int
    hook_words: int
    nearest_pattern: str  # the knowledge base's closest hook pattern to this script's hook
    nearest_similarity: float
    attempts: int  # writer calls needed (a rule break is written again)


class TopicResult(BaseModel):
    topic: str
    language: str
    research: researcher.Research
    with_research: Arm
    without_research: Arm
    judged_with_first: Verdict  # A = with research
    judged_without_first: Verdict  # A = without research

    def scores(self, arm: Literal["with", "without"]) -> Scores:
        """The arm's scores from both orders added up (each criterion out of 20: sums avoid rounding)."""
        first = self.judged_with_first.a if arm == "with" else self.judged_with_first.b
        second = self.judged_without_first.b if arm == "with" else self.judged_without_first.a
        return Scores(**{k: getattr(first, k) + getattr(second, k) for k in Scores.model_fields})  # sum of 2 = out of 20

    def winners(self) -> tuple[str, str]:
        """Who won each order, named by arm, not by letter."""
        def name(winner: str, a_is_with: bool) -> str:
            return "tie" if winner == "tie" else ("with" if (winner == "A") == a_is_with else "without")
        return name(self.judged_with_first.winner, True), name(self.judged_without_first.winner, False)


def paced[T](call: Callable[[], T]) -> T:
    """One LLM call at a time, at most ~13 a minute; a 429 waits and tries again (not billed)."""
    for attempt in range(1, 5):
        time.sleep(PAUSE_S)
        try:
            return call()
        except openai.RateLimitError as e:
            if attempt == 4:
                raise
            print(f"  429 ({str(e)[:100]}...): waiting {20 * attempt} s")
            time.sleep(20 * attempt)
    raise AssertionError("unreachable")


def write(topic: str, language: str, notes: ResearchNotes | None) -> tuple[Script, int]:
    for attempt in range(1, WRITER_ATTEMPTS + 1):
        try:
            script, _ = paced(lambda: llm.write_script(topic, language, research=notes))
            return script, attempt
        except ValidationError as e:
            print(f"  the writer broke a rule ({e.errors()[0]['msg']}); attempt {attempt} of {WRITER_ATTEMPTS}")
            if attempt == WRITER_ATTEMPTS:
                raise
    raise AssertionError("unreachable")


def measure(script: Script, attempts: int, kb: rag.KnowledgeBase) -> Arm:
    [nearest] = kb.retrieve(script.hook, k=1)
    return Arm(script=script, words=sum(len(s.narration.split()) for s in script.scenes),
               hook_words=len(script.hook.split()), nearest_pattern=nearest.title,
               nearest_similarity=nearest.similarity, attempts=attempts)


def as_text(script: Script) -> str:
    """What the judge reads: the spoken words only (images are judged in Phase 6, by the Critic)."""
    lines = [f"Title: {script.title}", f"Hook: {script.hook}"]
    lines += [f"Scene {s.idx}: {s.narration}" for s in script.scenes]
    return "\n".join(lines + [f"Call to action: {script.cta}"])


def judge(topic: str, language: str, a: Script, b: Script) -> Verdict:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Topic: {topic}\nLanguage: {llm.LANGUAGES[language]}\n\n"
                                    f"Script A:\n{as_text(a)}\n\nScript B:\n{as_text(b)}"},
    ]
    verdict, _ = paced(lambda: llm.chat_json(messages, Verdict, model=JUDGE_MODEL, step="judge"))
    return verdict


def run_topic(topic: str, language: str, kb: rag.KnowledgeBase) -> TopicResult:
    print(f"\n== {topic} ({language})")
    found = paced(lambda: researcher.research(topic, language))
    print(researcher.summary(found))
    with_script, with_attempts = write(topic, language, found.notes)
    without_script, without_attempts = write(topic, language, None)
    print(f"  with research:    {with_script.hook!r}\n  without research: {without_script.hook!r}")
    return TopicResult(
        topic=topic, language=language, research=found,
        with_research=measure(with_script, with_attempts, kb),
        without_research=measure(without_script, without_attempts, kb),
        judged_with_first=judge(topic, language, with_script, without_script),
        judged_without_first=judge(topic, language, without_script, with_script),
    )


def report(results: list[TopicResult]) -> str:
    rows = [f"{'topic':<44}{'arm':<9}{'hook':>5}{'spec':>5}{'rel':>5}{'flow':>5}{'total':>6}  "
            f"{'words':>5}  nearest pattern (similarity)"]
    for r in results:
        for arm in ("with", "without"):
            s, a = r.scores(arm), (r.with_research if arm == "with" else r.without_research)
            recommended = arm == "with" and a.nearest_pattern in {h.pattern for h in r.research.notes.hooks}
            rows.append(f"{r.topic[:42]:<44}{arm:<9}{s.hook:>5}{s.specificity:>5}{s.relevance:>5}{s.flow:>5}"
                        f"{s.total:>6}  {a.words:>5}  {a.nearest_pattern} ({a.nearest_similarity:.2f})"
                        + (" = a recommended pattern" if recommended else ""))
        first, second = r.winners()
        rows.append(f"{'':<44}winner: {first} (with first), {second} (without first)")
    wins = [w for r in results for w in r.winners()]
    rows.append(f"\nverdicts: with research {wins.count('with')}, without {wins.count('without')}, "
                f"tie {wins.count('tie')} (out of {len(wins)}; scores are sums of 2 judgements, max 10 x 2)")
    return "\n".join(rows)


def main() -> None:
    out = Path("runs/experiments") / f"phase5_{datetime.now():%Y-%m-%d_%H%M%S}"
    out.mkdir(parents=True)
    kb = rag.default_kb()
    print(kb.sync())
    results = []
    for topic, language in TOPICS:
        results.append(run_topic(topic, language, kb))
        (out / "results.json").write_text(json.dumps([r.model_dump(mode="json") for r in results], indent=2,
                                                     ensure_ascii=False), encoding="utf-8")  # saved after every topic
    table = report(results)
    (out / "report.txt").write_text(table, encoding="utf-8")
    print("\n" + table + f"\n\nsaved: {out}")


if __name__ == "__main__":
    main()

"""The Researcher: real trends + retrieved hook patterns -> short notes for the Writer.

    topic -> YouTube (top Shorts, last 30 days)  \
          -> knowledge base (top-k hook patterns) -> one structured LLM call -> ResearchNotes
          -> brand guideline sections (optional) /

Gathering is plain code: no LLM decides what to search, so a research step costs exactly one
LLM call (plus one or two embedding requests and, unless cached, one YouTube search). The LLM
only does what code can't: judge which trends are really about the topic, pick the patterns
that fit, and write it all down in a few lines. The notes are short on purpose, because they
go into the writer's prompt (context engineering): the writer never sees view counts, URLs or
similarity scores, only what to do with them.

Phase 6's LangGraph Researcher node calls research() as it is.

Try it from the repo root (one LLM call; YouTube is cached for 24 h):
    uv run --env-file .env python -m shortsmith.agents.researcher "coffee shop in Tashkent, cozy autumn vibe" --brand anor-va-dolchin
"""

import argparse
import logging
import re

from pydantic import BaseModel

from shortsmith import llm
from shortsmith.models import ResearchNotes
from shortsmith.research import rag, youtube

log = logging.getLogger("shortsmith.researcher")

HOOKS_K = 6  # patterns shown to the researcher; it keeps the best 2-3
BRAND_K = rag.BRAND_K
TRENDS_SHOWN = 10
MIN_TRENDS = 3  # fewer Shorts than this: try once more with a broader query

RESEARCHER_SYSTEM_PROMPT = """You are the researcher of a short-video team. From real data you write SHORT \
notes for the scriptwriter.

You get: a topic and language; the most-viewed YouTube videos of Shorts length about it from the last 30 \
days (views, length, title); hook patterns retrieved from our library; and sometimes a brand's rules.

Write:
- trending: 2-4 observations on what gets views for this topic now, based only on the videos given; \
describe a video by what its title says and mention its view count. The search is by keyword, so ignore \
videos that are not about the topic (a word can mean something else in another language). If none are \
relevant, say so in one item.
- hooks: the 2-3 given patterns that fit best (their exact names). For each, a hook for THIS topic in the \
video's language, at most 8 words, and one sentence on why it fits.
- angles: 2-3 concrete angles or details the script can use.
- avoid: 1-4 things to avoid: clichés, angles that are already everywhere, and the brand's words to avoid.
- visual_style: the brand's style words for image prompts, word for word; "" if there is no brand.

Each item is one short line (at most 25 words). Never invent numbers or facts that are not in the data. \
The videos are evidence of what viewers like, not facts for our script: never reuse their addresses, \
names or claims (they belong to other people and businesses); a brand's facts come only from its rules."""


class Research(BaseModel):
    """Everything one research step saw and produced: saved as research.json in the run folder."""

    topic: str
    language: str
    brand: str | None
    trend_queries: list[str]  # the YouTube searches, in order (a second one = the first found too little)
    trends: youtube.TrendReport | None  # None if YouTube failed: research goes on without it
    trend_error: str | None = None
    hooks: list[rag.Retrieved]
    brand_rules: list[rag.Retrieved]
    notes: ResearchNotes
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float  # the LLM call; the embeddings are in the ledger as step "rag"
    latency_s: float


def trend_query(topic: str) -> str:
    """What to type into YouTube: the topic's first part, before a comma, dash or colon
    ("coffee shop in Tashkent, cozy autumn vibe" -> "coffee shop in Tashkent"). Long, wordy
    queries find almost nothing among the last 30 days' most-viewed videos."""
    return re.split(r"[,;:–—]| - ", topic, maxsplit=1)[0].strip()


def broader(query: str) -> str | None:
    """The same query without its last word, or None when only two words are left."""
    words = query.split()
    return " ".join(words[:-1]) if len(words) > 2 else None


def find_trends(topic: str, language: str) -> tuple[list[str], youtube.TrendReport | None, str | None]:
    """At most two searches: the topic's first part, then a broader one if that found under
    MIN_TRENDS Shorts. A YouTube failure is reported, not raised: a video can be written without trends."""
    queries = [trend_query(topic)]
    try:
        report = youtube.search_trends(queries[0], language, TRENDS_SHOWN)
        if report.shorts_length < MIN_TRENDS and (wider := broader(queries[0])):
            queries.append(wider)
            report = youtube.search_trends(wider, language, TRENDS_SHOWN)
        return queries, report, None
    except youtube.YouTubeError as e:
        log.warning("no trend data: %s", e)
        return queries, None, str(e)


def research_message(topic: str, language: str, queries: list[str], trends: youtube.TrendReport | None,
                     hooks: list[rag.Retrieved], brand_rules: list[rag.Retrieved]) -> str:
    """The data, as compact as it can be while staying readable."""
    lines = [f"Topic: {topic}", f"Language: {llm.LANGUAGES[language]} (language code '{language}')", ""]
    if trends and trends.videos:
        lines.append(f"Most-viewed Shorts-length YouTube videos since {trends.published_after:%Y-%m-%d} "
                     f"(search: {trends.query!r}):")
        lines += [f"- {v.views:,} views | {v.duration_s} s | {v.title}" for v in trends.videos]
    else:
        lines.append(f"YouTube trends: none found (searches: {', '.join(repr(q) for q in queries)}).")
    lines += ["", "Hook patterns from our library, closest first:"]
    lines += [f"### {h.title}\n{h.text}" for h in hooks]
    if brand_rules:
        lines += ["", f"Brand rules ({brand_rules[0].group}):"]
        lines += [f"### {r.title}\n{r.text}" for r in brand_rules]
    return "\n".join(lines)


def research(topic: str, language: str = "en", brand: str | None = None, model: str | None = None) -> Research:
    """Trends + hook patterns (+ brand rules) -> ResearchNotes, with everything that went in."""
    queries, trends, trend_error = find_trends(topic, language)
    brand_rules = rag.brand_rules(topic, brand, BRAND_K) if brand else []
    hooks = rag.retrieve(topic, HOOKS_K)
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": research_message(topic, language, queries, trends, hooks, brand_rules)},
    ]
    notes, result = llm.chat_json(messages, ResearchNotes, model=model, step="researcher")
    return Research(topic=topic, language=language, brand=brand, trend_queries=queries, trends=trends,
                    trend_error=trend_error, hooks=hooks, brand_rules=brand_rules, notes=notes,
                    prompt_tokens=result.prompt_tokens, completion_tokens=result.completion_tokens,
                    cost_usd=result.cost_usd, latency_s=result.latency_s)


def summary(r: Research) -> str:
    """A few lines for the console: where the evidence came from and what it cost."""
    if r.trends:
        source = "cache" if r.trends.from_cache else f"live, {r.trends.units_used} units"
        trends = f"{len(r.trends.videos)} videos for {r.trends.query!r} ({source})"
    else:
        trends = f"none ({r.trend_error})"
    picked = ", ".join(f"{h.title} {h.similarity:.2f}" for h in r.hooks)
    return (f"trends: {trends}\nretrieved: {picked}"
            + (f"\nbrand: {r.brand} ({', '.join(b.title.split(': ', 1)[-1] for b in r.brand_rules)})" if r.brand else "")
            + f"\nresearcher: {r.prompt_tokens} in + {r.completion_tokens} out tokens, ${r.cost_usd:.6f} list price, "
              f"{r.latency_s} s | notes for the writer: {llm.estimate_tokens(r.notes.for_writer())} tokens")


def main() -> None:
    parser = argparse.ArgumentParser(description="Research a topic: YouTube trends + hook patterns -> notes for the writer.")
    parser.add_argument("topic")
    parser.add_argument("--lang", choices=sorted(llm.LANGUAGES), default="en")
    parser.add_argument("--brand", help="a brand guideline in data/hooks/brands/, e.g. anor-va-dolchin")
    args = parser.parse_args()
    r = research(args.topic, args.lang, args.brand)
    print(summary(r))
    print("\n" + r.notes.for_writer())


if __name__ == "__main__":
    main()

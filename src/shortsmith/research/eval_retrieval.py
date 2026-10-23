"""Does retrieval find the right hook pattern? A small labelled test, five ways of indexing.

16 queries, written by hand to describe a pattern's idea WITHOUT its words (in English, Uzbek
and Russian), each with the pattern ids that count as correct. For each setup: hit@1 (the
first result is correct), hit@3 (a correct one is in the top 3), MRR (1 / rank of the first
correct result: 1, 0.5, 0.33, ...), and how many tokens the top 3 results would add to a prompt.

Setups:
- sections: one "## " section per chunk, gemini-embedding-2 task formats, 768 numbers (what rag.py does)
- no task formats: the same chunks and queries as plain text
- 3072 numbers: the same, at the model's full size
- whole files (too big): one chunk per category file (8 chunks of ~950 tokens)
- templates only (too small): one chunk per pattern, only its one-line template

Ranking is done here in plain Python (cosine over all vectors, exactly what Chroma's index
approximates), so the five setups don't need five Chroma collections. ~330 embedded texts:
run it at most once a day on the free tier (100 texts a minute; a 429 waits and retries).
    uv run --env-file .env python -m shortsmith.research.eval_retrieval
"""

import json
import math
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from shortsmith import llm
from shortsmith.research import rag

MODEL = "gemini-embedding-2"

# query -> pattern ids that count as a correct answer (a few queries fit two patterns equally well)
LABELLED: list[tuple[str, set[str]]] = [
    ("show the finished dish first and then how it was made", {"contrast/end-first"}),
    ("a list of three dishes you must order", {"numbers/three-things"}),
    ("make viewers guess what something at the market costs", {"numbers/guess-the-price", "place/what-money-buys-here"}),
    ("a place locals love that tourists never find", {"place/the-hidden-gem", "contrast/tourist-vs-local"}),
    ("the viewer imagines being there themselves", {"story/pov"}),
    ("Bozordagi bir chol menga muhim maslahat berdi", {"story/what-a-stranger-told-me"}),
    ("Раньше всё стоило дешевле, помните?", {"emotion/nostalgia"}),
    ("warn people about a common trick before they pay", {"emotion/the-friendly-warning", "practical/before-you-buy"}),
    ("begin with a sound instead of words", {"place/sound-first", "visual/silence-then-sound"}),
    ("something seasonal that will be gone soon", {"place/only-this-month", "emotion/last-chance"}),
    ("Bir hafta tajriba qildim, natijasi qanday bo'ldi?", {"story/i-tried-it", "curiosity/what-happens-if"}),
    ("people argue in the comments about which option is better", {"contrast/pick-a-side", "contrast/unpopular-opinion"}),
    ("Покажите ремесло крупным планом, медленно и ритмично", {"visual/the-satisfying-process", "visual/extreme-close-up-then-reveal"}),
    ("prove wrong something that everybody believes", {"contrast/myth-vs-fact"}),
    ("Samarqandga borishdan oldin bilish kerak bo'lgan narsalar ro'yxati", {"practical/the-checklist", "practical/save-this-for-later"}),
    ("a business that started very small and grew big", {"story/small-beginnings", "story/they-said-it-wouldn-t-work"}),
]


class SetupResult(BaseModel):
    setup: str
    chunks: int
    hit_at_1: float
    hit_at_3: float
    mrr: float
    top3_tokens: int  # median, what retrieval would add to a prompt
    embedded_texts: int
    misses: list[str]  # queries with no correct answer in the top 3, and what came first instead


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))


def evaluate(setup: str, docs: dict[str, str], owner: dict[str, str], queries: list[str], dimensions: int,
             token_text: dict[str, str]) -> SetupResult:
    """docs: chunk id -> embedded text; owner: chunk id -> the pattern id(s) it stands for, as a
    "|"-joined string (a whole-file chunk stands for all of its patterns)."""
    ids = list(docs)
    vectors = llm.embed([docs[i] for i in ids], MODEL, dimensions, step="rag-eval").vectors
    query_vectors = llm.embed(queries, MODEL, dimensions, step="rag-eval").vectors
    hits1 = hits3 = 0
    rr = 0.0
    tokens, misses = [], []
    for (query, correct), q in zip(LABELLED, query_vectors):
        ranked = sorted(ids, key=lambda i: -cosine(q, vectors[ids.index(i)]))
        is_correct = [bool(set(owner[i].split("|")) & correct) for i in ranked]
        hits1 += is_correct[0]
        hits3 += any(is_correct[:3])
        rr += next((1 / (rank + 1) for rank, ok in enumerate(is_correct) if ok), 0.0)
        tokens.append(sum(llm.estimate_tokens(token_text[i]) for i in ranked[:3]))
        if not any(is_correct[:3]):
            misses.append(f"{query!r} -> {ranked[0]}")
    n = len(LABELLED)
    return SetupResult(setup=setup, chunks=len(ids), hit_at_1=round(hits1 / n, 3), hit_at_3=round(hits3 / n, 3),
                       mrr=round(rr / n, 3), top3_tokens=sorted(tokens)[n // 2], embedded_texts=len(ids) + n,
                       misses=misses)


def main() -> None:
    chunks = [c for c in rag.load_chunks() if c.kind == "hook"]
    known = {c.id for c in chunks}
    assert all(correct <= known for _, correct in LABELLED), "a label names a pattern that doesn't exist"
    queries = [q for q, _ in LABELLED]
    plain = {c.id: c.text for c in chunks}
    sections = {c.id: rag.document_text(c, MODEL) for c in chunks}
    owner = {c.id: c.id for c in chunks}
    formatted_queries = [rag.query_text(q, MODEL) for q in queries]

    files: dict[str, list[rag.Chunk]] = {}
    for c in chunks:
        files.setdefault(c.source, []).append(c)
    whole = {src: "title: " + cs[0].group + " | text: " + "\n\n".join(f"{c.title}\n{c.text}" for c in cs)
             for src, cs in files.items()}
    whole_owner = {src: "|".join(c.id for c in cs) for src, cs in files.items()}
    templates = {c.id: f"title: {c.title} | text: " + c.text.split("\n")[0] for c in chunks}  # "Template: ..." only

    results = [
        evaluate("sections (rag.py)", sections, owner, formatted_queries, 768, plain),
        evaluate("no task formats", {c.id: f"{c.title}\n{c.text}" for c in chunks}, owner, queries, 768, plain),
        evaluate("3072 numbers", sections, owner, formatted_queries, 3072, plain),
        evaluate("whole files (too big)", whole, whole_owner, formatted_queries, 768, whole),
        evaluate("templates only (too small)", templates, owner, formatted_queries, 768, templates),
    ]
    out = Path("runs/experiments") / f"retrieval_eval_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'setup':<28}{'chunks':>7}{'hit@1':>7}{'hit@3':>7}{'MRR':>7}{'top-3 tokens':>14}")
    for r in results:
        print(f"{r.setup:<28}{r.chunks:>7}{r.hit_at_1:>7.2f}{r.hit_at_3:>7.2f}{r.mrr:>7.2f}{r.top3_tokens:>14}")
    for r in results:
        if r.misses:
            print(f"\n{r.setup}: not in the top 3:\n  " + "\n  ".join(r.misses))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()

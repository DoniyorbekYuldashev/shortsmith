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

--stores checks the vector stores themselves instead (the pgvector move, 2026-10-25): the same 16
queries, and the Phase 5 example queries, through Chroma's and Postgres's indexes, against an exact
cosine ranking of the stored vectors. It embeds only the queries (22 texts), once: their vectors
are cached in .cache/, so a re-run makes no API call.
    uv run --env-file .env python -m shortsmith.research.eval_retrieval --stores
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from shortsmith import llm
from shortsmith.research import rag, stores

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
    rankings = [sorted(ids, key=lambda i: -cosine(q, vectors[ids.index(i)])) for q in query_vectors]
    return score(setup, rankings, owner, token_text, len(ids), len(ids) + len(LABELLED))


def score(setup: str, rankings: list[list[str]], owner: dict[str, str], token_text: dict[str, str], chunks: int,
          embedded_texts: int) -> SetupResult:
    """hit@1, hit@3, MRR and top-3 tokens from one ranked id list per labelled query."""
    hits1 = hits3 = 0
    rr = 0.0
    tokens, misses = [], []
    for (query, correct), ranked in zip(LABELLED, rankings, strict=True):
        is_correct = [bool(set(owner[i].split("|")) & correct) for i in ranked]
        hits1 += is_correct[0]
        hits3 += any(is_correct[:3])
        rr += next((1 / (rank + 1) for rank, ok in enumerate(is_correct) if ok), 0.0)
        tokens.append(sum(llm.estimate_tokens(token_text[i]) for i in ranked[:3]))
        if not any(is_correct[:3]):
            misses.append(f"{query!r} -> {ranked[0]}")
    n = len(LABELLED)
    return SetupResult(setup=setup, chunks=chunks, hit_at_1=round(hits1 / n, 3), hit_at_3=round(hits3 / n, 3),
                       mrr=round(rr / n, 3), top3_tokens=sorted(tokens)[n // 2], embedded_texts=embedded_texts,
                       misses=misses)


# --- --stores: the same index in two stores ------------------------------------------------------------

QUERY_CACHE = Path(".cache") / f"query-vectors-{MODEL}-768.json"  # gitignored; query text -> vector

# The Phase 5 build notes' example queries (section 4), with their filters: (query, kind, brand, k).
EXAMPLES: list[tuple[str, rag.Kind | None, str | None, int]] = [
    ("cozy coffee shop in Tashkent, autumn vibe", "hook", None, 5),
    ("Toshkentdagi shinam qahvaxona, kuzgi kayfiyat", "hook", None, 5),
    ("Samarqand Registoni kechqurun", "hook", None, 3),
    ("coffee shop in Tashkent, cozy autumn vibe: tone, words to use and avoid, visual style", "brand", "anor-va-dolchin", 5),
    ("Samarqand Registoni kechqurun: tone, words to use and avoid, visual style", "brand", "samoviy-karvon", 5),
    ("guess the price of melons at the bazaar", None, None, 5),  # no filter: hooks and brand sections together
]
STORES = ("chroma", "pgvector", "exact")  # "exact": the cosine formula over every stored vector, no index


def cached_query_vectors(texts: list[str], dimensions: int = 768, cache: Path = QUERY_CACHE) -> tuple[list[list[float]], int]:
    """The vectors for these (already formatted) query texts, embedding only the ones not cached yet.
    Returns (vectors, how many texts were embedded now)."""
    known: dict[str, list[float]] = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    missing = [t for t in dict.fromkeys(texts) if t not in known]
    if missing:
        result = llm.embed(missing, MODEL, dimensions, step="rag-eval")
        known.update(zip(missing, result.vectors, strict=True))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known), encoding="utf-8")
    return [known[t] for t in texts], len(missing)


class ParityRow(BaseModel):
    query: str
    kind: str | None
    brand: str | None
    k: int
    ids: dict[str, list[str]]  # store -> ids, nearest first
    scores: dict[str, list[float]]  # store -> similarities, rounded to 4 places as KnowledgeBase does
    same_ids: bool  # every store returned the same ids in the same order
    max_score_difference: float  # of any store from the exact score, rank by rank


def exact(records: list[stores.Record], vector: list[float], k: int, kind: str | None, brand: str | None) -> list[rag.Retrieved]:
    """The ground truth: every stored vector scored by the cosine formula, no index involved."""
    pool = [r for r in records if (not kind or r.kind == kind) and (not brand or r.brand == brand)]
    ranked = sorted(pool, key=lambda r: -cosine(vector, r.embedding))[:k]
    return [rag.Retrieved(id=r.id, kind=r.kind, title=r.title, group=r.group, text=r.text,
                          similarity=round(cosine(vector, r.embedding), 4)) for r in ranked]


def compare_stores() -> tuple[int, list[ParityRow], list[SetupResult]]:
    """(query texts embedded now, the example queries in every store, the 16-query eval per store)."""
    from shortsmith.research import migrate

    def refuse(texts: list[str]) -> llm.EmbedResult:
        raise RuntimeError(f"a store is not in step with data/hooks/ ({len(texts)} chunks to embed): run "
                           "`python -m shortsmith.research.rag` and `python -m shortsmith.research.migrate` first")

    kbs = {name: rag.KnowledgeBase(store=store, model=MODEL, dimensions=768, embed=refuse)
           for name, store in migrate.open_both(MODEL, 768).items()}
    for kb in kbs.values():  # both must hold the same 79 chunks before anything is compared
        kb.sync()
    records = kbs["pgvector"].store.records()
    hooks = [r for r in records if r.kind == "hook"]

    texts = [rag.query_text(q, MODEL) for q, *_ in EXAMPLES] + [rag.query_text(q, MODEL) for q, _ in LABELLED]
    vectors, embedded = cached_query_vectors(texts)
    example_vectors, labelled_vectors = vectors[:len(EXAMPLES)], vectors[len(EXAMPLES):]

    parity = []
    for (query, kind, brand, k), v in zip(EXAMPLES, example_vectors, strict=True):
        found = {name: kb.nearest(v, k, kind, brand) for name, kb in kbs.items()} | {"exact": exact(records, v, k, kind, brand)}
        ids = {name: [r.id for r in rs] for name, rs in found.items()}
        scores = {name: [r.similarity for r in rs] for name, rs in found.items()}
        parity.append(ParityRow(
            query=query, kind=kind, brand=brand, k=k, ids=ids, scores=scores,
            same_ids=len({tuple(i) for i in ids.values()}) == 1,
            max_score_difference=round(max(abs(a - b) for s in scores.values() for a, b in zip(s, scores["exact"])), 6)))

    owner, plain = {r.id: r.id for r in hooks}, {r.id: r.text for r in hooks}
    n = len(hooks)  # rank every hook pattern: MRR needs the rank of the first correct one, wherever it is
    rankings = {name: [[r.id for r in kb.nearest(v, n, "hook")] for v in labelled_vectors] for name, kb in kbs.items()}
    rankings["exact"] = [[r.id for r in exact(hooks, v, n, "hook", None)] for v in labelled_vectors]
    return embedded, parity, [score(name, rankings[name], owner, plain, n, len(LABELLED)) for name in STORES]


def main_stores() -> None:
    embedded, parity, results = compare_stores()
    path = Path("runs/experiments") / f"store_parity_{datetime.now():%Y-%m-%d_%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"embedded_now": embedded, "parity": [p.model_dump() for p in parity],
                                "eval": [r.model_dump() for r in results]}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"query texts embedded now: {embedded} (the others came from {QUERY_CACHE})\n")
    for p in parity:
        where = f"kind={p.kind or 'any'}" + (f", brand={p.brand}" if p.brand else "")
        print(f"{p.query!r} ({where}, k={p.k}): same ids in every store: {p.same_ids}; "
              f"max score difference from exact: {p.max_score_difference:g}")
        print(f"  {'#':<3}" + "".join(f"{s:<50}" for s in STORES))
        for rank in range(p.k):
            print(f"  {rank + 1:<3}" + "".join(f"{f'{p.scores[s][rank]:.4f} {p.ids[s][rank]}' if rank < len(p.ids[s]) else '-':<50}"
                                             for s in STORES))
    print(f"\n{'16 labelled queries':<22}{'hit@1':>7}{'hit@3':>7}{'MRR':>8}")
    for r in results:
        print(f"{r.setup:<22}{r.hit_at_1:>7.2f}{r.hit_at_3:>7.2f}{r.mrr:>8.3f}" + (f"   not in the top 3: {r.misses}" if r.misses else ""))
    print(f"\nsaved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation (Phase 5), or --stores: Chroma vs pgvector.")
    parser.add_argument("--stores", action="store_true", help="compare the stores' indexes (embeds only the queries, once)")
    if parser.parse_args().stores:
        main_stores()
        return
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
    try:
        main()
    except stores.StoreUnavailable as e:  # one line and the next step, not a traceback
        raise SystemExit(f"error: {e}") from None

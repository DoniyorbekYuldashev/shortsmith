"""🧠 Exercise, Step 5.3: the core of RAG in plain Python, no API and no Chroma.

Four small functions are the whole idea behind src/shortsmith/research/rag.py:
1. split_sections: chunking (one "## " section of a markdown file = one chunk)
2. cosine_similarity: how close two embeddings are (what Chroma computes for us)
3. top_k: retrieval (the k chunks closest to a query vector)
4. plan_sync: why rebuilding the index is idempotent (embed only what changed)

The finished versions: split_sections() and KnowledgeBase.sync() in rag.py (Chroma does 2 and 3).
Write these first, then compare. Check them from the repo root:  uv run pytest exercises/phase-5
The tests use tiny hand-made vectors (2-3 numbers), so you can check every answer on paper.
"""


def split_sections(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    """A markdown file's "# " title and its "## " sections as (heading, body) pairs.

    Example: "# Hooks\\nintro\\n## Open loop\\nline 1\\nline 2\\n## POV\\nline 3"
          -> ("Hooks", [("Open loop", "line 1\\nline 2"), ("POV", "line 3")])

    Hints:
    1. Go through markdown.splitlines() one line at a time, remembering the current heading
       (None before the first "## ") and the body lines collected for it.
    2. A line starting with "## " closes the current section (if any) and opens a new one:
       the heading is the rest of the line, stripped.
    3. A line starting with "# " before any section is the title (only the first one).
    4. Lines before the first "## " (the intro) belong to no section: skip them.
    5. Don't forget the LAST section when the lines run out. Strip each body.
    """
    raise NotImplementedError("your turn: follow the hints above")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(angle between a and b) = (a . b) / (|a| * |b|).

    1 = same direction (same meaning), 0 = at a right angle (unrelated), -1 = opposite.
    The LENGTH of a vector doesn't matter: [1, 2] and [2, 4] have similarity 1.

    Hints:
    1. The dot product: sum(x * y for x, y in zip(a, b)).
    2. A vector's length (norm): the square root of the sum of its squares (math.sqrt or math.hypot(*v)).
    3. Vectors of different lengths are a bug: raise ValueError. So is a vector of only zeros
       (its direction is undefined: dividing by 0).
    """
    raise NotImplementedError("your turn: follow the hints above")


def top_k(query: list[float], vectors: dict[str, list[float]], k: int) -> list[tuple[str, float]]:
    """The k ids whose vectors are most similar to `query`, as (id, similarity), best first.

    Example: query [1, 0], vectors {"a": [1, 0], "b": [0, 1], "c": [1, 1]}, k=2
          -> [("a", 1.0), ("c", 0.7071...)]

    Hints:
    1. Score every vector with cosine_similarity(query, vector).
    2. Sort by similarity, highest first. For equal scores, sort by id, so the answer never
       depends on dict order (a key like (-score, id) does both).
    3. Return the first k. Asking for more than there are returns all of them.
    """
    raise NotImplementedError("your turn: follow the hints above")


def plan_sync(stored: dict[str, str], current: dict[str, str]) -> tuple[list[str], list[str]]:
    """What a re-index must do, given {chunk id: content hash} in the store and in the files now.

    Returns (to_embed, to_delete), each sorted:
    - to_embed: ids that are new, or whose hash changed (their text was edited)
    - to_delete: ids that are in the store but no longer in the files
    Unchanged chunks are in neither list: that is why a second run costs nothing.

    Hints:
    1. A chunk needs embedding if stored.get(id) != current hash (a new id gives None != hash).
    2. A set difference gives the deleted ids.
    """
    raise NotImplementedError("your turn: follow the hints above")

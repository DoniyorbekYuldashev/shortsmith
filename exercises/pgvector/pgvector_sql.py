"""Optional exercise (the move to pgvector, 2026-10-25): the SQL and the math behind PgVectorStore.

Three small functions are the core of src/shortsmith/research/stores.py:
1. similarity: turn what each pgvector distance operator returns into a cosine similarity
   (KnowledgeBase.nearest does 1 - distance for `<=>`).
2. topk_query: the filtered nearest-neighbour query as SQL text + its parameters (stores.topk_sql).
3. rows_left_after_filter: why a filtered HNSW search can return fewer than k rows, and why
   pgvector 0.8 added hnsw.iterative_scan (PgVectorStore.query turns it on).

Write these first, then compare with the real ones.
Check them from the repo root:  uv run pytest exercises/pgvector
No database needed: plain strings and numbers, small enough to check on paper.
"""

import re  # noqa: F401  (you will want re.fullmatch)

ALLOWED_FILTERS = ("kind", "brand")  # the only columns a filter may name


def similarity(distance: float, operator: str) -> float:
    """The cosine similarity of two NORMALISED vectors (length 1), from what a pgvector operator returned.

    pgvector's three distance operators (smaller = closer, so ORDER BY ... LIMIT k finds the nearest):
    - `<=>` cosine distance:          1 - cos(angle)   (0 = same direction, 1 = unrelated, 2 = opposite)
    - `<#>` NEGATIVE inner product:   -(a . b)         (negated so that "smaller is closer" holds here too)
    - `<->` Euclidean (L2) distance:  |a - b|          (the straight-line distance between the two points)

    Example: similarity(0.25, "<=>") -> 0.75;  similarity(-0.75, "<#>") -> 0.75

    Hints:
    1. For vectors of length 1, a . b IS the cosine of the angle between them.
    2. For vectors of length 1, |a - b|^2 = |a|^2 + |b|^2 - 2 a.b = 2 - 2 cos. Solve that for cos.
    3. Any other operator: raise ValueError. An unknown operator must never quietly give a number.
    """
    raise NotImplementedError("your turn: follow the hints above")


def topk_query(table: str, k: int, filters: dict[str, str]) -> tuple[str, list]:
    """The k rows nearest to a query vector, filtered, as (sql, params) for psycopg.

    The query vector is the first parameter, then one parameter per filter, then k:
        topk_query("kb", 5, {"kind": "brand", "brand": "anor"}) ->
        ('SELECT id, embedding <=> %s AS distance FROM "kb" WHERE kind = %s AND brand = %s ORDER BY distance LIMIT %s',
         ["<query>", "brand", "anor", 5])
    With no filters there is no WHERE at all:
        ('SELECT id, embedding <=> %s AS distance FROM "kb" ORDER BY distance LIMIT %s', ["<query>", 5])
    ("<query>" stands for the vector; the real code puts a numpy array there.)

    Rules (the security part):
    - a VALUE never goes into the SQL text, only into params: the database receives the text and the
      values separately, so a value like  x' OR '1'='1  is just a strange brand name, never SQL;
    - a filter COLUMN can't be a parameter (parameters are values only), so it is checked against
      ALLOWED_FILTERS instead: anything else is a ValueError;
    - the table name must match ^[a-z_][a-z0-9_]*$ (else ValueError) and is written in double quotes;
    - k must be at least 1 (else ValueError);
    - the filters keep the order they were given in.

    Hints: build a list of conditions ("kind = %s") and a list of params side by side;
    " AND ".join(conditions); one space between the parts, exactly as in the examples.
    """
    raise NotImplementedError("your turn: follow the hints above")


def rows_left_after_filter(ef_search: int, k: int, matching: float) -> float:
    """How many rows, on average, an HNSW index scan WITHOUT iterative scanning returns for
    `SELECT ... WHERE <filter> ORDER BY distance LIMIT k`.

    The index hands over its ef_search nearest candidates, and only then is the WHERE applied. If a
    fraction `matching` of all rows passes the filter (and passing has nothing to do with the
    distance), about ef_search * matching candidates survive, and LIMIT keeps at most k of them.

    Example: ef_search 40, k 10, matching 0.1 -> 4.0  (the 20,000-row experiment in the build notes
             returned 3 rows there: recall 0.395).  ef_search 100, k 5, matching 0.5 -> 5.

    Hint: one line. This is why pgvector 0.8 added hnsw.iterative_scan: it keeps walking the graph
    until k rows pass the filter (or hnsw.max_scan_tuples is reached).
    """
    raise NotImplementedError("your turn: follow the hints above")

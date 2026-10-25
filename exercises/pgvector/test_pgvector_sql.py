"""Checks for the optional pgvector exercise. Plain strings and numbers; no database.
Run from the repo root:  uv run pytest exercises/pgvector
"""

import math

import pytest

from pgvector_sql import rows_left_after_filter, similarity, topk_query  # pytest puts this folder on sys.path

# --- similarity --------------------------------------------------------------------------------------------


def test_cosine_distance_is_one_minus_the_similarity():
    assert similarity(0.25, "<=>") == pytest.approx(0.75)
    assert similarity(0.0, "<=>") == pytest.approx(1.0)  # the same direction
    assert similarity(2.0, "<=>") == pytest.approx(-1.0)  # opposite directions


def test_the_inner_product_operator_is_negated():
    assert similarity(-0.75, "<#>") == pytest.approx(0.75)
    assert similarity(0.2, "<#>") == pytest.approx(-0.2)


def test_euclidean_distance_of_unit_vectors_gives_the_cosine_too():
    assert similarity(0.0, "<->") == pytest.approx(1.0)
    assert similarity(math.sqrt(2), "<->") == pytest.approx(0.0)  # at right angles: unrelated
    assert similarity(2.0, "<->") == pytest.approx(-1.0)


def test_all_three_operators_agree_on_one_pair():
    a, b = (1.0, 0.0), (0.6, 0.8)  # both of length 1; cos = 0.6
    cosine_distance = 1 - (a[0] * b[0] + a[1] * b[1])
    negative_inner = -(a[0] * b[0] + a[1] * b[1])
    l2 = math.dist(a, b)
    assert [similarity(cosine_distance, "<=>"), similarity(negative_inner, "<#>"), similarity(l2, "<->")] == \
        pytest.approx([0.6, 0.6, 0.6])


def test_an_unknown_operator_is_an_error():
    with pytest.raises(ValueError):
        similarity(0.5, "<+>")  # <+> is L1 (taxicab) distance: no cosine to be had from it


# --- topk_query --------------------------------------------------------------------------------------------


def test_two_filters():
    assert topk_query("kb", 5, {"kind": "brand", "brand": "anor"}) == (
        'SELECT id, embedding <=> %s AS distance FROM "kb" WHERE kind = %s AND brand = %s ORDER BY distance LIMIT %s',
        ["<query>", "brand", "anor", 5])


def test_no_filter_has_no_where():
    assert topk_query("kb_gemini_embedding_2_768", 3, {}) == (
        'SELECT id, embedding <=> %s AS distance FROM "kb_gemini_embedding_2_768" ORDER BY distance LIMIT %s',
        ["<query>", 3])


def test_filters_keep_their_order():
    sql, params = topk_query("kb", 2, {"brand": "anor", "kind": "brand"})
    assert "WHERE brand = %s AND kind = %s" in sql and params == ["<query>", "anor", "brand", 2]


def test_a_value_never_becomes_sql():
    evil = "x' OR '1'='1"
    sql, params = topk_query("kb", 5, {"brand": evil})
    assert evil not in sql and "'" not in sql and params[1] == evil


def test_only_known_columns_can_be_filtered():
    with pytest.raises(ValueError):
        topk_query("kb", 5, {"kind = 'hook' OR 1=1 --": "x"})
    with pytest.raises(ValueError):
        topk_query("kb", 5, {"text": "x"})


@pytest.mark.parametrize("table", ['kb"; DROP TABLE kb; --', "KB", "1kb", "kb-768", ""])
def test_a_strange_table_name_is_refused(table):
    with pytest.raises(ValueError):
        topk_query(table, 5, {})


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError):
        topk_query("kb", 0, {})


# --- rows_left_after_filter ------------------------------------------------------------------------------------


def test_a_tenth_of_the_rows_matching_leaves_a_tenth_of_the_candidates():
    assert rows_left_after_filter(40, 10, 0.1) == pytest.approx(4.0)


def test_limit_caps_it_at_k():
    assert rows_left_after_filter(100, 5, 0.5) == pytest.approx(5)
    assert rows_left_after_filter(40, 10, 1.0) == pytest.approx(10)


def test_a_rare_filter_leaves_almost_nothing():
    assert rows_left_after_filter(10, 5, 0.0) == pytest.approx(0.0)
    assert rows_left_after_filter(40, 5, 5 / 79) == pytest.approx(40 * 5 / 79)  # one brand's 5 sections of our 79 rows

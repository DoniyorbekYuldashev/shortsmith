"""Checks for the Step 5.3 exercise. Tiny vectors you can check on paper; no API, no Chroma.
Run from the repo root:  uv run pytest exercises/phase-5
"""

import math

import pytest

from rag_math import cosine_similarity, plan_sync, split_sections, top_k  # pytest puts this folder on sys.path

MARKDOWN = """# Curiosity hooks

The intro: about the whole file, not a section.

## Open loop

**Template:** "[Strange fact]. Why? At the end."

## Guess the price

**Template:** "Guess how much this costs."
**Why it works:** everyone has a price intuition.
"""


# --- split_sections ---------------------------------------------------------------------------


def test_sections_and_title():
    title, sections = split_sections(MARKDOWN)
    assert title == "Curiosity hooks"
    assert sections == [
        ("Open loop", '**Template:** "[Strange fact]. Why? At the end."'),
        ("Guess the price", '**Template:** "Guess how much this costs."\n**Why it works:** everyone has a price intuition.'),
    ]


def test_the_intro_is_not_a_section():
    _, sections = split_sections(MARKDOWN)
    assert all("intro" not in body for _, body in sections)


def test_no_sections():
    assert split_sections("# Just a title\nsome text") == ("Just a title", [])
    assert split_sections("") == ("", [])


def test_a_hash_inside_a_body_is_not_a_heading():
    _, sections = split_sections("## A\ntext with a # sign and ### a small heading\n## B\nx")
    assert sections == [("A", "text with a # sign and ### a small heading"), ("B", "x")]


# --- cosine_similarity ------------------------------------------------------------------------


def test_same_direction_is_one_whatever_the_length():
    assert cosine_similarity([1, 2], [2, 4]) == pytest.approx(1.0)
    assert cosine_similarity([3, 0, 0], [0.5, 0, 0]) == pytest.approx(1.0)


def test_right_angle_is_zero_and_opposite_is_minus_one():
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 1], [-1, -1]) == pytest.approx(-1.0)


def test_forty_five_degrees():
    assert cosine_similarity([1, 0], [1, 1]) == pytest.approx(1 / math.sqrt(2))  # cos 45° = 0.7071


def test_bad_vectors_raise():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 1])


# --- top_k ----------------------------------------------------------------------------------------

VECTORS = {"open-loop": [1, 0, 0], "pov": [0.9, 0.1, 0], "hidden-gem": [0, 1, 0], "nostalgia": [0, 0, 1]}


def test_the_closest_come_first():
    found = top_k([1, 0, 0], VECTORS, 2)
    assert [i for i, _ in found] == ["open-loop", "pov"]
    assert found[0][1] == pytest.approx(1.0) and found[1][1] == pytest.approx(0.9 / math.hypot(0.9, 0.1))


def test_length_does_not_change_the_ranking():
    short, long = top_k([1, 0, 0], VECTORS, 4), top_k([5, 0, 0], VECTORS, 4)
    assert [i for i, _ in short] == [i for i, _ in long]
    assert [s for _, s in short] == pytest.approx([s for _, s in long])


def test_ties_are_broken_by_id_and_k_can_exceed_the_count():
    found = top_k([0, 1, 1], {"b": [0, 1, 0], "a": [0, 0, 1], "c": [1, 0, 0]}, 10)
    assert [i for i, _ in found] == ["a", "b", "c"]  # a and b tie at 0.7071: alphabetical


# --- plan_sync ------------------------------------------------------------------------------------


def test_nothing_changed_means_nothing_to_do():
    assert plan_sync({"a": "h1", "b": "h2"}, {"a": "h1", "b": "h2"}) == ([], [])


def test_new_changed_and_deleted_chunks():
    stored = {"a": "h1", "b": "h2", "gone": "h3"}
    current = {"a": "h1", "b": "EDITED", "new": "h4"}
    assert plan_sync(stored, current) == (["b", "new"], ["gone"])


def test_an_empty_store_embeds_everything():
    assert plan_sync({}, {"z": "1", "y": "2"}) == (["y", "z"], [])

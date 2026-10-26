"""Tests for the allowance exercise. They fail with NotImplementedError until you write the functions."""

from datetime import date, datetime, timedelta, timezone

import pytest

from allowance import allowance_day, billed  # pytest puts this folder on sys.path

UTC = timezone.utc
NOON = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)


def test_the_day_is_the_utc_date():
    tashkent = timezone(timedelta(hours=5))
    assert allowance_day(datetime(2026, 10, 27, 4, 59, tzinfo=tashkent)) == date(2026, 10, 26)
    assert allowance_day(datetime(2026, 10, 27, 5, 0, tzinfo=tashkent)) == date(2026, 10, 27)


def test_free_within_the_allowance_and_a_share_of_the_crossing_call():
    calls = [(NOON, "mini", 600, 0.004), (NOON + timedelta(minutes=1), "mini", 800, 0.006),
             (NOON + timedelta(minutes=2), "mini", 200, 0.002)]
    assert billed(calls, {"mini": 1000}) == pytest.approx([0.0, 0.003, 0.002])


def test_the_answer_keeps_the_callers_order():
    calls = [(NOON + timedelta(minutes=2), "mini", 200, 0.002), (NOON, "mini", 600, 0.004),
             (NOON + timedelta(minutes=1), "mini", 800, 0.006)]
    assert billed(calls, {"mini": 1000}) == pytest.approx([0.002, 0.0, 0.003])


def test_groups_have_separate_allowances_and_none_is_never_free():
    calls = [(NOON, "mini", 1000, 0.01), (NOON, "large", 500, 0.02), (NOON, None, 10, 0.0001)]
    assert billed(calls, {"mini": 1000, "large": 500}) == pytest.approx([0.0, 0.0, 0.0001])


def test_a_new_utc_day_starts_a_new_allowance():
    late = datetime(2026, 10, 26, 23, 59, tzinfo=UTC)
    calls = [(late, "mini", 1000, 0.01), (late + timedelta(minutes=2), "mini", 1000, 0.01)]
    assert billed(calls, {"mini": 1000}) == pytest.approx([0.0, 0.0])

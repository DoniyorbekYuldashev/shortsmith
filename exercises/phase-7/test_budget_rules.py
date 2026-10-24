"""Checks for the optional Phase 7 exercise. Plain numbers; no API, no files.
Run from the repo root:  uv run pytest exercises/phase-7
"""

import pytest

from budget_rules import estimate_call, exit_code, suggest_budget, would_exceed  # pytest puts this folder on sys.path

# --- estimate_call ---------------------------------------------------------------------------------------


def test_the_estimate_is_the_most_expensive_recent_call():
    assert estimate_call([0.001, 0.004, 0.002], first_call=0.5) == 0.004


def test_old_calls_fall_out_of_the_window():
    assert estimate_call([0.009, 0.001, 0.002], first_call=0.5, window=2) == 0.002


def test_the_default_window_is_10_calls():
    past = [0.9] + [0.001] * 10  # the expensive one is the 11th from the end
    assert estimate_call(past, first_call=0.5) == 0.001


def test_no_history_uses_the_first_call_guess():
    assert estimate_call([], first_call=0.004) == 0.004


# --- would_exceed ------------------------------------------------------------------------------------------


def test_no_budget_never_stops():
    assert would_exceed(100.0, 5.0, None) is False


def test_a_step_that_fits_runs():
    assert would_exceed(0.004, 0.001, 0.005) is False  # exactly at the limit still fits


def test_a_step_that_would_pass_the_limit_stops():
    assert would_exceed(0.004, 0.001, 0.0049) is True


def test_a_free_step_is_never_stopped_even_when_already_over():
    assert would_exceed(0.02, 0.0, 0.01) is False


# --- suggest_budget ----------------------------------------------------------------------------------------


def test_the_suggestion_covers_what_is_left_rounded_up_to_a_cent():
    assert suggest_budget(0.0005, [0.01, 0.01, 0.001]) == 0.03


def test_the_suggestion_is_at_least_one_cent():
    assert suggest_budget(0.0, []) == 0.01


def test_a_whole_cent_stays_as_it_is():
    assert suggest_budget(0.25, [0.5]) == 0.75


# --- exit_code -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("outcome, code", [("ok", 0), ("error", 1), ("usage", 2), ("paused", 3),
                                           ("over_budget", 4), ("interrupted", 130)])
def test_every_outcome_has_its_code(outcome, code):
    assert exit_code(outcome) == code


def test_an_unknown_outcome_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="crashed") as error:
        exit_code("crashed")
    assert not isinstance(error.value, NotImplementedError)

"""Tests for the pure helpers in the plotting scripts.

These functions parse values that arrive from wandb run configs, so they are
handed whatever was logged: strings, ``None``, missing keys, and occasionally a
non-finite float. Each one therefore guards its numeric conversion and falls
back to a default rather than raising.

The point of pinning them here is that those guards were bare ``except:``
clauses until recently. A bare except silently catches ``KeyboardInterrupt``
and ``SystemExit`` too, so they were narrowed to the exceptions the conversions
can actually raise -- and narrowing is exactly the kind of change that can
quietly drop a case. ``int(float(v))`` in particular raises ``OverflowError``
on an infinity, which is a subclass of neither ``ValueError`` nor
``TypeError``; the infinity cases below fail if that type is dropped from the
guards again.
"""

import pytest
from plotting.time_series_grid import _loose_match
from plotting.time_series_grid import _safe_int as safe_int_grid
from plotting.time_series_grid_presentation import _safe_int as safe_int_pres
from plotting.time_series_grid_presentation import get_style_for_combo

# Both modules define their own copy of _safe_int; they must behave identically.
SAFE_INT_IMPLS = pytest.mark.parametrize(
    "safe_int", [safe_int_grid, safe_int_pres], ids=["time_series_grid", "presentation"]
)

SENTINEL = -1  # what _safe_int returns when the value is unusable


@SAFE_INT_IMPLS
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (8, 8),
        ("8", 8),
        (8.0, 8),
        ("8.0", 8),
        ("8.7", 8),  # truncates toward zero, via float()
        (-3, -3),
    ],
)
def test_safe_int_parses_usable_values(safe_int, value, expected):
    assert safe_int(value) == expected


@SAFE_INT_IMPLS
@pytest.mark.parametrize(
    "value",
    [
        "abc",  # ValueError
        "",  # ValueError
        None,  # TypeError
        [1, 2],  # TypeError
        {},  # TypeError
        "nan",  # ValueError: cannot convert float NaN to integer
        float("nan"),
    ],
)
def test_safe_int_falls_back_on_unusable_values(safe_int, value):
    assert safe_int(value) == SENTINEL


@SAFE_INT_IMPLS
@pytest.mark.parametrize("value", ["inf", "-inf", "1e400", float("inf"), float("-inf")])
def test_safe_int_falls_back_on_infinities(safe_int, value):
    """int(float(inf)) raises OverflowError, which is neither ValueError nor
    TypeError. Dropping OverflowError from the guard makes this raise instead
    of returning the sentinel."""
    assert safe_int(value) == SENTINEL


@pytest.mark.parametrize(
    ("actual", "target"),
    [(8, 8), ("8", 8), (8.0, "8"), ("0.33", 0.33), ("abc", "abc"), (None, None)],
)
def test_loose_match_accepts_equivalent_values(actual, target):
    assert _loose_match(actual, target) is True


@pytest.mark.parametrize(
    ("actual", "target"),
    [(8, 9), ("8", 9), ("abc", 8), (None, 8), ([1], 8), ("0.33", 0.34)],
)
def test_loose_match_rejects_differing_values(actual, target):
    assert _loose_match(actual, target) is False


def test_loose_match_tolerates_float_representation_error():
    """The whole point of the 1e-6 tolerance: 0.1 + 0.2 != 0.3 exactly."""
    assert _loose_match(0.1 + 0.2, 0.3) is True


@pytest.mark.parametrize("n_val", ["not-a-number", None, float("inf"), [1]])
def test_get_style_for_combo_survives_unusable_subdomain_counts(n_val):
    """An unparseable count must fall back to n=1, not propagate an exception."""
    colour, linestyle = get_style_for_combo("apts_p", n_val)
    assert isinstance(colour, str) and isinstance(linestyle, str)


def test_get_style_for_combo_is_sensitive_to_the_subdomain_count():
    """Guard against the fallback masking real values: distinct counts must
    give distinct colours, or the plots would be silently uninformative."""
    styles = {n: get_style_for_combo("apts_p", n) for n in (2, 4, 8)}
    assert len({colour for colour, _ in styles.values()}) == 3


def test_get_style_for_combo_sgd_ignores_the_count():
    assert get_style_for_combo("sgd", 2) == get_style_for_combo("sgd", 8)

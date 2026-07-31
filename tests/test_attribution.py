import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.attribution import last_touch, linear, time_decay, markov_removal_effect, shapley_value


def test_last_touch_credits_final_channel():
    journeys = {
        1: {"touches": [("social", 0), ("email", 3)], "converted": True, "conversion_day": 4},
        2: {"touches": [("direct", 0)], "converted": False, "conversion_day": None},
    }
    result = last_touch(journeys)
    assert result == {"email": 1.0}


def test_linear_splits_evenly():
    journeys = {
        1: {"touches": [("social", 0), ("email", 3), ("direct", 4)], "converted": True, "conversion_day": 4},
    }
    result = linear(journeys)
    assert math.isclose(sum(result.values()), 1.0)
    assert all(math.isclose(v, 1 / 3) for v in result.values())


def test_time_decay_favors_recent_touch():
    journeys = {
        1: {"touches": [("social", 0), ("email", 9)], "converted": True, "conversion_day": 10},
    }
    result = time_decay(journeys, half_life_days=5.0)
    assert result["email"] > result["social"]
    assert math.isclose(result["email"] + result["social"], 1.0)


def test_markov_removal_effect_essential_vs_useless_channel():
    # Channel A always converts alone; channel B never converts. B's removal
    # effect should be ~0 and A should absorb ~all attributed conversions.
    journeys = {
        1: {"touches": [("A", 0)], "converted": True, "conversion_day": 1},
        2: {"touches": [("A", 0)], "converted": True, "conversion_day": 1},
        3: {"touches": [("B", 0)], "converted": False, "conversion_day": None},
        4: {"touches": [("B", 0)], "converted": False, "conversion_day": None},
    }
    result = markov_removal_effect(journeys, ["A", "B"])
    assert math.isclose(result["A"], 2.0, abs_tol=1e-6)
    assert math.isclose(result["B"], 0.0, abs_tol=1e-6)


def test_shapley_value_symmetric_case():
    # A-only, B-only, and A+B journeys all convert -> by symmetry A and B
    # should split the 3 conversions evenly at 1.5 each (hand-derived).
    journeys = {
        1: {"touches": [("A", 0)], "converted": True, "conversion_day": 1},
        2: {"touches": [("B", 0)], "converted": True, "conversion_day": 1},
        3: {"touches": [("A", 0), ("B", 1)], "converted": True, "conversion_day": 2},
    }
    result = shapley_value(journeys, ["A", "B"])
    assert math.isclose(result["A"], 1.5, abs_tol=1e-6)
    assert math.isclose(result["B"], 1.5, abs_tol=1e-6)
    assert math.isclose(sum(result.values()), 3.0, abs_tol=1e-6)


def test_shapley_value_channel_never_touched_gets_zero():
    journeys = {
        1: {"touches": [("A", 0)], "converted": True, "conversion_day": 1},
    }
    result = shapley_value(journeys, ["A", "B"])
    assert math.isclose(result["B"], 0.0, abs_tol=1e-6)

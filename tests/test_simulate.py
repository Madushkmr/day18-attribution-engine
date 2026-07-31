import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimizer import calibrate_curves, optimize_allocation
from src.simulate import run_scenarios


def test_scenario_summary_shape_and_percentile_ordering():
    spend = {"paid_search": 1000.0, "social": 500.0}
    conversions = {"paid_search": 60.0, "social": 20.0}
    curves = calibrate_curves(spend, conversions)
    total_budget = sum(spend.values())
    recommended, _ = optimize_allocation(curves, total_budget)

    summary = run_scenarios(curves, spend, recommended, n_sims=1000, seed=1)

    for key in ("current", "recommended", "improvement"):
        assert key in summary
        s = summary[key]
        assert s["p10"] <= s["median"] <= s["p90"]

    assert summary["n_sims"] == 1000
    assert 0.0 <= summary["improvement"]["p_improves"] <= 1.0


def test_scenario_is_deterministic_given_seed():
    spend = {"paid_search": 1000.0}
    conversions = {"paid_search": 60.0}
    curves = calibrate_curves(spend, conversions)
    a = run_scenarios(curves, spend, spend, n_sims=500, seed=7)
    b = run_scenarios(curves, spend, spend, n_sims=500, seed=7)
    assert a["current"]["mean"] == b["current"]["mean"]

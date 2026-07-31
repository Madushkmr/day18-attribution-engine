import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimizer import calibrate_curves, curve_value, optimize_allocation


def test_calibration_reproduces_current_point():
    curves = calibrate_curves({"paid_search": 1000.0}, {"paid_search": 50.0}, current_saturation=0.6)
    c = curves["paid_search"]
    assert math.isclose(curve_value(c, 1000.0), 50.0, rel_tol=1e-6)


def test_zero_spend_channel_excluded():
    curves = calibrate_curves({"organic": 0.0, "paid_search": 500.0}, {"organic": 10.0, "paid_search": 20.0})
    assert "organic" not in curves
    assert "paid_search" in curves


def test_optimizer_reallocation_not_worse_than_current():
    spend = {"paid_search": 1000.0, "social": 200.0, "display": 3000.0}
    conversions = {"paid_search": 80.0, "social": 40.0, "display": 15.0}
    curves = calibrate_curves(spend, conversions, current_saturation=0.6)
    total_budget = sum(spend.values())

    current_total = sum(curve_value(curves[ch], spend[ch]) for ch in curves)
    alloc, _ = optimize_allocation(curves, total_budget)
    optimized_total = sum(curve_value(curves[ch], alloc[ch]) for ch in curves)

    assert optimized_total >= current_total - 1e-6
    # budget conserved
    assert math.isclose(sum(alloc.values()), total_budget, rel_tol=1e-6)


def test_optimizer_shifts_budget_toward_higher_marginal_channel():
    # display has much lower conversions per dollar spent than paid_search
    spend = {"paid_search": 1000.0, "display": 1000.0}
    conversions = {"paid_search": 100.0, "display": 5.0}
    curves = calibrate_curves(spend, conversions)
    alloc, _ = optimize_allocation(curves, sum(spend.values()))
    assert alloc["paid_search"] > alloc["display"]

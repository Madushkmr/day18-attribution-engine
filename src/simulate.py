"""
Monte Carlo scenario simulation.

The optimizer's response curves are calibrated from a single assumed
saturation point, so their *shape* (the rate parameter k) is genuinely
uncertain. This module perturbs k with lognormal noise across many
simulated "alternate worlds" and reports the resulting distribution of
total conversions under the current allocation vs. the recommended
allocation -- so a decision-maker sees a plausible range and a win
probability, not a single point estimate presented as certain.
"""
import numpy as np

from .optimizer import curve_value


def run_scenarios(curves, current_alloc, recommended_alloc, n_sims=5000,
                   k_relative_sigma=0.25, seed=42):
    rng = np.random.default_rng(seed)
    channels = list(curves.keys())

    current_totals = np.zeros(n_sims)
    recommended_totals = np.zeros(n_sims)

    for ch in channels:
        c = curves[ch]
        k_draws = c["k"] * rng.lognormal(mean=0.0, sigma=k_relative_sigma, size=n_sims)
        cur_spend = current_alloc.get(ch, 0.0)
        rec_spend = recommended_alloc.get(ch, 0.0)
        current_totals += c["A"] * (1 - np.exp(-k_draws * cur_spend))
        recommended_totals += c["A"] * (1 - np.exp(-k_draws * rec_spend))

    improvement = recommended_totals - current_totals

    def pct(a, q):
        return float(np.percentile(a, q))

    return {
        "n_sims": n_sims,
        "current": {
            "mean": float(current_totals.mean()), "p10": pct(current_totals, 10),
            "median": pct(current_totals, 50), "p90": pct(current_totals, 90),
        },
        "recommended": {
            "mean": float(recommended_totals.mean()), "p10": pct(recommended_totals, 10),
            "median": pct(recommended_totals, 50), "p90": pct(recommended_totals, 90),
        },
        "improvement": {
            "mean": float(improvement.mean()), "p10": pct(improvement, 10),
            "median": pct(improvement, 50), "p90": pct(improvement, 90),
            "p_improves": float((improvement > 0).mean()),
        },
    }

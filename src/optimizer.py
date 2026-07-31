"""
Budget reallocation optimizer.

Each channel's response is modeled as a saturating (diminishing-returns)
curve:

    conversions(spend) = A * (1 - exp(-k * spend))

Both A (asymptotic ceiling) and k (saturation rate) are calibrated per
channel from a single observed point -- its current spend and its current
attributed conversions (from one of the attribution models) -- under the
simplifying assumption that the channel is currently operating at
`current_saturation` (default 60%) of its ceiling. This lets us calibrate
a full curve from one data point; it is a modeling *assumption*, not a fit,
and is documented as a limitation in the README.

Given calibrated curves for every channel with spend > 0, the optimizer
finds the budget split across those channels that maximizes total expected
conversions for a fixed total budget. Because each curve is concave, the
optimum equalizes marginal conversions-per-dollar across channels -- the
classic "water-filling" solution -- found here by binary search on the
shared marginal-value threshold (lambda).
"""
import math


def calibrate_curves(channel_spend, channel_conversions, current_saturation=0.6):
    curves = {}
    for ch, s0 in channel_spend.items():
        c0 = channel_conversions.get(ch, 0.0)
        if s0 <= 0 or c0 <= 0:
            continue  # can't calibrate a spend-response curve with no spend or no credited conversions
        k = -math.log(1 - current_saturation) / s0
        denom = 1 - math.exp(-k * s0)
        A = c0 / denom if denom > 0 else c0
        curves[ch] = {"A": A, "k": k, "current_spend": s0, "current_conversions": c0}
    return curves


def curve_value(curve, spend):
    return curve["A"] * (1 - math.exp(-curve["k"] * max(0.0, spend)))


def curve_marginal(curve, spend):
    return curve["A"] * curve["k"] * math.exp(-curve["k"] * max(0.0, spend))


def optimize_allocation(curves, total_budget, iters=60):
    """Water-filling: find lambda such that sum_i x_i(lambda) == total_budget,
    where x_i(lambda) = max(0, ln(A_i k_i / lambda) / k_i)."""
    if not curves:
        return {}, 0.0

    def alloc_for_lambda(lam):
        alloc = {}
        for ch, c in curves.items():
            peak = c["A"] * c["k"]
            if lam >= peak or peak <= 0:
                alloc[ch] = 0.0
            else:
                alloc[ch] = max(0.0, math.log(peak / lam) / c["k"])
        return alloc

    lo, hi = 1e-12, max(c["A"] * c["k"] for c in curves.values())
    for _ in range(iters):
        mid = (lo + hi) / 2
        total = sum(alloc_for_lambda(mid).values())
        if total > total_budget:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2
    alloc = alloc_for_lambda(lam)

    total_alloc = sum(alloc.values())
    if total_alloc > 0:
        scale = total_budget / total_alloc
        alloc = {ch: x * scale for ch, x in alloc.items()}
    return alloc, lam


def recommend(channel_spend, channel_conversions, current_saturation=0.6):
    """Full optimizer step: calibrate curves, reallocate the same total budget,
    and report predicted conversions before/after per channel."""
    curves = calibrate_curves(channel_spend, channel_conversions, current_saturation)
    total_budget = sum(c["current_spend"] for c in curves.values())
    recommended, lam = optimize_allocation(curves, total_budget)

    rows = []
    for ch, c in curves.items():
        rows.append({
            "channel": ch,
            "current_spend": c["current_spend"],
            "recommended_spend": recommended.get(ch, 0.0),
            "current_predicted_conversions": curve_value(c, c["current_spend"]),
            "recommended_predicted_conversions": curve_value(c, recommended.get(ch, 0.0)),
        })
    rows.sort(key=lambda r: r["channel"])
    return {
        "curves": curves,
        "recommended_spend": recommended,
        "lambda": lam,
        "total_budget": total_budget,
        "rows": rows,
    }

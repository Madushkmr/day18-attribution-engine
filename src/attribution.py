"""
Multi-touch attribution models.

Five models, from simplest heuristic to game-theoretically "fair":

  last_touch, linear, time_decay   -- classic weighting heuristics
  markov_removal_effect            -- absorbing Markov chain over the
                                       touchpoint graph; a channel's credit
                                       is how much the overall conversion
                                       probability drops if that channel is
                                       removed from the graph.
  shapley_value                    -- exact Shapley value over channels
                                       treated as coalition "players", using
                                       a value function defined on which
                                       journeys' full channel-set is covered
                                       by the coalition.

All models return {channel: attributed_conversions} summing to the total
number of converted journeys (Markov and Shapley may allocate a small
negative/positive rounding residual which is corrected by rescaling).
"""
from collections import defaultdict
import itertools
import numpy as np

START, CONVERSION, NULL = "__start__", "__conversion__", "__null__"


# --------------------------------------------------------------------------
# Heuristic weighting models
# --------------------------------------------------------------------------

def last_touch(journeys):
    credit = defaultdict(float)
    for j in journeys.values():
        if not j["converted"] or not j["touches"]:
            continue
        credit[j["touches"][-1][0]] += 1.0
    return dict(credit)


def linear(journeys):
    credit = defaultdict(float)
    for j in journeys.values():
        if not j["converted"] or not j["touches"]:
            continue
        w = 1.0 / len(j["touches"])
        for ch, _ in j["touches"]:
            credit[ch] += w
    return dict(credit)


def time_decay(journeys, half_life_days=5.0):
    credit = defaultdict(float)
    for j in journeys.values():
        if not j["converted"] or not j["touches"]:
            continue
        conv_day = j["conversion_day"] if j["conversion_day"] is not None else j["touches"][-1][1]
        weights = [2 ** (-(conv_day - day) / half_life_days) for _, day in j["touches"]]
        total = sum(weights) or 1.0
        for (ch, _), w in zip(j["touches"], weights):
            credit[ch] += w / total
    return dict(credit)


# --------------------------------------------------------------------------
# Markov chain removal-effect attribution
# --------------------------------------------------------------------------

def _build_transition_counts(journeys, active_channels):
    """Counts[src][dst] over Start -> touches (restricted to active_channels,
    with skipped/removed channels redirected to Null) -> Conversion/Null."""
    counts = defaultdict(lambda: defaultdict(float))
    for j in journeys.values():
        touches = [ch for ch, _ in j["touches"]]
        prev = START
        reached_null = False
        for ch in touches:
            node = ch if ch in active_channels else NULL
            counts[prev][node] += 1.0
            if node == NULL:
                reached_null = True
                break
            prev = node
        if not reached_null:
            end = CONVERSION if j["converted"] else NULL
            counts[prev][end] += 1.0
    return counts


def _absorption_prob_to_conversion(counts, active_channels):
    transient = [START] + list(active_channels)
    idx = {s: i for i, s in enumerate(transient)}
    n = len(transient)
    Q = np.zeros((n, n))
    R = np.zeros((n, 2))  # columns: Conversion, Null
    for src in transient:
        row = counts.get(src, {})
        total = sum(row.values())
        if total == 0:
            continue
        for dst, c in row.items():
            p = c / total
            if dst in idx:
                Q[idx[src], idx[dst]] += p
            elif dst == CONVERSION:
                R[idx[src], 0] += p
            elif dst == NULL:
                R[idx[src], 1] += p
    I = np.eye(n)
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        N = np.linalg.pinv(I - Q)
    B = N @ R
    return float(B[idx[START], 0])


def markov_removal_effect(journeys, channels):
    counts_full = _build_transition_counts(journeys, set(channels))
    p_full = _absorption_prob_to_conversion(counts_full, list(channels))
    total_conversions = sum(1 for j in journeys.values() if j["converted"])

    effects = {}
    for ch in channels:
        active = [c for c in channels if c != ch]
        counts_wo = _build_transition_counts(journeys, set(active))
        p_wo = _absorption_prob_to_conversion(counts_wo, active)
        effects[ch] = max(0.0, p_full - p_wo)

    total_effect = sum(effects.values())
    if total_effect <= 0:
        # fall back to even split if the graph is degenerate
        return {ch: total_conversions / len(channels) for ch in channels}
    return {ch: (eff / total_effect) * total_conversions for ch, eff in effects.items()}


# --------------------------------------------------------------------------
# Shapley value attribution
# --------------------------------------------------------------------------

def _journey_channel_masks(journeys, channels):
    """Bitmask per channel; returns dict mask -> converted_count for each
    journey's distinct touched-channel set."""
    ch_index = {c: i for i, c in enumerate(channels)}
    mask_counts = defaultdict(int)
    for j in journeys.values():
        if not j["converted"]:
            continue
        mask = 0
        for ch, _ in j["touches"]:
            mask |= 1 << ch_index[ch]
        mask_counts[mask] += 1
    return mask_counts


def _coverage_value(mask_counts, n):
    """v[S] = number of converted journeys whose channel-mask is a subset of S,
    computed for every S in 0..2^n-1 via a subset-sum (zeta) transform."""
    v = np.zeros(1 << n)
    for m, c in mask_counts.items():
        v[m] += c
    # standard subset-sum / zeta transform: v[S] should accumulate all subsets
    for i in range(n):
        bit = 1 << i
        for S in range(1 << n):
            if S & bit:
                v[S] += v[S ^ bit]
    return v


def shapley_value(journeys, channels):
    n = len(channels)
    mask_counts = _journey_channel_masks(journeys, channels)
    v = _coverage_value(mask_counts, n)

    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i

    phi = np.zeros(n)
    full_mask = (1 << n) - 1
    for i in range(n):
        bit = 1 << i
        others = [b for b in range(n) if b != i]
        for s in range(1 << (n - 1)):
            # build subset S of `others` from s's bits
            S = 0
            for k, b in enumerate(others):
                if s & (1 << k):
                    S |= (1 << b)
            size = bin(S).count("1")
            weight = fact[size] * fact[n - size - 1] / fact[n]
            phi[i] += weight * (v[S | bit] - v[S])

    total_conversions = sum(1 for j in journeys.values() if j["converted"])
    total_phi = phi.sum()
    if total_phi <= 0:
        return {ch: total_conversions / n for ch in channels}
    return {ch: float(phi[i] / total_phi * total_conversions) for i, ch in enumerate(channels)}


MODELS = {
    "last_touch": last_touch,
    "linear": linear,
    "time_decay": time_decay,
}

GRAPH_MODELS = {
    "markov_removal_effect": markov_removal_effect,
    "shapley_value": shapley_value,
}


def run_all(journeys, channels):
    results = {name: fn(journeys) for name, fn in MODELS.items()}
    for name, fn in GRAPH_MODELS.items():
        results[name] = fn(journeys, channels)
    # ensure every channel present in every model's output (0 if missing)
    for name, res in results.items():
        for ch in channels:
            res.setdefault(ch, 0.0)
    return results

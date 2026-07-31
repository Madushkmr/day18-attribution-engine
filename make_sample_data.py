"""
Regenerates sample_data/touchpoints.csv and sample_data/spend.csv.

Simulates ~2,500 customer journeys across 7 marketing channels over a
90-day window, plus daily spend per channel, using a fixed random seed
so the checked-in CSVs are reproducible.
"""
import csv
import random

SEED = 42
N_JOURNEYS = 2500
DAYS = 90
CHANNELS = [
    "paid_search", "organic_search", "social", "email",
    "display", "referral", "direct",
]

# Rough relative "pull" of each channel toward conversion, and typical
# position in the journey (early / mid / late), used only to generate
# plausible synthetic sequences -- not read by the attribution code.
CHANNEL_WEIGHT = {
    "paid_search": 1.3, "organic_search": 1.1, "social": 0.9,
    "email": 1.4, "display": 0.6, "referral": 1.0, "direct": 1.2,
}
DAILY_SPEND_BASE = {
    "paid_search": 900, "organic_search": 0, "social": 500,
    "email": 80, "display": 650, "referral": 40, "direct": 0,
}


def gen_journey(rng, journey_id):
    n_touches = rng.choice([1, 2, 3, 4, 5, 6], p=[0.18, 0.27, 0.24, 0.16, 0.10, 0.05])
    day0 = rng.integers(0, DAYS - 14)
    touches = []
    t = day0
    for i in range(n_touches):
        ch = rng.choice(CHANNELS, p=_channel_probs())
        t += int(rng.integers(0, 4))
        touches.append((ch, min(t, DAYS - 1)))
    # crude conversion propensity: sum of channel weights + bonus for more touches
    score = sum(CHANNEL_WEIGHT[c] for c, _ in touches) + 0.35 * n_touches
    p_convert = 1 / (1 + pow(2.71828, -(score - 4.2)))
    converted = rng.random() < p_convert
    conv_day = touches[-1][1] + int(rng.integers(0, 3)) if converted else None
    if conv_day is not None:
        conv_day = min(conv_day, DAYS - 1)
    return touches, converted, conv_day


def _channel_probs():
    import numpy as np
    w = np.array([CHANNEL_WEIGHT[c] for c in CHANNELS], dtype=float)
    return w / w.sum()


def main():
    import numpy as np
    rng = np.random.default_rng(SEED)

    tp_path = "sample_data/touchpoints.csv"
    with open(tp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["journey_id", "channel", "day", "converted", "conversion_day"])
        for jid in range(1, N_JOURNEYS + 1):
            touches, converted, conv_day = gen_journey(rng, jid)
            for ch, day in touches:
                writer.writerow([jid, ch, day, int(converted), conv_day if conv_day is not None else ""])

    sp_path = "sample_data/spend.csv"
    with open(sp_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "channel", "spend"])
        for day in range(DAYS):
            for ch in CHANNELS:
                base = DAILY_SPEND_BASE[ch]
                if base == 0:
                    continue
                noise = rng.normal(1.0, 0.12)
                writer.writerow([day, ch, round(max(0.0, base * noise), 2)])

    print(f"Wrote {tp_path} and {sp_path}")


if __name__ == "__main__":
    main()

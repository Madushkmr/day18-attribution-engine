"""Load touchpoints + spend CSVs into journey/channel structures."""
import csv
from collections import defaultdict


def load_journeys(path):
    """Returns dict journey_id -> {"touches": [(channel, day), ...] (time-ordered),
    "converted": bool, "conversion_day": int|None}."""
    raw = defaultdict(list)
    meta = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            jid = int(row["journey_id"])
            raw[jid].append((row["channel"], int(row["day"])))
            meta[jid] = {
                "converted": bool(int(row["converted"])),
                "conversion_day": int(row["conversion_day"]) if row["conversion_day"] else None,
            }
    journeys = {}
    for jid, touches in raw.items():
        touches_sorted = sorted(touches, key=lambda t: t[1])
        journeys[jid] = {
            "touches": touches_sorted,
            "converted": meta[jid]["converted"],
            "conversion_day": meta[jid]["conversion_day"],
        }
    return journeys


def load_spend(path):
    """Returns dict channel -> total spend, and channel -> list of (day, spend)."""
    totals = defaultdict(float)
    by_day = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ch, day, spend = row["channel"], int(row["day"]), float(row["spend"])
            totals[ch] += spend
            by_day[ch].append((day, spend))
    return dict(totals), dict(by_day)


def channel_set(journeys):
    chans = set()
    for j in journeys.values():
        for ch, _ in j["touches"]:
            chans.add(ch)
    return sorted(chans)

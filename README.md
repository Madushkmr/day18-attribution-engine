# Day 18 — Marketing Attribution & Budget Optimization Engine

Day 18 of a daily AI-app series (BI focus). Given a set of customer journeys
(sequences of marketing-channel touches ending in conversion or not) and
per-channel ad spend, this app answers the two questions marketing/BI teams
actually get asked: **"which channels are really driving conversions?"** and
**"if I could move budget around, where should it go?"** — with uncertainty
quantified rather than presented as a single confident number.

## Why this matters for BI work

Most attribution setups stop at last-touch or a simple linear split, which
either wildly over-credits whichever channel happens to close the deal or
ignores channel interactions entirely. This project implements five models
side by side — from naive heuristics up to two principled, interaction-aware
methods (a Markov-chain removal-effect model and an exact Shapley value) —
so a BI analyst can show a stakeholder *why* the numbers differ between
models, not just report one number. It then goes a step further than pure
measurement: a budget optimizer converts the attribution output into a
concrete spend reallocation recommendation, and a Monte Carlo layer stress
tests that recommendation against uncertainty in how each channel actually
responds to more (or less) spend.

## Complexity tier: decision-support optimization + game-theoretic attribution

This is a step up from Day 17's cohort/CLV modeling pipeline. Day 17 combined
several *estimation* techniques (empirical-Bayes shrinkage, Monte Carlo
forecasting) to answer "what will happen." Day 18 adds a genuine
**decision/optimization layer on top of estimation**: it doesn't just
attribute credit, it recommends an action (reallocate budget) and quantifies
the risk of that action under uncertainty. It also introduces attribution
techniques not used elsewhere in the series — an absorbing Markov chain over
the touchpoint graph, and an exact Shapley value computed via subset-sum
(zeta) transform — plus a closed-form "water-filling" solution to the
concave budget-allocation problem, verified against small hand-derived
examples in the test suite.

## Architecture

```
day18-attribution-engine/
├── app.py                  # Flask REST API + dashboard
├── cli.py                  # command-line interface
├── make_sample_data.py     # regenerates sample_data/*.csv (fixed seed)
├── config/
│   └── settings.yaml       # data paths, primary model, saturation assumption, sim count
├── src/
│   ├── ingest.py           # loads touchpoints + spend CSVs into journey/channel structures
│   ├── attribution.py      # last_touch, linear, time_decay, markov_removal_effect, shapley_value
│   ├── optimizer.py        # per-channel response-curve calibration + water-filling budget allocation
│   ├── simulate.py         # Monte Carlo scenario simulation over response-curve uncertainty
│   ├── narrative.py        # rule-based NLG summary (no external LLM API, runs offline)
│   ├── db.py               # SQLite schema: runs, channel_attribution, budget_recommendation, scenario_summary
│   └── engine.py           # orchestrates ingest -> attribute -> optimize -> simulate -> narrate -> persist
├── templates/
│   └── dashboard.html      # attribution-by-model chart, budget before/after chart, scenario table, run history
├── sample_data/
│   ├── touchpoints.csv     # 2,500 synthetic customer journeys across 7 channels, 90-day window
│   └── spend.csv           # daily spend per channel over the same window
├── tests/
│   ├── test_attribution.py # incl. hand-derived Markov & Shapley correctness checks
│   ├── test_optimizer.py
│   ├── test_simulate.py
│   └── test_engine.py      # end-to-end pipeline + SQLite round trip
├── requirements.txt
└── Dockerfile
```

## The models, briefly

**Attribution (`src/attribution.py`)** — five models over the same journeys:
- `last_touch` / `linear` / `time_decay` — classic heuristic weightings.
- `markov_removal_effect` — builds an absorbing Markov chain (Start → channels
  → Conversion/Null) from the observed transition frequencies across *all*
  journeys (converting and non-converting), then computes each channel's
  credit as the drop in overall conversion probability when that channel is
  removed from the graph (its outgoing/incoming edges redirected to Null).
- `shapley_value` — treats each channel as a coalition "player" with value
  function v(S) = number of converted journeys whose full touched-channel
  set is contained in S, computed exactly for all 2ⁿ coalitions via a
  subset-sum transform, then combined with the standard Shapley weighting
  formula. Exact and tractable here because there are only 7 channels.

**Budget optimizer (`src/optimizer.py`)** — each channel's spend-to-conversions
response is modeled as a saturating curve `A·(1 − e^(−k·spend))`. Both `A`
(ceiling) and `k` (saturation rate) are calibrated from a single observed
point — current spend and current attributed conversions — under the
assumption that each channel is currently at `current_saturation` (default
60%) of its ceiling. Given calibrated curves, the optimal reallocation of a
fixed total budget equalizes marginal conversions-per-dollar across channels
(the classic concave-resource "water-filling" solution), found here by binary
search on the shared marginal-value threshold.

**Scenario simulation (`src/simulate.py`)** — the calibration above is a
modeling assumption, so `k` is genuinely uncertain. This module draws
thousands of lognormal perturbations of `k` per channel and reports the
resulting distribution (mean, median, 10th/90th percentile) of total
conversions under the current vs. recommended allocation, plus the fraction
of simulated worlds in which the recommendation actually wins.

## Running it

```
cd day18-attribution-engine
pip install -r requirements.txt

# (optional) regenerate sample data — already checked in with a fixed seed
python make_sample_data.py

# CLI: run the full pipeline once
python cli.py compute
python cli.py list-runs
python cli.py show-run 1

# Dashboard + API
python app.py   # http://localhost:5000
```

### REST API
```
curl -X POST localhost:5000/api/run
curl localhost:5000/api/runs
curl localhost:5000/api/runs/1
```

### Tests
```
pytest tests/ -v
```
14 tests covering: last-touch/linear/time-decay math, a hand-derived Markov
removal-effect example (one essential channel, one useless one — the
essential channel should absorb ~all attributed conversions), a hand-derived
symmetric Shapley example (two equally-important channels split conversions
evenly, verified against the value computed by hand), response-curve
calibration correctness, the water-filling optimizer (reallocation is never
worse than the current allocation for the same total budget, and shifts
budget toward the higher-marginal-value channel), Monte Carlo scenario shape
and determinism under a fixed seed, and an end-to-end engine run with SQLite
persistence and multi-run history.

### Docker
```
docker build -t attribution-engine .
docker run -p 5000:5000 attribution-engine
```

## Sample data

`make_sample_data.py` simulates 2,500 customer journeys over a 90-day window
across 7 channels (paid search, organic search, social, email, display,
referral, direct), with per-channel "pull" weights driving both which
channels appear in a journey and a logistic conversion probability, plus
daily spend per paid channel with multiplicative noise — all with a fixed
random seed for reproducibility. Both CSVs are checked in so the app runs
immediately without regeneration.

## Notes / limitations

- The budget-response curve is calibrated from **one** assumed point
  (current spend/conversions at an assumed 60% saturation), not fit from
  historical spend-vs-conversion variation — a production version would fit
  the curve from many historical (spend, conversions) observations per
  channel instead of assuming a saturation level.
- The Markov chain treats each journey's sequence independently and ignores
  time between touches (only order matters); a richer model could use
  higher-order transitions or explicit time decay within the chain.
- The Shapley value function is defined on touched-channel *sets*, not
  ordered sequences, so it (like the Markov model) captures channel
  interaction effects but not ordering effects the way `time_decay` does.
- This is a demo/portfolio project over synthetic data with a fixed seed,
  not a production attribution platform — a real version would need
  incremental data ingestion, authentication on the API, and validation of
  the response-curve assumption against realized results after actually
  reallocating budget.

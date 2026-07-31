"""Orchestrates the full pipeline: ingest -> attribute -> optimize -> simulate -> narrate -> persist."""
from . import attribution, db, ingest, narrative, optimizer, simulate


def run_pipeline(touchpoints_path, spend_path, db_path, primary_model="markov_removal_effect",
                  current_saturation=0.6, n_sims=5000, budget_multiplier=1.0):
    journeys = ingest.load_journeys(touchpoints_path)
    channels = ingest.channel_set(journeys)
    spend_totals, _ = ingest.load_spend(spend_path)

    attribution_results = attribution.run_all(journeys, channels)
    primary_credits = attribution_results[primary_model]

    rec = optimizer.recommend(spend_totals, primary_credits, current_saturation=current_saturation)
    if budget_multiplier != 1.0 and rec["curves"]:
        new_budget = rec["total_budget"] * budget_multiplier
        new_alloc, lam = optimizer.optimize_allocation(rec["curves"], new_budget)
        rows = []
        for ch, c in rec["curves"].items():
            rows.append({
                "channel": ch,
                "current_spend": c["current_spend"],
                "recommended_spend": new_alloc.get(ch, 0.0),
                "current_predicted_conversions": optimizer.curve_value(c, c["current_spend"]),
                "recommended_predicted_conversions": optimizer.curve_value(c, new_alloc.get(ch, 0.0)),
            })
        rows.sort(key=lambda r: r["channel"])
        rec["rows"] = rows
        rec["recommended_spend"] = new_alloc
        rec["total_budget"] = new_budget

    current_alloc = {ch: c["current_spend"] for ch, c in rec["curves"].items()}
    scenario = None
    if rec["curves"]:
        scenario = simulate.run_scenarios(
            rec["curves"], current_alloc, rec["recommended_spend"], n_sims=n_sims
        )

    text = narrative.build_narrative(attribution_results, rec["rows"], scenario, primary_model=primary_model)

    n_conversions = sum(1 for j in journeys.values() if j["converted"])
    conn = db.connect(db_path)
    run_id = db.save_run(
        conn, len(journeys), n_conversions, primary_model,
        attribution_results, rec["rows"], scenario or {},
    )
    conn.close()

    return {
        "run_id": run_id,
        "n_journeys": len(journeys),
        "n_conversions": n_conversions,
        "channels": channels,
        "attribution": attribution_results,
        "budget_recommendation": rec["rows"],
        "total_budget": rec["total_budget"],
        "scenario": scenario,
        "narrative": text,
    }

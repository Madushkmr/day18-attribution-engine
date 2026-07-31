"""Rule-based (no external LLM) narrative generation over the run results."""


def build_narrative(attribution_results, recommendation_rows, scenario, primary_model="markov_removal_effect"):
    lines = []

    primary = attribution_results.get(primary_model, {})
    if primary:
        ranked = sorted(primary.items(), key=lambda kv: kv[1], reverse=True)
        top_ch, top_credit = ranked[0]
        total = sum(primary.values()) or 1.0
        lines.append(
            f"Under the {primary_model.replace('_', ' ')} model, {top_ch} is the single largest "
            f"contributor to conversions, credited with {top_credit:.1f} of {total:.1f} "
            f"total attributed conversions ({100 * top_credit / total:.0f}%)."
        )

    lt = attribution_results.get("last_touch", {})
    if lt and primary:
        lt_top = max(lt.items(), key=lambda kv: kv[1])[0] if lt else None
        if lt_top and lt_top != top_ch:
            lines.append(
                f"Note that last-touch attribution alone would over-credit {lt_top}, "
                f"which drops in relative rank once earlier, assist-heavy touches are "
                f"accounted for by {primary_model.replace('_', ' ')}."
            )

    if recommendation_rows:
        biggest_increase = max(recommendation_rows, key=lambda r: r["recommended_spend"] - r["current_spend"])
        biggest_decrease = min(recommendation_rows, key=lambda r: r["recommended_spend"] - r["current_spend"])
        if biggest_increase["recommended_spend"] > biggest_increase["current_spend"]:
            lines.append(
                f"The budget optimizer recommends increasing {biggest_increase['channel']} spend from "
                f"${biggest_increase['current_spend']:.0f} to ${biggest_increase['recommended_spend']:.0f}, "
                f"where marginal conversions per dollar are currently highest."
            )
        if biggest_decrease["recommended_spend"] < biggest_decrease["current_spend"]:
            lines.append(
                f"Conversely, {biggest_decrease['channel']} spend looks over-saturated and is "
                f"recommended to drop from ${biggest_decrease['current_spend']:.0f} to "
                f"${biggest_decrease['recommended_spend']:.0f}."
            )

    if scenario:
        imp = scenario["improvement"]
        lines.append(
            f"Simulating {scenario['n_sims']:,} alternate scenarios with uncertain channel "
            f"response rates, the reallocation improves total expected conversions by "
            f"{imp['mean']:.1f} on average (10th-90th percentile: {imp['p10']:.1f} to {imp['p90']:.1f}), "
            f"and comes out ahead of the current allocation in {100 * imp['p_improves']:.0f}% of scenarios."
        )

    return " ".join(lines) if lines else "No conversions were available to attribute in this run."

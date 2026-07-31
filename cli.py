"""Command-line interface for the attribution engine."""
import argparse
import json

import yaml

from src import db, engine


def load_config(path="config/settings.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def cmd_compute(args):
    cfg = load_config(args.config)
    result = engine.run_pipeline(
        cfg["data"]["touchpoints_path"], cfg["data"]["spend_path"], cfg["data"]["db_path"],
        primary_model=cfg["model"]["primary_attribution_model"],
        current_saturation=cfg["model"]["current_saturation"],
        n_sims=cfg["model"]["monte_carlo_sims"],
        budget_multiplier=cfg["model"].get("budget_multiplier", 1.0),
    )
    print(f"Run #{result['run_id']} — {result['n_journeys']} journeys, {result['n_conversions']} conversions\n")
    print(result["narrative"])
    print("\nBudget recommendation:")
    for row in result["budget_recommendation"]:
        print(f"  {row['channel']:<16} ${row['current_spend']:>9,.0f} -> ${row['recommended_spend']:>9,.0f}"
              f"   (predicted conv {row['current_predicted_conversions']:.1f} -> {row['recommended_predicted_conversions']:.1f})")


def cmd_list_runs(args):
    cfg = load_config(args.config)
    conn = db.connect(cfg["data"]["db_path"])
    for r in db.list_runs(conn):
        print(f"#{r['id']}  {r['created_at']}  journeys={r['n_journeys']} conversions={r['n_conversions']} model={r['primary_model']}")
    conn.close()


def cmd_show_run(args):
    cfg = load_config(args.config)
    conn = db.connect(cfg["data"]["db_path"])
    run = db.get_run(conn, args.run_id)
    conn.close()
    if run is None:
        print(f"No run #{args.run_id}")
        return
    print(json.dumps(run, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Marketing attribution & budget optimization CLI")
    parser.add_argument("--config", default="config/settings.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("compute", help="run the full pipeline once").set_defaults(func=cmd_compute)
    sub.add_parser("list-runs", help="list past runs").set_defaults(func=cmd_list_runs)

    p_show = sub.add_parser("show-run", help="show full detail for one run")
    p_show.add_argument("run_id", type=int)
    p_show.set_defaults(func=cmd_show_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

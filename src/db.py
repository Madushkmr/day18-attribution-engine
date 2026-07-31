"""SQLite persistence for attribution engine runs."""
import json
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    n_journeys INTEGER,
    n_conversions INTEGER,
    primary_model TEXT
);

CREATE TABLE IF NOT EXISTS channel_attribution (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    model TEXT NOT NULL,
    channel TEXT NOT NULL,
    attributed_conversions REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS budget_recommendation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    channel TEXT NOT NULL,
    current_spend REAL NOT NULL,
    recommended_spend REAL NOT NULL,
    current_predicted_conversions REAL NOT NULL,
    recommended_predicted_conversions REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS scenario_summary (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    payload_json TEXT NOT NULL
);
"""


def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_run(conn, n_journeys, n_conversions, primary_model, attribution_results,
             recommendation_rows, scenario_summary):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs (created_at, n_journeys, n_conversions, primary_model) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), n_journeys, n_conversions, primary_model),
    )
    run_id = cur.lastrowid

    for model, channel_credits in attribution_results.items():
        for channel, credit in channel_credits.items():
            cur.execute(
                "INSERT INTO channel_attribution (run_id, model, channel, attributed_conversions) "
                "VALUES (?, ?, ?, ?)",
                (run_id, model, channel, credit),
            )

    for row in recommendation_rows:
        cur.execute(
            "INSERT INTO budget_recommendation "
            "(run_id, channel, current_spend, recommended_spend, "
            "current_predicted_conversions, recommended_predicted_conversions) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, row["channel"], row["current_spend"], row["recommended_spend"],
             row["current_predicted_conversions"], row["recommended_predicted_conversions"]),
        )

    cur.execute(
        "INSERT INTO scenario_summary (run_id, payload_json) VALUES (?, ?)",
        (run_id, json.dumps(scenario_summary)),
    )
    conn.commit()
    return run_id


def list_runs(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()]


def get_run(conn, run_id):
    run = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        return None
    run = dict(run)
    run["attribution"] = {}
    for row in conn.execute(
        "SELECT model, channel, attributed_conversions FROM channel_attribution WHERE run_id = ?",
        (run_id,),
    ):
        run["attribution"].setdefault(row["model"], {})[row["channel"]] = row["attributed_conversions"]
    run["budget_recommendation"] = [
        dict(r) for r in conn.execute(
            "SELECT channel, current_spend, recommended_spend, current_predicted_conversions, "
            "recommended_predicted_conversions FROM budget_recommendation WHERE run_id = ?",
            (run_id,),
        )
    ]
    scenario = conn.execute(
        "SELECT payload_json FROM scenario_summary WHERE run_id = ?", (run_id,)
    ).fetchone()
    run["scenario"] = json.loads(scenario["payload_json"]) if scenario else None
    return run

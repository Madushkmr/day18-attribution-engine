"""Flask REST API + dashboard for the attribution engine."""
from flask import Flask, jsonify, render_template, request

from cli import load_config
from src import db, engine

app = Flask(__name__)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    cfg = load_config()
    body = request.get_json(silent=True) or {}
    result = engine.run_pipeline(
        cfg["data"]["touchpoints_path"], cfg["data"]["spend_path"], cfg["data"]["db_path"],
        primary_model=body.get("primary_model", cfg["model"]["primary_attribution_model"]),
        current_saturation=body.get("current_saturation", cfg["model"]["current_saturation"]),
        n_sims=body.get("n_sims", cfg["model"]["monte_carlo_sims"]),
        budget_multiplier=body.get("budget_multiplier", cfg["model"].get("budget_multiplier", 1.0)),
    )
    return jsonify(result)


@app.route("/api/runs")
def api_runs():
    cfg = load_config()
    conn = db.connect(cfg["data"]["db_path"])
    runs = db.list_runs(conn)
    conn.close()
    return jsonify(runs)


@app.route("/api/runs/<int:run_id>")
def api_run_detail(run_id):
    cfg = load_config()
    conn = db.connect(cfg["data"]["db_path"])
    run = db.get_run(conn, run_id)
    conn.close()
    if run is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(run)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import db, engine

BASE = os.path.join(os.path.dirname(__file__), "..")
TOUCHPOINTS = os.path.join(BASE, "sample_data", "touchpoints.csv")
SPEND = os.path.join(BASE, "sample_data", "spend.csv")


def test_end_to_end_pipeline_and_persistence():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        result = engine.run_pipeline(TOUCHPOINTS, SPEND, db_path, n_sims=200)

        assert result["n_journeys"] > 0
        assert result["n_conversions"] > 0
        expected_models = {"last_touch", "linear", "time_decay", "markov_removal_effect", "shapley_value"}
        assert expected_models.issubset(result["attribution"].keys())

        for model, credits in result["attribution"].items():
            assert set(credits.keys()) == set(result["channels"])

        assert len(result["budget_recommendation"]) > 0
        assert result["scenario"] is not None
        assert result["narrative"]

        conn = db.connect(db_path)
        stored = db.get_run(conn, result["run_id"])
        conn.close()
        assert stored is not None
        assert stored["n_journeys"] == result["n_journeys"]
        assert "markov_removal_effect" in stored["attribution"]


def test_second_run_appends_history():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        r1 = engine.run_pipeline(TOUCHPOINTS, SPEND, db_path, n_sims=100)
        r2 = engine.run_pipeline(TOUCHPOINTS, SPEND, db_path, n_sims=100)
        assert r2["run_id"] == r1["run_id"] + 1

        conn = db.connect(db_path)
        runs = db.list_runs(conn)
        conn.close()
        assert len(runs) == 2

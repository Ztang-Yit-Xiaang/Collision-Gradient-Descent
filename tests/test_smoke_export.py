import csv
import json

from collision_optim import smoke


def test_smoke_exports_both_methods_and_actual_costs(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "_plots", lambda *args, **kwargs: None)
    report = smoke.run_experiments(tmp_path, seeds=1, steps=20)
    assert (tmp_path / "config.json").is_file()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["checks"] == report["checks"]
    methods = {entry["method"] for entry in saved["aggregate"]}
    assert {"active_reflection", "hybrid_contact", "momentum_sgd", "momentum_restart"} <= methods
    for trace in (tmp_path / "traces").glob("*.csv"):
        rows = list(csv.DictReader(trace.open()))
        assert rows[0]["step"] == "0"
        assert float(rows[-1]["gradient_calls"]) > 0
        assert float(rows[-1]["value_calls"]) >= 2
    assert report["geometry"]["passed"]


def test_smoke_initial_conditions_are_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "_plots", lambda *args, **kwargs: None)
    first = smoke.run_experiments(tmp_path / "first", seeds=1, steps=10)
    second = smoke.run_experiments(tmp_path / "second", seeds=1, steps=10)
    for a, b in zip(first["aggregate"], second["aggregate"]):
        assert a["median_best_value"] == b["median_best_value"]
        assert a["median_value_calls"] == b["median_value_calls"]
        assert a["median_gradient_calls"] == b["median_gradient_calls"]


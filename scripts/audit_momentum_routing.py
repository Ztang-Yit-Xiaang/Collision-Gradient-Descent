"""Audit saved raw routing artifacts; never select or retrain a model."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def audit(output, suite):
    output = Path(output); config = load(output/"config.json"); runs = load(output/"runs.json")
    expected = len(config["seeds"])*len(config["methods"])*(len(config["cases"]) if suite == "synthetic" else 1)
    assert len(runs) == expected
    for name, digest in config["source_sha256"].items():
        assert hashlib.sha256((ROOT/"src/collision_optim"/name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256(Path(config["selection_source"]).read_bytes()).hexdigest() == config["selection_sha256"]
    lookup = {(r.get("case"), r["seed"], r["method"]): r for r in runs}
    no_action_matches = inactive_pulse_damping_matches = ledger_checks = action_checks = 0
    max_raw_energy_error = 0.
    for r in runs:
        name = (f"{r['case']}__s{r['seed']}__{r['method']}" if suite == "synthetic" else f"s{r['seed']}__{r['method']}")
        with gzip.open(output/"trajectories"/(name+".csv.gz"), "rt") as stream:
            trace = list(csv.DictReader(stream))
        actions = load(output/"actions"/(name+".json"))
        pulses = [a for a in actions if a["kind"] == "pulse"]
        dampings = [a for a in actions if a["kind"] in {"damp", "restart"}]
        assert len(pulses) == r["shots"] <= config["routing"]["max_shots"]
        assert len(dampings) == r["dampings"] <= config["routing"]["max_dampings"]
        assert np.isclose(sum(a["delta_kinetic"] for a in pulses), r["supplied"], atol=1e-9)
        assert np.isclose(-sum(a["delta_kinetic"] for a in dampings), r["dissipated"], atol=1e-9)
        assert r["supplied"] <= r["supply_cap"]+1e-8
        for a in actions:
            assert a["delta_kinetic"] > 0 if a["kind"] == "pulse" else a["delta_kinetic"] <= 0
            if a["kind"] == "pulse":
                assert not any(a["step"] <= d["step"] < a["step"]+2*config["routing"]["window"] for d in dampings)
            action_checks += 1
        base = lookup.get((r.get("case"), r["seed"], "momentum"))
        if suite == "synthetic":
            assert not r["failure"]
            assert len(trace) == r["steps"]+1 == 750
            assert r["total_oracle_calls"] == config["budget"] == 1500
            assert r["gradient_calls"] == 750 and r["value_calls"] == 750
            assert r["path_calls"] == r["planning_calls"] == 0
            assert np.array_equal(r["initial_x"], base["initial_x"])
            assert float(trace[-1]["value"]) == r["final_f"]
            assert min(float(t["value"]) for t in trace) == r["best_f"]
            if r["method"].startswith("momentum"):
                assert r["max_energy_balance_error"] < 1e-10
                assert r["max_momentum_balance_error"] < 1e-10
                for t in trace:
                    h = float(t["kinetic"])+r["lr"]*(float(t["value"])-r["initial_f"])
                    accounted = float(t["supplied"])-float(t["dissipated"])+float(t["natural_damping"])+float(t["integration_residual"])
                    error = abs(h-accounted); max_raw_energy_error = max(max_raw_energy_error, error)
                    assert error < 1e-10
                    ledger_checks += 1
        else:
            assert r["status"] == "steps" and r["final_loss"] is not None
            assert r["actual_steps"] == len(trace) == config["steps"] == 400
            assert r["real_train_units"] == r["used_units"] == r["budget_units"] == 1200
            assert r["virtual_train_units"] == r["real_probe_units"] == r["virtual_probe_units"] == 0
            assert r["eval_split"] == "test" and r["eval_batches"] == 32
            assert r["initial_hash"] == base["initial_hash"]
            assert (output/"checkpoints"/(name+".pt")).is_file()
            if not actions and r["method"].startswith("momentum_"):
                assert r["final_hash"] == base["final_hash"] and r["final_loss"] == base["final_loss"]
                no_action_matches += 1
            if r["method"] == "momentum_adaptive" and r["shots"] == 0:
                damp = lookup[None, r["seed"], "momentum_damp"]
                assert r["final_hash"] == damp["final_hash"] and r["final_loss"] == damp["final_loss"]
                inactive_pulse_damping_matches += 1
            assert r["max_actuator_error"] < 1e-6
            assert r["max_recurrence_error"] < 1e-4
    if suite == "nanogpt":
        assert hashlib.sha256((ROOT/"data/tinyshakespeare.txt").read_bytes()).hexdigest() == config["data_sha256"]
        assert hashlib.sha256((ROOT/"third_party/nanogpt/model.py").read_bytes()).hexdigest() == config["vendor_sha256"]
    result = {"passed": True, "suite": suite, "runs_checked": len(runs), "actions_checked": action_checks,
              "raw_energy_rows_checked": ledger_checks, "max_raw_energy_error": max_raw_energy_error,
              "no_action_weight_hash_matches": no_action_matches, "source_hashes_checked": len(config["source_sha256"]),
              "inactive_pulse_vs_damping_weight_matches": inactive_pulse_damping_matches,
              "extra_diagnostic_model_queries": 0 if suite == "nanogpt" else None}
    (output/"audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("synthetic", "nanogpt")); parser.add_argument("output", type=Path)
    args = parser.parse_args(); print(json.dumps(audit(args.output, args.suite), indent=2))

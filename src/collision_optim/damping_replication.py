"""Frozen NanoGPT damping replication with a cheap state-independent control."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .momentum_routing import RoutingConfig
from .nanogpt_benchmark import (LMConfig, TextData, TrainingCost, UnitLimit, VENDOR,
                               make_model, make_optimizer, model_hash, probe_loss)
from .nanogpt_routing import (RoutedTraining, scale_torch_momentum, tensor_norm,
                             train_routed)
from .routing_experiment import ROOT, prepare, source_hashes
from .strategic_experiment import _csv, _json


METHODS = ("momentum", "momentum_restart", "momentum_damp", "momentum_periodic", "adamw")
SCHEDULE = (80, 160, 240, 320)
ROUTING = RoutingConfig(window=20, epsilon_gain=1e-3, noise_aware=True, radii=(.001, .003))


class PeriodicTraining(RoutedTraining):
    """Same actuation as damping, at fixed times and without state diagnostics."""

    def __init__(self, *args, schedule=SCHEDULE, factor=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if self.router.mode != "none" or not isinstance(self.optimizer, torch.optim.SGD):
            raise ValueError("Periodic control requires plain native Momentum SGD.")
        if not 0 <= factor < 1 or len(set(schedule)) != len(schedule) or any(t < 0 for t in schedule):
            raise ValueError("Invalid periodic schedule or factor.")
        self.schedule, self.factor = tuple(schedule), factor

    def step(self, batch, cost, step):
        # Do not mutate state if an update cannot be paid. No scope cap is used
        # by this real-training experiment, but honor it for API consistency.
        if cost.used+3 > cost.limit or (cost.scope_limit is not None and cost.used-cost.scope_start+3 > cost.scope_limit):
            raise UnitLimit("Insufficient budget for the next real update.")
        if step in self.schedule:
            stamp = perf_counter()
            before = [b+q for b, q in zip(self.control.base, self.control.external)]
            norm = tensor_norm(before)
            scale_torch_momentum(self.optimizer, self.control, self.factor)
            after = [b+q for b, q in zip(self.control.base, self.control.external)]
            delta = .5*(tensor_norm(after)**2-norm**2)
            expected = .5*(self.factor**2-1)*norm**2
            self.max_actuator_error = max(self.max_actuator_error, abs(delta-expected))
            self.dissipated -= delta; self.router.dampings += 1
            self.actions.append({"step": step, "kind": "periodic_damp", "factor": self.factor,
                                 "delta_kinetic": delta, "expected_delta_kinetic": expected,
                                 "impulse_norm": tensor_norm([a-b for a, b in zip(after, before)])})
            self.action_seconds += perf_counter()-stamp
        return super().step(batch, cost, step)


def train_periodic(data, config, lr, decay, seed, *, steps=400, eval_batches=32, checkpoint=None):
    model = make_model(data, config, seed); optimizer = make_optimizer(model, "momentum", lr, decay, config)
    training = PeriodicTraining(model, optimizer, config, "none", ROUTING, seed)
    initial_hash = model_hash(model); cost = TrainingCost(3*steps); rows = []; status = "steps"
    started = perf_counter()
    for step in range(steps):
        try:
            loss, movement, grad_norm = training.step(data.batch("train", seed, step), cost, step)
            rows.append({"step": step+1, "train_loss": loss, "relative_movement": movement,
                         "gradient_norm_before_clip": grad_norm, "dampings": training.router.dampings,
                         "shots": 0, "supplied": 0., "dissipated": training.dissipated,
                         "units": cost.used, **cost.counts})
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            status = f"numerical_error: {exc}"; break
    seconds = perf_counter()-started
    final_loss = probe_loss(model, [data.batch("test", 81001, j) for j in range(eval_batches)]) if status == "steps" else None
    if checkpoint is not None:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "external": training.control.external, "base": training.control.base,
                    "seed": seed, "lm_config": asdict(config), "schedule": SCHEDULE,
                    "kind": "momentum", "mode": "periodic", "factor": .5,
                    "lr": lr, "weight_decay": decay}, checkpoint)
    summary = {"kind": "momentum", "mode": "periodic", "seed": seed, "lr": lr, "weight_decay": decay,
               "status": status, "initial_hash": initial_hash, "final_hash": model_hash(model),
               "parameter_count": sum(p.numel() for p in model.parameters()), "actual_steps": len(rows),
               "used_units": cost.used, "budget_units": cost.limit, "eval_batches": eval_batches,
               "eval_split": "test", "final_loss": final_loss,
               "perplexity": None if final_loss is None else float(np.exp(final_loss)),
               "shots": 0, "dampings": training.router.dampings, "supplied": 0.,
               "dissipated": training.dissipated, "feature_seconds": 0.,
               "action_seconds": training.action_seconds, "training_seconds": seconds,
               "max_actuator_error": training.max_actuator_error,
               "max_recurrence_error": None, "actual_train_tokens": len(rows)*config.batch*config.block,
               **cost.counts}
    return summary, rows, training.actions, []


def paired_summary(runs, method, reference, *, draws=20000):
    """Resample complete seed pairs; negative differences favor method."""
    left = {r["seed"]: r for r in runs if r["method"] == method}
    right = {r["seed"]: r for r in runs if r["method"] == reference}
    if set(left) != set(right):
        raise ValueError("Seed sets must match for paired comparisons.")
    valid = sorted(s for s in left if left[s]["final_loss"] is not None and right[s]["final_loss"] is not None)
    deltas = np.array([left[s]["final_loss"]-right[s]["final_loss"] for s in valid])
    if len(deltas) == 0:
        return {"method": method, "reference": reference, "valid_pairs": 0, "excluded_pairs": len(left)}
    rng = np.random.default_rng(5239400)
    boot = np.mean(deltas[rng.integers(0, len(deltas), size=(draws, len(deltas)))], axis=1)
    low, high = np.quantile(boot, [.025, .975])
    return {"method": method, "reference": reference, "valid_pairs": len(valid), "excluded_pairs": len(left)-len(valid),
            "mean_difference": float(np.mean(deltas)), "median_difference": float(np.median(deltas)),
            "bootstrap_95_low": float(low), "bootstrap_95_high": float(high), "bootstrap_draws": draws,
            "better": int(np.sum(deltas < -1e-8)), "worse": int(np.sum(deltas > 1e-8)),
            "ties": int(np.sum(np.abs(deltas) <= 1e-8)),
            "seed_differences": [{"seed": s, "difference": float(d)} for s, d in zip(valid, deltas)]}


def aggregate(output):
    output = Path(output); runs = json.loads((output/"runs.json").read_text()); rows = []
    for method in METHODS:
        group = [r for r in runs if r["method"] == method]; valid = [r for r in group if r["final_loss"] is not None]
        rows.append({"method": method, "runs": len(group), "failures": len(group)-len(valid),
            "mean_test_loss": float(np.mean([r["final_loss"] for r in valid])) if valid else None,
            "std_test_loss": float(np.std([r["final_loss"] for r in valid], ddof=1)) if len(valid) > 1 else None,
            "mean_training_seconds": float(np.mean([r["training_seconds"] for r in group])),
            "mean_feature_seconds": float(np.mean([r["feature_seconds"] for r in group])),
            "actions": sum(r["dampings"] for r in group),
            "min_actions": min(r["dampings"] for r in group), "max_actions": max(r["dampings"] for r in group)})
    comparisons = [paired_summary(runs, method, reference) for method, reference in
                   (("momentum_damp", "momentum"), ("momentum_damp", "momentum_periodic"),
                    ("momentum_restart", "momentum"), ("momentum_periodic", "momentum"), ("adamw", "momentum"))]
    _json(output/"aggregate.json", rows); _csv(output/"aggregate.csv", rows)
    _json(output/"paired.json", comparisons); _csv(output/"paired.csv", comparisons)
    return rows, comparisons


def run(output, *, seeds=range(400, 410), steps=400):
    output = prepare(output); lm = LMConfig(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    data = TextData(ROOT/"data/tinyshakespeare.txt", lm)
    selection_path = ROOT/"output/nanogpt_comparison/selected.json"; selected = json.loads(selection_path.read_text())
    orders = {s: list(np.random.default_rng(614000+s).permutation(METHODS)) for s in seeds}
    _json(output/"config.json", {"seeds": list(seeds), "steps": steps, "methods": METHODS,
          "method_order": orders, "model": asdict(lm), "routing": asdict(ROUTING), "schedule": SCHEDULE,
          "periodic_factor": .5, "source_sha256": source_hashes(), "selected": selected,
          "selection_source": str(selection_path), "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
          "data_sha256": data.sha256, "vendor_sha256": hashlib.sha256(VENDOR.read_bytes()).hexdigest(),
          "torch": torch.__version__, "device": "cpu", "dtype": "float32", "new_tuning_runs": 0,
          "evaluation": "Final test: same fixed 32 batches; no test-based routing or checkpoint selection",
          "cost": "400 actual updates; 1200 training units; zero diagnostic or virtual model forwards",
          "primary_pairs": [["momentum_damp", "momentum"], ["momentum_damp", "momentum_periodic"]],
          "bootstrap": {"draws": 20000, "seed": 5239400, "unit": "initialization/training seed pair"}})
    runs = []
    for seed in seeds:
        for method in orders[seed]:
            kind = "adamw" if method == "adamw" else "momentum"
            setting = selected[kind]; name = f"s{seed}__{method}"; checkpoint = output/"checkpoints"/(name+".pt")
            if method == "momentum_periodic":
                result = train_periodic(data, lm, setting["lr"], setting["weight_decay"], seed, steps=steps, checkpoint=checkpoint)
            else:
                mode = method.split("_")[1] if "_" in method else "none"
                result = train_routed(data, lm, kind, setting["lr"], setting["weight_decay"], seed,
                    mode=mode, routing=ROUTING, steps=steps, checkpoint=checkpoint)
            row, trace, actions, feedback = result; row["method"] = str(method); runs.append(row)
            _csv(output/"trajectories"/(name+".csv.gz"), trace, True)
            _json(output/"actions"/(name+".json"), actions); _json(output/"feedback"/(name+".json"), feedback)
            _json(output/"runs.json", runs)
            print(f"Damping replication {len(runs)}/{len(seeds)*len(METHODS)} {name}: loss={row['final_loss']}, actions={row['dampings']}", flush=True)
    _csv(output/"runs.csv", runs)
    return aggregate(output)


def audit(output):
    output = Path(output); config = json.loads((output/"config.json").read_text()); runs = json.loads((output/"runs.json").read_text())
    assert len(runs) == len(config["seeds"])*len(METHODS)
    lookup = {(r["seed"], r["method"]): r for r in runs}; events = 0
    for name, digest in config["source_sha256"].items():
        assert hashlib.sha256((ROOT/"src/collision_optim"/name).read_bytes()).hexdigest() == digest, name
    for r in runs:
        assert r["initial_hash"] == lookup[r["seed"], "momentum"]["initial_hash"]
        assert r["shots"] == r["supplied"] == 0
        assert r["actual_steps"] == config["steps"] == 400 and r["status"] == "steps"
        assert r["used_units"] == r["budget_units"] == r["real_train_units"] == 1200
        assert r["real_probe_units"] == r["virtual_probe_units"] == r["virtual_train_units"] == 0
        assert r["eval_split"] == "test" and r["eval_batches"] == 32
        name = f"s{r['seed']}__{r['method']}"
        with gzip.open(output/"trajectories"/(name+".csv.gz"), "rt") as stream: trace = list(csv.DictReader(stream))
        actions = json.loads((output/"actions"/(name+".json")).read_text())
        assert len(trace) == 400 and len(actions) == r["dampings"]
        assert np.isclose(-sum(a["delta_kinetic"] for a in actions), r["dissipated"], atol=1e-10)
        assert all(a["delta_kinetic"] <= 0 for a in actions)
        assert r["max_actuator_error"] < 1e-6
        if r["method"] == "momentum_periodic":
            assert [a["step"] for a in actions] == list(SCHEDULE)
            assert all(a["factor"] == .5 for a in actions) and r["feature_seconds"] == 0.
        elif r["method"] in {"momentum", "adamw"}: assert not actions
        assert (output/"checkpoints"/(name+".pt")).is_file(); events += len(actions)
    assert hashlib.sha256((ROOT/"data/tinyshakespeare.txt").read_bytes()).hexdigest() == config["data_sha256"]
    assert hashlib.sha256(VENDOR.read_bytes()).hexdigest() == config["vendor_sha256"]
    assert hashlib.sha256(Path(config["selection_source"]).read_bytes()).hexdigest() == config["selection_sha256"]
    saved = json.loads((output/"paired.json").read_text())
    for pair in saved: assert pair == paired_summary(runs, pair["method"], pair["reference"])
    result = {"passed": True, "runs_checked": len(runs), "actions_checked": events,
              "initialization_pairs": len(config["seeds"]), "extra_model_queries": 0,
              "source_hashes_checked": len(config["source_sha256"]), "paired_summaries_recomputed": len(saved)}
    _json(output/"audit.json", result); return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true"); args = parser.parse_args(argv)
    if args.audit: print(json.dumps(audit(args.output), indent=2))
    else: run(args.output)
    return 0

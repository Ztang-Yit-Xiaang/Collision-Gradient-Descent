"""Frozen equal-query and equal-training-update smoke tests for momentum routing."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .momentum_routing import RoutingConfig, run_routed
from .optimizer_comparison import CASES, make_case
from .optimizer_controls import OptimizerConfig
from .population import PopulationOracle
from .strategic_experiment import _csv, _json


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("momentum", "momentum_restart", "momentum_damp", "momentum_random", "momentum_adaptive", "adam")


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}


def prepare(output):
    output = Path(output)
    if (output/"config.json").exists():
        raise ValueError("Use a fresh output directory.")
    for folder in (output, output/"trajectories", output/"actions", output/"feedback", output/"checkpoints"):
        folder.mkdir(parents=True, exist_ok=True)
    return output


def synthetic(output, *, seeds=range(300, 310), budget=1500, cases=CASES):
    output = prepare(output); routing = RoutingConfig()
    selection_path = ROOT/"output/optimizer_comparison/selected.json"
    selected = json.loads(selection_path.read_text())
    _json(output/"config.json", {"routing": asdict(routing), "seeds": list(seeds), "budget": budget,
          "methods": METHODS, "cases": list(cases), "source_sha256": source_hashes(), "python": sys.version,
          "numpy": np.__version__, "selection_source": str(selection_path),
          "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(), "selected": selected,
          "new_tuning_runs": 0, "cost": "Every actual value/gradient costs one; router makes zero extra queries",
          "ledger": "Deterministic zero-decay Momentum only; explicit Euler residual is recorded"})
    runs = []
    for case in cases:
        for seed in seeds:
            for method in METHODS:
                kind, *suffix = method.split("_")
                mode = suffix[0] if suffix else "none"
                objective, minimum, boundary, initial = make_case(case, seed)
                x0 = initial.particles[0].x
                config = OptimizerConfig(kind=kind, lr=selected[case][kind]["setting"])
                oracle = PopulationOracle(objective, total_budget=budget)
                row, trace, actions, feedback, _ = run_routed(oracle, x0, config, mode=mode, routing=routing, seed=seed)
                gap = max(row["initial_f"]-minimum, 1e-12)
                hits = []; final_hit = False
                for point in trace:
                    x = np.array(point["x"])
                    point["relative_best_gap"] = max(0., (point["best_value"]-minimum)/gap)
                    point["relative_current_gap"] = max(0., (point["value"]-minimum)/gap)
                    hit = point["relative_current_gap"] <= .1 and (boundary is None or x[0] < boundary)
                    if case == "wells20":
                        hit = hit and float(np.mean(objective.coordinates(x) < 0)) >= .9
                    if hit: hits.append(point["total_oracle_calls"])
                    final_hit = bool(hit)
                row.update({"case": case, "seed": seed, "method": method, "lr": config.lr,
                            "minimum": minimum, "initial_x": x0.tolist(),
                            "relative_best_gap": max(0., (row["best_f"]-minimum)/gap),
                            "relative_final_gap": max(0., (row["final_f"]-minimum)/gap),
                            "target_hit": bool(hits), "final_target_hit": final_hit,
                            "first_target_calls": min(hits) if hits else None})
                runs.append(row); name = f"{case}__s{seed}__{method}"
                _csv(output/"trajectories"/(name+".csv.gz"), trace, True)
                _json(output/"actions"/(name+".json"), actions); _json(output/"feedback"/(name+".json"), feedback)
            _json(output/"runs.json", runs)
            print(f"Routing synthetic {case} seed {seed}: {len(runs)} runs", flush=True)
    _csv(output/"runs.csv", runs)
    aggregate = []
    for case in cases:
        baseline = {r["seed"]: r for r in runs if r["case"] == case and r["method"] == "momentum"}
        for method in METHODS:
            group = [r for r in runs if r["case"] == case and r["method"] == method]
            deltas = [r["relative_final_gap"]-baseline[r["seed"]]["relative_final_gap"] for r in group]
            aggregate.append({"case": case, "method": method, "runs": len(group),
                "failures": sum(r["failure"] for r in group), "target_hits": sum(r["target_hit"] for r in group),
                "final_target_hits": sum(r["final_target_hit"] for r in group),
                "median_relative_best_gap": float(np.median([r["relative_best_gap"] for r in group])),
                "median_relative_final_gap": float(np.median([r["relative_final_gap"] for r in group])),
                "better_final_than_momentum": sum(d < -1e-8 for d in deltas),
                "worse_final_than_momentum": sum(d > 1e-8 for d in deltas),
                "shots": sum(r["shots"] for r in group), "dampings": sum(r["dampings"] for r in group),
                "mean_supplied": float(np.mean([r["supplied"] for r in group])),
                "mean_dissipated": float(np.mean([r["dissipated"] for r in group])),
                "median_seconds": float(np.median([r["wall_seconds"] for r in group]))})
    _json(output/"aggregate.json", aggregate); _csv(output/"aggregate.csv", aggregate)
    return aggregate


def nanogpt(output, *, seeds=(300, 301, 302), steps=400):
    import torch
    from .nanogpt_benchmark import LMConfig, TextData, VENDOR
    from .nanogpt_routing import train_routed

    output = prepare(output); lm = LMConfig()
    routing = RoutingConfig(window=20, epsilon_gain=1e-3, noise_aware=True, radii=(.001, .003))
    data = TextData(ROOT/"data/tinyshakespeare.txt", lm)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    selection_path = ROOT/"output/nanogpt_comparison/selected.json"
    selected = json.loads(selection_path.read_text())
    methods = METHODS[:-1]+("adamw",)
    _json(output/"config.json", {"model": asdict(lm), "routing": asdict(routing), "seeds": list(seeds),
          "steps": steps, "methods": methods, "source_sha256": source_hashes(), "torch": torch.__version__,
          "device": "cpu", "dtype": "float32", "data_sha256": data.sha256,
          "split_lengths": {k: len(v) for k, v in data.splits.items()}, "selected": selected,
          "selection_source": str(selection_path), "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
          "vendor_sha256": hashlib.sha256(VENDOR.read_bytes()).hexdigest(), "new_tuning_runs": 0,
          "cost": "400 real updates, train forward=1 backward=2 units, zero diagnostic/virtual forwards",
          "evaluation": "32 final test forwards per run outside training budget",
          "conservation": "Actuator kinetic increments only; minibatch loss is not a conservative potential"})
    runs = []
    for seed in seeds:
        for method in methods:
            kind, *suffix = method.split("_"); mode = suffix[0] if suffix else "none"
            setting = selected[kind]; name = f"s{seed}__{method}"
            row, trace, actions, feedback = train_routed(data, lm, kind, setting["lr"], setting["weight_decay"], seed,
                mode=mode, steps=steps, routing=routing, checkpoint=output/"checkpoints"/(name+".pt"))
            row["method"] = method; runs.append(row)
            _csv(output/"trajectories"/(name+".csv.gz"), trace, True)
            _json(output/"actions"/(name+".json"), actions); _json(output/"feedback"/(name+".json"), feedback)
            _json(output/"runs.json", runs)
            print(f"Routing NanoGPT {len(runs)}/{len(seeds)*len(methods)} {name}: loss={row['final_loss']}, pulses={row['shots']}, damping={row['dampings']}", flush=True)
    _csv(output/"runs.csv", runs); aggregate = []
    for method in methods:
        group = [r for r in runs if r["method"] == method]; valid = [r for r in group if r["final_loss"] is not None]
        aggregate.append({"method": method, "runs": len(group), "failures": len(group)-len(valid),
            "mean_test_loss": float(np.mean([r["final_loss"] for r in valid])) if valid else None,
            "std_test_loss": float(np.std([r["final_loss"] for r in valid], ddof=1)) if len(valid) > 1 else None,
            "mean_perplexity": float(np.mean([r["perplexity"] for r in valid])) if valid else None,
            "shots": sum(r["shots"] for r in group), "dampings": sum(r["dampings"] for r in group),
            "mean_seconds": float(np.mean([r["training_seconds"] for r in group])),
            "mean_feature_seconds": float(np.mean([r["feature_seconds"] for r in group])),
            "mean_action_seconds": float(np.mean([r["action_seconds"] for r in group]))})
    _json(output/"aggregate.json", aggregate); _csv(output/"aggregate.csv", aggregate)
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("synthetic", "nanogpt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=300)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args(argv)
    seeds = range(args.seed_start, args.seed_start+(args.seeds or (10 if args.suite == "synthetic" else 3)))
    if args.suite == "synthetic": synthetic(args.output, seeds=seeds)
    else: nanogpt(args.output, seeds=seeds, steps=args.steps)
    return 0

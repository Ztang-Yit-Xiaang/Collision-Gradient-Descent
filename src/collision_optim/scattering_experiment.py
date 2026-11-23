"""Frozen physical versus conserving-random scattering experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np

from .core import elastic_pair_collision
from .gravity import GravityConfig
from .population import PopulationConfig, PopulationGravity, PopulationOracle, run_population
from .population_experiment import initial_population, make_objective, _summarize
from .scattering_population import ScatteringPopulation, random_pair_collision, pair_diagnostics
from .strategic_experiment import _csv, _json


ROOT = Path(__file__).resolve().parents[2]
CASES = ("quadratic", "double_well", "rosenbrock")
RHOS = (.12, .24)


def mechanism_scan(output):
    rows = []
    u = np.array([-.4, 0., 0.])
    for center_name, center in (("zero", np.zeros(3)), ("parallel", np.array([.4, 0., 0.])),
                                 ("transverse", np.array([0., .4, 0.]))):
        for angle in (-60, -30, 0, 30, 60):
            radians = np.deg2rad(angle); n = np.array([np.cos(radians), np.sin(radians), 0.])
            vi, vj = center+u, center-u
            for mode, repeats in (("elastic", 1), ("random", 100)):
                for repeat in range(repeats):
                    rng = np.random.default_rng(np.random.SeedSequence([52391108, angle+60, repeat]))
                    ai, aj = elastic_pair_collision(vi, vj, n) if mode == "elastic" else random_pair_collision(vi, vj, n, rng)
                    row = {"center": center_name, "angle_degrees": angle, "mode": mode, "repeat": repeat,
                           "before_i": vi.tolist(), "before_j": vj.tolist(), "after_i": ai.tolist(), "after_j": aj.tolist(),
                           "normal": n.tolist(), **pair_diagnostics(vi, vj, ai, aj, n),
                           "energy_error": float(.5*(ai @ ai+aj @ aj-vi @ vi-vj @ vj)),
                           "momentum_error": float(np.linalg.norm(ai+aj-vi-vj)),
                           "random_expected_gain_i": float(center @ (.5*np.linalg.norm(u)*n-u))}
                    rows.append(row)
    _json(output/"mechanism_maps.json", rows); _csv(output/"mechanism_maps.csv", rows)
    return rows


def snapshot(output):
    files = set()
    for directory in ("src", "tests", "scripts"):
        files.update(p for p in (ROOT/directory).rglob("*.py") if "__pycache__" not in p.parts)
    files.update(ROOT.glob("*.py"))
    files.update(ROOT/p for p in ("AGENTS.md", "pyproject.toml", "docs/superpowers/specs/2026-11-08-collision-scattering-design.md"))
    manifest = {}
    with zipfile.ZipFile(output/"source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            relative = str(path.relative_to(ROOT)); raw = path.read_bytes()
            archive.writestr(relative, raw); manifest[relative] = hashlib.sha256(raw).hexdigest()
    _json(output/"source_manifest.json", manifest)
    return manifest


def summarize_pairs(runs):
    lookup = {(r["case"], r["seed"], r["method"], r["rho"]): r for r in runs}
    pairs = []
    for r in runs:
        if r["method"] != "elastic": continue
        for ref in ("independent", "random"):
            base = lookup[r["case"], r["seed"], ref, None if ref == "independent" else r["rho"]]
            valid = not r["failed"] and not base["failed"]
            diff = r["relative_final_gap"]-base["relative_final_gap"]
            pairs.append({"case": r["case"], "seed": r["seed"], "rho": r["rho"], "reference": ref,
                          "valid": valid, "final_gap_difference": diff if valid else None,
                          "best_gap_difference": r["relative_best_gap"]-base["relative_best_gap"] if valid else None,
                          "outcome": "unavailable" if not valid else "better" if diff < -1e-6 else "worse" if diff > 1e-6 else "tie",
                          "elastic_final_hit": r["final_target_hit"] and not r["failed"],
                          "reference_final_hit": base["final_target_hit"] and not base["failed"]})
    return pairs


def aggregate(runs):
    rows = []
    for case in CASES:
        for mode, rho in [("independent", None)]+[(m, r) for r in RHOS for m in ("elastic", "random")]:
            group = [r for r in runs if (r["case"], r["method"], r["rho"]) == (case, mode, rho)]
            valid = [r for r in group if not r["failed"]]
            rows.append({"case": case, "method": mode, "rho": rho, "runs": len(group), "valid": len(valid),
                         "failures": len(group)-len(valid),
                         "historical_hits": sum(r["population_target_hit"] for r in valid),
                         "final_hits": sum(r["final_target_hit"] for r in valid),
                         "median_relative_best_gap": float(np.median([r["relative_best_gap"] for r in valid])) if valid else None,
                         "median_relative_final_gap": float(np.median([r["relative_final_gap"] for r in valid])) if valid else None,
                         "pair_collisions": sum(r["pair_collisions"] for r in group),
                         "runs_with_collision": sum(r["pair_collisions"] > 0 for r in group),
                         "median_used_calls": float(np.median([r["total_oracle_calls"] for r in group])),
                         "median_observed_time": float(np.median([r["completed_time"] for r in group])),
                         "median_wall_seconds": float(np.median([r["wall_seconds"] for r in group]))})
    return rows


def run(output, *, seeds=range(500, 520), steps=500, budget=6000):
    output = Path(output)
    if (output/"config.json").exists(): raise ValueError("Use a fresh output directory.")
    for path in (output, output/"traces", output/"events", output/"initial_states", output/"map_attempts"):
        path.mkdir(parents=True, exist_ok=True)
    physics = GravityConfig(h=.05, beta=.995)
    manifest = snapshot(output)
    _json(output/"config.json", {"seeds": list(seeds), "steps": steps, "cases": CASES, "rhos": RHOS,
          "methods": ["independent", "elastic", "random"], "physics": asdict(physics),
          "population_size": 4, "initial_total_kinetic": .6, "oracle_budget": budget,
          "pair_prediction_budget": 100000, "source_manifest": "source_manifest.json",
          "source_archive_sha256": hashlib.sha256((output/"source.zip").read_bytes()).hexdigest(),
          "source_sha256": manifest, "launch_argv": sys.argv, "python": sys.version, "numpy": np.__version__,
          "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH")},
          "scoring": "Last complete observed state at a true shared 6000 query cap; history separate from final",
          "random_map": "Uniform outgoing hemisphere preserving V and norm(u); generally noncentral, angular momentum not guaranteed",
          "reference": "E06 initialization and physical scales, new seeds; no tuning, heat or state resets",
          "mechanism": "15 incoming states, 1 elastic + 100 random each; constructed kernel diagnostics only"})
    mechanism_scan(output); runs = []
    for case in CASES:
        objective, minimum, boundary = make_objective(case)
        for seed in seeds:
            initial = initial_population(case, objective, seed, rho_max=max(RHOS))
            f0 = min(objective.value(p.x) for p in initial.particles); target = minimum+.1*(f0-minimum)
            _json(output/"initial_states"/f"{case}__s{seed}.json", {
                "positions": [np.r_[p.x,p.z].tolist() for p in initial.particles],
                "velocities": [p.velocity.tolist() for p in initial.particles],
                "initial_energy": initial.kinetic+physics.gravity*sum(p.z for p in initial.particles),
                "known_minimum": minimum, "boundary": boundary, "target": target})
            for mode, rho in [("independent", None)]+[(m, r) for r in RHOS for m in ("elastic", "random")]:
                cfg = PopulationConfig(physics=physics, rho=rho or max(RHOS), collisions=mode != "independent", max_pair_tests=100000)
                rng_seed = np.random.SeedSequence([20261108, seed, int(1000*(rho or 0))])
                method = PopulationGravity(cfg) if mode == "independent" else ScatteringPopulation(cfg, rule=mode, seed=rng_seed)
                oracle = PopulationOracle(objective, total_budget=budget)
                result = run_population(method, oracle, initial, steps, basin_boundary=boundary, target_value=target)
                row = _summarize(result, initial, minimum, case=case, seed=seed, method=mode, rho=rho)
                final = result.rows[-1]
                gap = max(f0-minimum, 1e-12)
                for point in result.rows:
                    point["relative_current_gap"] = max(0., (min(point["values"])-minimum)/gap)
                final_hit = any(f <= target and (boundary is None or p.x[0] < boundary)
                                for f,p in zip(final["values"],result.final_state.particles))
                impacts = [e for e in result.events if e["kind"] == "pair"]
                attempts = getattr(method, "map_attempts", [])
                row.update({"minimum": minimum, "target": target, "final_value": min(final["values"]),
                            "relative_final_gap": max(0.,(min(final["values"])-minimum)/gap),
                            "final_target_hit": bool(final_hit), "final_observation_calls": final["total_oracle_calls"],
                            "map_attempts": len(attempts), "uncommitted_maps": len(attempts)-len(impacts),
                            "max_transfer_formula_error": max((e["transfer_formula_error"] for e in impacts), default=0.),
                            "max_angular_impulse": max((e["angular_impulse_frobenius"] for e in impacts), default=0.),
                            "positive_supply": 0., "no_new_particles": len(result.final_state.particles) == 4})
                name = f"{case}__s{seed}__{mode}__rho{rho}"
                _csv(output/"traces"/(name+".csv.gz"), result.rows, True)
                _json(output/"events"/(name+".json"), result.events)
                _json(output/"map_attempts"/(name+".json"), [{**a,"committed": i < len(impacts)} for i,a in enumerate(attempts)])
                runs.append(row); _json(output/"runs.json", runs)
            print(f"Scattering {case} seed {seed}: {len(runs)}/{len(CASES)*len(seeds)*5} runs, failures={sum(r['failed'] for r in runs)}", flush=True)
    pairs = summarize_pairs(runs); summaries = aggregate(runs)
    for name, data in (("runs", runs), ("paired", pairs), ("aggregate", summaries)):
        _json(output/(name+".json"), data); _csv(output/(name+".csv"), data)
    return summaries


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed-start",type=int,default=500); parser.add_argument("--seeds",type=int,default=20)
    parser.add_argument("--steps",type=int,default=500)
    args=parser.parse_args(argv)
    run(args.output,seeds=range(args.seed_start,args.seed_start+args.seeds),steps=args.steps)
    return 0

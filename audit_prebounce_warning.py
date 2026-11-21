"""Audit E28 traces and score alerts without using future loss as input."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_shared_prefix_transfer import segment_audit
from collision_optim.prebounce_warning import directional_quadratic, flight_factors, forecast_rise
from finalize_initial_budget import create, sha
from run_prebounce_warning import ENERGIES, SEEDS, read

SINGLE = {"speed": 0, "loss_step1": 1, "loss_step2": 2}
PHYSICAL = {"speed_and_loss_step1": 1, "quadratic_step2": 2}
STEPS = SINGLE | PHYSICAL


def case_from_trace(context, run, raw, factors, limit):
    q, v, loss, calls, events = raw
    assert np.array_equal(q[0, :, :2], context["points"])
    assert np.array_equal(v[0, :, :2], context["initial_horizontal_velocities"])
    assert np.array_equal(loss[0], context["values"])
    assert np.max(abs(q[0, :, 2] - loss[0] - context["heights"])) < 1e-12
    assert context["heights"][0] == .025 and np.max(abs(v[0, :, 2])) == 0
    assert abs(.5 * np.sum(v[0]**2) - .6) < 1e-12
    assert np.linalg.norm(v[0].sum(axis=0)) < 1e-12
    speed = float(np.linalg.norm(v[0, 0, :2]))
    assert abs(.5 * speed**2 - context["incumbent_energy"]) < 1e-14
    assert calls[0] == 8 and context["setup_reference_values"] == 4
    first_bounce = next((event["time"] for event in events if event["i"] == 0), None)
    times = np.arange(len(loss)) * .05
    increases = loss[:, 0] - loss[0, 0]
    valid = not run["failed"] and first_bounce is not None
    first_rise = None
    max_error = 0.
    if valid:
        precontact = np.flatnonzero(times < first_bounce)
        assert precontact[-1] < len(factors)
        expected = q[0, 0, :2] + factors[precontact, None] * v[0, 0, :2]
        max_error = float(np.max(abs(expected - q[precontact, 0, :2])))
        assert max_error < 1e-12
        first_rise = next((float(times[i]) for i in precontact if increases[i] > limit), None)
    return {
        "run_id": run["run_id"], "seed": run["seed"], "energy": context["incumbent_energy"],
        "valid": valid, "speed": speed, "loss_increases": increases.tolist(), "observation_calls": calls,
        "first_bounce": first_bounce, "first_deviation": first_rise, "positive": first_rise is not None if valid else None,
        "max_ballistic_position_error": max_error,
    }


def features(case, factors, curvature):
    increases = case["loss_increases"]
    available = len(increases)
    if available >= 3:
        quadratic = forecast_rise(increases[1:3], case["speed"], factors)
    else:
        quadratic = None
    return {
        "speed": round(case["speed"], 14),
        "loss_step1": increases[1] if available >= 2 else None,
        "loss_step2": increases[2] if available >= 3 else None,
        "speed_and_loss_step1": forecast_rise(increases[1:2], case["speed"], factors, curvature)
            if available >= 2 else None,
        "quadratic_step2": quadratic,
    }


def eligible(case, policy):
    time = STEPS[policy] * .05
    return (time < case["first_bounce"] and
            (case["first_deviation"] is None or time < case["first_deviation"]) and
            case["scores"][policy] is not None)


def prediction(case, policy, rule):
    if not eligible(case, policy):
        return False
    if rule["kind"] == "threshold":
        return case["scores"][policy] > rule["cutoff"]
    return rule["kind"] == "always"


def summarize(cases, policy, rule):
    pos = sum(c["positive"] for c in cases)
    neg = len(cases) - pos
    alerted = [c for c in cases if prediction(c, policy, rule)]
    on_time = [c for c in alerted if c["positive"]]
    fp = len(alerted) - len(on_time)
    recall = len(on_time) / pos if pos else None
    fpr = fp / neg if neg else None
    cutoff_time = STEPS[policy] * .05
    late = sum(c["positive"] and c["first_deviation"] <= cutoff_time for c in cases)
    leads = [c["first_deviation"] - cutoff_time for c in on_time]
    charges = [c["observation_calls"][STEPS[policy]] for c in cases if len(c["observation_calls"]) > STEPS[policy]]
    return {
        "n": len(cases), "positives": pos, "negatives": neg, "timely_alerts": len(on_time), "false_alerts": fp,
        "missed_or_late": pos - len(on_time), "positives_already_deviated": late,
        "recall": recall, "false_positive_rate": fpr,
        "balanced_accuracy": (recall + 1 - fpr) / 2 if pos and neg else None,
        "alert_time": cutoff_time, "median_lead": median(leads) if leads else None,
        "lead_range": [min(leads), max(leads)] if leads else None,
        "solver_queries_min_median_max": [min(charges), median(charges), max(charges)] if charges else None,
    }


def tune(cases, policy):
    scores = sorted({c["scores"][policy] for c in cases if eligible(c, policy)})
    rules = [{"kind": "never"}, {"kind": "always"}]
    rules.extend({"kind": "threshold", "cutoff": (a + b) / 2} for a, b in zip(scores, scores[1:]))
    candidates = [{"rule": rule, "performance": summarize(cases, policy, rule)} for rule in rules]
    best = min(candidates, key=lambda x: (-x["performance"]["balanced_accuracy"], x["performance"]["false_alerts"]))
    return best["rule"], candidates


def seed_bootstrap(cases, rules, comparisons):
    seeds = sorted({c["seed"] for c in cases})
    grouped = [[c for c in cases if c["seed"] == s] for s in seeds]
    positives = np.array([sum(c["positive"] for c in group) for group in grouped])
    negatives = np.array([len(group) for group in grouped]) - positives
    selected = np.random.default_rng(2280).integers(0, len(seeds), size=(2000, len(seeds)))
    p = positives[selected].sum(axis=1)
    n = negatives[selected].sum(axis=1)
    valid = (p > 0) & (n > 0)
    estimates = {}
    for policy, rule in rules.items():
        tp = np.array([sum(c["positive"] and prediction(c, policy, rule) for c in g) for g in grouped])
        fp = np.array([sum(not c["positive"] and prediction(c, policy, rule) for c in g) for g in grouped])
        estimates[policy] = .5 * (tp[selected[valid]].sum(axis=1) / p[valid] +
                                 1 - fp[selected[valid]].sum(axis=1) / n[valid])
    return [
        {"candidate": candidate, "reference": reference, "seed_groups": len(seeds),
         "valid_resamples": int(valid.sum()),
         "balanced_accuracy_gain_95_interval":
             np.quantile(estimates[candidate] - estimates[reference], [.025, .975]).tolist() if valid.any() else None}
        for candidate, reference in comparisons
    ]


def assess(cases, lock):
    rules = lock["rules"]
    overall = {p: summarize(cases, p, r) for p, r in rules.items()}
    by_energy = {str(e): {p: summarize([c for c in cases if c["energy"] == e], p, r)
                          for p, r in rules.items()} for e in ENERGIES}
    best = max(overall[p]["balanced_accuracy"] for p in SINGLE)
    screens = {}
    for name in PHYSICAL:
        s = overall[name]
        gain = s["balanced_accuracy"] - best
        screens[name] = {
            "gain_over_best_single_input": gain, "timely_recall_at_least_85_percent": s["recall"] >= .85,
            "false_positive_rate_at_most_10_percent": s["false_positive_rate"] <= .10,
            "median_lead_at_least_one_step": s["median_lead"] is not None and s["median_lead"] >= .05 - 1e-12,
            "balanced_accuracy_gain_at_least_0_05": gain >= .05,
        }
    comparisons = [(p, r) for p, same_step in zip(PHYSICAL, ("loss_step1", "loss_step2"))
                   for r in (same_step, "speed")]
    predictions = []
    for c in cases:
        predictions.append({
            "run_id": c["run_id"], "seed": c["seed"], "energy": c["energy"], "positive": c["positive"],
            "alert": {p: prediction(c, p, rule) for p, rule in rules.items()},
            "eligible": {p: eligible(c, p) for p in rules},
        })
    return {
        "overall": overall, "by_energy": by_energy, "screen": screens,
        "any_physical_rule_passed": any(all(v for k, v in s.items() if k != "gain_over_best_single_input")
                                       for s in screens.values()),
        "paired_seed_bootstrap": seed_bootstrap(cases, rules, comparisons),
        "predictions": predictions,
    }


def audit(root):
    config, contexts, runs, costs = (read(root / (n + ".json")) for n in ("config", "contexts", "runs", "execution_counts"))
    stage = config["stage"]
    expected = {(s, e) for s in SEEDS[stage] for e in ENERGIES}
    assert len(contexts) == len(runs) == len(expected) == costs["trajectory_executions"]
    assert {(r["seed"], r["incumbent_energy"]) for r in runs} == expected
    assert config["seeds"] == SEEDS[stage]
    assert sha(root / "source.zip") == config["source_archive_sha256"]
    for path, digest in config["dependency_data_sha256"].items():
        assert sha(Path(path)) == digest
    with zipfile.ZipFile(root / "source.zip") as archive:
        for name, digest in config["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    assert sum(r["counts"]["total_oracle_calls"] for r in runs) == costs["actual_solver_queries"]
    assert costs["setup_reference_values"] == len(runs) * 4
    assert costs["actual_queries_including_setup"] == costs["actual_solver_queries"] + costs["setup_reference_values"]
    totals = Counter()
    peaks = {"energy": 0., "momentum": 0., "reflection": 0.}
    factors = flight_factors(steps=config["max_steps"])
    forecast = factors[:config["forecast_end_step"] + 1]
    by_id = {c["context_id"]: c for c in contexts}
    cases, ledgers = [], []
    limit = config["threshold"]
    for r in runs:
        assert r["counts"]["total_oracle_calls"] <= config["query_cap"]
        c = by_id[r["run_id"]]
        raw, ledger = segment_audit(root, r, c, totals, peaks)
        ledgers.append(ledger)
        cases.append(case_from_trace(c, r, raw, factors, limit))
    eligible_cases = [c for c in cases if c["valid"]]
    if stage != "preflight":
        if stage == "development":
            fit = [directional_quadratic(c["speed"] * forecast[1:3], c["loss_increases"][1:3])[1]
                   for c in eligible_cases if c["first_bounce"] > .10]
            curvature = median(fit)
        else:
            lock = read(Path(config["previous_stage"]) / "model_lock.json")
            curvature = lock["prior_curvature"]
        for case in eligible_cases:
            case["scores"] = features(case, forecast, curvature)
        if stage == "development":
            rules, candidates = {}, {}
            for name in SINGLE:
                rules[name], candidates[name] = tune(eligible_cases, name)
            rules.update({name: {"kind": "threshold", "cutoff": limit} for name in PHYSICAL})
            lock = {
                "prior_curvature": float(curvature), "prior_examples": len(fit), "rules": rules, "all_candidates": candidates,
                "development_performance": {p: summarize(eligible_cases, p, rule) for p, rule in rules.items()},
            }
            create(root / "model_lock.json", lock)
        else:
            create(root / "results.json", assess(eligible_cases, lock))
    create(root / "labeled_observations.json", cases)
    create(root / "physical_ledgers.json", {"segments": ledgers, "runtime_interventions": 0, "new_oracle_calls_for_export": 0})
    result = {
        "passed": True, "trajectories": len(runs), "valid_labels": len(eligible_cases),
        "positive_labels": sum(c["positive"] for c in eligible_cases),
        "bounce_time_range": [min(c["first_bounce"] for c in eligible_cases),
                              max(c["first_bounce"] for c in eligible_cases)] if eligible_cases else None,
        "bounces_outside_nominal_window": sum(not .30 < c["first_bounce"] < .35 for c in eligible_cases),
        "max_ballistic_position_error": max(c["max_ballistic_position_error"] for c in cases),
        "solver_failures": [r["run_id"] for r in runs if r["failed"]], "max_residuals": peaks, **totals,
    }
    create(root / "audit.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

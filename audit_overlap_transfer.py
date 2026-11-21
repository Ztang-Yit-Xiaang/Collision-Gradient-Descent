"""Independent raw-state checks and paired E26 diagnostics."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_conditional_transfer import trace_audit
from audit_initial_budget import raw_metrics
from finalize_initial_budget import create, sha
from run_overlap_transfer import CONDITIONS, KINETICS, MODES, PREFIX_STEPS, SCORES, SEEDS, read

FEATURES = ("loss_score", "initial_kinetic", "early_horizontal_impulse")


def context_audit(c):
    q = np.array(c["points"]) @ np.array(c["rotation"]).T
    roots = np.sort(np.roots([1., 0., -1., .05]).real)
    refs = .25 * (roots**2 - 1)**2 + .05 * roots
    assert np.max(abs(refs - c["reference_values"])) < 1e-12
    assert abs(c["minimum"] - refs[0]) < 1e-12 and abs(c["boundary"] - roots[1]) < 1e-12
    good = c["condition"] == "good_present"
    assert abs(np.linalg.norm(q[0] - [roots[0 if good else 2], 0]) - .001) < 1e-12
    assert np.max(abs(q[1:, 0] - roots[-1])) < 1e-12
    a, b, d, e = c["values"]
    width = .11 / c["score_target"]
    assert np.max(abs(np.array([b-a, d-a, e-a]) - [.11-width/2, .11, .11+width/2])) < 1e-12
    assert abs((d - a) / (e - b) - c["score"]) < 1e-12
    assert abs(c["score"] - c["score_target"]) < 1e-8
    assert c["conditional_source_policy"] == ("transfer" if c["score"] >= 100 else "keep")
    assert abs(c["target"] - refs[0] - (1e-4 if good else .1 * (a - refs[0]))) < 1e-12
    velocity = np.array(c["initial_horizontal_velocities"])
    assert abs(.5 * np.sum(velocity[0]**2) - KINETICS[c["seed"] % 4]) < 1e-14
    assert abs(.5 * np.sum(velocity**2) - .6) < 1e-12
    assert np.linalg.norm(velocity.sum(axis=0)) < 1e-12
    assert c["probe_counts"] == {"value_calls": 4, "gradient_calls": 0, "path_calls": 0}
    assert c["setup_reference_values"] == 4


def success(row):
    if row["failed"]:
        return False
    if row["condition"] == "good_present":
        return row["particle0_final_hit"] and not row["particle0_harmed"]
    return row["final_hit"]


def feature_row(root, c, branches, raw):
    keep, transfer = (branches[mode] for mode in MODES)
    available = len(raw[2]) > PREFIX_STEPS and not (keep["failed"] or transfer["failed"])
    impulse = calls = None
    if available:
        events = read(root / "events" / (keep["run_id"] + ".json"))
        deltas = [np.array(e["after_i"][:2]) - e["before_i"][:2] for e in events
                  if e["i"] == 0 and e["step"] <= PREFIX_STEPS]
        impulse = float(np.linalg.norm(np.sum(deltas, axis=0))) if deltas else 0.
        calls = int(raw[3][PREFIX_STEPS])
    return {
        "context_id": c["context_id"], "condition": c["condition"], "seed": c["seed"],
        "loss_score": c["score_target"], "initial_kinetic": c["incumbent_energy"],
        "early_horizontal_impulse": impulse, "charged_calls_to_observe_prefix": calls,
        "keep_run_id": keep["run_id"], "transfer_run_id": transfer["run_id"],
        "available": available, "failed_pair": keep["failed"] or transfer["failed"],
        "keep_success": success(keep), "transfer_success": success(transfer),
    }


def auc(rows, feature, direction):
    positive = [r[feature] for r in rows if r["transfer_success"]]
    negative = [r[feature] for r in rows if r["keep_success"]]
    if not positive or not negative:
        return None
    return sum(float(direction * (p - n) > 0) + .5 * (p == n)
               for p in positive for n in negative) / (len(positive) * len(negative))


def compare_features(features, directions=None):
    result = {}
    for condition in CONDITIONS:
        eligible = [r for r in features if r["condition"] == condition and r["available"]]
        discordant = [r for r in eligible if r["transfer_success"] != r["keep_success"]]
        positive = {r["seed"] for r in discordant if r["transfer_success"]}
        negative = {r["seed"] for r in discordant if r["keep_success"]}
        enough = min(len(positive), len(negative)) >= 5
        dev = None if directions is None else directions[condition]
        ranks = {}
        for name in FEATURES:
            if dev is None:
                forward = auc(discordant, name, 1)
                direction = -1 if forward is not None and forward < .5 else 1
            else:
                direction = dev["features"][name]["direction"]
            ranks[name] = {"direction": direction, "auc": auc(discordant, name, direction)}
        interpretable = enough and dev is not None and dev["enough_distinct_seeds"]
        baseline = ranks["loss_score"]["auc"]
        if interpretable:
            for name in FEATURES:
                gain = ranks[name]["auc"] - baseline
                ranks[name].update(auc_gain_over_loss_score=gain, follow_up_candidate=gain >= .10)
        prefix = [r["charged_calls_to_observe_prefix"] for r in eligible]
        result[condition] = {
            "eligible_contexts": len(eligible), "discordant_contexts": len(discordant),
            "transfer_only": sum(r["transfer_success"] for r in discordant),
            "keep_only": sum(r["keep_success"] for r in discordant),
            "transfer_only_distinct_seeds": len(positive), "keep_only_distinct_seeds": len(negative),
            "enough_distinct_seeds": enough, "interpretable_held_out_comparison": interpretable,
            "prefix_charged_calls_min_median_max": [min(prefix), median(prefix), max(prefix)] if prefix else None,
            "features": ranks,
        }
    return result


def summarize(rows):
    output = []
    for condition in CONDITIONS:
        for score in SCORES:
            for mode in (*MODES, "conditional"):
                group = [r for r in rows if (r["condition"], r["score_target"], r["policy"]) ==
                         (condition, score, mode)]
                valid = [r for r in group if not r["failed"]]
                output.append({
                    "condition": condition, "score_target": score, "policy": mode, "n": len(group),
                    "failures": len(group) - len(valid), "particle0_harmed": sum(r["particle0_harmed"] for r in group),
                    "particle0_final_hits": sum(r["particle0_final_hit"] for r in valid),
                    "final_hits": sum(r["final_hit"] for r in valid),
                    "historical_hits": sum(r["historical_hit"] for r in group),
                    "median_final_value": median(r["final_value"] for r in valid) if valid else None,
                    "median_best_value": median(r["best_value"] for r in valid) if valid else None,
                })
    return output


def paired_summary(features, rows):
    groups = []
    for condition in CONDITIONS:
        for score in SCORES:
            selected = [r for r in features if (r["condition"], r["loss_score"]) == (condition, score)]
            outcomes = [(r["keep_success"], r["transfer_success"]) for r in selected if not r["failed_pair"]]
            branches = {(r["seed"], r["policy"]): r for r in rows
                        if (r["condition"], r["score_target"]) == (condition, score)}
            missed = sum(
                branches[r["seed"], "keep"]["particle0_harmed"] and r["transfer_success"]
                and not branches[r["seed"], "keep"]["failed"] for r in selected
            ) if condition == "good_present" and score == 30 else None
            unwanted = outcomes.count((True, False)) if condition == "bad_only" and score == 300 else None
            groups.append({
                "condition": condition, "score_target": score, "n": len(selected),
                "failed_pairs": len(selected) - len(outcomes),
                "both_success": outcomes.count((True, True)), "keep_only": outcomes.count((True, False)),
                "transfer_only": outcomes.count((False, True)), "neither": outcomes.count((False, False)),
                "missed_protection": missed, "unwanted_transfer": unwanted,
            })
    return groups


def audit(root):
    config = read(root / "config.json")
    runs = read(root / "runs.json")
    contexts = read(root / "contexts.json")
    costs = read(root / "execution_counts.json")
    expected = {(s, c, g) for s in SEEDS[config["stage"]] for c in CONDITIONS for g in SCORES}
    assert len(contexts) == len(expected) and len(runs) == 2 * len(expected)
    assert {(c["seed"], c["condition"], c["score_target"]) for c in contexts} == expected
    assert len({r["run_id"] for r in runs}) == len(runs) and len(runs) == costs["trajectory_executions"]
    assert costs["contexts"] == len(contexts)
    assert costs["shared_probe_calls"] == costs["setup_reference_values"] == 4 * len(contexts)
    assert costs["actual_branch_calls"] == sum(r["actual_branch_calls"] for r in runs)
    assert costs["actual_solver_queries"] == costs["actual_branch_calls"] + costs["shared_probe_calls"]
    assert costs["actual_queries_including_setup"] == costs["actual_solver_queries"] + costs["setup_reference_values"]
    assert costs["charged_control_queries"] == sum(r["total_oracle_calls"] for r in runs)
    assert sha(root / "source.zip") == config["source_archive_sha256"]
    for path, digest in config["data_sha256"].items():
        assert sha(Path(path)) == digest
    with zipfile.ZipFile(root / "source.zip") as archive:
        for name, digest in config["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest

    shared = defaultdict(list)
    run_groups = defaultdict(dict)
    for c in contexts:
        shared[c["seed"], c["score_target"]].append(c)
    for row in runs:
        run_groups[row["context_id"]][row["policy"]] = row
    overlap = max_score_difference = 0
    for (seed, score), (a, b) in shared.items():
        assert {a["condition"], b["condition"]} == set(CONDITIONS)
        assert a["initial_horizontal_velocities"] == b["initial_horizontal_velocities"]
        assert np.max(abs(np.diff(a["values"]) - np.diff(b["values"]))) < 1e-12
        assert a["conditional_source_policy"] == b["conditional_source_policy"]
        max_score_difference = max(max_score_difference, abs(a["score"] - b["score"]))
        overlap += 1
    assert overlap == 2 * len(SEEDS[config["stage"]])

    peaks = dict(energy=0., momentum=0., reflection=0., matched_energy=0.)
    counts = dict(posthoc_value_calls=0, posthoc_gradient_calls=0, saved_observations_checked=0, wall_events_checked=0)
    ledgers, features, views, common = [], [], [], []
    for c in sorted(contexts, key=lambda item: item["context_id"]):
        context_audit(c)
        counts["posthoc_value_calls"] += 3
        branches = run_groups[c["context_id"]]
        assert set(branches) == set(MODES)
        raw = {}
        for mode, run in branches.items():
            assert (run["condition"], run["seed"], run["score_target"]) == (c["condition"], c["seed"], c["score_target"])
            raw[mode], ledger = trace_audit(root, c, run, counts, peaks, config)
            ledgers.append(ledger)
        difference = abs(branches["keep"]["initial_energy"] - branches["transfer"]["initial_energy"])
        assert difference < 1e-12
        peaks["matched_energy"] = max(peaks["matched_energy"], difference)
        features.append(feature_row(root, c, branches, raw["keep"]))
        last = min(len(a[2]) for a in raw.values())
        for mode in (*MODES, "conditional"):
            source = c["conditional_source_policy"] if mode == "conditional" else mode
            row = {**branches[source], "policy": mode, "source_policy": source, "derived": mode == "conditional"}
            views.append(row)
            values, positions, times, calls = raw[source]
            common.append({**row, **raw_metrics(values[:last], positions[:last], times[:last], c),
                           "physical_time": float(times[last-1]), "charged_calls_at_view": int(calls[last-1])})
    create(root / "conditional.json", [r for r in views if r["derived"]])
    create(root / "common_time.json", common)
    create(root / "physical_ledgers.json", {"new_oracle_calls_for_export": 0, "runs": ledgers})
    create(root / "features.json", features)
    create(root / "summary.json", {"same_budget": summarize(views), "common_time": summarize(common),
                                   "paired_actions": paired_summary(features, views)})
    if config["stage"] == "development":
        create(root / "feature_directions.json", compare_features(features))
    if config["stage"] == "formal":
        development = read(Path(config["previous_stage"]) / "feature_directions.json")
        create(root / "heldout_feature_results.json", compare_features(features, development))
    result = {
        "passed": True, "contexts_checked": len(contexts), "trajectories_checked": len(runs),
        "matched_score_pairs": overlap, "max_matched_score_difference": max_score_difference,
        "max_residuals": peaks, **counts, "solver_failures": [r["run_id"] for r in runs if r["failed"]],
    }
    create(root / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

"""Audit E25 raw states, then derive the locked policy and paired results."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_initial_budget import raw_metrics
from finalize_initial_budget import create, sha
from run_conditional_transfer import MODES, SEEDS, read

POLICIES = (*MODES, "conditional")


def trace_audit(root, context, run, audit_counts, peaks, config):
    key = run["run_id"]
    initial = read(root / "initial_states" / (key + ".json"))
    with gzip.open(root / "traces" / (key + ".csv.gz"), "rt") as stream:
        rows = list(csv.DictReader(stream))
    q = np.array([json.loads(r["positions"]) for r in rows])
    v = np.array([json.loads(r["velocities"]) for r in rows])
    loss = np.array([json.loads(r["values"]) for r in rows])
    times = np.array([float(r["time"]) for r in rows])
    calls = np.array([int(r["total_oracle_calls"]) for r in rows])
    rotation = np.array(context["rotation"])
    coords = q[:, :, :2] @ rotation.T

    def objective(x):
        return .25 * (x[..., 0] ** 2 - 1) ** 2 + .05 * x[..., 0] + 4 * x[..., 1] ** 2

    true_loss = objective(coords)
    audit_counts["posthoc_value_calls"] += true_loss.size
    assert np.max(abs(true_loss - loss)) < 1e-9 and np.min(q[:, :, 2] - loss) > -1e-8
    assert np.array_equal(q[0], initial["positions"]) and np.array_equal(v[0], initial["velocities"])
    assert np.array_equal(q[0, :, :2], context["points"])
    assert np.max(abs(loss[0] - context["values"])) < 1e-12
    j = int(np.argmin(context["values"]))
    heights = np.full(4, .8 + (.8 - .025) / 3)
    heights[j] = .025
    assert j == run["selected_index"] and np.max(abs(q[0, :, 2] - loss[0] - heights)) < 1e-12
    base = np.c_[context["initial_horizontal_velocities"], np.zeros(4)]
    expected = base.copy()
    if run["policy"] == "transfer":
        others = np.arange(4) != j
        centered = base[others] + base[j] / 3
        expected[others] = centered * np.sqrt(np.sum(base**2) / np.sum(centered**2))
        expected[j] = 0
    assert np.max(abs(v[0] - expected)) < 1e-14
    angular_before = np.cross(q[0], base).sum(axis=0)
    angular_after = np.cross(q[0], v[0]).sum(axis=0)
    assert np.max(abs(angular_after - angular_before -
                      initial["allocation"]["angular_change_at_same_positions"])) < 1e-12

    kinetic = .5 * np.sum(v * v, axis=(1, 2))
    momentum = v.sum(axis=1)
    energy = kinetic + .5 * q[:, :, 2].sum(axis=1)
    assert abs(kinetic[0] - .6) < 1e-12 and np.linalg.norm(momentum[0]) < 1e-12
    assert abs(energy[0] - run["initial_energy"]) < 1e-12
    assert np.max(abs(energy - [float(r["energy"]) for r in rows])) < 1e-9
    assert calls[0] == 8 and np.all(np.diff(calls) > 0)
    assert calls[-1] <= run["total_oracle_calls"] <= config["query_cap"]
    assert np.array_equal(times, np.arange(len(rows)) * config["h"])
    assert len(rows) - 1 <= config["max_steps"]
    if config["stage"] == "formal" and not run["failed"]:
        assert run["total_oracle_calls"] == config["query_cap"]
    assert run["total_oracle_calls"] == 4 + run["actual_branch_calls"]
    for r, call_count in zip(rows, calls):
        assert call_count == sum(int(r[k]) for k in ("value_calls", "gradient_calls", "path_calls"))

    events = read(root / "events" / (key + ".json"))
    wall_impulses = np.zeros((len(rows), 3))
    wall_work = 0.
    for event in events:
        assert event["kind"] == "wall"
        point = np.array(event["point_i"])
        xy = rotation @ point[:2]
        normal = np.r_[-rotation.T @ np.array([xy[0]**3 - xy[0] + .05, 8 * xy[1]]), 1.]
        normal /= np.linalg.norm(normal)
        before, after = np.array(event["before_i"]), np.array(event["after_i"])
        error = max(np.max(abs(normal - event["normal"])), np.max(abs(after - before + 2 * (before @ normal) * normal)))
        peaks["reflection"] = max(peaks["reflection"], float(error))
        assert error < 1e-9 and abs(point[2] - objective(xy)) < 1e-8
        audit_counts["posthoc_value_calls"] += 1
        audit_counts["posthoc_gradient_calls"] += 1
        wall_impulses[event["step"]] += after - before
        wall_work += .5 * float(after @ after - before @ before)
    audit_counts["wall_events_checked"] += len(events)
    audit_counts["saved_observations_checked"] += len(rows)

    time_step = config["h"]
    base_rate = -np.log(.995) / time_step
    rate = base_rate + (.5 - base_rate) * np.minimum((times[1:] - time_step / 2) / 25, 1)
    beta = np.exp(-rate * time_step)
    assert np.allclose(beta, [float(r["drag_beta"]) for r in rows[1:]], rtol=0, atol=1e-14)
    damping_impulses = (beta[:, None] - 1) * momentum[:-1]
    gravity_impulse_per_step = np.array([0., 0., -.1])
    momentum_error = np.diff(momentum, axis=0) - damping_impulses - wall_impulses[1:] - gravity_impulse_per_step
    damping_energy = (beta**2 - 1) * kinetic[:-1]
    assert np.allclose(damping_energy, [float(r["damping_energy"]) for r in rows[1:]], rtol=0, atol=1e-9)
    impulse_energy = np.array([float(r["impulse_energy"]) for r in rows[1:]])
    energy_error = energy[1:] - energy[0] - np.cumsum(damping_energy + impulse_energy)
    pe = float(np.max(abs(momentum_error), initial=0))
    ee = float(np.max(abs(energy_error), initial=0))
    assert pe < 1e-9 and ee < 1e-8
    peaks["energy"] = max(peaks["energy"], ee)
    peaks["momentum"] = max(peaks["momentum"], pe)
    gravity_work = -.5 * float(np.sum(q[-1, :, 2] - q[0, :, 2]))
    assert abs(kinetic[-1] - kinetic[0] - gravity_work - damping_energy.sum() - impulse_energy.sum()) < 1e-8
    assert abs(wall_work - impulse_energy.sum()) < 1e-8
    ledger = {
        "run_id": key, "initial_energy": float(energy[0]), "final_energy": float(energy[-1]),
        "initial_kinetic": float(kinetic[0]), "final_kinetic": float(kinetic[-1]),
        "linear_P_before_initial_allocation": base.sum(axis=0).tolist(),
        "linear_P_after_initial_allocation": momentum[0].tolist(),
        "angular_P_before_initial_allocation": angular_before.tolist(),
        "angular_P_after_initial_allocation": angular_after.tolist(),
        "angular_P_change": (angular_after - angular_before).tolist(),
        "gravity_work": gravity_work, "wall_work": wall_work,
        "damping_heat": -float(damping_energy.sum()),
        "gravity_impulse": (gravity_impulse_per_step * (len(rows) - 1)).tolist(),
        "wall_impulse": wall_impulses.sum(axis=0).tolist(),
        "damping_impulse": damping_impulses.sum(axis=0).tolist(),
        "max_energy_residual": ee, "max_linear_momentum_component_residual": pe,
        "runtime_external_control_supply": 0., "binary_interventions": 0,
        "initial_transfer": run["policy"] == "transfer", "committed_steps": len(rows) - 1,
        "charged_uncommitted_queries": run["total_oracle_calls"] - int(calls[-1]),
    }
    metrics = raw_metrics(loss, q, times, context)
    for name, value in metrics.items():
        if isinstance(value, bool) or value is None:
            assert run[name] == value
        else:
            assert abs(run[name] - value) < 1e-8
    return (loss, q, times, calls), ledger


def summaries(rows):
    output = []
    for condition in ("good_present", "bad_only"):
        for policy in POLICIES:
            subset = [r for r in rows if r["condition"] == condition and r["policy"] == policy]
            successes = [r for r in subset if not r["failed"]]
            output.append({
                "condition": condition, "policy": policy, "n": len(subset),
                "failures": len(subset) - len(successes),
                "transfers": sum(r["source_policy"] == "transfer" for r in subset),
                "particle0_harmed": sum(r["particle0_harmed"] for r in subset),
                "particle0_harmed_or_failed": sum(r["particle0_harmed"] or r["failed"] for r in subset),
                "particle0_final_hits": sum(r["particle0_final_hit"] for r in successes),
                "final_hits": sum(r["final_hit"] for r in successes),
                "historical_hits_observed": sum(r["historical_hit"] for r in subset),
                "median_current_best_loss": median(r["final_value"] for r in successes) if successes else None,
                "median_historical_best_loss": median(r["best_value"] for r in successes) if successes else None,
                "median_particle0_max_rise": median(r["particle0_max_rise"] for r in successes) if successes else None,
            })
    return output


def compare(rows, summary):
    groups = {(r["condition"], r["policy"]): r for r in summary}
    good_keep, good_transfer, good_conditional = (groups["good_present", p] for p in POLICIES)
    bad_keep, bad_transfer, bad_conditional = (groups["bad_only", p] for p in POLICIES)
    checks = {
        "no_solver_failures": all(r["failures"] == 0 for r in summary),
        "baseline_bad_hits_at_least_five": bad_keep["final_hits"] >= 5,
        "baseline_good_deviations_differ_by_at_least_five":
            good_keep["particle0_harmed_or_failed"] - good_transfer["particle0_harmed_or_failed"] >= 5,
        "good_deviations_within_two_of_transfer":
            good_conditional["particle0_harmed_or_failed"] - good_transfer["particle0_harmed_or_failed"] <= 2,
        "good_final_hits_within_two_of_transfer":
            good_transfer["particle0_final_hits"] - good_conditional["particle0_final_hits"] <= 2,
        "bad_final_hits_within_two_of_keep": bad_keep["final_hits"] - bad_conditional["final_hits"] <= 2,
    }
    paired = []
    for condition, metric in (("good_present", "particle0_harmed"), ("good_present", "particle0_final_hit"),
                              ("bad_only", "final_hit"), ("bad_only", "historical_hit")):
        for reference in MODES:
            lookup = {(r["seed"], r["policy"]): r for r in rows if r["condition"] == condition}
            states = []
            for seed in sorted({seed for seed, _ in lookup}):
                pair = [lookup[seed, p] for p in ("conditional", reference)]
                states.append(tuple(bool(r[metric] and not r["failed"]) for r in pair))
            paired.append({
                "condition": condition, "metric": metric, "reference": reference, "pairs": len(states),
                "both": states.count((True, True)), "only_conditional": states.count((True, False)),
                "only_reference": states.count((False, True)), "neither": states.count((False, False)),
            })
    return {
        "checks": checks, "passed_descriptive_screen": all(checks.values()),
        "bad_final_hit_change_from_keep": bad_conditional["final_hits"] - bad_keep["final_hits"],
        "bad_final_hit_change_from_transfer": bad_conditional["final_hits"] - bad_transfer["final_hits"],
        "paired": paired,
    }


def audit(root):
    config = read(root / "config.json")
    runs = read(root / "runs.json")
    contexts = read(root / "contexts.json")
    counts = read(root / "execution_counts.json")
    development = read(root / "development.json")
    assert config["seeds"] == SEEDS[config["stage"]]
    expected = {(s, c) for s in config["seeds"] for c in config["conditions"]}
    assert len(contexts) == len(expected) and {(c["seed"], c["condition"]) for c in contexts} == expected
    assert len(runs) == 2 * len(contexts) == counts["trajectory_executions"]
    assert len({r["run_id"] for r in runs}) == len(runs)
    assert sha(root / "development.json") == config["development_sha256"]
    assert config["threshold"] == development["threshold"]
    assert sha(root / "source.zip") == config["source_archive_sha256"]
    with zipfile.ZipFile(root / "source.zip") as archive:
        import hashlib
        for name, digest in config["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    assert counts["shared_probe_calls"] == 4 * len(contexts)
    assert counts["setup_reference_values"] == 3 * len(contexts)
    assert counts["actual_branch_calls"] == sum(r["actual_branch_calls"] for r in runs)
    assert counts["actual_solver_queries"] == counts["actual_branch_calls"] + counts["shared_probe_calls"]
    assert counts["actual_queries_including_setup"] == counts["actual_solver_queries"] + counts["setup_reference_values"]
    assert sum(r["total_oracle_calls"] for r in runs) - counts["actual_solver_queries"] == 4 * len(contexts)
    peaks = dict(energy=0., momentum=0., reflection=0., matched_energy=0.)
    audit_counts = dict(posthoc_value_calls=0, posthoc_gradient_calls=0, saved_observations_checked=0, wall_events_checked=0)
    ledgers, conditional, common, all_policies = [], [], [], []
    for context in sorted(contexts, key=lambda c: c["context_id"]):
        key = context["context_id"]
        branch = {r["policy"]: r for r in runs if r["context_id"] == key}
        assert set(branch) == set(MODES)
        assert context["probe_counts"] == {"value_calls": 4, "gradient_calls": 0, "path_calls": 0}
        vals = np.sort(context["values"])
        score = (vals[2] - vals[0]) / max(vals[3] - vals[1], 1e-12)
        assert score == context["score"]
        chosen = "transfer" if score >= config["threshold"] else "keep"
        assert chosen == context["conditional_source_policy"]
        cache = {}
        for mode, row in branch.items():
            assert (row["seed"], row["condition"]) == (context["seed"], context["condition"])
            cache[mode], ledger = trace_audit(root, context, row, audit_counts, peaks, config)
            ledgers.append(ledger)
        energy_diff = abs(branch["keep"]["initial_energy"] - branch["transfer"]["initial_energy"])
        assert energy_diff < 1e-12
        peaks["matched_energy"] = max(peaks["matched_energy"], energy_diff)
        last = min(len(raw[2]) for raw in cache.values())
        for policy in POLICIES:
            mode = chosen if policy == "conditional" else policy
            row = {**branch[mode], "policy": policy, "source_policy": mode,
                   "derived_policy_result": policy == "conditional"}
            all_policies.append(row)
            if policy == "conditional":
                conditional.append(row)
            loss, q, times, calls = cache[mode]
            common.append({**row, **raw_metrics(loss[:last], q[:last], times[:last], context),
                           "physical_time": float(times[last - 1]), "charged_calls_at_view": int(calls[last - 1])})
    summary = summaries(all_policies)
    common_summary = summaries(common)
    create(root / "conditional.json", conditional)
    create(root / "common_time.json", common)
    create(root / "physical_ledgers.json", {"runs": ledgers, "new_oracle_calls_for_ledger_export": 0})
    create(root / "summary.json", {"same_budget": summary, "common_time": common_summary})
    if config["stage"] == "formal":
        create(root / "screen.json", compare(all_policies, summary))
        create(root / "common_time_screen.json", compare(common, common_summary))
    result = {
        "passed": True, "contexts_checked": len(contexts), "trajectories_checked": len(runs),
        "derived_policy_rows": len(conditional), "max_residuals": peaks, **audit_counts,
        "solver_failures": [r["run_id"] for r in runs if r["failed"]],
    }
    create(root / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

"""E29 audit of frozen alerts, matched continuations, and late cases."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_shared_prefix_transfer import load_segment, segment_audit
from finalize_initial_budget import create, sha
from run_warning_triggered_transfer import CAP, CHECKPOINT_STEP, END_STEP, H, HORIZON_STEP, LIMIT, MODES, read


def audit_map(root, key, checkpoint, branch, velocities, positions, peaks):
    before = np.array(checkpoint["velocities"])
    delta = velocities - before
    assert np.array_equal(checkpoint["positions"], positions)
    assert np.array_equal(before[:, 2], velocities[:, 2])
    info = read(root / "interventions" / (key + ".json"))
    assert np.array_equal(info["start"]["positions"], positions)
    assert np.array_equal(info["start"]["velocities"], velocities)
    assert np.array_equal(info["velocity_delta"], delta)
    assert info["damping_clock_at_start"] == CHECKPOINT_STEP and info["selected"] == 0
    if branch == "keep":
        assert np.array_equal(velocities, before)
    else:
        center = before[:, :2].mean(axis=0)
        relative = before[:, :2] - center
        centered = relative[1:] - relative[1:].mean(axis=0)
        expected = center + centered * np.sqrt(np.sum(relative**2) / np.sum(centered**2))
        assert np.max(abs(velocities[1:, :2] - expected)) < 1e-12
        assert np.max(abs(velocities[0, :2] - center)) < 1e-12
    linear = delta.sum(axis=0)
    angular = np.cross(positions, delta).sum(axis=0)
    kinetic = .5 * float(np.sum(velocities**2) - np.sum(before**2))
    assert abs(kinetic) < 1e-12 and np.max(abs(linear)) < 1e-12
    peaks["map_energy"] = max(peaks["map_energy"], abs(kinetic))
    peaks["map_linear_momentum"] = max(peaks["map_linear_momentum"], float(np.max(abs(linear))))
    return {
        "run_id": key, "policy": branch, "instantaneous_energy_change": kinetic,
        "linear_impulse": linear.tolist(), "angular_momentum_change": angular.tolist(),
        "particle_impulses": delta.tolist(), "horizontal_speed_before": float(np.linalg.norm(before[0, :2])),
        "horizontal_speed_after": float(np.linalg.norm(velocities[0, :2])),
        "external_supply": 0., "binary_collisions": 0,
    }


def branch_metrics(prefix, initial_loss, raw, run):
    _, _, loss, calls, events = raw
    rises = np.r_[prefix, loss[1:, 0] - initial_loss]
    times = H * np.arange(len(rises))
    first_bounce = next((CHECKPOINT_STEP * H + e["time"] for e in events if e["i"] == 0), None)
    crossings = np.flatnonzero(rises > LIMIT)
    first = int(crossings[0]) if crossings.size else None
    returned = next((float(times[i]) for i in range(first + 1, len(rises)) if rises[i] <= LIMIT),
                    None) if first is not None else None
    valid = len(rises) == END_STEP + 1 and not run["failed"]
    return {
        "run_id": run["run_id"], "valid_complete": valid, "end_time": float(times[-1]),
        "deviated_by_point30": bool(np.max(rises[:HORIZON_STEP+1]) > LIMIT) if len(rises) > HORIZON_STEP else None,
        "deviated_before_own_bounce": bool(np.any(rises[times < first_bounce] > LIMIT)) if first_bounce is not None else None,
        "deviated_by_end": bool(crossings.size) if valid else None,
        "first_deviation_time": float(times[first]) if first is not None else None,
        "first_own_bounce_time": first_bounce, "first_return_within_limit": returned,
        "peak_loss_rise": float(np.max(rises)), "last_loss_rise": float(rises[-1]),
        "within_limit_at_end": bool(rises[-1] <= LIMIT) if valid else None,
        "policy_calls": run["counts"]["total_oracle_calls"], "actual_suffix_calls": run["actual_suffix_calls"],
        "calls_through_point30": calls[HORIZON_STEP-CHECKPOINT_STEP] if len(rises) > HORIZON_STEP else None,
    }


def summarize(rows):
    valid = [r for r in rows if r["valid_complete"]]
    return {
        "n": len(rows), "complete": len(valid),
        "deviated_by_point30": sum(r["deviated_by_point30"] for r in valid),
        "deviated_before_own_bounce": sum(r["deviated_before_own_bounce"] for r in valid
                                        if r["deviated_before_own_bounce"] is not None),
        "known_own_bounce": sum(r["first_own_bounce_time"] is not None for r in valid),
        "deviated_by_end": sum(r["deviated_by_end"] for r in valid),
        "within_limit_at_end": sum(r["within_limit_at_end"] for r in valid),
        "median_peak_loss_rise": median(r["peak_loss_rise"] for r in valid) if valid else None,
        "median_last_loss_rise": median(r["last_loss_rise"] for r in valid) if valid else None,
    }


def paired(rows, metric):
    comparable = [r for r in rows if all(r["branches"][m]["valid_complete"] for m in MODES)]
    states = [(r["branches"]["keep"][metric], r["branches"]["transfer"][metric]) for r in comparable]
    return {
        "pairs": len(rows), "complete_pairs": len(states),
        "both_deviated": states.count((True, True)), "keep_only_deviated": states.count((True, False)),
        "transfer_only_deviated": states.count((False, True)), "neither_deviated": states.count((False, False)),
    }


def analyze(rows):
    groups = {}
    for name in ("alert", "already_deviated", "no_alert"):
        selected = [r for r in rows if r["status"] == name]
        groups[name] = {
            "contexts": len(selected), **{mode: summarize([r["branches"][mode] for r in selected
                                                          if mode in r["branches"]]) for mode in MODES},
        }
        if selected and all("transfer" in r["branches"] for r in selected):
            groups[name]["paired_point30"] = paired(selected, "deviated_by_point30")
            groups[name]["paired_end"] = paired(selected, "deviated_by_end")
    point = groups["alert"].get("paired_point30")
    end = groups["alert"].get("paired_end")
    checks = None
    if point and end:
        risks = point["both_deviated"] + point["keep_only_deviated"]
        checks = {
            "all_alert_pairs_complete": point["complete_pairs"] == point["pairs"],
            "at_least_twenty_control_deviations": risks >= 20,
            "prevented_at_least_90_percent_at_point30": bool(risks and point["keep_only_deviated"] / risks >= .90),
            "at_least_80_percent_never_deviated_through_end": (
                end["keep_only_deviated"] + end["neither_deviated"]) / max(1, end["pairs"]) >= .80,
            "no_transfer_only_deviations_by_point30": point["transfer_only_deviated"] == 0,
        }
    return {
        "groups": groups, "success_checks": checks, "passed_screen": bool(checks and all(checks.values())),
        "whole_cohort_keep": summarize([r["branches"]["keep"] for r in rows]),
        "whole_cohort_warning_policy": summarize([
            r["branches"]["transfer" if r["status"] == "alert" else "keep"] for r in rows
        ]),
    }


def audit(root):
    config, checkpoints, runs, counts = (read(root / (name + ".json"))
                                         for name in ("config", "checkpoints", "runs", "execution_counts"))
    formal = config["stage"] == "formal"
    parent = Path(config["parent"])
    assert len(checkpoints) == config["reused_parent_contexts"] == counts["reused_prefixes"] == (240 if formal else 6)
    assert len(runs) == counts["suffix_executions"] == (344 if formal else 12)
    assert len({r["run_id"] for r in runs}) == len(runs)
    assert counts["new_prefix_executions"] == counts["actual_new_setup_queries"] == 0
    assert counts["historical_prefix_queries"] == 32 * len(checkpoints)
    assert sum(r["actual_suffix_calls"] for r in runs) == counts["actual_new_solver_queries"]
    assert counts["charged_policy_queries"] == counts["actual_new_solver_queries"] + 32 * len(runs)
    assert sha(root / "source.zip") == config["source_archive_sha256"]
    for file, digest in config["dependency_data_sha256"].items():
        assert sha(Path(file)) == digest
    with zipfile.ZipFile(root / "source.zip") as archive:
        for name, digest in config["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    contexts = {c["context_id"]: c for c in read(parent / "contexts.json")}
    labels = {c["run_id"]: c for c in read(parent / "labeled_observations.json")}
    predictions = {r["run_id"]: r for r in read(parent / "results.json")["predictions"]} if formal else {}
    parent_manifest = {a["path"]: a["sha256"] for a in read(parent / "raw_artifact_manifest.json")["files"]}
    by_context = defaultdict(dict)
    for run in runs:
        by_context[run["context_id"]][run["policy"]] = run

    totals = Counter()
    peaks = dict(energy=0., momentum=0., reflection=0., map_energy=0., map_linear_momentum=0., parent_overlap=0.)
    outputs, ledgers, transfers = [], [], []
    for c in checkpoints:
        key = c["context_id"]
        parent_rows, pq, pv, pf = load_segment(parent, key)
        p_trace, p_events = (parent / folder / (key + extension)
                             for folder, extension in (("traces", ".csv.gz"), ("events", ".json")))
        assert sha(p_trace) == c["source_parent_trace_sha256"] == parent_manifest[str(p_trace)]
        assert sha(p_events) == c["source_parent_event_sha256"] == parent_manifest[str(p_events)]
        prefix = pf[:CHECKPOINT_STEP+1, 0] - pf[0, 0]
        assert np.array_equal(prefix, c["prefix_loss_rises"])
        assert c["initial_loss"] == pf[0, 0]
        assert np.array_equal(c["checkpoint"]["positions"], pq[CHECKPOINT_STEP])
        assert np.array_equal(c["checkpoint"]["velocities"], pv[CHECKPOINT_STEP])
        assert c["prefix_calls"] == int(parent_rows[CHECKPOINT_STEP]["total_oracle_calls"]) == 32
        status = ("already_deviated" if max(prefix) > LIMIT else
                  ("alert" if c["risk"] > LIMIT else "no_alert"))
        assert status == c["status"]
        if formal:
            label = labels[key]
            assert label["valid"] and c["risk"] == label["scores"]["quadratic_step2"]
            assert (status == "alert") == predictions[key]["alert"]["quadratic_step2"]
            assert (status == "already_deviated") == (label["positive"] and label["first_deviation"] <= CHECKPOINT_STEP * H)
        assert set(by_context[key]) == ({"keep"} if formal and status == "no_alert" else set(MODES))
        output = {k: c[k] for k in ("context_id", "seed", "status", "incumbent_energy", "risk", "prefix_loss_rises")}
        output["branches"] = {}
        for mode, run in by_context[key].items():
            raw, ledger = segment_audit(root, run, contexts[key], totals, peaks, start_step=CHECKPOINT_STEP)
            ledgers.append(ledger)
            assert raw[3][0] == c["prefix_calls"] + 4
            assert np.array_equal(raw[2][0], pf[CHECKPOINT_STEP])
            assert run["counts"]["total_oracle_calls"] == run["actual_suffix_calls"] + c["prefix_calls"]
            assert run["counts"]["total_oracle_calls"] <= CAP and run["committed_steps"] <= END_STEP - CHECKPOINT_STEP
            transfers.append(audit_map(root, run["run_id"], c["checkpoint"], mode, raw[1][0], raw[0][0], peaks))
            metrics = branch_metrics(prefix, c["initial_loss"], raw, run)
            if mode == "keep":
                overlap = min(len(pf) - CHECKPOINT_STEP, len(raw[2]))
                assert overlap == len(pf) - CHECKPOINT_STEP
                error = max(float(np.max(abs(p[CHECKPOINT_STEP:CHECKPOINT_STEP+overlap] - r[:overlap])))
                            for p, r in zip((pq, pv, pf), raw[:3]))
                assert error < 1e-12
                peaks["parent_overlap"] = max(peaks["parent_overlap"], error)
                for i, count in enumerate(raw[3][:overlap]):
                    assert count == int(parent_rows[CHECKPOINT_STEP + i]["total_oracle_calls"]) + 4
                assert metrics["deviated_before_own_bounce"] == labels[key]["positive"]
            output["branches"][mode] = metrics
        outputs.append(output)
    group_counts = Counter(x["status"] for x in outputs)
    if formal:
        assert group_counts == {"alert": 102, "already_deviated": 2, "no_alert": 136}
        assert {x["seed"] for x in outputs if x["status"] == "already_deviated"} == {2249, 2266}
        create(root / "results.json", analyze(outputs))
    create(root / "paired_outcomes.json", outputs)
    create(root / "already_deviated_cases.json", [x for x in outputs if x["status"] == "already_deviated"])
    create(root / "physical_ledgers.json", {"segments": ledgers, "intervention_maps": transfers,
                                            "new_oracle_calls_for_export": 0})
    result = {
        "passed": True, "checkpoints_checked": len(checkpoints), "continuations_checked": len(runs),
        "group_counts": dict(group_counts), "solver_failures": [r["run_id"] for r in runs if r["failed"]],
        "incomplete": [r["run_id"] for r in runs if r["committed_steps"] != END_STEP - CHECKPOINT_STEP],
        "max_residuals": peaks, **totals,
    }
    create(root / "audit.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

"""Verify E30 forecasting, matched states, prevention, and recovery."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_shared_prefix_transfer import load_segment, segment_audit
from finalize_initial_budget import create, sha
from run_postbounce_warning_transfer import CAP, END_STEP, H, LIMIT, MODELS, MODES, PARENT_STEP, read


def map_audit(root, key, mode, c, q, v, peaks):
    before = np.array(c["checkpoint"]["velocities"])
    delta = v - before
    assert np.array_equal(q, c["checkpoint"]["positions"])
    assert np.array_equal(v[:, 2], before[:, 2])
    record = read(root / "interventions" / (key + ".json"))
    assert np.array_equal(record["velocity_delta"], delta)
    assert record["damping_clock_at_start"] == c["checkpoint_step"] and record["selected"] == 0
    assert np.array_equal(record["start"]["positions"], q) and np.array_equal(record["start"]["velocities"], v)
    center = before[:, :2].mean(axis=0)
    if mode == "keep":
        assert np.array_equal(v, before)
    else:
        relative = before[:, :2] - center
        centered = relative[1:] - relative[1:].mean(axis=0)
        expected = center + centered * np.sqrt(np.sum(relative**2) / np.sum(centered**2))
        assert np.max(abs(v[1:, :2] - expected)) < 1e-12 and np.max(abs(v[0, :2] - center)) < 1e-12
    kinetic = .5 * float(np.sum(v**2) - np.sum(before**2))
    linear = delta.sum(axis=0)
    assert abs(kinetic) < 1e-12 and np.max(abs(linear)) < 1e-12
    peaks["map_energy"] = max(peaks["map_energy"], abs(kinetic))
    peaks["map_linear_momentum"] = max(peaks["map_linear_momentum"], float(np.max(abs(linear))))
    return {
        "run_id": key, "policy": mode, "instantaneous_energy_change": kinetic,
        "linear_impulse": linear.tolist(), "angular_momentum_change": np.cross(q, delta).sum(axis=0).tolist(),
        "particle_impulses": delta.tolist(), "horizontal_speed_after": float(np.linalg.norm(v[0, :2])),
        "external_supply": 0., "binary_collisions": 0,
    }


def metrics(c, f, run):
    rises = np.r_[c["prefix_rises"], f[1:, 0] - c["initial_loss"]]
    crossed = np.flatnonzero(rises > LIMIT)
    first = int(crossed[0]) if crossed.size else None
    start = c["checkpoint_step"]
    returned = next((i for i in range(start + 1, len(rises)) if rises[i] <= LIMIT),
                    None) if first is not None and first <= start else None
    complete = len(rises) == END_STEP + 1 and not run["failed"]
    return {
        "run_id": run["run_id"], "complete": complete, "deviated": bool(crossed.size) if complete else None,
        "first_deviation": first * H if first is not None else None,
        "first_return_after_checkpoint": returned * H if returned is not None else None,
        "peak_rise": float(np.max(rises)), "last_rise": float(rises[-1]),
        "within_limit_at_end": bool(rises[-1] <= LIMIT) if complete else None,
        "policy_calls": run["counts"]["total_oracle_calls"], "actual_suffix_calls": run["actual_suffix_calls"],
    }


def summarize(branches):
    valid = [b for b in branches if b["complete"]]
    return {
        "n": len(branches), "complete": len(valid), "deviated": sum(b["deviated"] for b in valid),
        "within_limit_at_end": sum(b["within_limit_at_end"] for b in valid),
        "returned_at_least_once_after_checkpoint": sum(b["first_return_after_checkpoint"] is not None for b in valid),
        "median_peak_rise": median(b["peak_rise"] for b in valid) if valid else None,
        "median_last_rise": median(b["last_rise"] for b in valid) if valid else None,
    }


def paired(rows):
    comparisons = [(r["branches"]["keep"]["deviated"], r["branches"]["transfer"]["deviated"])
                   for r in rows if all(b["complete"] for b in r["branches"].values())]
    return {
        "pairs": len(rows), "complete_pairs": len(comparisons),
        "both_deviated": comparisons.count((True, True)), "keep_only_deviated": comparisons.count((True, False)),
        "transfer_only_deviated": comparisons.count((False, True)), "neither_deviated": comparisons.count((False, False)),
        **{mode: summarize([r["branches"][mode] for r in rows]) for mode in MODES},
    }


def evaluate(rows):
    late = [r for r in rows if r["already_deviated"]]
    available = [r for r in rows if not r["already_deviated"] and all(b["complete"] for b in r["branches"].values())]
    forecasts = {}
    for name in MODELS:
        positive = [r for r in available if r["branches"]["keep"]["deviated"]]
        negative = [r for r in available if not r["branches"]["keep"]["deviated"]]
        timely = [r for r in positive if r["warnings"][name]]
        false = [r for r in negative if r["warnings"][name]]
        leads = [r["branches"]["keep"]["first_deviation"] - r["checkpoint_time"] for r in timely]
        forecasts[name] = {
            "available": len(available), "positives": len(positive), "negatives": len(negative),
            "timely_alerts": len(timely), "missed": len(positive) - len(timely), "false_alerts": len(false),
            "median_lead": median(leads) if leads else None, "lead_range": [min(leads), max(leads)] if leads else None,
            "paired_alerts": paired(timely), "paired_false_alerts": paired(false),
            "paired_missed": paired([r for r in positive if not r["warnings"][name]]),
            "actual_predictor_queries": 0 if name == "normal_quadratic" else (
                len(available) if name == "one_path" else sum(sum(r["forecast_calls"].values()) for r in available)),
            "cohort_warning_policy": summarize([
                r["branches"]["transfer" if not r["already_deviated"] and r["warnings"][name] else "keep"]
                for r in rows
            ]),
        }
    full = forecasts["through_next_reflection"]
    p, n, hits, fp = (full[k] for k in ("positives", "negatives", "timely_alerts", "false_alerts"))
    checks = {
        "no_incomplete_pairs": len(available) + len(late) == len(rows)
                               and all(b["complete"] for r in late for b in r["branches"].values()),
        "forecast_completed": all(r["forecast_complete"] for r in available),
        "recall_at_least_90_percent": bool(p and hits / p >= .90),
        "false_alert_rate_at_most_10_percent": bool(n and fp / n <= .10),
        "median_lead_at_least_one_step": full["median_lead"] is not None and full["median_lead"] >= H - 1e-12,
        "transfer_prevents_at_least_90_percent_of_timely_positive_alerts":
            bool(hits and full["paired_alerts"]["keep_only_deviated"] / hits >= .90),
        "no_transfer_only_deviation_among_false_alerts": full["paired_false_alerts"]["transfer_only_deviated"] == 0,
    }
    return {
        "forecasts": forecasts, "already_deviated": paired(late),
        "late_at_exact_contact": sum(r["first_contact_loss_rise"] > LIMIT for r in late),
        "no_alert_cohort_always_keep": summarize([r["branches"]["keep"] for r in rows]),
        "no_alert_cohort_transfer_when_still_within_limit": summarize([
            r["branches"]["keep" if r["already_deviated"] else "transfer"] for r in rows]),
        "checks": checks, "passed_screen": all(checks.values()),
    }


def audit(root):
    config, checkpoints, runs, counts = (read(root / (name + ".json"))
                                         for name in ("config", "checkpoints", "runs", "execution_counts"))
    formal = config["stage"] == "formal"
    parent = Path(config["parent"])
    assert len(checkpoints) == counts["reused_prefixes"] == (136 if formal else 4)
    assert len(runs) == counts["suffix_executions"] == 2 * len(checkpoints)
    assert counts["new_prefix_executions"] == config["new_independent_seed_groups"] == 0
    assert len({r["run_id"] for r in runs}) == len(runs)
    assert sha(root / "source.zip") == config["source_archive_sha256"]
    for path, digest in config["dependency_data_sha256"].items():
        assert sha(Path(path)) == digest
    with zipfile.ZipFile(root / "source.zip") as archive:
        for name, digest in config["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    original = Path(read(parent / "config.json")["parent"])
    contexts = {c["context_id"]: c for c in read(original / "contexts.json")}
    older = {c["context_id"]: c for c in read(parent / "paired_outcomes.json") if c["status"] == "no_alert"}
    initial = {c["context_id"]: c for c in read(parent / "checkpoints.json") if c["status"] == "no_alert"}
    parent_hashes = {a["path"]: a["sha256"] for a in read(parent / "raw_artifact_manifest.json")["files"]}
    assert {c["context_id"] for c in checkpoints} == older.keys() == initial.keys()
    forecast_cost = sum(sum(c["forecast_calls"].values()) for c in checkpoints)
    suffix_cost = sum(r["actual_suffix_calls"] for r in runs)
    assert forecast_cost == counts["actual_forecast_queries"] == counts["forecast_path_queries"] + counts["forecast_gradient_queries"]
    assert suffix_cost == counts["actual_suffix_queries"]
    assert counts["actual_new_solver_queries_including_forecast"] == forecast_cost + suffix_cost
    assert counts["historical_prefix_queries"] == sum(c["historical_prefix_calls"] for c in checkpoints)
    assert counts["charged_policy_queries"] == suffix_cost + 2 * (counts["historical_prefix_queries"] + forecast_cost)
    totals = Counter()
    peaks = dict(energy=0., momentum=0., reflection=0., map_energy=0., map_linear_momentum=0., parent_overlap=0.,
                 predicted_loss=0., predicted_next_contact_time=0.)
    records, ledgers, maps = [], [], []
    run_groups = defaultdict(dict)
    for r in runs:
        run_groups[r["context_id"]][r["policy"]] = r
    for c in checkpoints:
        key = c["context_id"]
        source = c["source_keep_run"]
        trace = parent / "traces" / (source + ".csv.gz")
        events = parent / "events" / (source + ".json")
        assert sha(trace) == c["source_trace_sha256"] == parent_hashes[str(trace)]
        assert sha(events) == c["source_event_sha256"] == parent_hashes[str(events)]
        old_rows, old_q, old_v, old_f = load_segment(parent, source)
        contacts = [e for e in read(events) if e["i"] == 0]
        time = contacts[0]["time"] + PARENT_STEP * H
        step = c["checkpoint_step"]
        idx = step - PARENT_STEP
        assert step == 7 and (step - 1) * H <= time < step * H
        assert c["first_contact_time"] == time and c["checkpoint_time"] == step * H
        assert c["initial_loss"] == initial[key]["initial_loss"]
        assert c["first_contact_loss_rise"] == contacts[0]["point_i"][2] - c["initial_loss"]
        assert c["historical_prefix_calls"] == int(old_rows[idx]["total_oracle_calls"])
        assert np.array_equal(c["checkpoint"]["positions"], old_q[idx]) and np.array_equal(c["checkpoint"]["velocities"], old_v[idx])
        prefix = np.r_[initial[key]["prefix_loss_rises"], old_f[1:idx+1, 0] - c["initial_loss"]]
        assert np.array_equal(c["prefix_rises"], prefix)
        assert c["already_deviated"] == (np.max(prefix) > LIMIT)
        forecast = c["forecast"]
        cost = sum(c["forecast_calls"].values())
        assert c["charged_at_branch"] == c["historical_prefix_calls"] + cost
        if forecast is None:
            assert c["already_deviated"] and cost == 0
        else:
            assert not c["already_deviated"] and cost == (1 if forecast["next_contact"] is None else 3)
            assert c["forecast_calls"]["value_calls"] == 0
            if forecast["next_contact"] is not None:
                error = abs(forecast["next_contact"] - contacts[1]["time"] - PARENT_STEP * H)
                assert error < 1e-9
                peaks["predicted_next_contact_time"] = max(peaks["predicted_next_contact_time"], error)
            obs = forecast["observations"]
            for point in obs:
                error = abs(point["loss_rise"] - (old_f[point["step"]-PARENT_STEP, 0] - c["initial_loss"]))
                assert error < 1e-9
                peaks["predicted_loss"] = max(peaks["predicted_loss"], error)
            assert c["warnings"] == {
                "normal_quadratic": forecast["normal_quadratic_score"] > LIMIT,
                "one_path": any(x["loss_rise"] > LIMIT and not x["after_next_contact"] for x in obs),
                "through_next_reflection": any(x["loss_rise"] > LIMIT for x in obs),
            }
        r = {k: c[k] for k in ("context_id", "seed", "incumbent_energy", "already_deviated", "checkpoint_time",
                                "first_contact_loss_rise", "warnings", "forecast_calls")}
        r.update(checkpoint_loss_rise=float(prefix[-1]), forecast_complete=forecast["complete"] if forecast else None,
                 branches={})
        assert set(run_groups[key]) == set(MODES)
        for mode, run in run_groups[key].items():
            raw, ledger = segment_audit(root, run, contexts[key], totals, peaks, start_step=step)
            ledgers.append(ledger)
            maps.append(map_audit(root, run["run_id"], mode, c, raw[0][0], raw[1][0], peaks))
            assert np.array_equal(raw[2][0], old_f[idx])
            assert raw[3][0] == c["charged_at_branch"] + 4
            assert run["counts"]["total_oracle_calls"] == run["actual_suffix_calls"] + c["charged_at_branch"] <= CAP
            if mode == "keep":
                n = min(len(old_f) - idx, len(raw[2]))
                error = max(float(np.max(abs(p[idx:idx+n] - q[:n]))) for p, q in zip((old_q, old_v, old_f), raw[:3]))
                assert error < 1e-12
                peaks["parent_overlap"] = max(peaks["parent_overlap"], error)
                for i, query in enumerate(raw[3][:n]):
                    assert query == int(old_rows[idx+i]["total_oracle_calls"]) + cost + 4
            r["branches"][mode] = metrics(c, raw[2], run)
        keep = r["branches"]["keep"]
        if keep["complete"]:
            assert keep["deviated"] == older[key]["branches"]["keep"]["deviated_by_end"]
            assert keep["first_deviation"] == older[key]["branches"]["keep"]["first_deviation_time"]
        records.append(r)
    late = [r for r in records if r["already_deviated"]]
    create(root / "paired_outcomes.json", records)
    create(root / "already_deviated_cases.json", late)
    create(root / "physical_ledgers.json", {"segments": ledgers, "intervention_maps": maps, "new_oracle_calls_for_export": 0})
    if formal:
        assert len(late) == 10 and sum(r["branches"]["keep"]["deviated"] for r in records) == 76
        create(root / "results.json", evaluate(records))
    audit_result = {
        "passed": True, "checkpoints_checked": len(checkpoints), "continuations_checked": len(runs),
        "already_deviated": len(late), "incomplete_forecasts": sum(r["forecast_complete"] is False for r in records),
        "solver_failures": [r["run_id"] for r in runs if r["failed"]],
        "incomplete_continuations": [b["run_id"] for r in records for b in r["branches"].values() if not b["complete"]],
        "max_residuals": peaks, **totals,
    }
    create(root / "audit.json", audit_result)
    print(audit_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

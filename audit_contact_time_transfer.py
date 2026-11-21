"""Audit substep replay, warning timing, prevention and late recovery."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_shared_prefix_transfer import load_segment, segment_audit
from finalize_initial_budget import create, sha
from run_contact_time_transfer import CAP, END_STEP, H, LIMIT, MODES, PARENT_STEP, read


def physical(q, v):
    kinetic = .5 * float(np.sum(v**2))
    return kinetic, kinetic + .5 * float(q[:, 2].sum()), v.sum(axis=0)


def audit_contact(c, parent_event, parent_q, parent_v, rotation, peaks):
    step_before = c["previous_step"]
    idx = step_before - PARENT_STEP
    q0, v0 = parent_q[idx], parent_v[idx]
    assert np.array_equal(c["before_step"]["positions"], q0) and np.array_equal(c["before_step"]["velocities"], v0)
    contact = c["contact"]
    q1 = np.array(c["post_reflection"]["positions"])
    v1 = np.array(c["post_reflection"]["velocities"])
    assert contact["index"] == parent_event["i"] == 0
    offset = contact["offset"]
    assert c["remaining_duration"] == H - offset
    for a, b in ((q1[0], parent_event["point_i"]), (v1[0], parent_event["after_i"]),
                 (contact["incoming_velocity"], parent_event["before_i"]), (contact["normal"], parent_event["normal"])):
        assert np.max(abs(np.array(a) - b)) < 1e-12
    time_error = abs(c["contact_time"] - (PARENT_STEP*H + parent_event["time"]))
    assert time_error < 1e-12
    peaks["parent_event_time"] = max(peaks["parent_event_time"], time_error)
    xy = np.array(rotation) @ q1[0, :2]
    loss = .25 * (xy[0]**2 - 1)**2 + .05 * xy[0] + 4 * xy[1]**2
    assert abs(loss - contact["contact_value"]) < 1e-12
    assert c["contact_rise"] == contact["contact_value"] - c["initial_loss"]
    assert c["already_deviated"] == (c["contact_rise"] > LIMIT)
    predicted = c["predicted_boundary_rise"]
    assert c["warning"] == (predicted is not None and predicted > LIMIT)
    assert c["actual_event_reconstruction_calls"] == 10
    assert c["actual_forecast_calls"] == int(not c["already_deviated"])
    assert c["charged_at_branch"] == c["historical_prefix_calls"] + 10 + c["actual_forecast_calls"]
    h = H
    base = -np.log(.995) / h
    rate = base + (.5 - base) * min((step_before+.5) * h / 25, 1)
    beta = float(np.exp(-rate * h))
    assert abs(contact["damping_beta"] - beta) < 1e-14
    damped = beta * v0
    incoming = damped.copy()
    incoming[:, 2] -= .5 * offset
    q_expected = q0 + offset * damped
    q_expected[:, 2] -= .25 * offset**2
    assert np.max(abs(q_expected - q1)) < 1e-12
    assert np.max(abs(incoming[0] - contact["incoming_velocity"])) < 1e-12
    assert np.max(abs(incoming[1:] - v1[1:])) < 1e-12
    k0, e0, p0 = physical(q0, v0)
    k1, e1, p1 = physical(q1, v1)
    err_e = abs(e1 - e0 - contact["damping_energy"] - contact["terrain_work"])
    err_p = float(np.max(abs(p1 - p0 - np.array(contact["damping_impulse"])
                             - contact["gravity_impulse"] - contact["terrain_impulse"])))
    assert err_e < 1e-12 and err_p < 1e-12
    assert abs(k1 - k0 - contact["gravity_work"] - contact["damping_energy"] - contact["terrain_work"]) < 1e-12
    peaks["event_energy"] = max(peaks["event_energy"], err_e)
    peaks["event_momentum"] = max(peaks["event_momentum"], err_p)


def audit_fraction(frac, c, mode, peaks):
    q = np.array(frac["start"]["positions"])
    v = np.array(frac["start"]["velocities"])
    old_v = np.array(c["post_reflection"]["velocities"])
    assert np.array_equal(c["post_reflection"]["positions"], q)
    assert np.array_equal(v[:, 2], old_v[:, 2])
    if mode == "keep":
        assert np.array_equal(v, old_v)
    else:
        center = old_v[:, :2].mean(axis=0)
        relative = old_v[:, :2] - center
        rest = relative[1:] - relative[1:].mean(axis=0)
        expected = center + rest * np.sqrt(np.sum(relative**2) / np.sum(rest**2))
        assert np.max(abs(v[0, :2] - center)) < 1e-12 and np.max(abs(v[1:, :2] - expected)) < 1e-12
    delta = v - old_v
    k_before, e_before, p_before = physical(q, old_v)
    k0, e0, p0 = physical(q, v)
    energy = abs(e0 - e_before)
    momentum = float(np.max(abs(p0 - p_before)))
    assert energy < 1e-12 and momentum < 1e-12
    peaks["map_energy"] = max(peaks["map_energy"], energy)
    peaks["map_momentum"] = max(peaks["map_momentum"], momentum)
    assert frac["duration"] == c["remaining_duration"] and frac["damping_beta"] == 1.
    diag = frac["diagnostics"]
    assert diag["damping_energy"] == 0. and np.array_equal(diag["damping_impulse"], np.zeros(3))
    assert not frac["events"]
    dt = frac["duration"]
    gravity = np.array([0., 0., -.5])
    end_q, end_v = (np.array(frac["end"][k]) for k in ("positions", "velocities"))
    assert np.max(abs(end_q - q - dt*v - .5*dt**2*gravity)) < 1e-12
    assert np.max(abs(end_v - v - dt*gravity)) < 1e-12
    k1, e1, p1 = physical(end_q, end_v)
    err_e = abs(e1 - e0)
    err_p = float(np.max(abs(p1 - p0 - len(v)*gravity*dt)))
    assert err_e < 1e-12 and err_p < 1e-12
    peaks["fraction_energy"] = max(peaks["fraction_energy"], err_e)
    peaks["fraction_momentum"] = max(peaks["fraction_momentum"], err_p)
    return {
        "policy": mode, "map_energy_change": e0-e_before, "map_total_linear_impulse": (p0-p_before).tolist(),
        "map_particle_impulses": delta.tolist(), "map_angular_momentum_change": np.cross(q, delta).sum(axis=0).tolist(),
        "fraction_gravity_work": -.5 * float((end_q[:, 2]-q[:, 2]).sum()),
        "fraction_kinetic_change": k1-k0, "fraction_gravity_impulse": (len(v)*gravity*dt).tolist(),
        "fraction_damping_heat": 0., "fraction_terrain_impulse": [0., 0., 0.],
        "external_energy_supply": 0., "binary_collisions": 0,
    }


def metrics(c, raw, run):
    q, v, f, calls, events = raw
    future = f[:, 0] - c["initial_loss"]
    rises = [*c["history_rises"], c["contact_rise"], *future]
    complete = len(f) == END_STEP - c["boundary_step"] + 1 and not run["failed"]
    return_steps = np.flatnonzero(future <= LIMIT)
    return {
        "run_id": run["run_id"], "complete": complete, "boundary_rise": float(future[0]),
        "deviated_at_contact_or_through_boundary": bool(max(rises[:len(c["history_rises"])+2]) > LIMIT),
        "deviated_through_end": bool(max(rises) > LIMIT) if complete else None,
        "first_saved_recovery": float((c["boundary_step"] + return_steps[0]) * H)
            if c["already_deviated"] and return_steps.size else None,
        "peak_rise": float(max(rises)), "final_rise": float(future[-1]),
        "within_limit_at_end": bool(future[-1] <= LIMIT) if complete else None,
    }


def result(rows):
    prevent = [r for r in rows if not r["already_deviated"]]
    recovery = [r for r in rows if r["already_deviated"]]
    summaries = {}
    for name, group in (("prevention", prevent), ("recovery", recovery)):
        stats = {"cases": len(group), "warnings": sum(r["warning"] for r in group)}
        for mode in MODES:
            runs = [r["branches"][mode] for r in group]
            valid = [r for r in runs if r["complete"]]
            stats[mode] = {
                "complete": len(valid), "deviated_through_boundary": sum(r["deviated_at_contact_or_through_boundary"] for r in valid),
                "deviated_through_end": sum(r["deviated_through_end"] for r in valid),
                "within_limit_at_end": sum(r["within_limit_at_end"] for r in valid),
                "observed_recoveries": sum(r["first_saved_recovery"] is not None for r in valid),
                "median_peak_rise": median(r["peak_rise"] for r in valid) if valid else None,
                "median_final_rise": median(r["final_rise"] for r in valid) if valid else None,
            }
        summaries[name] = stats
    paired = sum(r["warning"] and r["branches"]["keep"]["deviated_at_contact_or_through_boundary"]
                 and r["branches"]["transfer"]["complete"] and not r["branches"]["transfer"]["deviated_through_end"]
                 for r in prevent)
    comparables = [r for r in recovery if r["branches"]["transfer"]["first_saved_recovery"] is not None
                   and r["delayed_E30_transfer"]["first_return_after_checkpoint"] is not None]
    difference = [r["delayed_E30_transfer"]["first_return_after_checkpoint"] -
                  r["branches"]["transfer"]["first_saved_recovery"] for r in comparables]
    lead = [r["boundary_time"] - r["contact_time"] for r in prevent if r["warning"]]
    return {
        **summaries, "warned_and_prevented_pairs": paired, "passed_screen": (
            paired >= 3 and summaries["recovery"]["transfer"]["within_limit_at_end"] >= 5),
        "warning_lead_min_median_max": [min(lead), median(lead), max(lead)] if lead else None,
        "recovery_vs_delayed_E30": {"comparable": len(difference),
                                    "earlier": sum(v > 1e-12 for v in difference),
                                    "same": sum(abs(v) <= 1e-12 for v in difference),
                                    "later": sum(v < -1e-12 for v in difference)},
    }


def audit(root):
    config, checkpoints, runs, counts = (read(root / (n + ".json"))
                                         for n in ("config", "checkpoints", "runs", "execution_counts"))
    formal = config["stage"] == "formal"
    parent = Path(config["parent"])
    source = Path(config["raw_parent_source"])
    original = Path(read(source / "config.json")["parent"])
    contexts = {c["context_id"]: c for c in read(original / "contexts.json")}
    delayed = {r["context_id"]: r for r in read(parent / "paired_outcomes.json")}
    expected = (10 if formal else 4)
    assert len(checkpoints) == expected == counts["new_event_reconstructions"] == counts["reused_prefixes"]
    assert len(runs) == 2 * expected == counts["suffix_executions"]
    assert len({r["run_id"] for r in runs}) == len(runs)
    assert sha(root / "source.zip") == config["source_archive_sha256"]
    for path, digest in config["dependency_data_sha256"].items():
        assert sha(Path(path)) == digest
    with zipfile.ZipFile(root / "source.zip") as archive:
        for name, digest in config["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    actual = (sum(c["actual_event_reconstruction_calls"] + c["actual_forecast_calls"] for c in checkpoints) +
              sum(r["actual_fraction_calls"] + r["actual_tail_calls"] for r in runs))
    assert counts["actual_new_solver_queries"] == actual
    assert counts["charged_policy_queries"] == sum(r["counts"]["total_oracle_calls"] for r in runs)
    assert counts["charged_policy_queries"] == actual + 2*counts["historical_prefix_queries"] + (
        counts["actual_event_reconstruction_queries"] + counts["actual_forecast_queries"])
    groups = defaultdict(dict)
    for run in runs:
        groups[run["context_id"]][run["policy"]] = run
    totals = Counter()
    peaks = dict(energy=0., momentum=0., reflection=0., event_energy=0., event_momentum=0., parent_event_time=0.,
                 map_energy=0., map_momentum=0., fraction_energy=0., fraction_momentum=0., parent_overlap=0.,
                 boundary_forecast_error=0.)
    records, ledgers = [], []
    for c in checkpoints:
        key = c["context_id"]
        trace = source / "traces" / (key + "__keep.csv.gz")
        events = source / "events" / (key + "__keep.json")
        assert sha(trace) == c["source_trace_sha256"] and sha(events) == c["source_event_sha256"]
        parent_rows, q, v, f = load_segment(source, key + "__keep")
        event = next(e for e in read(events) if e["i"] == 0)
        assert event["step"] + PARENT_STEP == c["boundary_step"] == 7 and c["previous_step"] == 6
        audit_contact(c, event, q, v, contexts[key]["rotation"], peaks)
        totals["value_checks"] += 1
        assert c["historical_prefix_calls"] == int(parent_rows[c["previous_step"]-PARENT_STEP]["total_oracle_calls"])
        assert max(c["history_rises"]) <= LIMIT
        assert set(groups[key]) == set(MODES)
        record = {k: c[k] for k in ("context_id", "seed", "incumbent_energy", "contact_time", "contact_rise",
                                    "already_deviated", "predicted_boundary_rise", "warning")}
        record.update(boundary_time=c["boundary_step"]*H, branches={}, delayed_E30_transfer=delayed[key]["branches"]["transfer"])
        for mode, run in groups[key].items():
            name = run["run_id"]
            frac = read(root / "fractions" / (name + ".json"))
            frac_ledger = audit_fraction(frac, c, mode, peaks)
            raw, tail = segment_audit(root, run, contexts[key], totals, peaks, start_step=c["boundary_step"])
            ledgers.append({"run_id": name, "fraction_and_map": frac_ledger, "tail": tail})
            assert np.array_equal(raw[0][0], frac["end"]["positions"])
            assert np.array_equal(raw[1][0], frac["end"]["velocities"])
            assert np.array_equal(raw[2][0], frac["values"])
            assert frac["total_oracle_calls_at_end"] == c["charged_at_branch"] + run["actual_fraction_calls"]
            assert raw[3][0] == frac["total_oracle_calls_at_end"] + 4
            assert run["counts"]["total_oracle_calls"] == frac["total_oracle_calls_at_end"] + run["actual_tail_calls"] <= CAP
            record["branches"][mode] = metrics(c, raw, run)
            if mode == "keep":
                idx = c["boundary_step"] - PARENT_STEP
                for p, new in zip((q, v, f), raw[:3]):
                    error = float(np.max(abs(p[idx:idx+len(new)] - new)))
                    assert error < 1e-12
                    peaks["parent_overlap"] = max(peaks["parent_overlap"], error)
                for i, charge in enumerate(raw[3]):
                    assert charge == int(parent_rows[idx+i]["total_oracle_calls"]) + 4 + c["actual_forecast_calls"]
                predicted = c["predicted_boundary_rise"]
                if predicted is not None:
                    error = abs(predicted - record["branches"][mode]["boundary_rise"])
                    assert error < 1e-12
                    peaks["boundary_forecast_error"] = max(peaks["boundary_forecast_error"], error)
        records.append(record)
    if formal:
        assert sum(r["already_deviated"] for r in records) == 6
        assert {r["context_id"] for r in records} == {r["context_id"] for r in read(parent / "already_deviated_cases.json")}
        create(root / "results.json", result(records))
    create(root / "paired_outcomes.json", records)
    create(root / "contact_already_deviated_cases.json", [r for r in records if r["already_deviated"]])
    create(root / "physical_ledgers.json", {
        "events": [{"context_id": c["context_id"], **c["contact"]} for c in checkpoints],
        "continuations": ledgers, "new_oracle_calls_for_export": 0,
    })
    audit_result = {
        "passed": True, "contacts": len(checkpoints), "continuations": len(runs),
        "already_deviated_at_contact": sum(r["already_deviated"] for r in records),
        "unavailable_forecasts": sum(c["predicted_boundary_rise"] is None and not c["already_deviated"] for c in checkpoints),
        "solver_failures": [r["run_id"] for r in runs if r["failed"]],
        "incomplete": [r["run_id"] for r in runs if r["committed_steps"] != END_STEP-7],
        "max_residuals": peaks, **totals,
    }
    create(root / "audit.json", audit_result)
    print(audit_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

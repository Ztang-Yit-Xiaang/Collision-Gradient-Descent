"""Audit E34 matched timing policies and the E33 high-kinetic diagnosis."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_contact_time_transfer import audit_fraction, physical
from audit_contact_warning_replication import event_audit
from audit_shared_prefix_transfer import segment_audit
from collision_optim.prebounce_warning import flight_factors, forecast_rise
from finalize_initial_budget import create, sha
from run_high_kinetic_intervention_timing import CAP, END_STEP, ENERGY, H, LIMIT, PARENT, PRECONTACT_STEP, PRIMARY_STEP, SEEDS, read

POLICIES = ("keep", "transfer", "early")


def range3(values):
    return [min(values), median(values), max(values)] if values else None


def prior_diagnosis():
    parent = [r for r in read(PARENT / "paired_outcomes.json") if r["incumbent_energy"] == ENERGY]
    manifest = {r["path"]: r["sha256"] for r in read(PARENT / "raw_artifact_manifest.json")["files"]}
    assert len(parent) == 40 and all(r["already_deviated"] for r in parent)
    rows = []
    for r in parent:
        name = r["context_id"]
        c_path = PARENT/"checkpoints"/(name+".json")
        f_path = PARENT/"fractions"/(name+"__transfer.json")
        for path in (c_path, f_path):
            assert sha(path) == manifest[str(path)]
        c, f = read(c_path), read(f_path)
        contact, final = r["contact_rise"], r["branches"]["transfer"]["final_rise"]
        assert contact > LIMIT and final > LIMIT
        rows.append({
            "seed": r["seed"], "context_id": name, "contact_rise": contact,
            "final_rise_transfer": final, "final_rise_keep": r["branches"]["keep"]["final_rise"],
            "fraction_of_drop_needed_to_reach_limit": (contact-final)/(contact-LIMIT),
            "pretransfer_horizontal_speed": float(np.linalg.norm(c["post_reflection"]["velocities"][0][:2])),
            "posttransfer_horizontal_speed": float(np.linalg.norm(f["start"]["velocities"][0][:2])),
            "first_saved_deviation": next(i*H for i, rise in enumerate(c["history_rises"]) if rise > LIMIT),
        })
    return {
        "cases": rows, "new_oracle_calls": 0,
        "min_median_max": {key: range3([r[key] for r in rows]) for key in (
            "contact_rise", "final_rise_transfer", "final_rise_keep", "fraction_of_drop_needed_to_reach_limit",
            "pretransfer_horizontal_speed", "posttransfer_horizontal_speed", "first_saved_deviation")},
    }


def early_map(c, q, v, peaks):
    state = c["early_state"]
    before = np.array(state["velocities"])
    assert np.array_equal(q, state["positions"]) and np.array_equal(v[:, 2], before[:, 2])
    center = before[:, :2].mean(axis=0)
    rel = before[:, :2] - center
    rest = rel[1:] - rel[1:].mean(axis=0)
    expected = center + rest * np.sqrt(np.sum(rel**2) / np.sum(rest**2))
    assert np.max(abs(v[0, :2] - center)) < 1e-12 and np.max(abs(v[1:, :2] - expected)) < 1e-12
    _, e0, p0 = physical(q, before)
    _, e1, p1 = physical(q, v)
    assert abs(e1-e0) < 1e-12 and np.max(abs(p1-p0)) < 1e-12
    peaks["map_energy"] = max(peaks["map_energy"], abs(e1-e0))
    peaks["map_momentum"] = max(peaks["map_momentum"], float(np.max(abs(p1-p0))))
    return {"energy_change": e1-e0, "linear_impulse": (p1-p0).tolist(),
            "particle_impulses": (v-before).tolist(), "angular_momentum_change": np.cross(q, v-before).sum(axis=0).tolist()}


def outcome(c, raw, run, mode):
    early = mode == "early"
    step = 1 if early else c["boundary_step"]
    rises = [*c["history_rises"][:step], *(raw[2][:, 0]-c["initial_loss"])]
    if early:
        own = next((e for e in raw[4] if e["i"] == 0), None)
        hit = (own["time"] + H, own["point_i"][2] - c["initial_loss"]) if own else None
        decision = H
        already = c["already_deviated_at_step1"]
    else:
        hit = (c["contact_time"], c["contact_rise"])
        decision = c["contact_time"]
        already = c["already_deviated_at_reflection"]
    complete = len(rises) == END_STEP+1 and not run["failed"]
    saved = [(i*H, float(rise)) for i, rise in enumerate(rises)]
    augmented = sorted([*saved, hit]) if hit else saved
    first = next((time for time, rise in augmented if rise > LIMIT), None)
    recovered = next((time for time, rise in saved if time > decision and rise <= LIMIT), None) if already else None
    phases = {}
    for name, horizon in (("primary", PRIMARY_STEP), ("extended", END_STEP)):
        if len(rises) <= horizon or run["failed"]:
            phases[name] = None
            continue
        observed = [rise for time, rise in augmented if time <= horizon*H]
        phases[name] = {
            "deviated": max(observed) > LIMIT, "peak_rise": max(observed), "final_rise": float(rises[horizon]),
            "within_limit": rises[horizon] <= LIMIT, "recovered_by_horizon": recovered is not None and recovered <= horizon*H,
        }
        phases[name]["within_limit"] = bool(phases[name]["within_limit"])
    return {"run_id": run["run_id"], "complete": complete, "first_deviation": first,
            "first_saved_recovery": recovered, "first_own_contact": hit, **phases}


def summarize(rows, horizon):
    valid = [r[horizon] for r in rows if r[horizon] is not None]
    return {
        "n": len(rows), "complete": len(valid), "deviated": sum(r["deviated"] for r in valid),
        "within_limit": sum(r["within_limit"] for r in valid),
        "recovered": sum(r["recovered_by_horizon"] for r in valid),
        "median_peak_rise": median(r["peak_rise"] for r in valid) if valid else None,
        "median_final_rise": median(r["final_rise"] for r in valid) if valid else None,
    }


def evaluate(rows):
    late = [r for r in rows if r["already_deviated_at_reflection"]]
    primary = {p: summarize([r["branches"][p] for r in rows], "primary") for p in POLICIES}
    long = {p: summarize([r["branches"][p] for r in rows], "extended") for p in POLICIES}
    prevented = {}
    for policy in ("early", "transfer"):
        prevented[policy] = sum(r["branches"]["keep"]["primary"]["deviated"] and
                                not r["branches"][policy]["primary"]["deviated"]
                                for r in rows if r["branches"]["keep"]["primary"] and r["branches"][policy]["primary"])
    positive = primary["keep"]["deviated"]
    return {
        "primary": primary, "extended": long, "n": len(rows), "paired_prevented_at_primary": prevented,
        "early_warnings": sum(r["early_warning"] for r in rows),
        "already_deviated_at_step1": sum(r["already_deviated_at_step1"] for r in rows),
        "already_deviated_at_reflection": len(late),
        "late_recovery": {
            stage: {p: summarize([r["branches"][p] for r in late], stage) for p in ("keep", "transfer")}
            for stage in ("primary", "extended")
        },
        "late_recovery_time_min_median_max": {
            p: range3([r["branches"][p]["first_saved_recovery"] for r in late
                       if r["branches"][p]["first_saved_recovery"] is not None]) for p in ("keep", "transfer")
        },
        "new_contact_rise_min_median_max": range3([r["contact_rise"] for r in rows]),
        "passed_screen": positive >= 20 and prevented["early"] >= .9*positive and prevented["early"]-prevented["transfer"] >= 10,
    }


def audit(root):
    cfg, contexts, prefixes, runs, direct, costs = (read(root/(n+".json")) for n in (
        "config", "contexts", "prefixes", "runs", "uninterrupted", "execution_counts"))
    assert len(contexts) == len(prefixes) == len(SEEDS[cfg["stage"]]) == costs["contexts"]
    assert {c["seed"] for c in contexts} == set(SEEDS[cfg["stage"]])
    assert len(runs) == costs["continuations"] == 2*len(contexts) + sum(c["early_warning"] for c in contexts)
    assert len(direct) == (len(contexts) if cfg["stage"] == "preflight" else 0)
    assert sha(root/"source.zip") == cfg["source_archive_sha256"]
    for path, digest in cfg["dependency_data_sha256"].items():
        assert sha(Path(path)) == digest
    with zipfile.ZipFile(root/"source.zip") as archive:
        for path, digest in cfg["source_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    assert costs["actual_forecast_queries"] == 0
    assert costs["actual_prefix_queries"] == sum(r["counts"]["total_oracle_calls"] for r in prefixes)
    assert costs["actual_contact_queries"] == 10*len(contexts)
    assert costs["actual_continuation_queries"] == sum(r["actual_suffix_calls"] for r in runs)
    assert costs["actual_uninterrupted_queries_excluding_shared_initial_values"] == sum(
        r["counts"]["total_oracle_calls"]-4 for r in direct)
    assert costs["actual_solver_queries"] == sum(costs[k] for k in (
        "actual_prefix_queries", "actual_contact_queries", "actual_continuation_queries",
        "actual_uninterrupted_queries_excluding_shared_initial_values"))
    assert costs["actual_queries_including_setup"] == costs["actual_solver_queries"] + 4*len(contexts)
    by_prefix = {r["context_id"]: r for r in prefixes}
    by_direct = {r["context_id"]: r for r in direct}
    groups = defaultdict(dict)
    for r in runs:
        groups[r["context_id"]][r["policy"]] = r
    totals = Counter()
    peaks = dict(energy=0., momentum=0., reflection=0., map_energy=0., map_momentum=0., fraction_energy=0.,
                 fraction_momentum=0., contact_energy=0., contact_momentum=0., uninterrupted_overlap=0.)
    rows, ledgers = [], []
    for c in contexts:
        name = c["context_id"]
        p, p_ledger = segment_audit(root, by_prefix[name], c, totals, peaks)
        ledgers.append({"run_id": by_prefix[name]["run_id"], "prefix": p_ledger})
        q, v, f, calls, events = p
        assert np.array_equal(q[0], c["start"]["positions"]) and np.array_equal(v[0], c["start"]["velocities"])
        assert not events and calls[0] == 8 and calls[1] == c["early_prefix_calls"] == 20
        assert abs(.5*np.sum(v[0,0]**2)-ENERGY) < 1e-14
        assert abs(.5*np.sum(v[0]**2)-.6) < 1e-12 and np.linalg.norm(v[0].sum(axis=0)) < 1e-12
        event_audit(c, q[-1], v[-1], totals, peaks)
        history = f[:,0]-f[0,0]
        assert np.array_equal(history,c["history_rises"])
        assert np.array_equal(q[1],c["early_state"]["positions"]) and np.array_equal(v[1],c["early_state"]["velocities"])
        assert c["already_deviated_at_step1"] == (max(history[:2]) > LIMIT)
        assert c["already_deviated_at_reflection"] == (max(*history,c["contact_rise"]) > LIMIT)
        score = forecast_rise(history[1:2], float(np.linalg.norm(v[0,0,:2])), flight_factors(), cfg["early_prior_curvature"])
        assert c["early_score"] == score
        assert c["early_warning"] == (not c["already_deviated_at_step1"] and score > LIMIT)
        assert c["charged_at_reflection"] == calls[-1]+10
        rec = {k:c[k] for k in ("context_id", "seed", "contact_time", "contact_rise", "early_score", "early_warning",
                                "already_deviated_at_step1", "already_deviated_at_reflection")}
        rec["branches"] = {}
        assert set(groups[name]) == (set(POLICIES) if c["early_warning"] else {"keep","transfer"})
        if name in by_direct:
            full, ledger = segment_audit(root, by_direct[name], c, totals, peaks)
            ledgers.append({"run_id": by_direct[name]["run_id"], "uninterrupted": ledger})
            first = next(e for e in full[4] if e["i"] == 0)
            assert abs(first["time"]-c["contact_time"]) < 1e-12
        else:
            full = None
        for mode, run in groups[name].items():
            early = mode == "early"
            step = 1 if early else c["boundary_step"]
            raw, tail = segment_audit(root, run, c, totals, peaks, start_step=step)
            if early:
                ledger = early_map(c, raw[0][0], raw[1][0], peaks)
                assert np.array_equal(raw[2][0], f[1]) and raw[3][0] == calls[1]+4
                assert run["counts"]["total_oracle_calls"] == run["actual_suffix_calls"] + calls[1]
            else:
                frac = read(root/"fractions"/(run["run_id"]+".json"))
                ledger = audit_fraction(frac,c,mode,peaks)
                assert np.array_equal(raw[0][0],frac["end"]["positions"]) and np.array_equal(raw[1][0],frac["end"]["velocities"])
                assert np.array_equal(raw[2][0],frac["values"])
                assert raw[3][0] == frac["total_oracle_calls_at_end"]+4
                assert run["counts"]["total_oracle_calls"] == run["actual_suffix_calls"]+c["charged_at_reflection"]
            assert run["counts"]["total_oracle_calls"] <= CAP
            ledgers.append({"run_id":run["run_id"], "intervention_or_fraction":ledger, "tail":tail})
            rec["branches"][mode] = outcome(c,raw,run,mode)
            if mode == "keep" and full:
                for target, got in zip(full[:3],raw[:3]):
                    error = float(np.max(abs(target[step:step+len(got)]-got)))
                    assert error < 1e-12
                    peaks["uninterrupted_overlap"] = max(peaks["uninterrupted_overlap"],error)
                assert all(charge == full[3][step+i]+4 for i,charge in enumerate(raw[3]))
        if not c["early_warning"]:
            rec["branches"]["early"] = {**rec["branches"]["keep"],"derived_no_warning_keep":True}
        rows.append(rec)
    if cfg["stage"] == "formal":
        create(root/"prior_high_kinetic_diagnostics.json",prior_diagnosis())
        create(root/"results.json",evaluate(rows))
    create(root/"paired_outcomes.json",rows)
    create(root/"already_deviated_cases.json",[r for r in rows if r["already_deviated_at_reflection"] or r["already_deviated_at_step1"]])
    create(root/"physical_ledgers.json",{
        "events":[{"context_id":c["context_id"],**c["contact"]} for c in contexts],
        "segments":ledgers,"new_oracle_calls_for_export":0,
    })
    result = {
        "passed":True,"generated":len(contexts),"continuations":len(runs),
        "solver_failures":[r["run_id"] for r in prefixes+runs+direct if r["failed"]],
        "incomplete_continuations":[r["run_id"] for r in rows for r in r["branches"].values() if not r["complete"]],
        "max_residuals":peaks,**totals,
    }
    create(root/"audit.json",result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)

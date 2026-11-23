"""Audit E32 fresh first-contact labels, forecasting and paired outcomes."""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_contact_time_transfer import audit_fraction, metrics, physical
from audit_shared_prefix_transfer import segment_audit
from finalize_initial_budget import create, sha
from run_contact_warning_replication import BOUNDARY_STEP, CAP, END_STEP, ENERGIES, H, LIMIT, MODES, PRECONTACT_STEP, SEEDS, read

SHORT = "deviated_at_contact_or_through_boundary"
LONG = "deviated_through_end"


def event_audit(c, q, v, counts, peaks):
    assert np.array_equal(q, c["before_step"]["positions"]) and np.array_equal(v, c["before_step"]["velocities"])
    q1 = np.array(c["post_reflection"]["positions"])
    v1 = np.array(c["post_reflection"]["velocities"])
    event = c["contact"]
    assert event["index"] == 0
    beta = np.exp(-H * (-np.log(.995)/H + (.5 + np.log(.995)/H) * (PRECONTACT_STEP+.5)*H/25))
    assert abs(event["damping_beta"] - beta) < 1e-14
    dt = event["offset"]
    gravity = np.array([0., 0., -.5])
    assert 0 < dt < H and abs(c["contact_time"] - (PRECONTACT_STEP*H + dt)) < 1e-14
    assert np.max(abs(q1 - q - dt*beta*v - .5*dt**2*gravity)) < 1e-12
    pre = beta*v + dt*gravity
    assert np.max(abs(pre[0] - event["incoming_velocity"])) < 1e-12
    assert np.max(abs(pre[1:] - v1[1:])) < 1e-12
    Q = np.array(c["rotation"])
    x, y = Q @ q1[0, :2]
    val = .25*(x*x-1)**2 + .05*x + 4*y*y
    grad = Q.T @ [x**3-x+.05, 8*y]
    normal = np.r_[-grad, 1.]
    normal /= np.linalg.norm(normal)
    assert abs(val - event["contact_value"]) < 1e-12 and abs(val - q1[0, 2]) < 1e-9
    assert np.max(abs(normal - event["normal"])) < 1e-12
    assert np.max(abs(v1[0] - pre[0] + 2*(pre[0]@normal)*normal)) < 1e-12
    _, e0, p0 = physical(q, v)
    _, e1, p1 = physical(q1, v1)
    e = abs(e1-e0-event["damping_energy"]-event["terrain_work"])
    p = float(np.max(abs(p1-p0-np.array(event["damping_impulse"])-event["gravity_impulse"]-event["terrain_impulse"])))
    assert e < 1e-12 and p < 1e-12
    peaks["contact_energy"] = max(peaks["contact_energy"], e)
    peaks["contact_momentum"] = max(peaks["contact_momentum"], p)
    counts.update({"value_checks": 1, "gradient_checks": 1, "reconstructed_first_reflections": 1})


def classify(rows, metric):
    positive = [r for r in rows if r["branches"]["keep"][metric]]
    negative = [r for r in rows if not r["branches"]["keep"][metric]]
    tp = [r for r in positive if r["warning"]]
    fp = [r for r in negative if r["warning"]]
    return {
        "n": len(rows), "positive": len(positive), "negative": len(negative), "true_warnings": len(tp),
        "false_warnings": len(fp), "misses": len(positive)-len(tp), "true_negatives": len(negative)-len(fp),
        "unavailable_forecasts": sum(r["predicted_boundary_rise"] is None for r in rows),
        "prevented_true_warnings": sum(not r["branches"]["transfer"][metric] for r in tp),
        "transfer_only_deviations_among_warnings": sum(r["branches"]["transfer"][metric] for r in fp),
    }


def summarize(branches):
    valid = [b for b in branches if b["complete"]]
    recover = [b["first_saved_recovery"] for b in valid if b["first_saved_recovery"] is not None]
    return {
        "n": len(branches), "complete": len(valid), "deviated_by_point35": sum(b[SHORT] for b in valid),
        "deviated_by_end": sum(b[LONG] for b in valid),
        "within_limit_at_end": sum(b["within_limit_at_end"] for b in valid),
        "observed_recoveries": len(recover), "recovery_time_min_median_max":
            [min(recover), median(recover), max(recover)] if recover else None,
        "median_peak_rise": median(b["peak_rise"] for b in valid) if valid else None,
        "median_final_rise": median(b["final_rise"] for b in valid) if valid else None,
    }


def analyze(rows):
    eligible = [r for r in rows if r["eligible"]]
    valid = [r for r in eligible if all(b["complete"] for b in r["branches"].values())]
    at_risk = [r for r in valid if not r["already_deviated"]]
    earlier_unwarned = [r for r in at_risk if r["earlier_E28_status"] == "no_alert"]
    warning = [r for r in at_risk if r["warning"]]
    no_warning = [r for r in at_risk if not r["warning"]]
    late = [r for r in valid if r["already_deviated"]]
    groups = {
        name: {mode: summarize([r["branches"][mode] for r in group]) for mode in MODES}
        for name, group in (("warning", warning), ("no_warning", no_warning), ("already_deviated", late))
    }
    groups["already_deviated"].update(
        earlier_saved=sum(r["earlier_saved_deviation"] for r in late),
        above_limit_at_contact=sum(r["above_limit_at_contact"] for r in late),
        first_crossing_at_contact=sum(not r["earlier_saved_deviation"] for r in late),
    )
    short, long = classify(at_risk, SHORT), classify(at_risk, LONG)
    checks = {
        "five_positive_controls": short["positive"] >= 5,
        "false_warning_rate_at_most_10_percent":
            bool(short["negative"] and short["false_warnings"] / short["negative"] <= .10),
        "miss_rate_at_most_10_percent":
            bool(short["positive"] and short["misses"] / short["positive"] <= .10),
        "prevent_at_least_90_percent_of_true_warnings":
            bool(short["true_warnings"] and short["prevented_true_warnings"]/short["true_warnings"] >= .90),
        "no_transfer_only_near_term_deviations_among_warnings":
            short["transfer_only_deviations_among_warnings"] == 0,
    }
    return {
        "eligible": len(eligible), "complete_pairs": len(valid), "unique_seed_groups": len({r["seed"] for r in eligible}),
        "ineligible": [{"context_id": r["context_id"], "reason": r["unavailable_reason"]} for r in rows if not r["eligible"]],
        "classification_point35": short, "classification_through_end": long,
        "earlier_E28_no_alert_point35": classify(earlier_unwarned, SHORT),
        "earlier_E28_no_alert_through_end": classify(earlier_unwarned, LONG),
        "groups": groups, "checks": checks, "passed_near_term_screen": all(checks.values()),
        "cohort_always_keep": summarize([r["branches"]["keep"] for r in valid]),
        "cohort_warning_only": summarize([r["branches"]["transfer" if r["warning"] else "keep"] for r in valid]),
        "cohort_warning_and_recovery": summarize([
            r["branches"]["transfer" if r["warning"] or r["already_deviated"] else "keep"] for r in valid
        ]),
    }


def audit(root, *, seed_sets=SEEDS, evaluator=analyze, extension_audit=None):
    config, contexts, prefixes, runs, direct, costs = (read(root / (n + ".json")) for n in (
        "config", "contexts", "prefixes", "runs", "uninterrupted", "execution_counts"))
    stage = config["stage"]
    seeds = seed_sets[stage]
    assert len(contexts) == len(prefixes) == costs["contexts"] == len(seeds)*len(ENERGIES)
    assert {(c["seed"], c["incumbent_energy"]) for c in contexts} == {(s, e) for s in seeds for e in ENERGIES}
    assert len(runs) == costs["continuations"] == 2 * sum(c["eligible"] for c in contexts)
    assert len(direct) == (len(contexts) if stage == "preflight" else 0)
    assert len({r["run_id"] for r in runs}) == len(runs)
    assert sha(root / "source.zip") == config["source_archive_sha256"]
    for path, digest in config["dependency_data_sha256"].items():
        assert sha(Path(path)) == digest
    with zipfile.ZipFile(root / "source.zip") as archive:
        for path, digest in config["source_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    assert costs["actual_prefix_queries"] == sum(p["counts"]["total_oracle_calls"] for p in prefixes)
    assert costs["actual_contact_queries"] == sum(c["actual_event_reconstruction_calls"] for c in contexts)
    assert costs["actual_forecast_queries"] == sum(c["actual_forecast_calls"] for c in contexts)
    assert costs["actual_fraction_queries"] == sum(r["actual_fraction_calls"] for r in runs)
    assert costs["actual_tail_queries"] == sum(r["actual_tail_calls"] for r in runs)
    assert costs["actual_uninterrupted_queries_excluding_shared_initial_values"] == sum(
        r["counts"]["total_oracle_calls"]-4 for r in direct)
    assert costs["actual_solver_queries"] == sum(v for k, v in costs.items()
                                                if k.startswith("actual_") and k not in ("actual_solver_queries", "actual_queries_including_setup"))
    assert costs["actual_queries_including_setup"] == costs["actual_solver_queries"] + 4 * len(contexts)
    by_prefix = {r["context_id"]: r for r in prefixes}
    by_direct = {r["context_id"]: r for r in direct}
    by_run = defaultdict(dict)
    for r in runs:
        by_run[r["context_id"]][r["policy"]] = r
    totals = Counter()
    peaks = dict(energy=0., momentum=0., reflection=0., map_energy=0., map_momentum=0., fraction_energy=0.,
                 fraction_momentum=0., contact_energy=0., contact_momentum=0., forecast_loss=0.,
                 uninterrupted_overlap=0., uninterrupted_event_time=0.)
    rows, ledgers = [], []
    for c in contexts:
        name = c["context_id"]
        raw, ledger = segment_audit(root, by_prefix[name], c, totals, peaks)
        ledgers.append({"run_id": by_prefix[name]["run_id"], "prefix": ledger})
        q, v, f, calls, events = raw
        assert np.array_equal(q[0], c["start"]["positions"]) and np.array_equal(v[0], c["start"]["velocities"])
        assert np.array_equal(f[0], c["values"]) and calls[0] == 8
        assert np.array_equal(q[0, :, :2], c["points"])
        assert abs(.5*np.sum(v[0]**2) - .6) < 1e-12
        assert abs(.5*np.sum(v[0, 0]**2) - c["incumbent_energy"]) < 1e-12
        row = {"context_id": name, "seed": c["seed"], "incumbent_energy": c["incumbent_energy"], "eligible": c["eligible"]}
        if not c["eligible"]:
            assert not by_run[name]
            rows.append({**row, "unavailable_reason": c["unavailable_reason"]})
            continue
        assert len(q) == PRECONTACT_STEP+1 and not any(e["i"] == 0 for e in events)
        event_audit(c, q[-1], v[-1], totals, peaks)
        assert c["actual_event_reconstruction_calls"] == 10
        if extension_audit is None:
            assert c["actual_forecast_calls"] == int(not c["already_deviated"])
        assert c["historical_prefix_calls"] == calls[-1]
        assert c["charged_at_branch"] == calls[-1] + 10 + c["actual_forecast_calls"]
        rises = f[:, 0] - f[0, 0]
        assert np.array_equal(rises, c["history_rises"])
        assert c["contact_rise"] == c["contact"]["contact_value"] - f[0, 0]
        assert c["earlier_saved_deviation"] == (np.max(rises) > LIMIT)
        assert c["above_limit_at_contact"] == (c["contact_rise"] > LIMIT)
        assert c["already_deviated"] == (c["earlier_saved_deviation"] or c["above_limit_at_contact"])
        forecast = c["predicted_boundary_rise"]
        assert c["warning"] == (forecast is not None and forecast > LIMIT)
        assert set(by_run[name]) == set(MODES)
        row.update({k: c[k] for k in ("contact_rise", "contact_time", "already_deviated", "earlier_saved_deviation",
                                     "above_limit_at_contact", "earlier_E28_status", "predicted_boundary_rise", "warning")})
        row["branches"] = {}
        if name in by_direct:
            d, d_ledger = segment_audit(root, by_direct[name], c, totals, peaks)
            ledgers.append({"run_id": by_direct[name]["run_id"], "uninterrupted": d_ledger})
            direct_event = next(e for e in d[4] if e["i"] == 0)
            err = abs(direct_event["time"] - c["contact_time"])
            assert err < 1e-12 and direct_event["step"] == BOUNDARY_STEP
            peaks["uninterrupted_event_time"] = max(peaks["uninterrupted_event_time"], err)
        else:
            d = None
        for mode, run in by_run[name].items():
            frac = read(root / "fractions" / (run["run_id"] + ".json"))
            frac_ledger = audit_fraction(frac, c, mode, peaks)
            continuation, tail = segment_audit(root, run, c, totals, peaks, start_step=BOUNDARY_STEP)
            ledgers.append({"run_id": run["run_id"], "fraction_and_map": frac_ledger, "tail": tail})
            assert np.array_equal(continuation[0][0], frac["end"]["positions"])
            assert np.array_equal(continuation[1][0], frac["end"]["velocities"])
            assert np.array_equal(continuation[2][0], frac["values"])
            assert frac["total_oracle_calls_at_end"] == c["charged_at_branch"] + run["actual_fraction_calls"]
            assert continuation[3][0] == frac["total_oracle_calls_at_end"] + 4
            assert run["counts"]["total_oracle_calls"] == frac["total_oracle_calls_at_end"] + run["actual_tail_calls"] <= CAP
            row["branches"][mode] = metrics(c, continuation, run)
            if mode == "keep":
                if extension_audit is not None:
                    row.update(extension_audit(c, continuation, peaks))
                if forecast is not None:
                    error = abs(forecast - row["branches"][mode]["boundary_rise"])
                    assert error < 1e-12
                    peaks["forecast_loss"] = max(peaks["forecast_loss"], error)
                if d:
                    for a, b in zip(d[:3], continuation[:3]):
                        error = float(np.max(abs(a[BOUNDARY_STEP:BOUNDARY_STEP+len(b)] - b)))
                        assert error < 1e-12
                        peaks["uninterrupted_overlap"] = max(peaks["uninterrupted_overlap"], error)
                    for i, query in enumerate(continuation[3]):
                        assert query == d[3][BOUNDARY_STEP+i] + 4 + c["actual_forecast_calls"]
        rows.append(row)
    if stage == "formal":
        create(root / "results.json", evaluator(rows))
    create(root / "paired_outcomes.json", rows)
    create(root / "already_deviated_cases.json", [r for r in rows if r.get("already_deviated")])
    create(root / "physical_ledgers.json", {
        "events": [{"context_id": c["context_id"], **c["contact"]} for c in contexts if c["eligible"]],
        "segments": ledgers, "new_oracle_calls_for_export": 0,
    })
    result = {
        "passed": True, "generated": len(contexts), "eligible": sum(c["eligible"] for c in contexts),
        "solver_failures": [r["run_id"] for r in prefixes + runs + direct if r["failed"]],
        "ineligible": [{"context_id": c["context_id"], "reason": c["unavailable_reason"]} for c in contexts if not c["eligible"]],
        "incomplete_continuations": [r["run_id"] for r in runs if r["committed_steps"] != END_STEP - BOUNDARY_STEP],
        "max_residuals": peaks, **totals,
    }
    create(root / "audit.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

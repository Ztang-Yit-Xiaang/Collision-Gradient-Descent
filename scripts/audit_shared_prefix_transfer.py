"""E27 physical and query ledger, development lock, and held-out policy check."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median
import zipfile

import numpy as np

from audit_initial_budget import raw_metrics
from finalize_initial_budget import create, sha
from run_shared_prefix_transfer import ALL_FEATURES, MODEL_FEATURES, MODES, PREFIX_STEPS, SEEDS, read


def load_segment(root, key):
    with gzip.open(root / "traces" / (key + ".csv.gz"), "rt") as stream:
        rows = list(csv.DictReader(stream))
    q = np.array([json.loads(r["positions"]) for r in rows])
    v = np.array([json.loads(r["velocities"]) for r in rows])
    f = np.array([json.loads(r["values"]) for r in rows])
    return rows, q, v, f


def segment_audit(root, rec, context, totals, peaks, *, start_step=0):
    key = rec["run_id"]
    rows, q, v, loss = load_segment(root, key)
    Q = np.array(context["rotation"])

    def value(r):
        return .25 * (r[..., 0] ** 2 - 1) ** 2 + .05 * r[..., 0] + 4 * r[..., 1] ** 2

    assert np.max(abs(value(q[:, :, :2] @ Q.T) - loss)) < 1e-9
    assert np.min(q[:, :, 2] - loss) > -1e-8
    K = .5 * np.sum(v * v, axis=(1, 2))
    E = K + .5 * q[:, :, 2].sum(axis=1)
    P = v.sum(axis=1)
    assert np.max(abs(E - [float(r["energy"]) for r in rows])) < 1e-9
    assert rec["committed_steps"] == len(rows) - 1
    assert [int(r["step"]) for r in rows] == list(range(len(rows)))
    calls = [int(r["total_oracle_calls"]) for r in rows]
    assert all(a < b for a, b in zip(calls, calls[1:]))
    for row, charge in zip(rows, calls):
        assert sum(int(row[f]) for f in ("value_calls", "gradient_calls", "path_calls")) == charge
    assert calls[-1] <= rec["counts"]["total_oracle_calls"]

    events = read(root / "events" / (key + ".json"))
    wall = np.zeros((len(rows), 3))
    wall_work = 0.
    for e in events:
        assert e["kind"] == "wall"
        point = np.array(e["point_i"])
        xy = Q @ point[:2]
        gradient = Q.T @ [xy[0]**3 - xy[0] + .05, 8 * xy[1]]
        normal = np.r_[-gradient, 1.]
        normal /= np.linalg.norm(normal)
        before, after = np.array(e["before_i"]), np.array(e["after_i"])
        error = max(np.max(abs(normal - e["normal"])), np.max(abs(after - before + 2 * (before @ normal) * normal)))
        assert error < 1e-9 and abs(point[-1] - value(xy)) < 1e-8
        peaks["reflection"] = max(peaks["reflection"], float(error))
        wall[e["step"]] += after - before
        wall_work += .5 * float(after @ after - before @ before)
    totals.update({"value_checks": loss.size + len(events), "gradient_checks": len(events),
                   "saved_observations": len(rows), "wall_events": len(events)})

    h = .05
    base = -np.log(.995) / h
    rate = base + (.5 - base) * np.minimum((np.arange(1, len(rows)) + start_step - .5) * h / 25, 1)
    beta = np.exp(-rate * h)
    assert np.allclose(beta, [float(r["drag_beta"]) for r in rows[1:]], rtol=0, atol=1e-14)
    dissipation = (beta**2 - 1) * K[:-1]
    assert np.allclose(dissipation, [float(r["damping_energy"]) for r in rows[1:]], rtol=0, atol=1e-9)
    drag_P = (beta[:, None] - 1) * P[:-1]
    gravity_P = np.array([0., 0., -.1])
    impulse_E = np.array([float(r["impulse_energy"]) for r in rows[1:]])
    error_E = float(np.max(abs(E[1:] - E[0] - np.cumsum(dissipation + impulse_E)), initial=0))
    error_P = float(np.max(abs(np.diff(P, axis=0) - drag_P - wall[1:] - gravity_P), initial=0))
    peaks["energy"] = max(peaks["energy"], error_E)
    peaks["momentum"] = max(peaks["momentum"], error_P)
    assert error_E < 1e-8 and error_P < 1e-9
    gravity_work = -.5 * float(np.sum(q[-1, :, 2] - q[0, :, 2]))
    assert abs(K[-1] - K[0] - gravity_work - dissipation.sum() - impulse_E.sum()) < 1e-8
    assert abs(wall_work - impulse_E.sum()) < 1e-8
    ledger = {
        "run_id": key, "committed_steps": len(rows) - 1, "global_start_step": start_step,
        "initial_energy": float(E[0]), "final_energy": float(E[-1]),
        "initial_kinetic": float(K[0]), "final_kinetic": float(K[-1]),
        "damping_heat": -float(dissipation.sum()), "gravity_work": gravity_work, "wall_work": wall_work,
        "gravity_impulse": (gravity_P * (len(rows) - 1)).tolist(), "wall_impulse": wall.sum(axis=0).tolist(),
        "drag_impulse": drag_P.sum(axis=0).tolist(), "max_energy_residual": error_E,
        "max_linear_momentum_residual": error_P, "charged_uncommitted_queries": rec["counts"]["total_oracle_calls"] - calls[-1],
    }
    return (q, v, loss, calls, events), ledger


def map_audit(root, context, mode, run_id, raw, peaks):
    q, v, _, _, _ = raw
    packed = context["checkpoint"]
    before = np.array(packed["velocities"])
    after = v[0]
    assert np.array_equal(packed["positions"], q[0]) and np.array_equal(before[:, 2], after[:, 2])
    intervention = read(root / "interventions" / (run_id + ".json"))
    assert np.array_equal(intervention["start"]["positions"], q[0])
    assert np.array_equal(intervention["start"]["velocities"], after)
    assert intervention["damping_clock_at_start"] == PREFIX_STEPS
    delta = after - before
    assert np.array_equal(delta, intervention["velocity_delta"])
    expected = before.copy()
    if mode == "transfer":
        j = context["selected"]
        other = np.arange(4) != j
        center = before[:, :2].mean(axis=0)
        rel = before[:, :2] - center
        centered = rel[other] - rel[other].mean(axis=0)
        expected[j, :2] = center
        expected[other, :2] = center + centered * np.sqrt(np.sum(rel**2) / np.sum(centered**2))
    assert np.max(abs(expected - after)) < 1e-12
    K_change = .5 * float(np.sum(after**2) - np.sum(before**2))
    P_change = delta.sum(axis=0)
    angular_change = np.cross(q[0], delta).sum(axis=0)
    assert abs(K_change) < 1e-12 and np.max(abs(P_change)) < 1e-12
    peaks["map_kinetic_error"] = max(peaks["map_kinetic_error"], abs(K_change))
    peaks["map_linear_momentum_error"] = max(peaks["map_linear_momentum_error"], float(np.max(abs(P_change))))
    return {
        "run_id": run_id, "mode": mode, "instantaneous_energy_change": K_change,
        "total_linear_impulse": P_change.tolist(), "particle_impulses": delta.tolist(),
        "angular_momentum_change": angular_change.tolist(), "external_energy_supply": 0.,
        "binary_collisions": 0,
    }


def success(row):
    return row is not None and not row["failed"] and row["final_hit"]


def only_counts(examples):
    pairs = [(success(x["keep"]), success(x["transfer"])) for x in examples]
    return {
        "both": pairs.count((True, True)), "keep_only": pairs.count((True, False)),
        "transfer_only": pairs.count((False, True)), "neither": pairs.count((False, False)),
    }


def choose(rule, value):
    if rule["direction"] == 0:
        return rule["constant"]
    return "transfer" if rule["direction"] * (value - rule["cutoff"]) >= 0 else "keep"


def fit(examples, feature):
    observed = sorted({x["features"][feature] for x in examples})
    candidates = [{"direction": 0, "constant": mode} for mode in MODES]
    for a, b in zip(observed, observed[1:]):
        candidates.extend({"direction": d, "cutoff": (a + b) / 2} for d in (1, -1))
    for candidate in candidates:
        choices = [choose(candidate, x["features"][feature]) for x in examples]
        candidate["final_hits"] = sum(success(x[m]) for x, m in zip(examples, choices))
        candidate["transfers"] = choices.count("transfer")
    best = min(candidates, key=lambda x: (-x["final_hits"], x["transfers"]))
    return {"rule": best, "all_candidates": candidates}


def auc(examples, feature, direction):
    good = [x["features"][feature] for x in examples if success(x["transfer"]) and not success(x["keep"])]
    bad = [x["features"][feature] for x in examples if success(x["keep"]) and not success(x["transfer"])]
    if not good or not bad:
        return None
    return sum((direction * (a - b) > 0) + .5 * (a == b) for a in good for b in bad) / (len(good) * len(bad))


def development_lock(examples, energies):
    groups = {energy: [x for x in examples if x["incumbent_energy"] == energy and x["valid_pair"]] for energy in energies}
    counts = {energy: only_counts(group) for energy, group in groups.items()}
    energy = min(energies, key=lambda e: (-min(counts[e]["keep_only"], counts[e]["transfer_only"]),
                                          -counts[e]["keep_only"] - counts[e]["transfer_only"], e))
    chosen = groups[energy]
    ranks = {}
    for feature in ALL_FEATURES:
        forward = auc(chosen, feature, 1)
        direction = -1 if forward is not None and forward < .5 else 1
        ranks[feature] = {"direction": direction, "auc": auc(chosen, feature, direction)}
    return {
        "energy": energy, "regimes": counts,
        "enough_discordance": min(counts[energy]["keep_only"], counts[energy]["transfer_only"]) >= 5,
        "rules": {name: fit(chosen, name) for name in MODEL_FEATURES}, "ranks": ranks,
    }


def summaries(examples, lock, *, common_time=False):
    output = {}
    policies = (*MODES, *MODEL_FEATURES)
    for condition in ("bad_only", "good_present"):
        group = [x for x in examples if x["condition"] == condition]
        output[condition] = {}
        for policy in policies:
            chosen = [(x, policy if policy in MODES else
                       choose(lock["rules"][policy]["rule"], x["features"][policy])) for x in group if x["valid_pair"]]
            rows = [x[m + "_common"] if common_time else x[m] for x, m in chosen]
            output[condition][policy] = {
                "n": len(group), "valid_pairs": len(rows), "transfers": sum(m == "transfer" for _, m in chosen),
                "prefix_deviations": sum(x["prefix_deviation"] for x, _ in chosen),
                "deviations": sum(r["particle0_harmed"] for r in rows),
                "new_deviations_after_prefix": sum(r["particle0_harmed"] and not x["prefix_deviation"]
                                                   for (x, _), r in zip(chosen, rows)),
                "good_particle_final_hits": sum(r["particle0_final_hit"] for r in rows),
                "final_hits": sum(r["final_hit"] for r in rows), "historical_hits": sum(r["historical_hit"] for r in rows),
                "median_final_value": median(r["final_value"] for r in rows) if rows else None,
                "median_best_value": median(r["best_value"] for r in rows) if rows else None,
            }
    return output


def screen(examples, lock):
    bad = [x for x in examples if x["condition"] == "bad_only" and x["valid_pair"]]
    counts = only_counts(bad)
    enough = min(counts["keep_only"], counts["transfer_only"]) >= 5
    normal = summaries(examples, lock)
    escape = normal["bad_only"]
    protect = normal["good_present"]
    impulse = "horizontal_impulse"
    checks = {
        "all_pairs_valid": all(x["valid_pair"] for x in examples),
        "development_both_directions_at_least_five": lock["enough_discordance"],
        "held_out_both_directions_at_least_five": enough,
        "bad_hits_at_least_two_above_each_baseline":
            escape[impulse]["final_hits"] >= max(escape[p]["final_hits"] for p in (*MODES, "loss_score")) + 2,
        "good_deviations_within_two_of_transfer": protect[impulse]["deviations"] - protect["transfer"]["deviations"] <= 2,
        "good_final_hits_within_two_of_transfer":
            protect["transfer"]["good_particle_final_hits"] - protect[impulse]["good_particle_final_hits"] <= 2,
    }
    return {
        "checks": checks, "passed_descriptive_screen": all(checks.values()),
        "same_budget": normal, "common_time": summaries(examples, lock, common_time=True),
        "bad_paired_actions": counts,
        "bad_common_time_paired_actions": only_counts([
            {m: x[m + "_common"] for m in MODES} for x in bad]),
        "bad_ranks": {f: {"direction": lock["ranks"][f]["direction"],
                          "auc": auc(bad, f, lock["ranks"][f]["direction"])} for f in ALL_FEATURES},
        "rank_comparison_interpretable": enough and lock["enough_discordance"],
    }


def audit(root):
    config = read(root / "config.json")
    contexts, prefixes, runs, directs = (read(root / (n + ".json")) for n in ("contexts", "prefixes", "runs", "uninterrupted"))
    costs = read(root / "execution_counts.json")
    assert config["seeds"] == SEEDS[config["stage"]]
    expected = {(s, c, e) for s in config["seeds"] for c in config["conditions"] for e in config["initial_pivot_kinetics"]}
    assert len(contexts) == len(prefixes) == len(expected)
    assert {(c["seed"], c["condition"], c["incumbent_energy"]) for c in contexts} == expected
    assert len(runs) == costs["suffix_executions"] and len(directs) == costs["uninterrupted_replication_executions"]
    assert len(prefixes) == costs["shared_prefix_executions"] and len(contexts) == costs["contexts"]
    assert sha(root / "source.zip") == config["source_archive_sha256"]
    with zipfile.ZipFile(root / "source.zip") as archive:
        for path, digest in config["source_sha256"].items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == digest
    for path, digest in config["dependency_data_sha256"].items():
        assert sha(Path(path)) == digest
    actual = (sum(r["counts"]["total_oracle_calls"] for r in prefixes) +
              sum(r["actual_suffix_calls"] for r in runs) + sum(r["counts"]["total_oracle_calls"] - 4 for r in directs))
    assert actual == costs["actual_solver_queries"]
    assert actual + 4 * len(contexts) == costs["actual_queries_including_setup"]
    totals = Counter()
    peaks = dict(energy=0., momentum=0., reflection=0., map_kinetic_error=0., map_linear_momentum_error=0.,
                 uninterrupted_position_or_velocity_error=0.)
    ledgers, maps, examples = [], [], []
    by_prefix = {r["context_id"]: r for r in prefixes}
    by_direct = {r["context_id"]: r for r in directs}
    for c in sorted(contexts, key=lambda item: item["context_id"]):
        key = c["context_id"]
        p, ledger = segment_audit(root, by_prefix[key], c, totals, peaks)
        ledgers.append(ledger)
        assert p[3][0] == 8 and np.array_equal(c["start"]["positions"], p[0][0])
        assert np.array_equal(c["start"]["velocities"], p[1][0])
        assert abs(.5 * np.sum(p[1][0]**2) - .6) < 1e-12
        assert np.linalg.norm(p[1][0].sum(axis=0)) < 1e-12
        assert abs(.5 * np.sum(p[1][0, 0]**2) - c["incumbent_energy"]) < 1e-12
        assert c["setup_reference_values"] == 4
        x = {"context_id": key, "seed": c["seed"], "condition": c["condition"], "incumbent_energy": c["incumbent_energy"],
             "valid_pair": False}
        if c.get("unavailable_reason"):
            examples.append(x)
            continue
        x.update(features=c["features"], prefix_calls=c["prefix_calls"],
                 prefix_deviation=bool(np.max(p[2][:, 0] - c["values"][0]) > 1e-4))
        assert p[3][-1] == c["prefix_calls"] and len(p[3]) == PREFIX_STEPS + 1
        assert np.array_equal(p[0][-1], c["checkpoint"]["positions"])
        assert np.array_equal(p[1][-1], c["checkpoint"]["velocities"])
        vals = p[2][-1]
        j = int(np.argmin(vals))
        assert j == c["selected"] and np.array_equal(vals, c["checkpoint_values"])
        s = np.sort(vals)
        J = sum((np.array(e["after_i"][:2]) - e["before_i"][:2] for e in p[4] if e["i"] == j), np.zeros(2))
        u = p[1][-1, j, :2] - p[1][-1, :, :2].mean(axis=0)
        assert abs(c["features"]["loss_score"] - (s[2] - s[0]) / max(s[3] - s[1], 1e-12)) < 1e-12
        assert abs(c["features"]["horizontal_impulse"] - np.linalg.norm(J)) < 1e-12
        assert abs(c["features"]["relative_kinetic"] - .5 * (u @ u)) < 1e-12
        pair = {r["policy"]: r for r in runs if r["context_id"] == key}
        assert set(pair) == set(MODES)
        raw = {}
        for mode, row in pair.items():
            assert row["actual_suffix_calls"] + c["prefix_calls"] == row["counts"]["total_oracle_calls"]
            assert row["counts"]["total_oracle_calls"] <= config["query_cap"]
            if config["stage"] != "preflight" and not row["failed"]:
                assert row["counts"]["total_oracle_calls"] == config["query_cap"]
            data, ledger = segment_audit(root, row, c, totals, peaks, start_step=PREFIX_STEPS)
            ledgers.append(ledger)
            maps.append(map_audit(root, c, mode, row["run_id"], data, peaks))
            assert data[3][0] == c["prefix_calls"] + 4
            assert np.array_equal(data[2][0], vals)
            q = np.concatenate((p[0], data[0][1:]))
            f = np.concatenate((p[2], data[2][1:]))
            t = np.arange(len(f)) * config["h"]
            metrics = raw_metrics(f, q, t, c)
            for name, value in metrics.items():
                if isinstance(value, bool) or value is None:
                    assert value == row[name], (name, row["run_id"])
                else:
                    assert abs(value - row[name]) < 1e-8, (name, row["run_id"])
            x[mode] = row
            raw[mode] = (q, f, t, data)
        last = min(len(r[1]) for r in raw.values())
        for mode, (q, f, t, _) in raw.items():
            x[mode + "_common"] = {**pair[mode], **raw_metrics(f[:last], q[:last], t[:last], c)}
        x["valid_pair"] = not any(r["failed"] for r in pair.values())
        if key in by_direct:
            direct, ledger = segment_audit(root, by_direct[key], c, totals, peaks)
            ledgers.append(ledger)
            keep = raw["keep"][3]
            last = min(len(direct[3]), len(p[3]) + len(keep[3]) - 1)
            for before, after, target in zip(p[:3], keep[:3], direct[:3]):
                merged = np.concatenate((before, after[1:]))
                error = float(np.max(abs(merged[:last] - target[:last])))
                assert error < 1e-12
                peaks["uninterrupted_position_or_velocity_error"] = max(peaks["uninterrupted_position_or_velocity_error"], error)
            for i in range(len(p[3]), last):
                assert keep[3][i-PREFIX_STEPS] == direct[3][i] + 4
        examples.append(x)
    create(root / "paired_examples.json", examples)
    create(root / "physical_ledgers.json", {"segments": ledgers, "interventions": maps, "new_oracle_calls_for_export": 0})
    if config["stage"] == "development":
        create(root / "selection.json", development_lock(examples, config["initial_pivot_kinetics"]))
    if config["stage"] == "formal":
        lock = read(Path(config["previous_stage"]) / "selection.json")
        create(root / "screen.json", screen(examples, lock))
    result = {
        "passed": True, "contexts": len(contexts), "shared_prefixes": len(prefixes), "suffixes": len(runs),
        "uninterrupted_replications": len(directs), "invalid_pairs": sum(not x["valid_pair"] for x in examples),
        "solver_failures": [r["run_id"] for r in prefixes + runs + directs if r["failed"]],
        "max_residuals": peaks, **totals,
    }
    create(root / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    audit(parser.parse_args().output)

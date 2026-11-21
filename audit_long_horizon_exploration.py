"""Audit E41 restart continuity and report paired exploration and protection."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from audit_shared_prefix_transfer import segment_audit
from docs.research.e40_exploration import crossings, timeline
from finalize_initial_budget import create
from run_later_reflection_warning import END_STEP, H, LIMIT, decode_rows, read, sha
from run_long_horizon_exploration import ANALYSIS, PARENT, STOP_STEP

MODES = ("e37","capped_policy")


def good_outcome(knots, initial, previous):
    first = previous
    late_crossing = first_recovery = first_recross = None
    above = knots[0][1]-initial>LIMIT
    assert not above
    for t,f in knots[1:]:
        now = f-initial>LIMIT
        if now and not above:
            if late_crossing is None:
                late_crossing = t
            if first is None:
                first = t
            elif first_recross is None:
                first_recross = t
        if above and not now and first_recovery is None:
            first_recovery = t
        above = now
    return {"previous_first_crossing":previous,"first_crossing":first,"first_after_t2_crossing":late_crossing,
            "first_recovery_after_t2":first_recovery,"first_recross_after_t2":first_recross,
            "final_rise":knots[-1][1]-initial,"within_limit_at_end":not above}


def score(saved,events,context,previous,protection):
    axis = np.array(context["rotation"])[0]
    boundary,target = context["boundary"],context["target"]
    particles = []
    for p in previous["particles"]:
        i = p["particle"]
        snapshots = [(r["time"],float(axis@r["positions"][i][:2]),r["values"][i]) for r in saved]
        reflections = [(e["time"],float(axis@e["point_i"][:2]),e["point_i"][-1]) for e in events if e["i"]==i]
        knots = sorted(snapshots+reflections)
        first = next((t for t,x,f in knots if x<boundary and f<=target),None)
        assert p["first_target_time"] is None
        end = snapshots[-1]
        particles.append({"particle":i,"entered_by_t2":bool(p["entry_times"]),
                          "entry_times_after_t2":[t for t in crossings(knots,boundary) if t>2.],"first_target_time":first,
                          "final_loss":end[2],"best_observed_loss":min(p["best_observed_loss"],*(f for _,_,f in knots)),
                          "final_target_hit":end[1]<boundary and end[2]<=target})
    good = [(r["time"],r["values"][0]) for r in saved]
    contact = [(e["time"],e["point_i"][-1]) for e in events if e["i"]==0]
    times = [p["first_target_time"] for p in particles if p["first_target_time"] is not None]
    return {"particles":particles,"first_target_time":min(times,default=None),
            "best_explorer_final_loss":min(p["final_loss"] for p in particles),
            "best_explorer_historical_loss":min(p["best_observed_loss"] for p in particles),
            "best_population_final_loss":min(saved[-1]["values"]),
            "good_saved":good_outcome(good,context["initial_loss"],protection["first_crossing"]),
            "good_with_contacts":good_outcome(sorted(good+contact),context["initial_loss"],
                                               protection["first_crossing_including_contacts"])}


def med(vals):
    return median(vals) if vals else None


def particle_summary(particles):
    times = [p["first_target_time"] for p in particles if p["first_target_time"] is not None]
    return {"particles":len(particles),"hit":len(times),"final_hit":sum(p["final_target_hit"] for p in particles),
            "first_hit_min":min(times,default=None),"first_hit_median":med(times),"first_hit_max":max(times,default=None),
            "median_final_loss":med([p["final_loss"] for p in particles])}


def compare_hits(pairs):
    hit = [(a["first_target_time"],b["first_target_time"]) for a,b in pairs]
    differences = [b-a for a,b in hit if a is not None and b is not None]
    return {"e37_only":sum(a is not None and b is None for a,b in hit),
            "capped_only":sum(a is None and b is not None for a,b in hit),
            "both":len(differences),"neither":sum(a is None and b is None for a,b in hit),
            "capped_earlier":sum(d<-1e-10 for d in differences),"capped_later":sum(d>1e-10 for d in differences),
            "median_time_delta":med(differences),"min_time_delta":min(differences,default=None),
            "max_time_delta":max(differences,default=None)}


def compare_loss(pairs,key):
    delta = [b[key]-a[key] for a,b in pairs]
    return {"median_delta":med(delta),"min_delta":min(delta,default=None),"max_delta":max(delta,default=None),
            "capped_lower":sum(x<-1e-12 for x in delta),"capped_higher":sum(x>1e-12 for x in delta),
            "tied":sum(abs(x)<=1e-12 for x in delta)}


def summary(rows):
    results = {"contexts":len(rows)}
    all_pairs = [(r["e37"],r["capped_policy"]) for r in rows]
    particle_pairs = [p for a,b in all_pairs for p in zip(a["particles"],b["particles"],strict=True)]
    entrants = [(a,b) for a,b in particle_pairs if a["entered_by_t2"]]
    assert all(a["entered_by_t2"]==b["entered_by_t2"] for a,b in particle_pairs)
    for i,mode in enumerate(MODES):
        cases = [r[mode] for r in rows]
        particles = [p[i] for p in particle_pairs]
        times = [r["first_target_time"] for r in cases if r["first_target_time"] is not None]
        result = {"prior_entrants":particle_summary([p[i] for p in entrants]),
                  "all_explorers":particle_summary(particles),"contexts_with_target":len(times),
                  "contexts_with_final_target":sum(any(p["final_target_hit"] for p in r["particles"]) for r in cases),
                  "median_context_first_target_time":med(times),
                  "entry_events_after_t2":sum(len(p["entry_times_after_t2"]) for p in particles),
                  "new_entrants_after_t2":sum(not p["entered_by_t2"] and bool(p["entry_times_after_t2"]) for p in particles),
                  **{f"median_{k}":med([r[k] for r in cases]) for k in (
                      "best_explorer_final_loss","best_explorer_historical_loss","best_population_final_loss")}}
        for key in ("good_saved","good_with_contacts"):
            group = [r[key] for r in cases]
            result[key] = {"ever_through_t5":sum(g["first_crossing"] is not None for g in group),
                           "any_crossing_after_t2":sum(g["first_after_t2_crossing"] is not None for g in group),
                           "first_crossing_after_t2":sum(g["previous_first_crossing"] is None and g["first_crossing"] is not None for g in group),
                           "previously_crossed_and_recrossed":sum(g["previous_first_crossing"] is not None and g["first_after_t2_crossing"] is not None for g in group),
                           "any_recross_after_t2":sum(g["first_recross_after_t2"] is not None for g in group),
                           "within_limit_at_end":sum(g["within_limit_at_end"] for g in group),
                           "median_final_rise":med([g["final_rise"] for g in group])}
        results[mode] = result
    results["paired"] = {"prior_entrant_hits":compare_hits(entrants),"all_explorer_hits":compare_hits(particle_pairs),
                         "context_hits":compare_hits(all_pairs),"prior_entrant_final_loss":compare_loss(entrants,"final_loss"),
                         "all_explorer_final_loss":compare_loss(particle_pairs,"final_loss"),
                         "best_explorer_final_loss":compare_loss(all_pairs,"best_explorer_final_loss"),
                         "best_population_final_loss":compare_loss(all_pairs,"best_population_final_loss")}
    return results


def audit(root):
    cfg = read(root/"config.json")
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(p))==digest for p,digest in cfg["dependency_data_sha256"].items())
    contexts = {r["context_id"]:r for r in read(PARENT/"contexts.json")}
    prior = {r["context_id"]:r for r in read(ANALYSIS/"per_context.json")}
    protection = {r["context_id"]:r for r in read(PARENT/"second_reflection"/"paired_outcomes.json")}
    runs = read(root/"runs.json")
    cases = read(root/"contexts.json")
    costs = read(root/"execution_counts.json")
    assert len(runs)==costs["continuations"]==len(cases)+sum(c["transferred"] for c in cases)
    assert costs["actual_new_queries"]==sum(r["actual_new_queries"] for r in runs)<=cfg["new_query_cap"]
    outcomes,ledgers = {},[]
    counts,peaks = Counter(),dict(energy=0.,momentum=0.,reflection=0.)
    for r in runs:
        name,mode = r["context_id"],r["mode"]
        assert r["stop_reason"]=="completed" and r["committed_steps"]==STOP_STEP-END_STEP,r
        assert sha(Path(r["source_trace"]))==r["source_sha256"]
        parent = decode_rows(Path(r["source_folder"]),r["source_run_id"])[-1]
        ctx = contexts[name]
        raw,ledger = segment_audit(root,r,ctx,counts,peaks,start_step=END_STEP)
        q,v,f,calls,events = raw
        assert all(np.array_equal(data[0],parent[key]) for data,key in ((q,"positions"),(v,"velocities"),(f,"values")))
        assert calls[0]==r["historical_queries_at_fork"]+4 and calls[-1]-int(parent["total_oracle_calls"])==r["actual_new_queries"]
        trace = [{"step":i,"positions":x.tolist(),"values":y.tolist()} for i,(x,y) in enumerate(zip(q,f))]
        saved,exact = timeline(trace,events,END_STEP)
        assert saved[0]["time"]==2. and saved[-1]["time"]==5.
        outcome = score(saved,exact,ctx,prior[name][mode],protection[name]["baseline" if mode=="e37" else "policy"])
        outcomes[name,mode] = outcome
        ledgers.append({**ledger,"external_energy_supply":0.,"new_interventions":0})
    rows = [{"context_id":c["context_id"],"seed":c["seed"],"incumbent_energy":c["incumbent_energy"],
             "transferred":c["transferred"],"e37":outcomes[c["context_id"],"e37"],
             "capped_policy":outcomes[c["context_id"],"capped_policy" if c["transferred"] else "e37"]} for c in cases]
    results = {"transferred_only":summary([r for r in rows if r["transferred"]]),"all_contexts":summary(rows)}
    if cfg["stage"]=="formal":
        assert results["transferred_only"]["e37"]["prior_entrants"]["particles"]==5
        assert results["all_contexts"]["e37"]["prior_entrants"]["particles"]==174
    create(root/"paired_outcomes.json",rows)
    create(root/"physical_ledgers.json",ledgers)
    create(root/"results.json",results)
    result = {"passed":True,"contexts":len(rows),"continuations":len(runs),"solver_failures":[],"max_residuals":peaks,**counts}
    create(root/"audit.json",result)
    print(result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    audit(parser.parse_args().output)

"""Derive explorer outcomes from the sealed E40 policy paths, with no oracle."""

import argparse
from collections import Counter
from functools import cache
from pathlib import Path
from statistics import median

import numpy as np

from finalize_initial_budget import create, sha
from run_conditional_transfer import read
from run_later_reflection_warning import BOUNDARY_STEP, END_STEP, H, decode_rows

ROOT = Path("output/collision_second_reflection_capped_policy")
DEFAULT_OUTPUT = Path("docs/research/e40_exploration")
EXPLORERS = (1, 2, 3)
EPS = 1e-10


def point(time, positions, values):
    return {"time":time,"positions":positions,"values":values}


def timeline(rows, events, start_step):
    offset = start_step*H
    return ([point((start_step+int(r["step"]))*H,r["positions"],r["values"]) for r in rows],
            [{**e,"time":offset+e["time"]} for e in events])


def prefix_path(data, context, family):
    name = context["context_id"]
    base_saved,base_events = data.segment(ROOT,name+"__base_prefix",0)
    if family=="base":
        return base_saved,base_events
    fork = 1 if family=="early" else 2
    after,after_events = data.segment(ROOT,f"{name}__{family}_prefix",fork)
    assert base_saved[fork]==after[0]
    return base_saved[:fork]+after,[e for e in base_events if e["time"]<=fork*H+EPS]+after_events


class Archive:
    def __init__(self):
        self.manifest = ROOT/"raw_artifact_manifest.json"
        self.expected = {r["path"]:r["sha256"] for r in read(self.manifest)["files"]}
        self.dependencies = {str(self.manifest):sha(self.manifest)}

    @cache
    def verify(self, path):
        digest = sha(path)
        assert digest==self.expected[str(path)],path
        self.dependencies[str(path)] = digest
        return path

    def read(self, path):
        return read(self.verify(path))

    def segment(self, root, name, start):
        self.verify(root/"traces"/(name+".csv.gz"))
        events = self.read(root/"events"/(name+".json"))
        return timeline(decode_rows(root,name),events,start)


def baseline_path(data, context, decision):
    name = decision["source"]
    family = name.rsplit("__",2)[-2]
    saved,events = prefix_path(data,context,family)
    contact = context["families"][family]
    frac = data.read(ROOT/"fractions"/(name+".json"))
    # The first reflection is recorded separately from the tail event list.
    event = {"time":contact["contact_time"],"kind":"wall","i":0,
             "point_i":contact["post_reflection"]["positions"][0]}
    tail,tail_events = data.segment(ROOT,name,BOUNDARY_STEP)
    assert tail[0]["positions"]==frac["end"]["positions"] and tail[0]["values"]==frac["values"]
    assert len(saved)==BOUNDARY_STEP and len(tail)==END_STEP-BOUNDARY_STEP+1
    return saved+tail,[*events,event,*[{**e,"time":contact["contact_time"]+e["time"]} for e in frac["events"]],*tail_events]


def capped_path(data, decision, saved, events):
    cp = decision["checkpoint"]
    time,step = cp["time"],cp["boundary_step"]
    folder = ROOT/"second_reflection"
    name = cp["checkpoint_id"]+"__capped"
    frac = data.read(folder/"fractions"/(name+".json"))
    tail,tail_events = data.segment(folder,name,step)
    assert frac["start"]["positions"]==cp["post_reflection"]["positions"]
    assert tail[0]["positions"]==frac["end"]["positions"] and tail[0]["values"]==frac["values"]
    before,after = (np.array(x["velocities"])[:,:2] for x in (cp["post_reflection"],frac["start"]))
    individual = .5*np.sum(after**2-before**2,axis=1)
    assert abs(individual.sum())<1e-12 and np.allclose(after.sum(axis=0),before.sum(axis=0),atol=1e-12,rtol=0)
    joined = saved[:step]+tail
    events = [e for e in events if e["time"]<=time+EPS]
    events += [{"time":time,"kind":"transfer","i":i,"point_i":frac["start"]["positions"][i]}
               for i in EXPLORERS]
    events += [{**e,"time":time+e["time"]} for e in frac["events"]]+tail_events
    return joined,events,individual.tolist()


def crossings(knots, boundary):
    """Horizontal paths are piecewise linear between saved points and wall events."""
    entries = []
    previous_side = knots[0][1]>=boundary
    previous = knots[0]
    for time,x,loss in knots[1:]:
        side = x>=boundary
        if side!=previous_side and not side:
            t0,x0,_ = previous
            entries.append(t0+(time-t0)*(x0-boundary)/(x0-x))
        previous = (time,x,loss)
        previous_side = side
    return entries


def score(saved, events, context):
    assert [r["time"] for r in saved]==[step*H for step in range(END_STEP+1)]
    assert all(a["time"]<=b["time"]+EPS for a,b in zip(events,events[1:]))
    axis = np.array(context["rotation"])[0]
    boundary,target = context["boundary"],context["target"]
    results = []
    for i in EXPLORERS:
        points = [(r["time"],float(axis@r["positions"][i][:2]),r["values"][i]) for r in saved]
        contacts = [(e["time"],float(axis@e["point_i"][:2]),e["point_i"][-1])
                    for e in events if e["i"]==i and e["kind"]=="wall"]
        transfer = [(e["time"],float(axis@e["point_i"][:2]),None)
                    for e in events if e["i"]==i and e["kind"]=="transfer"]
        observations = sorted(points+contacts)
        knots = sorted(observations+transfer,key=lambda k:k[0])
        assert points[0][1]>boundary
        entries = crossings(knots,boundary)
        time = next((t for t,x,f in observations if x<boundary and f<=target),None)
        end = points[-1]
        results.append({"particle":i,"entry_times":entries,"first_target_time":time,"final_loss":end[2],
                        "best_observed_loss":min(f for _,_,f in observations),"final_in_good_well":end[1]<boundary,
                        "final_target_hit":end[1]<boundary and end[2]<=target,
                        "closest_longitudinal_margin":min(x for _,x,_ in knots)-boundary})
    return results


def median_or_none(values):
    return median(values) if values else None


def case_score(particles):
    hits = [p["first_target_time"] for p in particles if p["first_target_time"] is not None]
    return {"entry_events":sum(len(p["entry_times"]) for p in particles),
            "particles_ever_entered":sum(bool(p["entry_times"]) for p in particles),
            "particles_ever_hit":len(hits),"first_target_time":min(hits,default=None),
            "any_final_target":any(p["final_target_hit"] for p in particles),
            "best_explorer_final_loss":min(p["final_loss"] for p in particles),
            "best_explorer_historical_loss":min(p["best_observed_loss"] for p in particles),
            "closest_longitudinal_margin":min(p["closest_longitudinal_margin"] for p in particles)}


def summarize(rows):
    policies = {}
    for mode in ("e37","capped_policy"):
        cases = [r[mode] for r in rows]
        particles = [p for c in cases for p in c["particles"]]
        times = [c["first_target_time"] for c in cases if c["first_target_time"] is not None]
        policies[mode] = {"entry_events":sum(c["entry_events"] for c in cases),
            "particles_ever_entered":sum(c["particles_ever_entered"] for c in cases),
            "contexts_ever_entered":sum(c["entry_events"]>0 for c in cases),
            "particles_ever_hit":sum(c["particles_ever_hit"] for c in cases),
            "contexts_ever_hit":len(times),"median_first_target_time_among_hits":median_or_none(times),
            "earliest_target_time":min(times,default=None),"latest_first_target_time":max(times,default=None),
            "contexts_final_hit":sum(c["any_final_target"] for c in cases),
            "median_best_explorer_final_loss":median(c["best_explorer_final_loss"] for c in cases),
            "median_explorer_final_loss":median(p["final_loss"] for p in particles),
            "median_best_explorer_historical_loss":median(c["best_explorer_historical_loss"] for c in cases),
            "min_longitudinal_margin":min(c["closest_longitudinal_margin"] for c in cases)}
    paired = {}
    for metric in ("best_explorer_final_loss","best_explorer_historical_loss","entry_events"):
        diff = [r["capped_policy"][metric]-r["e37"][metric] for r in rows]
        paired[metric] = {"median_delta":median(diff),"min_delta":min(diff),"max_delta":max(diff),
                          "lower":sum(d<-1e-12 for d in diff),"higher":sum(d>1e-12 for d in diff),
                          "tied":sum(abs(d)<=1e-12 for d in diff)}
    gains = [r["horizontal_kinetic_change"] for r in rows if r["transferred"]]
    return {"contexts":len(rows),"explorer_pairs":len(rows)*len(EXPLORERS),**policies,"paired":paired,
            "energy_at_transfer":{
                "cases":len(gains),"median_total_explorer_gain":median_or_none([sum(x[1:]) for x in gains]),
                "min_total_explorer_gain":min((sum(x[1:]) for x in gains),default=None),
                "max_total_explorer_gain":max((sum(x[1:]) for x in gains),default=None),
                "individual_explorers_gained":sum(y>0 for x in gains for y in x[1:]),
                "individual_explorers_lost":sum(y<0 for x in gains for y in x[1:])}}


def derive(output):
    data = Archive()
    contexts = {c["context_id"]:c for c in data.read(ROOT/"contexts.json")}
    decisions = data.read(ROOT/"second_reflection"/"contexts.json")
    reference = {r["context_id"]:r for r in data.read(ROOT/"second_reflection"/"paired_outcomes.json")}
    assert len(decisions)==len(contexts)==len(reference)==480
    assert Counter(c["seed"] for c in decisions)=={s:6 for s in range(2720,2800)}
    rows = []
    for decision in decisions:
        name = decision["context_id"]
        context = contexts[name]
        saved,events = baseline_path(data,context,decision)
        assert abs(saved[-1]["values"][0]-context["initial_loss"]-reference[name]["baseline"]["final_rise"])<1e-12
        old = score(saved,events,context)
        new,kinetic = old,None
        if decision["transferred"]:
            cap,cap_events,kinetic = capped_path(data,decision,saved,events)
            assert abs(cap[-1]["values"][0]-context["initial_loss"]-reference[name]["policy"]["final_rise"])<1e-12
            new = score(cap,cap_events,context)
        rows.append({"context_id":name,"seed":decision["seed"],"incumbent_energy":decision["incumbent_energy"],
                     "transferred":decision["transferred"],"horizontal_kinetic_change":kinetic,
                     "target":context["target"],"boundary":context["boundary"],
                     "e37":{**case_score(old),"particles":old},"capped_policy":{**case_score(new),"particles":new}})
    treated = [r for r in rows if r["transferred"]]
    assert len(treated)==20 and all(r["e37"]==r["capped_policy"] for r in rows if not r["transferred"])
    output.mkdir(parents=True)
    create(output/"per_context.json",rows)
    summary = {"transferred_only":summarize(treated),"all_contexts":summarize(rows),
               "new_trajectories":0,"new_oracle_calls":0,"source_files_checked":len(data.dependencies)}
    create(output/"summary.json",summary)
    create(output/"provenance.json",{"inputs_sha256":data.dependencies,"script_sha256":sha(Path(__file__)),
        "launch_command":"PYTHONPATH=src:. python3 docs/research/e40_exploration.py",
        "new_trajectories":0,"new_oracle_calls":0})
    print(summary)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT)
    derive(parser.parse_args().output)

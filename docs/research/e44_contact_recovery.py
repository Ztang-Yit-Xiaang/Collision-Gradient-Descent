"""Export E43 contact-only mechanisms and E44 recovery timelines without an oracle."""

from pathlib import Path

import numpy as np

from finalize_initial_budget import create
from run_capped_reflection_transfer import checked_hashes
from run_later_reflection_warning import H, LIMIT
from run_third_reflection_guard import contacts, decode_rows, history, points, sources, read, sha
from run_third_reflection_replication_branches import FreshArchive
from run_third_reflection_saved_window import BASE, PARENT


def load(root):
    paths = [root/"third_reflection"/"paired_outcomes.json",root/"second_reflection"/"contexts.json",root/"contexts.json"]
    rows,cases,contexts = [read(p) for p in paths]
    return rows,{r["context_id"]:r for r in cases},{r["context_id"]:r for r in contexts},paths


def e43():
    rows,cases,contexts,paths = load(PARENT)
    data,output = FreshArchive(),[]
    for r in rows:
        if not r["transferred"] or r["baseline"]["saved"] is not None:
            continue
        key = r["context_id"]
        case,ctx = cases[key],contexts[key]
        initial = ctx["initial_loss"]
        saved,hits,*_ = history(data,case,ctx,parent=PARENT,extension=PARENT/"extension")
        t,loss = hits[3]
        mode,_,_ = list(sources(case,PARENT))[-1]
        events = data.read(PARENT/"extension"/"events"/(key+"__"+mode+".json"))
        event = next(e for e in events if e["i"]==0 and abs(2+e["time"]-t)<1e-10)
        n = np.asarray(event["normal"])
        gradient = -n[:2]/n[-1]
        slopes = [float(gradient@event[k][:2]) for k in ("before_i","after_i")]
        before = next((s,f-initial) for s,f in reversed(saved) if s<t)
        after = next((s,f-initial) for s,f in saved if s>t)
        assert before[1]<LIMIT<loss-initial and after[1]<LIMIT and slopes[0]>0>slopes[1]
        output.append({"seed":r["seed"],"context_id":key,"previous_saved":before,"contact":[t,loss-initial],
            "next_saved":after,"directional_loss_rate_before":slopes[0],"directional_loss_rate_after":slopes[1]})
    assert len(output)==4
    deps = checked_hashes(PARENT,[*paths,*(Path(p) for p in data.dependencies)])
    create(BASE/"e43_contact_only_diagnosis.json",{"rows":output,"new_queries":0,"dependency_data_sha256":deps})


def e44():
    rows,cases,contexts,paths = load(BASE)
    data,output = FreshArchive(),[]
    target = BASE/"third_reflection"
    decisions = {r["context_id"]:r for r in data.read(target/"decisions.json")}
    for r in rows:
        modes = ("original_policy","saved_window_policy")
        if not any(r[m]["including_contacts"] is not None for m in modes):
            continue
        key = r["context_id"]
        case,ctx = cases[key],contexts[key]
        saved,hits,*_ = history(data,case,ctx,parent=BASE,extension=BASE/"extension")
        initial = ctx["initial_loss"]
        cp = decisions[key]["checkpoint"]
        timelines = {}
        for mode,transfer in zip(modes,("original_transfer","saved_window_transfer")):
            observed = saved+hits
            if r[transfer]:
                run_id = cp["checkpoint_id"]+"__capped"
                tail,events = data.trace(target,run_id)
                fraction = data.read(target/"fractions"/(run_id+".json"))
                observed = [(t,f) for t,f in observed if t<=cp["time"]]
                observed += points(tail,cp["boundary_step"])+contacts(events,cp["boundary_step"]*H)
                observed += contacts(fraction["events"],cp["time"])
            changes = []
            above = False
            for t,f in sorted(observed):
                now = f-initial>LIMIT
                if now!=above:
                    changes.append({"time":t,"loss_rise":f-initial,"type":"crossing" if now else "recovery"})
                    above = now
            assert bool(changes)==(r[mode]["including_contacts"] is not None)
            timelines[mode] = changes
        output.append({"seed":r["seed"],"context_id":key,"checkpoint_time":r["checkpoint_time"],
                       "transitions":timelines,"retained_recovery":r["retained_recovery"]})
    create(BASE/"third_reflection"/"recovery_timelines.json",{"rows":output,"new_queries":0,
        "dependency_data_sha256":{**data.dependencies,**{str(p):sha(p) for p in paths}}})


if __name__=="__main__":
    e43()
    e44()

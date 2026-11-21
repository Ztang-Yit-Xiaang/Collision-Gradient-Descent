"""Derive the common retained trajectories' crossing and recovery events."""

from finalize_initial_budget import create
from run_later_reflection_warning import LIMIT
from run_third_reflection_contact_gate import BASE
from run_third_reflection_guard import history, read
from run_third_reflection_replication_branches import FreshArchive


def export(root):
    target = root/"third_reflection"
    rows = read(target/"contact_gate_outcomes.json")
    cases = {r["context_id"]:r for r in read(root/"second_reflection"/"contexts.json")}
    contexts = {r["context_id"]:r for r in read(root/"contexts.json")}
    data = FreshArchive()
    output = []
    for r in rows:
        if r["saved_window_policy"]["including_contacts"] is None and r["gated_policy"]["including_contacts"] is None:
            continue
        assert r["saved_window_policy"]==r["gated_policy"]==r["baseline"]
        key = r["context_id"]
        ctx = contexts[key]
        saved,hits,*_ = history(data,cases[key],ctx,parent=root,extension=root/"extension")
        above = False
        transitions = []
        for time,loss in sorted(saved+hits):
            rise = loss-ctx["initial_loss"]
            now = rise>LIMIT
            if now!=above:
                transitions.append({"time":time,"loss_rise":rise,"type":"crossing" if now else "recovery"})
                above = now
        output.append({"seed":r["seed"],"context_id":key,"predecision_history":r["retained_recovery"] is not None,
                       "transitions":transitions})
    create(target/"recovery_timelines.json",{"rows":output,"new_queries":0,"dependency_data_sha256":data.dependencies})


if __name__=="__main__":
    export(BASE)

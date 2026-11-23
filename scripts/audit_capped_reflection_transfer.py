"""Compare E39 with the two sealed E38 controls and audit conservation."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median

import numpy as np

from audit_later_reflection_warning import audit_branch, check_fraction, update_peak
from finalize_initial_budget import create
from run_capped_reflection_transfer import BASE, checked_hashes
from run_later_reflection_warning import BOUNDARY_STEP, H, LIMIT, decode_rows, read, sha

MODES = ("keep","transfer","capped")


def capped_fraction(frac, checkpoint, mode, peaks):
    v = np.array(checkpoint["post_reflection"]["velocities"])[:,:2]
    after = np.array(frac["start"]["velocities"])[:,:2]
    target = after[0]
    before_k, after_k = .5*float(v[0]@v[0]), .5*float(target@target)
    assert after_k<=before_k+1e-14
    center = v.mean(axis=0)
    available = np.linalg.norm(v-center)**2
    feasible_min = max(0.,np.linalg.norm(center)-np.sqrt(.75*available))
    update_peak(peaks,"capped_speed_optimality",abs(np.linalg.norm(target)-feasible_min))
    required = float(np.sum(v**2)-target@target-np.linalg.norm(v.sum(axis=0)-target)**2/3)
    deviations = v[1:]-v[1:].mean(axis=0)
    spread = float(np.sum(deviations**2))
    mean = (v.sum(axis=0)-target)/3
    expected = mean+deviations*np.sqrt(max(required,0.)/spread) if spread else np.broadcast_to(mean,(3,2))
    update_peak(peaks,"spectator_velocity",np.max(abs(expected-after[1:])))
    ledger = check_fraction(frac,checkpoint,mode,peaks,expected_horizontal=after)
    return {**ledger,"incumbent_horizontal_kinetic_before":before_k,"incumbent_horizontal_kinetic_after":after_k,
            "feasible_min_horizontal_speed":feasible_min}


def summary(rows):
    warned = [r for r in rows if r["warning"]]
    recovery = [r for r in rows if r["already_deviated"]]
    prevention = {}
    for mode in MODES:
        group = [r["branches"][mode] for r in warned]
        saved = "first_saved_crossing"
        both = [(r["branches"]["keep"][saved],r["branches"][mode][saved]) for r in warned
                if r["branches"]["keep"][saved] is not None and r["branches"][mode][saved] is not None]
        prevention[mode] = {
            "states":len(group),"deviated_by_t1":sum(g[saved] is not None and g[saved]<=1.+1e-10 for g in group),
            "deviated_by_t2":sum(g[saved] is not None for g in group),
            "deviated_including_contacts":sum(g["first_crossing_including_contacts"] is not None for g in group),
            "earlier_than_keep":sum(b<a-1e-10 for a,b in both),"later_than_keep":sum(b>a+1e-10 for a,b in both),
            "within_limit_at_end":sum(g["within_limit_at_end"] for g in group),
            "median_final_rise":median(g["final_rise"] for g in group) if group else None,
            "median_peak_saved_rise":median(g["peak_saved_rise"] for g in group) if group else None,
            "median_best_saved_loss":median(g["best_saved_loss"] for g in group) if group else None,
            "median_horizontal_kinetic_after":median(g["horizontal_kinetic_after"] for g in group) if group else None,
        }
    comparisons = {}
    for baseline in ("keep","transfer"):
        baseline_crossed = [r["branches"][baseline][saved] is not None for r in warned]
        capped_crossed = [r["branches"]["capped"][saved] is not None for r in warned]
        a = sum(b and not c for b,c in zip(baseline_crossed,capped_crossed))
        b = sum(c and not b for b,c in zip(baseline_crossed,capped_crossed))
        comparisons[baseline] = {"additional_prevented":a,"newly_deviated":b,"net_additional_prevented":a-b}
    def recovery_metrics(group, mode):
        results = [r["branches"][mode] for r in group]
        return {"states":len(group),"recovered_or_already_within":sum(g["first_recovery"] is not None for g in results),
                "recrossed":sum(g["first_recross"] is not None for g in results),
                "within_limit_at_end":sum(g["within_limit_at_end"] for g in results)}
    checks = {"at_least_three_more_prevented_than_centroid":comparisons["transfer"]["net_additional_prevented"]>=3,
              "none_by_t1":prevention["capped"]["deviated_by_t1"]==0,
              "none_earlier_than_keep":prevention["capped"]["earlier_than_keep"]==0}
    return {"prevention":prevention,"capped_comparisons":comparisons,"recovery":{
        name:{mode:recovery_metrics(group,mode) for mode in MODES} for name,group in (
            ("all",recovery),("above_at_contact",[r for r in recovery if r["above_at_contact"]]),
            ("already_within",[r for r in recovery if not r["above_at_contact"]]))},
        "no_ops":sum(r["no_op"] for r in rows),"screen":checks,"passed_screen":bool(warned) and all(checks.values())}


def audit(root):
    cfg = read(root/"config.json")
    assert sha(root/"source.zip")==cfg["source_archive_sha256"]
    assert all(sha(Path(p))==digest for p,digest in cfg["dependency_data_sha256"].items())
    parent, e37 = Path(cfg["parent"]), Path(cfg["source_e37"])
    checks,runs,cost = (read(root/(name+".json")) for name in ("contexts","runs","execution_counts"))
    parents = {r["run_id"]:r for r in read(parent/"runs.json")}
    cases = {r["context_id"]:r for r in read(e37/"contexts.json")}
    parent_cases = {r["context_id"]:r for r in read(parent/"contexts.json")}
    dependencies = {}
    peaks, counts, reused_counts = dict(energy=0.,momentum=0.,reflection=0.),Counter(),Counter()
    outcomes,ledgers = [],[]
    assert len(checks)==len(runs)==cost["continuations"]==cost["reused_checkpoint_states"]
    assert len(checks)==(2 if cfg["stage"]=="preflight" else 40)
    for c,r in zip(checks,runs,strict=True):
        key = c["checkpoint_id"]
        assert r["checkpoint_id"]==key and not r["failed"]
        ctx = cases[c["context_id"]]
        history = parent_cases[c["context_id"]]
        src = history["source_branch"]["run_id"]
        kind = src.rsplit("__",2)[-2]
        dependencies.update(checked_hashes(e37,[e37/"traces"/(src+".csv.gz")]))
        trace = decode_rows(e37,src)
        before_tail = [ctx["initial_loss"]+rise for rise in ctx["families"][kind]["history_rises"]]
        past = [*before_tail,*[x["values"][0] for x in trace[:c["boundary_step"]-BOUNDARY_STEP]]]
        results = {}
        for mode in MODES:
            folder = root if mode=="capped" else parent
            run = r if mode=="capped" else parents[key+"__"+mode]
            run_id = run["run_id"]
            paths = [folder/d/(run_id+s) for d,s in (
                ("traces",".csv.gz"),("events",".json"),("fractions",".json"))]
            if mode!="capped":
                dependencies.update(checked_hashes(parent,paths))
            res,*_,ledger = audit_branch(folder,run,c,ctx,counts if mode=="capped" else reused_counts,peaks,
                                         fraction_audit=capped_fraction if mode=="capped" else check_fraction)
            if mode!="capped":
                assert res==c["branches"][mode]
            v = np.array(read(paths[-1])["start"]["velocities"])[0,:2]
            res["horizontal_kinetic_after"] = .5*float(v@v)
            res["peak_saved_rise"] = max(res["peak_saved_rise"],max(past)-c["initial_loss"])
            res["best_saved_loss"] = min(res["best_saved_loss"],*past)
            results[mode] = res
            ledgers.append({"mode":mode,"source":str(folder),**ledger})
        outcomes.append({"checkpoint_id":key,"context_id":c["context_id"],"seed":r["seed"],"time":c["time"],
                         "warning":c["warning"],"already_deviated":c["already_deviated"],
                         "above_at_contact":c["above_at_contact"],"no_op":r["no_op"],"branches":results})
    assert cost["actual_new_solver_queries"]==cost["actual_fraction_queries"]+cost["actual_tail_queries"]<=cfg["total_new_query_cap"]
    assert cost["actual_new_solver_queries"]==sum(r["counts"]["total_oracle_calls"] for r in runs)
    result = {"passed":True,"solver_failures":[],"continuations":len(runs),"max_residuals":peaks,
              "new_continuation_checks":dict(counts),"reused_control_checks":dict(reused_counts),
              "value_checks":counts["value_checks"]+reused_counts["value_checks"],
              "gradient_checks":counts["gradient_checks"]+reused_counts["gradient_checks"]}
    create(root/"paired_outcomes.json",outcomes)
    create(root/"already_deviated_cases.json",[r for r in outcomes if r["already_deviated"]])
    create(root/"physical_ledgers.json",ledgers)
    create(root/"control_dependency_hashes.json",dependencies)
    create(root/"results.json",summary(outcomes))
    create(root/"audit.json",result)
    print(result)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path,default=BASE,nargs="?")
    audit(parser.parse_args().output)

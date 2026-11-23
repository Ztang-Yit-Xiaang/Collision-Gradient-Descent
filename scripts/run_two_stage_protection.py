"""E35: execute the shared prefix tree for four protection rules."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import sys
import traceback

import numpy as np

from collision_optim.contact_long_forecast import forecast_from_reflection
from collision_optim.contact_time_transfer import first_reflection
from collision_optim.contact_window import forecast_first_contact
from collision_optim.curvature_energy import IndependentEnergyPopulation, initial_state
from collision_optim.energy_generalization import controlled_velocities
from collision_optim.event_replay import pack, unpack
from collision_optim.gravity import GravityConfig
from collision_optim.hybrid import ContactResolutionError
from collision_optim.initial_budget import allocate_initial
from collision_optim.population import PopulationConfig, PopulationOracle, run_population
from collision_optim.prebounce_warning import flight_factors, forecast_rise
from collision_optim.shared_prefix import transfer_at_checkpoint
from collision_optim.strategic_experiment import _csv, _json
from run_conditional_transfer import read, sha
from run_contact_warning_replication import BOUNDARY_STEP, PRECONTACT_STEP, continue_branch, snapshot
from run_overlap_transfer import build_context
from run_postbounce_warning_transfer import load_counters
from run_prebounce_warning import ENERGIES
from run_shared_prefix_transfer import save_segment, stop_record

SPEC = Path("docs/superpowers/specs/2026-11-11-two-stage-protection-design.md")
WARNING_SOURCE = Path("output/collision_prebounce_warning_development/model_lock.json")
PARENTS = (Path("output/collision_contact_long_forecast/config.json"),
           Path("output/collision_high_kinetic_intervention_timing/config.json"))
SEEDS = {"preflight": [2490], "formal": list(range(2500, 2540))}
END_STEP, PRIMARY_STEP = 40, 20
CAP, H, LIMIT = 960, .05, 1e-4
POLICIES = ("keep", "early_only", "reflection_only", "two_stage")


def family(root, name, kind, initial_loss, rises, prefix, oracle, cfg):
    source = f"{name}__{kind}"
    prior_charge = sum(oracle.counts().values())
    record = {
        "family_id": source, "history_rises": rises, "before_step": pack(prefix.final_state),
        "previous_step": PRECONTACT_STEP, "boundary_step": BOUNDARY_STEP, "complete": False,
        "actual_contact_queries": 0, "actual_forecast_queries": 0, "initial_loss": initial_loss,
    }
    target_failure = None
    if any(e["i"] == 0 for e in prefix.events):
        target_failure = "first_reflection_before_step_7"
    if prefix.stop_reason != "completed":
        target_failure = prefix.stop_reason
    if target_failure:
        return {**record, "unavailable_reason": target_failure}, []
    try:
        outgoing, contact = first_reflection(oracle, prefix.final_state, cfg, PRECONTACT_STEP)
    except ContactResolutionError as exc:
        return {**record, "unavailable_reason": str(exc),
                "actual_contact_queries": sum(oracle.counts().values()) - prior_charge}, []
    after_contact = sum(oracle.counts().values())
    record["actual_contact_queries"] = after_contact-prior_charge
    if contact["index"] != 0:
        return {**record, "unavailable_reason": "first_event_in_step_7_is_another_particle"}, []
    remaining = H-contact["offset"]
    contact_rise = contact["contact_value"] - initial_loss
    already = max(*rises, contact_rise) > LIMIT
    prediction = None
    warning = False
    if not already:
        _, prediction = forecast_from_reflection(
            oracle, outgoing.particles[0], remaining, cfg, initial_loss, LIMIT, BOUNDARY_STEP, PRIMARY_STEP)
        warning = any(p["loss_rise"] > LIMIT for p in prediction["observations"])
    charges = oracle.counts()
    total = sum(charges.values())
    record.update(complete=True, post_reflection=pack(outgoing), contact=contact,
                  contact_time=PRECONTACT_STEP*H+contact["offset"], remaining_duration=remaining,
                  contact_rise=contact_rise, already_deviated=already,
                  above_at_contact=contact_rise>LIMIT, earlier_saved_deviation=max(rises)>LIMIT,
                  prediction=prediction, warning=warning, actual_forecast_queries=total-after_contact,
                  charged_at_fork=total)
    _json(root/"checkpoints"/(source+".json"), record)
    branches = []
    modes = ("keep", "transfer") if warning or already else ("keep",)
    for mode in modes:
        state = transfer_at_checkpoint(outgoing, 0) if mode == "transfer" else outgoing
        branch_oracle = PopulationOracle(oracle.objective, total_budget=CAP)
        load_counters(branch_oracle, charges)
        row = continue_branch(root, source, mode, state, branch_oracle, cfg, remaining, end_step=END_STEP)
        branches.append({**row, "initial_context_id": name, "family": kind,
                         "actual_suffix_calls": row["actual_fraction_calls"]+row["actual_tail_calls"]})
    return record, branches


def simulate(root, seed, energy, curvature, preflight, second_stage=False, guard_fraction=None):
    objective, points, values, _, c = build_context("good_present", seed, 300, 100)
    velocity = controlled_velocities(seed, energy)
    init = initial_state(points, values, allocate_initial(values, (0.,)*4, "fixed_ref_keep", velocity), velocity)
    cfg = PopulationConfig(physics=GravityConfig(), rho=.24, collisions=False)
    name = f"good_present__s{seed}__k{energy:g}"
    oracle = PopulationOracle(objective, total_budget=CAP)
    oracle.values = 4
    prefix = run_population(IndependentEnergyPopulation(cfg), oracle, init, PRECONTACT_STEP)
    key = name+"__base_prefix"
    save_segment(root, key, prefix)
    rows = [{"run_id": key, "initial_context_id": name, "family": "base",
             "actual_prefix_calls": prefix.counts["total_oracle_calls"], **stop_record(prefix)}]
    assert len(prefix.rows) == PRECONTACT_STEP+1 and not any(e["step"] <= 1 for e in prefix.events)
    rises = [r["values"][0]-values[0] for r in prefix.rows]
    step1 = prefix.rows[1]
    early_state = unpack(step1)
    early_score = forecast_rise(rises[1:2], float(np.linalg.norm(velocity[0])), flight_factors(), curvature)
    early_already = max(rises[:2]) > LIMIT
    early_alert = not early_already and early_score > LIMIT
    second_alert = False
    contact_alert = False
    if second_stage:
        second_already = not early_alert and max(rises[:3]) > LIMIT
        second_score = None if early_alert or second_already else forecast_rise(
            rises[1:3], float(np.linalg.norm(velocity[0])), flight_factors())
        second_alert = second_score is not None and second_score > LIMIT
        c.update(second_state=pack(unpack(prefix.rows[2])), second_score=second_score, second_warning=second_alert,
                 already_deviated_at_step2=second_already)
        if guard_fraction is not None:
            check = second_score is not None and guard_fraction*LIMIT <= second_score <= LIMIT
            prediction = None
            if check:
                contact_oracle = PopulationOracle(objective,total_budget=CAP)
                load_counters(contact_oracle,prefix.rows[2])
                prediction = forecast_first_contact(contact_oracle,unpack(prefix.rows[2]).particles[0],cfg,2,8)
                assert contact_oracle.paths == prefix.rows[2]["path_calls"]+1
                contact_alert = prediction is not None and prediction["loss"]-values[0]>LIMIT
            c.update(contact_guard_queries=int(check),contact_guard_prediction=prediction,contact_guard_alert=contact_alert)
    base_family, runs = family(root, name, "base", values[0], rises, prefix, oracle, cfg)
    families = {"base": base_family}
    if early_alert or second_alert or contact_alert:
        step, kind = (1, "early") if early_alert else (2, "second")
        decision_row = prefix.rows[step]
        start = transfer_at_checkpoint(unpack(decision_row), 0)
        branch_oracle = PopulationOracle(objective, total_budget=CAP)
        load_counters(branch_oracle, decision_row)
        if contact_alert:
            branch_oracle.paths+=1
        charged_at_decision=sum(branch_oracle.counts().values())
        method = IndependentEnergyPopulation(cfg)
        method.committed_steps = step
        after = run_population(method, branch_oracle, start, PRECONTACT_STEP-step)
        after_id = f"{name}__{kind}_prefix"
        save_segment(root, after_id, after)
        rows.append({"run_id": after_id, "initial_context_id": name, "family": kind,
                     "actual_prefix_calls": after.counts["total_oracle_calls"]-charged_at_decision,
                     **stop_record(after)})
        changed_rises = [*rises[:step], *[r["values"][0]-values[0] for r in after.rows]]
        families[kind], more = family(root, name, kind, values[0], changed_rises, after, branch_oracle, cfg)
        runs.extend(more)
    direct = []
    if preflight:
        check = PopulationOracle(objective, total_budget=CAP)
        check.values = 4
        control = run_population(IndependentEnergyPopulation(cfg), check, init, END_STEP)
        key = name+"__uninterrupted"
        save_segment(root, key, control)
        direct.append({"run_id": key, "initial_context_id": name, **stop_record(control)})
    c.update(context_id=name, incumbent_energy=energy, initial_horizontal_velocities=velocity.tolist(),
             start=pack(init), early_state=pack(early_state),
             initial_loss=values[0], early_score=early_score, early_warning=early_alert,
             already_deviated_at_step1=early_already, families=families)
    return c, rows, runs, direct


def save(root, contexts, prefixes, runs, direct):
    for name, rows in (("contexts", contexts), ("prefixes", prefixes), ("runs", runs), ("uninterrupted", direct)):
        _json(root/(name+".json"), rows)
    if runs:
        _csv(root/"runs.csv", runs)
    families = [f for c in contexts for f in c["families"].values()]
    counts = {
        "contexts": len(contexts), "prefix_executions": len(prefixes),
        "base_prefixes": len(contexts), "early_transfer_prefixes": sum(p["family"]=="early" for p in prefixes),
        "exact_contacts": sum(f["complete"] for f in families), "continuations": len(runs),
        "uninterrupted_preflight_executions": len(direct),
        "actual_prefix_queries": sum(p["actual_prefix_calls"] for p in prefixes),
        "actual_contact_queries": sum(f["actual_contact_queries"] for f in families),
        "actual_forecast_queries": sum(f["actual_forecast_queries"] for f in families),
        "actual_continuation_queries": sum(r["actual_suffix_calls"] for r in runs),
        "actual_uninterrupted_queries_excluding_shared_initial_values":
            sum(r["counts"]["total_oracle_calls"]-4 for r in direct),
        "setup_reference_queries": sum(c["setup_reference_values"] for c in contexts),
        "charged_policy_queries": sum(r["counts"]["total_oracle_calls"] for r in runs),
    }
    if any("second_warning" in c for c in contexts):
        counts["second_transfer_prefixes"] = sum(p["family"]=="second" for p in prefixes)
    if any("contact_guard_queries" in c for c in contexts):
        counts["actual_contact_guard_queries"]=sum(c["contact_guard_queries"] for c in contexts)
    counts["actual_solver_queries"] = sum(v for k,v in counts.items() if k.startswith("actual_"))
    counts["actual_queries_including_setup"] = counts["actual_solver_queries"]+counts["setup_reference_queries"]
    _json(root/"execution_counts.json", counts)
    return counts


def run(args, *, phase="E35", spec=SPEC, seed_sets=SEEDS, parents=PARENTS, second_stage=False, guard_fraction=None):
    root, stage = args.output, args.stage
    dependencies = {str(p): sha(p) for p in (WARNING_SOURCE,*parents)}
    if args.lock:
        audit = args.lock/"audit.json"
        assert read(audit)["passed"]
        dependencies[str(audit)] = sha(audit)
    root.mkdir(parents=True)
    for folder in ("traces","events","uncommitted","checkpoints","fractions"):
        (root/folder).mkdir()
    source = snapshot(root,spec)
    if args.lock:
        assert source == read(args.lock/"config.json")["source_sha256"]
    seeds = seed_sets[stage]
    curvature = read(WARNING_SOURCE)["prior_curvature"]
    config={
        "phase":phase,"stage":stage,"seeds":seeds,"energies":list(ENERGIES),
        "policies":[*POLICIES,"two_stage_plus_step2"] if second_stage else list(POLICIES),
        "early_prior_curvature":curvature,"query_cap":CAP,"threshold":LIMIT,
        "h":H,"gravity":.5,"stiffness":8,"primary_step":PRIMARY_STEP,"end_step":END_STEP,"binary_interactions":False,
        "source_sha256":source,"source_archive_sha256":sha(root/"source.zip"),"dependency_data_sha256":dependencies,
        "launch_command":["python3",*sys.argv],"cwd":str(Path.cwd()),"platform":platform.platform(),
        "python":sys.version,"numpy":np.__version__,"workers":args.workers,"launch_utc":datetime.now(timezone.utc).isoformat(),
        "thread_environment":{k:os.environ.get(k) for k in ("PYTHONPATH","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
    }
    if guard_fraction is not None:
        config["contact_guard_fraction"]=guard_fraction
        config["policies"].append("second_step_contact_guard")
    _json(root/"config.json",config)
    contexts,prefixes,runs,direct = [],[],[],[]
    save(root,contexts,prefixes,runs,direct)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(simulate,root,s,e,curvature,stage=="preflight",second_stage,guard_fraction)
                for s in seeds for e in ENERGIES]
        try:
            for i,job in enumerate(jobs,1):
                c,p,r,d = job.result()
                contexts.append(c)
                prefixes.extend(p)
                runs.extend(r)
                direct.extend(d)
                if i%30==0 or i==len(jobs):
                    counts=save(root,contexts,prefixes,runs,direct)
                    print(f"{stage}: {i}/{len(jobs)} matched initial states, {len(runs)} actual continuations",flush=True)
        except BaseException:
            save(root,contexts,prefixes,runs,direct)
            _json(root/"task_failure.json",{"traceback":traceback.format_exc(),
                  "unreturned_cost":"Unknown if the worker did not return its ledger."})
            raise
    _json(root/"completion_observation.json",{**counts,"observed_utc":datetime.now(timezone.utc).isoformat()})
    print(counts,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--stage",choices=SEEDS,required=True)
    parser.add_argument("--lock",type=Path)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.workers<1 or (args.stage=="preflight")==bool(args.lock):
        parser.error("Positive worker count required. Formal requires the audited preflight --lock.")
    run(args)

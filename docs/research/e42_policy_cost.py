"""Compare paid continuation costs at the common E42 fork."""

from finalize_initial_budget import create
from run_long_horizon_exploration import sources
from run_third_reflection_guard import Archive, BASE, CFG, E40, E41, history, read, sha


def export():
    data = Archive()
    old_runs = {r["run_id"]:r for r in data.read(E41/"runs.json")}
    contexts = {r["context_id"]:r for r in data.read(E40/"contexts.json")}
    cases = {r["context_id"]:r for r in read(BASE/"contexts.json")}
    runs = {r["context_id"]:r for r in read(BASE/"runs.json")}
    decisions = read(BASE/"decisions.json")
    differences = []
    for d in decisions:
        if not d["transferred"]:
            continue
        name = d["context_id"]
        cp = d["checkpoint"]
        _,_,rows,events,start = history(data,cases[name],contexts[name])
        step = cp["boundary_step"]-start
        key = next(i for i,e in enumerate(events) if e==cp["event"])
        paid = [e for e in events[:key+1] if e["step"]==step]
        groups = 1+sum(b["time"]-a["time"]>CFG.physics.time_tolerance for a,b in zip(paid,paid[1:]))
        prefix = int(rows[step-1]["total_oracle_calls"])+8*groups+2*len(paid)
        mode,_,_ = list(sources(cases[name]))[-1]
        retained = int(rows[-1]["total_oracle_calls"])-prefix+old_runs[name+"__"+mode]["actual_new_queries"]
        moved = runs[name]["counts"]["total_oracle_calls"]
        differences.append({"context_id":name,"retained_suffix":retained,"capped_suffix":moved,"delta":moved-retained})
    forecasts = sum(d["forecast_queries"] for d in decisions)
    suffix_delta = sum(d["delta"] for d in differences)
    deps = dict(data.dependencies)
    for name in ("runs","contexts","decisions"):
        path = BASE/(name+".json")
        deps[str(path)] = sha(path)
    create(BASE/"policy_query_comparison.json",{"rows":differences,"forecast_queries_added":forecasts,
        "net_suffix_query_change":suffix_delta,"net_policy_query_change":forecasts+suffix_delta,
        "new_oracle_calls_for_export":0,"dependency_data_sha256":deps})


if __name__=="__main__":
    export()

"""Summarize changes in good-particle energy and loss using paid E38 artifacts."""

from pathlib import Path
from statistics import median

from finalize_initial_budget import create, sha
from run_conditional_transfer import read


def export(root):
    checkpoint_path = root/"warning_checkpoints.json"
    checkpoints = read(checkpoint_path)
    dependencies = {str(checkpoint_path):sha(checkpoint_path)}
    rows = []
    for c in checkpoints:
        if not c["branches"]:
            continue
        outcomes = {}
        for mode in ("keep","transfer"):
            file = root/"fractions"/(c["checkpoint_id"]+"__"+mode+".json")
            frac = read(file)
            dependencies[str(file)] = sha(file)
            velocity = frac["start"]["velocities"][0][:2]
            outcomes[mode] = {**c["branches"][mode],"horizontal_velocity":velocity,
                              "horizontal_kinetic":.5*sum(v*v for v in velocity)}
        rows.append({"checkpoint_id":c["checkpoint_id"],"ordinal":c["ordinal"],
                     "warning":c["warning"],"already_deviated":c["already_deviated"],**outcomes})
    warned = [r for r in rows if r["warning"]]
    late = [r for r in rows if r["already_deviated"]]
    statistics = {name:{mode:{
        "median_horizontal_kinetic":median(r[mode]["horizontal_kinetic"] for r in group),
        "median_final_rise":median(r[mode]["final_rise"] for r in group),
        "max_peak_saved_rise":max(r[mode]["peak_saved_rise"] for r in group),
    } for mode in ("keep","transfer")} for name,group in (("warned",warned),("already_deviated",late))}
    early = sum(r["transfer"]["first_saved_crossing"] is not None
                and r["transfer"]["first_saved_crossing"]<=1. for r in warned)
    worse = sum(r["transfer"]["final_rise"]>r["keep"]["final_rise"] for r in warned)
    create(root/"transfer_effects.json",{"rows":rows,"statistics":statistics,"new_solver_queries":0,
            "warned_cases_crossing_by_t1_after_transfer":early,"warned_cases_worse_final_loss":worse,
            "source_sha256":dependencies})


if __name__=="__main__":
    export(Path("output/collision_later_reflection_warning"))

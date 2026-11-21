"""Summarize recovery pairs at the actual two-stage reflection states."""

from pathlib import Path

from audit_high_kinetic_intervention_timing import summarize
from audit_two_stage_protection import WINDOWS
from finalize_initial_budget import create, sha
from run_conditional_transfer import read


def selected_recovery(root):
    source=root/"paired_outcomes.json"
    rows=[]
    for r in read(source):
        selection=r["policies"]["two_stage"]
        family=r["families"][selection["family"]]
        if family["complete"] and family["already_deviated"]:
            rows.append({
                "context_id":r["context_id"],"seed":r["seed"],"incumbent_energy":r["incumbent_energy"],
                "family":selection["family"],**family,
            })
    stats={}
    for mode in ("keep","transfer"):
        branch=[r["branches"][mode] for r in rows]
        stats[mode]={
            name:{**summarize(branch,name),
                  "recrossed":sum(b["first_recross"] is not None and b["first_recross"]<=h for b in branch)}
            for name,h in WINDOWS.items()
        }
    create(root/"two_stage_recovery_cases.json",{
        "source_sha256":sha(source),"rows":rows,"summary":stats,"new_oracle_calls":0,
    })


if __name__=="__main__":
    selected_recovery(Path("output/collision_two_stage_protection"))

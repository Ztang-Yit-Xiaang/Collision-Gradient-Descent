"""Check which E44 forecasts each candidate gate would omit, without queries."""

from pathlib import Path

from finalize_initial_budget import create
from run_capped_reflection_transfer import checked_hashes
from run_later_reflection_warning import LIMIT
from run_third_reflection_contact_gate import BASE, CONTACT_FRACTION, E44
from run_third_reflection_guard import read, sha


def scan(output):
    source = E44/"third_reflection"
    paths = [source/(name+".json") for name in ("paired_outcomes","decisions")]
    hashes = checked_hashes(E44,paths)
    rows,decisions = [read(p) for p in paths]
    eligible = []
    for row,d in zip(rows,decisions,strict=True):
        assert row["context_id"]==d["context_id"]
        if row["extended_forecast"]:
            prediction = d["prediction"]
            initial = d["checkpoint"]["initial_loss"]
            eligible.append({"context_id":row["context_id"],"seed":row["seed"],
                "incumbent_energy":row["incumbent_energy"],
                "contact_rise":prediction["next_contact"]["loss"]-initial,
                "after_contact_saved_rise":prediction["after_contact_saved"]["loss"]-initial,
                "saved_window_warning":row["saved_window_warning"],"saved_window_transfer":row["saved_window_transfer"],
                "baseline":row["baseline"],"saved_window_policy":row["saved_window_policy"]})
    assert len(rows)==480 and len(eligible)==464
    report = []
    for fraction in (.5,.7,.8,.9,.95,1.):
        skipped = [r for r in eligible if r["contact_rise"]<fraction*LIMIT]
        missed = [r for r in skipped if r["saved_window_warning"]]
        lost = [r for r in missed if r["saved_window_transfer"] and r["baseline"]["saved"] is not None
                and r["saved_window_policy"]["saved"] is None]
        report.append({"fraction":fraction,"retained_extensions":len(eligible)-len(skipped),
                       "queries_saved":2*len(skipped),"missed_warnings":missed,"lost_saved_preventions":lost})
    create(output/"e44_gate_scan.json",{"contexts":len(rows),"extensions":eligible,"candidates":report,
        "selected_fraction":CONTACT_FRACTION,"new_queries":0,"dependency_data_sha256":hashes,
        "source_sha256":sha(Path(__file__))})
    print([(r["fraction"],r["retained_extensions"],len(r["missed_warnings"])) for r in report])


if __name__=="__main__":
    scan(BASE)

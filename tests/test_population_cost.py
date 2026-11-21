import csv
import gzip
import json
import pytest

from collision_optim.population_cost import last_observation,cost_matched_report


def test_cost_cap_uses_observed_loss_without_interpolation_or_future_values():
    rows=[{"total_oracle_calls":4,"loss":1.},
          {"total_oracle_calls":14,"loss":.8},
          {"total_oracle_calls":28,"loss":.3}]
    result,covered=last_observation(rows,20)
    assert covered and result["loss"]==.8
    result,covered=last_observation(rows,30)
    assert not covered and result["loss"]==.3
    with pytest.raises(ValueError): last_observation(rows,3)


def test_equal_cost_report_flags_short_trajectories_instead_of_extrapolating(tmp_path):
    config={"population_size":4,"steps":10,"cases":["quadratic"],"seeds":1,"rhos":[.12]}
    (tmp_path/"config.json").write_text(json.dumps(config))
    (tmp_path/"traces").mkdir()
    summaries=[{"case":"quadratic","seed":0,"rho":"","method":"independent","stop_reason":"completed"},
               {"case":"quadratic","seed":0,"rho":.12,"method":"collisions","stop_reason":"contact_error: unsupported"}]
    with (tmp_path/"runs.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    names=["total_oracle_calls","relative_best_gap","time","crossed_particles","target_particles"]
    for name,end in [("independent",150),("collisions__rho0.12",80)]:
        with gzip.open(tmp_path/"traces"/f"quadratic__s00__{name}.csv.gz","wt") as stream:
            writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
            for cost,loss in [(4,1.),(end,.2)]:
                writer.writerow(dict(zip(names,[cost,loss,1.,0,0])))
    report=cost_matched_report(tmp_path)
    assert report["call_cap"]==120
    row=report["aggregate"][0]
    assert row["both_covered"]==0 and row["not_covered"]==1
    assert row["median_collision_gap"] is None

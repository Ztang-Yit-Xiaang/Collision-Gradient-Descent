import numpy as np
import pytest

from collision_optim.optimizer_comparison import CASES,make_case,run_case


def test_height_tuning_preserves_trapped_energy_ratio_and_initial_parameters():
    original=None;ratio=None
    for scale in [.1,.3,1.,3.]:
        objective,_,boundary,state=make_case("bad_well",200,scale)
        p=state.particles[0];energy=state.kinetic+.5*p.z
        barrier=.5*scale*objective.value(np.array([boundary,0.]))
        assert energy<barrier
        if original is not None:
            np.testing.assert_array_equal(p.x,original)
            assert energy/scale==pytest.approx(ratio,abs=1e-14)
        original=p.x;ratio=energy/scale


@pytest.mark.parametrize("case",CASES)
def test_parameter_initialization_is_shared_across_physical_scalings(case):
    a=make_case(case,200,.1)[-1].particles[0]
    b=make_case(case,200,3.)[-1].particles[0]
    np.testing.assert_array_equal(a.x,b.x)
    assert b.z==pytest.approx(30*a.z)


@pytest.mark.parametrize("method,setting",[("momentum",.01),("adam",.005),("gravity",1.)])
def test_primary_comparison_stops_at_actual_shared_query_cap(method,setting):
    row,trace,*_=run_case("good_well",200,method,setting,100)
    assert row["total_oracle_calls"]==100 and not row["failure"]
    assert row["planning_calls"]+row["dynamics_calls"]==100
    assert row["best_f"]==min(r["value"] for r in trace if r["record"]=="observation")

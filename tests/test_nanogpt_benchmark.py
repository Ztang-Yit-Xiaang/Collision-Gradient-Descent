from dataclasses import replace
import copy
import numpy as np
import pytest

torch=pytest.importorskip("torch")
from collision_optim.nanogpt_benchmark import (
    LMConfig,TextData,TorchControl,TrainingCost,clone_training,make_model,make_optimizer,
    model_hash,train_step,preview_training,train_run,
)


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(1)
    path=tmp_path/"text.txt";path.write_text("abcd efgh ijkl mnop.\n"*500)
    cfg=LMConfig(block=8,batch=2,layers=1,width=16,heads=2)
    return TextData(path,cfg),cfg


def test_split_and_batch_replay_are_disjoint_and_repeatable(setup):
    data,cfg=setup
    assert sum(len(x) for x in data.splits.values())==len(data.path.read_text())
    a,b=data.batch("train",200,5);c,d=data.batch("train",200,5)
    assert torch.equal(a,c) and torch.equal(b,d)
    assert a.shape==b.shape==(cfg.batch,cfg.block)
    assert data.splits["train"].data_ptr()!=data.splits["test"].data_ptr()


def test_joint_clone_keeps_tied_weights_and_optimizer_links_without_mutation(setup):
    data,cfg=setup;model=make_model(data,cfg,3);opt=make_optimizer(model,"adamw",.001,.01,cfg);control=TorchControl(model)
    cost=TrainingCost(300);train_step(model,opt,control,data.batch("train",3,0),cfg,cost)
    original=model_hash(model);snapshot=copy.deepcopy(opt.state_dict())
    cloned,other,external=clone_training(model,opt,control)
    assert cloned.lm_head.weight is cloned.transformer.wte.weight
    assert set(id(p) for group in other.param_groups for p in group["params"])==set(id(p) for p in cloned.parameters())
    train_step(cloned,other,external,data.batch("train",3,1),cfg,cost,virtual=True)
    assert model_hash(model)==original and model_hash(cloned)!=original
    for key,group in snapshot["state"].items():
        for name,value in group.items():
            if torch.is_tensor(value):assert torch.equal(value,opt.state_dict()["state"][key][name])


def test_preview_only_reads_train_and_does_not_commit_virtual_model(setup):
    data,cfg=setup;cfg=replace(cfg,horizon=2)
    model=make_model(data,cfg,3);opt=make_optimizer(model,"momentum",.1,0.,cfg);control=TorchControl(model)
    before=model_hash(model);probe=data.probe();cost=TrainingCost(1000)
    original=data.batch;seen=[]
    def batch(split,seed,index):
        assert split=="train";seen.append(index);return original(split,seed,index)
    data.batch=batch
    _,log=preview_training(model,opt,control,data,cfg,cost,3,5,[.01,.03],.1,np.random.default_rng(3),probe)
    assert model_hash(model)==before and all(torch.count_nonzero(q)==0 for q in control.external)
    assert cost.planning_used==log["units"]>0
    assert all(seen[i:i+2]==[5,6] for i in range(0,len(seen),2))


def test_zero_decay_adam_and_adamw_training_match(setup):
    data,cfg=setup
    a,*_=train_run(data,cfg,"adam",.001,0.,3,equivalent_steps=6,eval_batches=2,eval_split="tuning")
    b,*_=train_run(data,cfg,"adamw",.001,0.,3,equivalent_steps=6,eval_batches=2,eval_split="tuning")
    assert a["initial_hash"]==b["initial_hash"] and a["final_hash"]==b["final_hash"]
    assert a["final_loss"]==b["final_loss"]


def test_paid_diagnostics_and_previews_reduce_real_updates_under_same_budget(setup):
    data,cfg=setup;cfg=replace(cfg,diagnostic_interval=2,horizon=2,epsilon_gain=10.,epsilon_move=10.,max_decisions=1)
    base,*_=train_run(data,cfg,"momentum",.03,0.,3,equivalent_steps=30,eval_batches=2,eval_split="tuning")
    active,*_=train_run(data,cfg,"momentum",.03,0.,3,mode="predictive",equivalent_steps=30,eval_batches=2,eval_split="tuning")
    assert base["actual_training_steps"]==30
    assert active["actual_training_steps"]<30
    assert active["virtual_train_units"]>0
    assert active["used_units"]<=active["budget_units"]==base["budget_units"]
    assert sum(active[k] for k in ["real_train_units","virtual_train_units","real_probe_units","virtual_probe_units"])==active["used_units"]

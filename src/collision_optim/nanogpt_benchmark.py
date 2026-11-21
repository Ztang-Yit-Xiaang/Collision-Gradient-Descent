"""Small licensed NanoGPT benchmark, with paid training-only intervention probes."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import copy
from dataclasses import dataclass,asdict
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from .strategic_experiment import _json,_csv


ROOT=Path(__file__).resolve().parents[2]
VENDOR=ROOT/"third_party"/"nanogpt"/"model.py"
MODES=("momentum","adam","adamw","momentum_random","momentum_predictive","adamw_random","adamw_predictive")


def vendor_model():
    name="ee5239_vendor_nanogpt"
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,VENDOR)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class LMConfig:
    block: int=64
    batch: int=8
    layers: int=2
    heads: int=2
    width: int=64
    dropout: float=0.
    beta1: float=.9
    beta2: float=.95
    clip: float=1.
    diagnostic_interval: int=20
    horizon: int=10
    epsilon_gain: float=1e-3
    epsilon_move: float=1e-3
    radii: tuple[float,...]=(.001,.003)
    max_decisions: int=2
    max_shots: int=2
    planning_units: int=360


class UnitLimit(Exception):pass


class TrainingCost:
    """Forward-equivalent batch units, with one backward approximated as two."""
    def __init__(self,limit):
        self.limit=limit
        self.counts={k:0 for k in ["real_train_units","virtual_train_units","real_probe_units","virtual_probe_units"]}
        self.scope_limit=None;self.scope_start=0
    @property
    def used(self):return sum(self.counts.values())
    @property
    def planning_used(self):return self.counts["virtual_train_units"]+self.counts["virtual_probe_units"]
    def charge(self,kind,amount):
        if self.used+amount>self.limit or self.scope_limit is not None and self.used-self.scope_start+amount>self.scope_limit:
            raise UnitLimit("Training or preview compute allowance exhausted.")
        self.counts[kind]+=amount


class TextData:
    def __init__(self,path,config):
        self.path=Path(path);raw=self.path.read_bytes();text=raw.decode("utf-8")
        chars=sorted(set(text));self.vocabulary=chars;mapping={c:i for i,c in enumerate(chars)}
        encoded=torch.tensor([mapping[c] for c in text],dtype=torch.long)
        n=len(encoded);a=int(.8*n);b=int(.9*n)
        self.splits={"train":encoded[:a],"tuning":encoded[a:b],"test":encoded[b:]}
        self.config=config;self.sha256=hashlib.sha256(raw).hexdigest()
        if min(map(len,self.splits.values()))<=config.block+1:raise ValueError("Text splits are too short.")
    def batch(self,split,seed,index):
        domain={"train":1,"tuning":2,"test":3}[split]
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,domain,5239]))
        data=self.splits[split];cfg=self.config
        offsets=rng.integers(0,len(data)-cfg.block,size=cfg.batch)
        x=torch.stack([data[int(i):int(i)+cfg.block] for i in offsets])
        y=torch.stack([data[int(i)+1:int(i)+cfg.block+1] for i in offsets])
        return x,y
    def probe(self):return [self.batch("train",71001,i) for i in range(2)]


class TorchControl:
    def __init__(self,model,decay=.9):
        self.external=[torch.zeros_like(p) for p in model.parameters()]
        self.base=[torch.zeros_like(p) for p in model.parameters()]
        self.decay=decay
    def velocity_projection(self,direction):
        return sum(float(torch.sum((b+q).double()*d.double())) for b,q,d in zip(self.base,self.external,direction))
    def inject(self,direction,radius):
        projection=self.velocity_projection(direction)
        amplitude=-projection+np.sqrt(projection*projection+radius*radius)
        self.external=[q+float(amplitude)*d for q,d in zip(self.external,direction)]
        return float(.5*radius*radius),float(amplitude)


def make_model(data,config,seed):
    torch.manual_seed(seed)
    module=vendor_model()
    cfg=module.GPTConfig(block_size=config.block,vocab_size=len(data.vocabulary),n_layer=config.layers,
                         n_head=config.heads,n_embd=config.width,dropout=config.dropout,bias=True)
    with redirect_stdout(io.StringIO()):model=module.GPT(cfg)
    return model


def make_optimizer(model,kind,lr,decay,config):
    groups=[{"params":[p for p in model.parameters() if p.dim()>=2],"weight_decay":decay},
            {"params":[p for p in model.parameters() if p.dim()<2],"weight_decay":0.}]
    if kind=="momentum":return torch.optim.SGD(groups,lr=lr,momentum=config.beta1,foreach=False)
    klass=torch.optim.AdamW if kind=="adamw" else torch.optim.Adam
    return klass(groups,lr=lr,betas=(config.beta1,config.beta2),eps=1e-8,foreach=False)


def model_hash(model):
    digest=hashlib.sha256()
    for p in model.parameters():digest.update(p.detach().numpy().tobytes())
    return digest.hexdigest()


def train_step(model,optimizer,control,batch,config,cost,*,virtual=False):
    cost.charge("virtual_train_units" if virtual else "real_train_units",3)
    model.train();optimizer.zero_grad(set_to_none=True)
    _,loss=model(*batch)
    if not torch.isfinite(loss):raise FloatingPointError("Nonfinite NanoGPT loss.")
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip,error_if_nonfinite=True)
    before=[p.detach().clone() for p in model.parameters()]
    optimizer.step()
    relative_sq=0.;before_sq=0.
    with torch.no_grad():
        for i,(p,old) in enumerate(zip(model.parameters(),before)):
            control.base[i]=p.detach().clone()-old
            control.external[i].mul_(control.decay);p.add_(control.external[i])
            relative_sq+=float(torch.sum((p-old).double()**2));before_sq+=float(torch.sum(old.double()**2))
    return float(loss.detach()),float(np.sqrt(relative_sq)/(1+np.sqrt(before_sq))),float(norm)


def probe_loss(model,batches,cost=None,*,virtual=False):
    was_training=model.training;model.eval();values=[]
    try:
        with torch.no_grad():
            for batch in batches:
                if cost is not None:cost.charge("virtual_probe_units" if virtual else "real_probe_units",1)
                _,loss=model(*batch);values.append(float(loss))
        if not np.all(np.isfinite(values)):raise FloatingPointError("Nonfinite probe loss.")
        return float(np.mean(values))
    finally:model.train(was_training)


def clone_training(model,optimizer,control):
    """Joint deepcopy preserves cloned optimizer links and tied embedding weights."""
    return copy.deepcopy((model,optimizer,control))


def preview_training(model,optimizer,control,data,config,cost,seed,step,radii,remaining_energy,rng,probe):
    log={"step":step,"reason":"no_benefit","candidates":[],"selected":None,"units":0}
    start=cost.used
    allowance=min(config.planning_units-cost.planning_used,cost.limit-cost.used-60)
    if allowance<=0:log["reason"]="planning_limit";return None,log
    generator=torch.Generator().manual_seed(int(rng.integers(0,2**31)))
    direction=[torch.randn(p.shape,generator=generator,dtype=p.dtype) for p in model.parameters()]
    length=np.sqrt(sum(float(torch.sum(d.double()**2)) for d in direction));direction=[d/length for d in direction]
    cost.scope_start=cost.used;cost.scope_limit=allowance
    def rollout(branch):
        m,o,q=branch
        for j in range(config.horizon):train_step(m,o,q,data.batch("train",seed,step+j),config,cost,virtual=True)
        return probe_loss(m,probe,cost,virtual=True)
    try:
        baseline=rollout(clone_training(model,optimizer,control));log["baseline_loss"]=baseline
        for radius in radii:
            if .5*radius*radius>remaining_energy+1e-12:continue
            acceptable=[]
            for sign in [1.,-1.]:
                candidate={"radius":radius,"sign":sign,"status":"incomplete"};log["candidates"].append(candidate)
                try:
                    branch=clone_training(model,optimizer,control);branch[2].inject([sign*d for d in direction],radius)
                    loss=rollout(branch);gain=(baseline-loss)/(1+abs(baseline))
                    candidate.update({"status":"completed","loss":loss,"gain":gain})
                    if gain>config.epsilon_gain:acceptable.append((gain,len(log["candidates"])-1,sign))
                except (FloatingPointError,RuntimeError) as exc:
                    candidate.update({"status":"numerical_error","error":str(exc)})
            if acceptable:
                gain,index,sign=max(acceptable,key=lambda z:(z[0],-z[1]));log.update({"reason":"accepted","selected":index,"gain":gain})
                return ([sign*d for d in direction],radius),log
        return None,log
    except UnitLimit:
        log["reason"]="planning_limit";return None,log
    finally:
        cost.scope_limit=None;log["units"]=cost.used-start


def train_run(data,config,kind,lr,decay,seed,*,mode="none",equivalent_steps=400,eval_split="test",eval_batches=32,checkpoint=None):
    model=make_model(data,config,seed);optimizer=make_optimizer(model,kind,lr,decay,config);control=TorchControl(model,config.beta1)
    initial_hash=model_hash(model);parameter_norm=np.sqrt(sum(float(torch.sum(p.detach().double()**2)) for p in model.parameters()))
    radii=[r*(1+parameter_norm) for r in config.radii];energy_budget=config.max_shots*.5*radii[-1]**2
    cost=TrainingCost(3*equivalent_steps);rows=[];actions=[];decisions=[];probe=data.probe()
    step=shots=streak=attempts=0;allowed=0;supplied=0.;probes=[];moves=[];status="budget"
    rng=np.random.default_rng(9006+seed);started=perf_counter()
    if mode!="none":probes.append((0,probe_loss(model,probe,cost)))
    while cost.used+3<=cost.limit:
        try:
            if mode!="none" and step and step%config.diagnostic_interval==0 and (not probes or probes[-1][0]!=step):
                loss=probe_loss(model,probe,cost);progress=(probes[-1][1]-loss)/(1+abs(probes[-1][1]));probes.append((step,loss))
                movement=float(np.mean(moves[-config.diagnostic_interval:]))
                streak=streak+1 if progress<=config.epsilon_gain and movement<=config.epsilon_move else 0
                if streak>=2 and step>=allowed and shots<config.max_shots:
                    selection=None
                    if mode=="random" and supplied+.5*radii[-1]**2<=energy_budget+1e-12:
                        generator=torch.Generator().manual_seed(int(rng.integers(0,2**31)))
                        direction=[torch.randn(p.shape,generator=generator) for p in model.parameters()]
                        length=np.sqrt(sum(float(torch.sum(d.double()**2)) for d in direction))
                        selection=([d/length for d in direction],radii[-1])
                    elif mode=="predictive" and attempts<config.max_decisions:
                        attempts+=1
                        selection,log=preview_training(model,optimizer,control,data,config,cost,seed,step,radii,energy_budget-supplied,rng,probe)
                        decisions.append(log)
                    if selection is not None:
                        energy,amplitude=control.inject(*selection);supplied+=energy;shots+=1;streak=0
                        actions.append({"step":step,"energy_diagnostic":energy,"impulse_norm":amplitude,"radius":selection[1]})
                        allowed=step+2*config.diagnostic_interval
                    else:allowed=step+4*config.diagnostic_interval
            loss,movement,grad_norm=train_step(model,optimizer,control,data.batch("train",seed,step),config,cost)
            step+=1;moves.append(movement)
            rows.append({"step":step,"train_loss":loss,"relative_movement":movement,"gradient_norm_before_clip":grad_norm,
                         "units":cost.used,"shots":shots,**cost.counts})
        except UnitLimit:break
        except (FloatingPointError,RuntimeError) as exc:
            status=f"numerical_error: {exc}";break
    training_seconds=perf_counter()-started
    final_loss=None
    if not status.startswith("numerical_error"):
        final_loss=probe_loss(model,[data.batch(eval_split,81001,j) for j in range(eval_batches)])
    if checkpoint is not None:
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"external":control.external,
                    "config":asdict(config),"kind":kind,"lr":lr,"weight_decay":decay,"seed":seed},checkpoint)
    summary={"kind":kind,"control":mode,"lr":lr,"weight_decay":decay,"seed":seed,"status":status,
             "initial_hash":initial_hash,"final_hash":model_hash(model),"parameter_count":sum(p.numel() for p in model.parameters()),
             "actual_training_steps":step,"budget_units":cost.limit,"used_units":cost.used,
             "shots":shots,"decisions":attempts,"supplied_model_kinetic":supplied,"training_seconds":training_seconds,
             "eval_split":eval_split,"eval_batches":eval_batches,"final_loss":final_loss,
             "perplexity":None if final_loss is None else float(np.exp(final_loss)),**cost.counts,
             "actual_train_tokens":step*config.batch*config.block,
             "virtual_train_tokens":cost.counts["virtual_train_units"]//3*config.batch*config.block,
             "probe_tokens":(cost.counts["real_probe_units"]+cost.counts["virtual_probe_units"])*config.batch*config.block}
    return summary,rows,actions,decisions


def run_benchmark(output,*,data_path=ROOT/"data"/"tinyshakespeare.txt",config=None,seeds=(200,201,202),tune_steps=150,test_steps=400):
    config=config or LMConfig();output=Path(output)
    if (output/"config.json").exists():raise ValueError("Use a new benchmark output directory.")
    for folder in [output,output/"trajectories",output/"actions",output/"decisions",output/"checkpoints"]:folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    data=TextData(data_path,config)
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    _json(output/"config.json",{"model":asdict(config),"tuning_seed":0,"test_seeds":list(seeds),"tune_steps":tune_steps,"test_steps":test_steps,
          "torch":torch.__version__,"device":"cpu","dtype":"float32","data_sha256":data.sha256,
          "split_lengths":{k:len(v) for k,v in data.splits.items()},"vocabulary":data.vocabulary,
          "vendor_commit":"3adf61e154c3fe3fca428ad6bc3818b27a3b8291","vendor_sha256":hashlib.sha256(VENDOR.read_bytes()).hexdigest(),
          "source_sha256":source,"cost_unit":"One batch forward; backward approximated as two; train=3, probe=1",
          "evaluation_cost":"Final tuning/test evaluation outside training budget, same batch count within each phase"})
    tuning=[];selected={}
    for kind in ["momentum","adam","adamw"]:
        lrs=(.03,.1,.3) if kind=="momentum" else (.0001,.0003,.001)
        candidates=[]
        for lr,decay in itertools.product(lrs,(0.,.01,.1)):
            row,*_=train_run(data,config,kind,lr,decay,0,equivalent_steps=tune_steps,eval_split="tuning",eval_batches=8)
            tuning.append(row);candidates.append(row);_json(output/"tuning.json",tuning)
            print(f"NanoGPT tune {len(tuning)}/27 {kind} lr={lr} wd={decay}: {row['final_loss']}",flush=True)
        valid=[r for r in candidates if r["final_loss"] is not None]
        if not valid:raise RuntimeError(f"All tuning configurations failed: {kind}")
        chosen=min(valid,key=lambda r:r["final_loss"])
        selected[kind]={"lr":chosen["lr"],"weight_decay":chosen["weight_decay"]}
    _json(output/"selected.json",selected);_csv(output/"tuning.csv",tuning)
    runs=[]
    for seed in seeds:
        for method in MODES:
            kind,*suffix=method.split("_");mode=suffix[0] if suffix else "none"
            row,trace,actions,decisions=train_run(data,config,kind,**{"lr":selected[kind]["lr"],"decay":selected[kind]["weight_decay"]},seed=seed,
                mode=mode,equivalent_steps=test_steps,checkpoint=output/"checkpoints"/f"{method}__s{seed}.pt")
            row["method"]=method;runs.append(row);_json(output/"runs.json",runs)
            name=f"{method}__s{seed}"
            _csv(output/"trajectories"/(name+".csv.gz"),trace,True)
            _json(output/"actions"/(name+".json"),actions);_json(output/"decisions"/(name+".json"),decisions)
            print(f"NanoGPT test {len(runs)}/{len(seeds)*len(MODES)} {name}: loss={row['final_loss']}, actual steps={row['actual_training_steps']}, shots={row['shots']}",flush=True)
    _csv(output/"runs.csv",runs)
    aggregate=[]
    for method in MODES:
        group=[r for r in runs if r["method"]==method];valid=[r for r in group if r["final_loss"] is not None]
        aggregate.append({"method":method,"runs":len(group),"failures":len(group)-len(valid),
                          "mean_test_loss":float(np.mean([r["final_loss"] for r in valid])) if valid else None,
                          "std_test_loss":float(np.std([r["final_loss"] for r in valid],ddof=1)) if len(valid)>1 else None,
                          "mean_test_perplexity":float(np.mean([r["perplexity"] for r in valid])) if valid else None,
                          "mean_actual_steps":float(np.mean([r["actual_training_steps"] for r in group])),
                          "mean_training_seconds":float(np.mean([r["training_seconds"] for r in group])),
                          "shots":sum(r["shots"] for r in group),"virtual_train_units":sum(r["virtual_train_units"] for r in group)})
    _json(output/"aggregate.json",aggregate);_csv(output/"aggregate.csv",aggregate)
    return aggregate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=Path("output/nanogpt_comparison"))
    parser.add_argument("--tune-steps",type=int,default=150);parser.add_argument("--test-steps",type=int,default=400)
    args=parser.parse_args(argv)
    print(json.dumps(run_benchmark(args.output,tune_steps=args.tune_steps,test_steps=args.test_steps),indent=2));return 0

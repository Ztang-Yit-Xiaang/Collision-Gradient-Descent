"""NanoGPT training with scalar routing and no diagnostic model evaluations."""

from dataclasses import asdict
from time import perf_counter

import numpy as np
import torch

from .momentum_routing import Action, MomentumRouter, RoutingConfig
from .nanogpt_benchmark import (TorchControl, TrainingCost, make_model, make_optimizer,
                               model_hash, probe_loss, train_step)


def tensor_norm(values):
    return float(np.sqrt(sum(float(torch.sum(v.detach().double()**2)) for v in values)))


def scale_torch_momentum(optimizer, control, factor):
    """Native SGD buffer and both recorded velocity components must agree."""
    if not np.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError("Invalid dissipation factor.")
    if not isinstance(optimizer, torch.optim.SGD):
        raise ValueError("Only native Momentum SGD can be dissipated.")
    with torch.no_grad():
        for group in optimizer.param_groups:
            for p in group["params"]:
                buffer = optimizer.state[p].get("momentum_buffer")
                if buffer is not None:
                    buffer.mul_(factor)
        for b, q in zip(control.base, control.external):
            b.mul_(factor); q.mul_(factor)


class RoutedTraining:
    def __init__(self, model, optimizer, lm_config, mode, routing, seed):
        self.model, self.optimizer, self.lm = model, optimizer, lm_config
        self.params = list(model.parameters())
        self.control = TorchControl(model, lm_config.beta1)
        self.router = MomentumRouter(mode, routing)
        self.generator = torch.Generator().manual_seed(20261106+seed)
        self.radii = np.array(routing.radii)*(1+tensor_norm(self.params))
        self.supply_cap = routing.max_shots*.5*self.radii[-1]**2
        self.supplied = self.dissipated = self.movement = 0.
        self.direction = None
        self.actions = []
        self.feature_seconds = self.action_seconds = 0.
        self.max_actuator_error = self.max_recurrence_error = 0.

    def step(self, batch, cost, step):
        if self.router.mode == "none":
            loss, movement, grad_norm = train_step(self.model, self.optimizer, self.control, batch, self.lm, cost)
            self.movement = movement
            return loss, movement, grad_norm
        cost.charge("real_train_units", 3)
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        _, loss = self.model(*batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.lm.clip, error_if_nonfinite=True)
        with torch.no_grad():
            stamp = perf_counter()
            group_config = {id(p): group for group in self.optimizer.param_groups for p in group["params"]}
            gradients = [p.grad.detach()+group_config[id(p)]["weight_decay"]*p.detach() for p in self.params]
            velocities = [b+q for b, q in zip(self.control.base, self.control.external)]
            vnorm, gnorm, xnorm = tensor_norm(velocities), tensor_norm(gradients), tensor_norm(self.params)
            projection = sum(float(torch.sum(v.double()*g.double())) for v, g in zip(velocities, gradients))
            lr = self.optimizer.param_groups[0]["lr"]
            action = self.router.decide(step, loss=float(loss.detach()), movement=self.movement,
                       relative_velocity=vnorm/(1+xnorm),
                       cosine=projection/(vnorm*gnorm) if vnorm*gnorm > 1e-300 else 0.,
                       speed_ratio=vnorm/max(lr*gnorm, 1e-300))
            self.feature_seconds += perf_counter()-stamp
            if action.kind != "continue":
                stamp = perf_counter()
                expected = 0.
                if action.kind == "pulse":
                    if action.reverse:
                        self.direction = [-d for d in self.direction]
                    else:
                        direction = [torch.randn(p.shape, dtype=p.dtype, generator=self.generator) for p in self.params]
                        length = tensor_norm(direction); self.direction = [d/length for d in direction]
                    radius = float(self.radii[action.radius_index]); expected = .5*radius**2
                    if self.supplied+expected > self.supply_cap+1e-8:
                        raise FloatingPointError("Supply cap exceeded.")
                    self.control.inject(self.direction, radius)
                else:
                    expected = .5*(action.factor**2-1)*vnorm**2
                    scale_torch_momentum(self.optimizer, self.control, action.factor)
                changed = [b+q for b, q in zip(self.control.base, self.control.external)]
                delta = .5*(tensor_norm(changed)**2-vnorm**2)
                impulse_norm = tensor_norm([a-b for a, b in zip(changed, velocities)])
                self.max_actuator_error = max(self.max_actuator_error, abs(delta-expected))
                if action.kind == "pulse": self.supplied += delta
                else: self.dissipated -= delta
                self.actions.append({"step": step, **asdict(action), "delta_kinetic": delta,
                                     "expected_delta_kinetic": expected, "impulse_norm": impulse_norm})
                self.action_seconds += perf_counter()-stamp
            previous_velocity = [b+q for b, q in zip(self.control.base, self.control.external)]
            before = [p.detach().clone() for p in self.params]
        self.optimizer.step()
        with torch.no_grad():
            for i, (p, old) in enumerate(zip(self.params, before)):
                self.control.base[i] = p.detach().clone()-old
                self.control.external[i].mul_(self.control.decay); p.add_(self.control.external[i])
            self.movement = tensor_norm([p-old for p, old in zip(self.params, before)])/(1+tensor_norm(before))
            errors = [b+q-self.lm.beta1*v+lr*g for b, q, v, g in
                      zip(self.control.base, self.control.external, previous_velocity, gradients)]
            self.max_recurrence_error = max(self.max_recurrence_error, tensor_norm(errors))
        return float(loss.detach()), self.movement, float(grad_norm)


def train_routed(data, lm_config, kind, lr, decay, seed, *, mode="none", steps=400,
                 routing=None, eval_batches=32, checkpoint=None):
    routing = routing or RoutingConfig(window=20, epsilon_gain=1e-3, noise_aware=True, radii=(.001, .003))
    if mode != "none" and kind != "momentum":
        raise ValueError("Routing is defined for Momentum SGD.")
    model = make_model(data, lm_config, seed)
    optimizer = make_optimizer(model, kind, lr, decay, lm_config)
    training = RoutedTraining(model, optimizer, lm_config, mode, routing, seed)
    initial_hash = model_hash(model)
    cost = TrainingCost(3*steps); rows = []; status = "steps"
    started = perf_counter()
    for step in range(steps):
        try:
            loss, move, grad_norm = training.step(data.batch("train", seed, step), cost, step)
            rows.append({"step": step+1, "train_loss": loss, "relative_movement": move,
                         "gradient_norm_before_clip": grad_norm, "shots": training.router.shots,
                         "dampings": training.router.dampings, "supplied": training.supplied,
                         "dissipated": training.dissipated, "units": cost.used, **cost.counts})
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            status = f"numerical_error: {exc}"; break
    training_seconds = perf_counter()-started
    final_loss = None if status != "steps" else probe_loss(model, [data.batch("test", 81001, j) for j in range(eval_batches)])
    if checkpoint is not None:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "external": training.control.external, "base": training.control.base,
                    "lm_config": asdict(lm_config), "routing": asdict(routing), "seed": seed,
                    "kind": kind, "mode": mode, "lr": lr, "weight_decay": decay}, checkpoint)
    summary = {"kind": kind, "mode": mode, "seed": seed, "lr": lr, "weight_decay": decay,
               "status": status, "initial_hash": initial_hash, "final_hash": model_hash(model),
               "parameter_count": sum(p.numel() for p in model.parameters()), "actual_steps": len(rows),
               "used_units": cost.used, "budget_units": cost.limit, "eval_batches": eval_batches,
               "eval_split": "test", "final_loss": final_loss,
               "perplexity": None if final_loss is None else float(np.exp(final_loss)),
               "shots": training.router.shots, "dampings": training.router.dampings,
               "supplied": training.supplied, "supply_cap": float(training.supply_cap),
               "dissipated": training.dissipated, "feature_seconds": training.feature_seconds,
               "action_seconds": training.action_seconds, "training_seconds": training_seconds,
               "max_actuator_error": training.max_actuator_error, "max_recurrence_error": training.max_recurrence_error,
               "actual_train_tokens": len(rows)*lm_config.batch*lm_config.block, **cost.counts}
    return summary, rows, training.actions, training.router.events

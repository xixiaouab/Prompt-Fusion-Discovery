from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch.func import functional_call


def cosine_temperature(epoch: int, search_epochs: int, maximum: float = 5.0, minimum: float = 0.001) -> float:
    if search_epochs < 1 or minimum <= 0 or maximum < minimum:
        raise ValueError("Use positive search epochs and 0 < minimum <= maximum.")
    progress = min(max(epoch / search_epochs, 0.0), 1.0)
    return minimum + 0.5 * (maximum - minimum) * (1 + math.cos(math.pi * progress))


def architecture_regularizers(model, temperature: float):
    probabilities = model.probabilities(temperature)
    log_probabilities = (model.architecture_logits / temperature).log_softmax(dim=-1)
    negative_entropy = (probabilities * log_probabilities).sum()
    expected_cost = (probabilities * model.operator_costs).sum()
    return negative_entropy, expected_cost


def _batch(batch):
    if isinstance(batch, Mapping):
        return batch["images"], batch["labels"]
    return batch[0], batch[1]


class Architect:
    """One-step unrolled architecture update with an exact mixed Hessian product."""

    def __init__(self, model, optimizer, entropy_weight: float = 0.01, cost_weight: float = 0.05, grad_clip: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.entropy_weight = entropy_weight
        self.cost_weight = cost_weight
        self.grad_clip = grad_clip

    def unrolled_objective(self, train_batch, val_batch, inner_lr: float, temperature: float):
        if self.model.is_discrete:
            raise ValueError("Architecture search cannot update a discretized model.")
        if inner_lr < 0:
            raise ValueError("inner_lr must be nonnegative.")
        train_images, train_labels = _batch(train_batch)
        val_images, val_labels = _batch(val_batch)
        weights = dict(self.model.named_weight_parameters())
        train_loss = F.cross_entropy(self.model(train_images, temperature=temperature), train_labels)
        gradients = torch.autograd.grad(train_loss, tuple(weights.values()), create_graph=True)
        virtual_weights = {name: value - inner_lr * gradient for (name, value), gradient in zip(weights.items(), gradients)}
        val_logits = functional_call(self.model, virtual_weights, (val_images,), {"temperature": temperature}, strict=False)
        val_loss = F.cross_entropy(val_logits, val_labels)
        entropy, cost = architecture_regularizers(self.model, temperature)
        objective = val_loss + self.entropy_weight * entropy + self.cost_weight * cost
        return objective, {"train_loss": train_loss, "val_loss": val_loss, "negative_entropy": entropy, "expected_cost": cost}

    def step(self, train_batch, val_batch, inner_lr: float, temperature: float = 1.0):
        architecture_parameters = self.model.architecture_parameters()
        if not architecture_parameters:
            raise ValueError("No trainable architecture parameters are available.")
        self.optimizer.zero_grad(set_to_none=True)
        objective, metrics = self.unrolled_objective(train_batch, val_batch, inner_lr, temperature)
        gradients = torch.autograd.grad(objective, architecture_parameters)
        for parameter, gradient in zip(architecture_parameters, gradients):
            parameter.grad = gradient.detach()
        norm = torch.nn.utils.clip_grad_norm_(architecture_parameters, self.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        return {"outer_loss": float(objective.detach()), "architecture_grad_norm": float(norm), **{name: float(value.detach()) for name, value in metrics.items()}}

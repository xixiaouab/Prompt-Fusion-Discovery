import argparse
import copy
import json
import math
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F
import yaml

from .data import build_loaders
from .model import create_model
from .search import Architect, cosine_temperature
from .utils import atomic_save, load_config, resolve_device, restore_rng, rng_state, seed_everything, write_json


def to_device(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def validation_metrics(model, loader, device, temperature=1.0):
    model.eval()
    loss_sum = correct = count = 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            logits = model(batch["images"], temperature=temperature)
            loss = F.cross_entropy(logits, batch["labels"], reduction="sum")
            loss_sum += float(loss)
            correct += int((logits.argmax(-1) == batch["labels"]).sum())
            count += batch["labels"].numel()
    if not count:
        raise ValueError("The validation split is empty.")
    return {"loss": loss_sum / count, "top1": 100 * correct / count, "samples": count}


def make_weight_optimizer(model, settings):
    return torch.optim.AdamW(model.weight_parameters(), lr=float(settings.get("learning_rate", 1e-3)),
                             weight_decay=float(settings.get("weight_decay", 0.01)))


def best_name(phase):
    return "best_search.pt" if phase == "search" else "best.pt"


def run_training(config, output_dir, resume=None, device=None, stop_after_epoch=None):
    run_config = copy.deepcopy(config)
    settings = run_config.setdefault("training", {})
    seed_everything(int(settings.get("seed", 0)))
    device = resolve_device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = build_loaders(run_config)
    if not {"train", "val"}.issubset(loaders):
        raise ValueError("Search requires disjoint train and validation splits.")
    num_classes = loaders["train"].dataset.num_classes
    model_options = run_config.setdefault("model", {})
    if model_options.get("num_classes", num_classes) != num_classes:
        raise ValueError("model.num_classes does not match the dataset.")
    model_options["num_classes"] = num_classes
    search_epochs = int(settings.get("search_epochs", 90))
    finetune_epochs = int(settings.get("finetune_epochs", 10))
    warmup = int(settings.get("alpha_warmup_epochs", 10))
    total_epochs = search_epochs + finetune_epochs
    if search_epochs < 0 or finetune_epochs < 1 or not 0 <= warmup <= search_epochs:
        raise ValueError("Use search_epochs >= alpha_warmup_epochs >= 0 and finetune_epochs >= 1.")
    if search_epochs == 0 and not model_options.get("blueprint"):
        raise ValueError("Fine-tuning without search requires model.blueprint.")
    if search_epochs and model_options.get("blueprint"):
        raise ValueError("A supplied model.blueprint requires training.search_epochs=0.")
    checkpoint = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if checkpoint:
        for section in ("model", "data"):
            if checkpoint["run_config"].get(section, {}) != run_config.get(section, {}):
                raise ValueError(f"Resume {section} settings differ from the checkpoint.")
        for key in ("search_epochs", "alpha_warmup_epochs"):
            if checkpoint["run_config"].get("training", {}).get(key, settings.get(key)) != settings.get(key):
                raise ValueError(f"Resume training.{key} differs from the checkpoint.")
    active_config = copy.deepcopy(checkpoint["config"] if checkpoint else run_config)
    active_config["training"] = copy.deepcopy(settings)
    model = create_model(active_config, pretrained=False if checkpoint else None).to(device)
    phase = checkpoint["phase"] if checkpoint else ("search" if search_epochs else "finetune")
    architecture_variables = list(model.architecture_parameters())
    weight_optimizer = make_weight_optimizer(model, settings)
    architecture_optimizer = None
    architect = None
    if phase == "search":
        architecture_optimizer = torch.optim.AdamW(
            model.architecture_parameters(), lr=float(settings.get("architecture_learning_rate", 3e-4)),
            weight_decay=float(settings.get("architecture_weight_decay", 0)))
        architect = Architect(model, architecture_optimizer,
                              entropy_weight=float(settings.get("entropy_weight", 0.01)),
                              cost_weight=float(settings.get("cost_weight", 0.05)),
                              grad_clip=float(settings.get("architecture_grad_clip", 1.0)))
    start_epoch, best_top1 = 0, -math.inf
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        weight_optimizer.load_state_dict(checkpoint["weight_optimizer"])
        if architecture_optimizer is not None:
            architecture_optimizer.load_state_dict(checkpoint["architecture_optimizer"])
        start_epoch, best_top1 = checkpoint["epoch"] + 1, checkpoint["best_top1"]
        restore_rng(checkpoint["rng"])
        for split, state in checkpoint["loader_rng"].items():
            if split in loaders:
                loaders[split].generator.set_state(state)
        destination = output_dir / best_name(phase)
        if not destination.exists():
            original_best = Path(resume).parent / best_name(phase)
            if original_best.is_file():
                shutil.copyfile(original_best, destination)
            else:
                best_top1 = -math.inf
    if start_epoch >= total_epochs:
        raise ValueError("The checkpoint has completed the requested training schedule.")
    if stop_after_epoch is not None and stop_after_epoch <= start_epoch:
        raise ValueError("stop_after_epoch must be greater than the resumed epoch count.")
    (output_dir / "config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    print(json.dumps({"device": str(device), "start_epoch": start_epoch,
                      "trainable_parameters": sum(p.numel() for p in model.weight_parameters())}), flush=True)
    completed = start_epoch
    for epoch in range(start_epoch, total_epochs):
        if epoch >= search_epochs and phase == "search":
            blueprint = model.blueprint()
            model.discretize(blueprint=blueprint, prune=True)
            active_config["model"]["blueprint"] = blueprint
            write_json({"operators": blueprint}, output_dir / "blueprint.json")
            weight_optimizer = make_weight_optimizer(model, settings)
            architecture_optimizer = architect = None
            architecture_variables = []
            phase, best_top1 = "finetune", -math.inf
        learning_rate = float(settings.get("learning_rate", 1e-3)) * 0.5 * (
            1 + math.cos(math.pi * epoch / total_epochs))
        for group in weight_optimizer.param_groups:
            group["lr"] = learning_rate
        temperature = cosine_temperature(epoch, search_epochs,
                                         float(settings.get("temperature_max", 5.0)),
                                         float(settings.get("temperature_min", 0.001))) if phase == "search" else 1.0
        model.train()
        loss_sum = correct = count = architecture_steps = 0
        architecture_metrics = {}
        validation_iterator = iter(loaders["val"]) if phase == "search" and epoch >= warmup else None
        for batch in loaders["train"]:
            batch = to_device(batch, device)
            for parameter in architecture_variables:
                parameter.requires_grad_(False)
            weight_optimizer.zero_grad(set_to_none=True)
            logits = model(batch["images"], temperature=temperature)
            loss = F.cross_entropy(logits, batch["labels"])
            if not torch.isfinite(loss):
                raise FloatingPointError("The training loss is non-finite.")
            loss.backward()
            weight_optimizer.step()
            if validation_iterator is not None:
                for parameter in architecture_variables:
                    parameter.requires_grad_(True)
                try:
                    validation_batch = next(validation_iterator)
                except StopIteration:
                    validation_iterator = iter(loaders["val"])
                    validation_batch = next(validation_iterator)
                architecture_metrics = architect.step(batch, to_device(validation_batch, device),
                                                      inner_lr=learning_rate, temperature=temperature)
                architecture_steps += 1
            size = batch["labels"].numel()
            loss_sum += float(loss.detach()) * size
            correct += int((logits.argmax(-1) == batch["labels"]).sum())
            count += size
        if not count:
            raise ValueError("The training split is empty.")
        validation = validation_metrics(model, loaders["val"], device, temperature)
        improved = validation["top1"] > best_top1
        best_top1 = max(best_top1, validation["top1"])
        record = {"epoch": epoch + 1, "phase": phase, "learning_rate": learning_rate,
                  "temperature": temperature, "architecture_steps": architecture_steps,
                  "train": {"loss": loss_sum / count, "top1": 100 * correct / count, "samples": count},
                  "val": validation, "best_val_top1": best_top1,
                  "architecture": architecture_metrics, "blueprint": model.blueprint()}
        print(json.dumps(record), flush=True)
        with (output_dir / "metrics.jsonl").open("a" if epoch else "w") as handle:
            handle.write(json.dumps(record) + "\n")
        state = {"format_version": 1, "epoch": epoch, "phase": phase,
                 "config": active_config, "run_config": run_config,
                 "model": model.state_dict(), "weight_optimizer": weight_optimizer.state_dict(),
                 "architecture_optimizer": architecture_optimizer.state_dict() if architecture_optimizer else None,
                 "schedule": {"epoch": epoch + 1, "total_epochs": total_epochs}, "temperature": temperature,
                 "best_top1": best_top1, "rng": rng_state(),
                 "loader_rng": {split: loader.generator.get_state() for split, loader in loaders.items()}}
        atomic_save(state, output_dir / "last.pt")
        if improved:
            atomic_save(state, output_dir / best_name(phase))
        completed = epoch + 1
        if stop_after_epoch is not None and completed >= stop_after_epoch:
            break
    result = {"phase": phase, "epochs": completed, "best_val_top1": best_top1,
              "best_checkpoint": str(output_dir / best_name(phase)),
              "last_checkpoint": str(output_dir / "last.pt"), "blueprint": model.blueprint()}
    write_json(result, output_dir / "summary.json")
    return result


def main():
    parser = argparse.ArgumentParser(description="Search prompt fusion and fine-tune the discrete architecture.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device")
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    run_training(load_config(args.config, args.set), args.output, args.resume, args.device, args.stop_after_epoch)


if __name__ == "__main__":
    main()

import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn

from .data import build_loader
from .model import create_model
from .utils import resolve_device, write_json


def load_checkpoint_model(checkpoint, device=None):
    device = resolve_device(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = copy.deepcopy(state["config"])
    model = create_model(config, pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, config, device, float(state.get("temperature", 1.0))


def evaluate_checkpoint(checkpoint, manifest=None, data_root=None, split="test", device=None,
                        output=None, batch_size=None):
    if split not in {"val", "test"}:
        raise ValueError("Evaluation split must be val or test.")
    model, config, device, temperature = load_checkpoint_model(checkpoint, device)
    if manifest is not None:
        config["data"]["manifest"] = str(manifest)
    if data_root is not None:
        config["data"]["root"] = str(data_root)
    if batch_size is not None:
        config.setdefault("training", {})["batch_size"] = batch_size
    loader = build_loader(config, split)
    correct = count = 0
    loss_sum = 0.0
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["images"].to(device), temperature=temperature)
            labels = batch["labels"].to(device)
            loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
            predicted = logits.argmax(-1)
            loss_sum += float(loss)
            correct += int((predicted == labels).sum())
            count += labels.numel()
            predictions.extend({"id": identifier, "label": label, "prediction": prediction}
                               for identifier, label, prediction in
                               zip(batch["ids"], labels.cpu().tolist(), predicted.cpu().tolist()))
    if not count:
        raise ValueError(f"The {split} split is empty.")
    metrics = {"split": split, "samples": count, "loss": loss_sum / count,
               "top1": 100 * correct / count}
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        write_json(metrics, output / "metrics.json")
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions))
    print(json.dumps(metrics), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved prompt fusion model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", default="outputs/evaluation")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.manifest, args.root, args.split, args.device,
                        args.output, args.batch_size)


if __name__ == "__main__":
    main()

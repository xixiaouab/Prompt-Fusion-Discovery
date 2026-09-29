import copy
import csv
import json

import numpy as np
from PIL import Image
import pytest
import torch

from prompt_fusion.data import build_loader, read_manifest
from prompt_fusion.evaluate import evaluate_checkpoint, load_checkpoint_model
from prompt_fusion.predict import predict
from prompt_fusion.train import run_training


@pytest.fixture
def tiny_config(tmp_path):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    records = []
    for split, count in (("train", 4), ("val", 2), ("test", 2)):
        for index in range(count):
            array = np.zeros((40, 40, 3), dtype=np.uint8)
            array[..., index % 2] = 150
            array[2 + index:15 + index, 10:20, 2] = 170
            path = tmp_path / f"{split}-{index}.png"
            Image.fromarray(array).save(path)
            records.append({"path": path.name, "label": index % 2, "split": split})
    manifest = tmp_path / "images.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "label", "split"))
        writer.writeheader()
        writer.writerows(records)
    yield {
        "model": {"backbone": "vit_tiny_patch16_224", "pretrained": False, "num_classes": 2,
                  "backbone_kwargs": {"img_size": 32, "patch_size": 8, "embed_dim": 24,
                                      "depth": 2, "num_heads": 3, "mlp_ratio": 2},
                  "prompt_length": 2, "sparse_topk": 4},
        "data": {"manifest": str(manifest), "protocol": "official", "image_size": 32,
                 "resize_size": 40, "num_workers": 0},
        "training": {"search_epochs": 2, "finetune_epochs": 2, "alpha_warmup_epochs": 1,
                     "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.01,
                     "architecture_learning_rate": 0.0003, "seed": 19},
    }
    torch.set_num_threads(threads)


def test_search_discretize_evaluate_and_resume(tiny_config, tmp_path):
    output = tmp_path / "run"
    warmup = run_training(tiny_config, output, device="cpu", stop_after_epoch=1)
    state0 = torch.load(output / "last.pt", weights_only=True)
    assert warmup["phase"] == "search"
    assert torch.count_nonzero(state0["model"]["architecture_logits"]) == 0
    original_backbone = state0["model"]["backbone.patch_embed.proj.weight"].clone()

    run_training(tiny_config, output, resume=output / "last.pt", device="cpu", stop_after_epoch=3)
    state1 = torch.load(output / "last.pt", weights_only=True)
    assert state1["phase"] == "finetune"
    assert state1["epoch"] == 2
    assert state1["architecture_optimizer"] is None
    assert len(state1["config"]["model"]["blueprint"]) == 2
    assert torch.count_nonzero(state1["model"]["architecture_logits"]) > 0
    assert (output / "blueprint.json").is_file()
    assert (output / "best.pt").is_file()

    result = run_training(tiny_config, output, resume=output / "last.pt", device="cpu")
    final = torch.load(output / "last.pt", weights_only=True)
    assert final["epoch"] == 3 and result["epochs"] == 4
    assert torch.equal(original_backbone, final["model"]["backbone.patch_embed.proj.weight"])
    assert not torch.equal(state1["model"]["head.weight"], final["model"]["head.weight"])
    assert all(int(value["step"]) == 4 for value in final["weight_optimizer"]["state"].values())
    model, config, device, temperature = load_checkpoint_model(output / "last.pt", "cpu")
    assert model.is_discrete
    assert model.blueprint() == final["config"]["model"]["blueprint"]
    batch = next(iter(build_loader(config, "test")))
    with torch.no_grad():
        logits = model(batch["images"], temperature=temperature)
    metrics = evaluate_checkpoint(output / "last.pt", device="cpu", output=tmp_path / "evaluation")
    assert metrics["samples"] == 2
    assert metrics["top1"] == pytest.approx(100 * (logits.argmax(-1) == batch["labels"]).float().mean().item())
    predicted = predict(output / "last.pt", tmp_path / "test-0.png", top_k=2, device="cpu")
    probabilities, labels = logits[0].softmax(-1).sort(descending=True)
    assert [row["class_id"] for row in predicted] == labels.tolist()
    assert [row["probability"] for row in predicted] == pytest.approx(probabilities.tolist())
    history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["phase"] for row in history] == ["search", "search", "finetune", "finetune"]
    assert [row["architecture_steps"] for row in history] == [0, 2, 0, 0]

    uninterrupted = tmp_path / "continuous"
    run_training(tiny_config, uninterrupted, device="cpu")
    expected = torch.load(uninterrupted / "last.pt", weights_only=True)
    assert all(torch.equal(value, expected["model"][name]) for name, value in final["model"].items())


def test_resume_rejects_changed_search_model(tiny_config, tmp_path):
    output = tmp_path / "run"
    run_training(tiny_config, output, device="cpu", stop_after_epoch=1)
    changed = copy.deepcopy(tiny_config)
    changed["model"]["prompt_length"] = 5
    with pytest.raises(ValueError, match="Resume model settings"):
        run_training(changed, output, resume=output / "last.pt", device="cpu")


def test_manifest_rejects_train_validation_overlap(tmp_path):
    Image.new("RGB", (16, 16)).save(tmp_path / "shared.png")
    manifest = tmp_path / "images.csv"
    manifest.write_text("path,label,split\nshared.png,0,train\nshared.png,0,val\n")
    with pytest.raises(ValueError, match="Duplicate image"):
        read_manifest(manifest)


def test_generated_partitions_do_not_change_with_training_seed(tmp_path, monkeypatch):
    from prompt_fusion import data

    records = [data.Sample(tmp_path / f"{index}.png", index % 2, "train", str(index))
               for index in range(40)]
    monkeypatch.setattr(data, "read_manifest", lambda *_: records)
    config = {"data": {"manifest": "unused.csv", "protocol": "fgvc", "split_seed": 42},
              "training": {"seed": 0}}
    first = data.dataset_records(config)
    config["training"]["seed"] = 2
    second = data.dataset_records(config)
    assert [(row.id, row.split) for row in first] == [(row.id, row.split) for row in second]
    assert sum(row.split == "val" for row in first) == 4


def test_vtab_preserves_supplied_partition_and_test_set(tmp_path):
    from prompt_fusion.data import Sample, apply_protocol

    records = [Sample(tmp_path / f"train-{index}", index % 10, "train", f"train-{index}")
               for index in range(800)]
    records += [Sample(tmp_path / f"val-{index}", index % 10, "val", f"val-{index}")
                for index in range(200)]
    records += [Sample(tmp_path / "test", 0, "test", "test")]
    assert apply_protocol(records, "vtab", seed=5) == records
    with pytest.raises(ValueError, match="800 train / 200 val"):
        apply_protocol(records[1:], "vtab", seed=5)

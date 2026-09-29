from dataclasses import dataclass
import csv
import math
import os
from pathlib import Path
import random
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF



IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    split: str
    id: str


def validate_records(records: Sequence[Sample]) -> None:
    if not records:
        raise ValueError("The dataset contains no images.")
    paths, identifiers = set(), set()
    for sample in records:
        if sample.split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {sample.split!r}; use train, val, or test.")
        if sample.label < 0:
            raise ValueError("Class labels must be nonnegative integers.")
        path = sample.path.resolve()
        if path in paths or sample.id in identifiers:
            raise ValueError(f"Duplicate image or id across dataset records: {sample.id}")
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.add(path)
        identifiers.add(sample.id)


def read_manifest(path: str | Path, root: str | Path | None = None) -> list[Sample]:
    path = Path(path).expanduser().resolve()
    image_root = Path(root).expanduser().resolve() if root else path.parent
    records = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "label", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("A manifest must contain path,label,split columns; id is optional.")
        for line_number, row in enumerate(reader, start=2):
            image_path = Path(row["path"]).expanduser()
            image_path = image_path if image_path.is_absolute() else image_root / image_path
            try:
                label = int(row["label"])
            except ValueError as error:
                raise ValueError(f"Manifest line {line_number}: label must be an integer.") from error
            records.append(Sample(image_path.resolve(), label, row["split"].strip().lower(),
                                  row.get("id", "").strip() or str(image_path.resolve())))
    validate_records(records)
    return records


def scan_imagefolder(root: str | Path) -> list[Sample]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    split_dirs = {split: root / split for split in ("train", "val", "test")
                  if (root / split).is_dir()}
    if not split_dirs:
        split_dirs = {"train": root}
    class_names = sorted({directory.name for folder in split_dirs.values()
                          for directory in folder.iterdir() if directory.is_dir()})
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    records = []
    for split, folder in split_dirs.items():
        for class_name in class_names:
            class_folder = folder / class_name
            if not class_folder.is_dir():
                continue
            for path in sorted(class_folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append(Sample(path.resolve(), class_to_idx[class_name], split,
                                          path.relative_to(root).as_posix()))
    validate_records(records)
    return records


def stratified_split(records: Sequence[Sample], val_count: int, seed: int = 42) -> list[Sample]:
    if not 0 < val_count < len(records):
        raise ValueError("Validation size must be between 1 and the training-set size minus 1.")
    if any(sample.split != "train" for sample in records):
        raise ValueError("Only training samples may be repartitioned.")
    groups = {}
    for sample in sorted(records, key=lambda item: (item.label, item.id)):
        groups.setdefault(sample.label, []).append(sample)
    ratio = val_count / len(records)
    quotas = {label: min(len(group) - 1, int(len(group) * ratio))
              for label, group in groups.items()}
    remaining = val_count - sum(quotas.values())
    order = sorted(groups, key=lambda label: (-(len(groups[label]) * ratio - quotas[label]), label))
    while remaining:
        progressed = False
        for label in order:
            if quotas[label] < len(groups[label]) - 1:
                quotas[label] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("Not enough samples to retain every class in training.")
    rng = np.random.default_rng(seed)
    validation_ids = set()
    for label, group in groups.items():
        selection = rng.permutation(len(group))[:quotas[label]]
        validation_ids.update(group[index].id for index in selection)
    return [Sample(sample.path, sample.label,
                   "val" if sample.id in validation_ids else "train", sample.id)
            for sample in records]


def apply_protocol(records: Sequence[Sample], protocol: str = "official", seed: int = 42,
                   val_fraction: float = 0.1) -> list[Sample]:
    protocol = protocol.lower().replace("-1k", "")
    if protocol not in {"official", "fgvc", "vtab", "hta"}:
        raise ValueError("protocol must be official, fgvc, vtab, or hta.")
    training = [sample for sample in records if sample.split == "train"]
    validation = [sample for sample in records if sample.split == "val"]
    holdout = [sample for sample in records if sample.split != "train"]
    if not training or protocol in {"official", "hta"}:
        return list(records)
    if validation:
        if protocol == "vtab" and (len(training), len(validation)) != (800, 200):
            raise ValueError("VTAB-1k requires its supplied 800 train / 200 val partition.")
        return list(records)
    if protocol == "vtab":
        if len(training) != 1000:
            raise ValueError("VTAB-1k needs the supplied 1000 training examples or an 800/200 manifest.")
        val_count = 200
    else:
        if not 0 < val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1).")
        val_count = math.ceil(len(training) * val_fraction)
    return stratified_split(training, val_count, seed) + holdout


def write_manifest(records: Sequence[Sample], path: str | Path,
                   root: str | Path | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(root).resolve() if root else path.parent.resolve()
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "label", "split", "id"))
        writer.writeheader()
        for sample in records:
            writer.writerow({"path": os.path.relpath(sample.path, root), "label": sample.label,
                             "split": sample.split, "id": sample.id})


class ImageTransform:
    def __init__(self, training: bool, image_size: int = 224, resize_size: int = 256,
                 horizontal_flip: float = 0.0, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        if image_size < 1 or resize_size < image_size:
            raise ValueError("resize_size must be >= image_size > 0.")
        if not 0 <= horizontal_flip <= 1:
            raise ValueError("horizontal_flip must be in [0, 1].")
        self.training = training
        self.image_size = image_size
        self.resize_size = resize_size
        self.horizontal_flip = horizontal_flip if training else 0.0
        self.mean, self.std = tuple(mean), tuple(std)
        self.deterministic = not training
        operations = [transforms.Resize(resize_size, InterpolationMode.BICUBIC),
                      transforms.RandomCrop(image_size) if training else transforms.CenterCrop(image_size)]
        if self.horizontal_flip:
            operations.append(transforms.RandomHorizontalFlip(self.horizontal_flip))
        self.spatial = transforms.Compose(operations)

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.spatial(image.convert("RGB"))

    def normalize(self, image: Image.Image) -> torch.Tensor:
        return TF.normalize(TF.to_tensor(image), self.mean, self.std)

    def settings(self) -> dict:
        return {"training": self.training, "image_size": self.image_size,
                "resize_size": self.resize_size, "horizontal_flip": self.horizontal_flip,
                "mean": self.mean, "std": self.std, "interpolation": "bicubic"}


class ImageDataset(Dataset):
    def __init__(self, records: Sequence[Sample], transform: ImageTransform,
                 num_classes: int | None = None):
        self.records = list(records)
        self.transform = transform
        self.num_classes = num_classes or max((item.label for item in records), default=-1) + 1

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        sample = self.records[index]
        with Image.open(sample.path) as image:
            view = self.transform(image)
        return {"images": self.transform.normalize(view), "labels": sample.label, "ids": sample.id}


def dataset_records(config: Mapping) -> list[Sample]:
    data = config.get("data", {})
    if data.get("manifest"):
        records = read_manifest(data["manifest"], data.get("root"))
    elif data.get("root"):
        records = scan_imagefolder(data["root"])
    else:
        raise ValueError("Set data.manifest or data.root.")
    seed = int(data.get("split_seed", 42))
    return apply_protocol(records, data.get("protocol", "official"), seed,
                          float(data.get("val_fraction", 0.1)))


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def _loader(config: Mapping, records: Sequence[Sample], split: str) -> DataLoader:
    data, training = config.get("data", {}), config.get("training", {})
    selected = [sample for sample in records if sample.split == split]
    if not selected:
        raise ValueError(f"The dataset has no {split!r} samples.")
    transform = ImageTransform(split == "train", int(data.get("image_size", 224)),
                               int(data.get("resize_size", 256)), float(data.get("horizontal_flip", 0)),
                               data.get("mean", IMAGENET_MEAN), data.get("std", IMAGENET_STD))
    num_classes = max(sample.label for sample in records) + 1
    dataset = ImageDataset(selected, transform, num_classes)
    seed = int(config.get("seed", training.get("seed", 42)))
    workers = int(data.get("num_workers", 4))
    return DataLoader(dataset, batch_size=int(training.get("batch_size", data.get("batch_size", 64))),
                      shuffle=split == "train", num_workers=workers, drop_last=False,
                      pin_memory=bool(data.get("pin_memory", torch.cuda.is_available())),
                      persistent_workers=bool(data.get("persistent_workers", False)) and workers > 0,
                      worker_init_fn=_seed_worker,
                      generator=torch.Generator().manual_seed(seed))


def build_loader(config: Mapping, split: str) -> DataLoader:
    return _loader(config, dataset_records(config), split)


def build_loaders(config: Mapping) -> dict[str, DataLoader]:
    records = dataset_records(config)
    available = {sample.split for sample in records}
    return {split: _loader(config, records, split) for split in ("train", "val", "test")
            if split in available}

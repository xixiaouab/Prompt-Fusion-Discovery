# Prompt Fusion Discovery

**Layer-Specific Prompt Fusion Discovery via Differentiable Search in Vision Foundation Models** · ECCV 2026

[Paper](https://arxiv.org/abs/2606.26379) · [Project](https://xixiaouab.github.io/Prompt-Fusion-Discovery/)

Layer-wise search over **Add, Affine, Concat, and Cross-Attention** for frozen vision transformers, followed by operator selection and discrete fine-tuning.

## Installation

Python 3.10+ and PyTorch with the CUDA version appropriate for your GPU.

```bash
git clone https://github.com/xixiaouab/Prompt-Fusion-Discovery.git
cd Prompt-Fusion-Discovery
pip install -e .
```

## Data

Create a CSV with paths relative to `data.root`. Supply the benchmark's train, validation, and test partitions; VTAB-1k uses 800 training and 200 validation images.

```csv
path,label,split
train/class_0/image_1.jpg,0,train
val/class_0/image_2.jpg,0,val
test/class_0/image_3.jpg,0,test
```

For folders organized as `train|val|test/class_name/image.jpg`:

```bash
python -m prompt_fusion.prepare --root data/task --output data/task.csv
```

## Search and Train

```bash
python -m prompt_fusion.train --config configs/vtab.yaml --output outputs/task \
  --set data.root=data/task data.manifest=data/task.csv training.seed=0
```

The default schedule runs 90 search epochs and 10 discrete fine-tuning epochs. Architecture logits remain fixed during the first 10 epochs. Training uses AdamW, cosine learning-rate decay, and batch size 64. Use seeds `0`, `1`, and `2` for separate runs.

Choose `configs/fgvc.yaml`, `configs/hta.yaml`, `configs/vtab_mae.yaml`, or `configs/vtab_swin.yaml` for the corresponding benchmark/backbone. All settings are editable in YAML.

For MoCo v3, download its [ViT-B checkpoint](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar) to `checkpoints/mocov3_vit_b.pth.tar`, then use `configs/vtab_mocov3.yaml`.

Resume either phase with the same configuration:

```bash
python -m prompt_fusion.train --config configs/vtab.yaml --output outputs/task \
  --resume outputs/task/last.pt
```

## Evaluate and Predict

```bash
python -m prompt_fusion.evaluate --checkpoint outputs/task/best.pt \
  --split test --output outputs/task/evaluation
python -m prompt_fusion.predict --checkpoint outputs/task/best.pt \
  --image path/to/image.jpg --top-k 5
```

`best.pt` contains the best discrete model; `best_search.pt` contains the best search model. `blueprint.json` records the selected operator at each layer. Evaluation writes top-1 accuracy and per-image predictions.

## Files

| File | Purpose |
| --- | --- |
| `prompt_fusion/model.py` | Frozen ViT/Swin, fusion operators, and pruning |
| `prompt_fusion/search.py` | Unrolled architecture gradients and regularizers |
| `prompt_fusion/train.py` | Search, fine-tuning, and checkpoint recovery |
| `prompt_fusion/data.py`, `prepare.py` | Image transforms and split manifests |
| `prompt_fusion/evaluate.py`, `predict.py` | Dataset evaluation and single-image inference |
| `configs/` | Benchmark and backbone settings |
| `tests/` | Operator, gradient, and training checks |

```bash
pip install -e '.[dev]'
pytest -q
```

## Citation

```bibtex
@article{xiao2026promptfusion,
  title={Layer-Specific Prompt Fusion Discovery via Differentiable Search in Vision Foundation Models},
  author={Xiao, Xi and Li, Xingjian and Zhang, Yunbei and Han, Cheng and Liu, Tianming and Wang, Tianyang and Jiang, Runmin and Hamm, Jihun and Wang, Xiao and Xu, Min},
  journal={arXiv preprint arXiv:2606.26379},
  year={2026}
}
```

Code: [MIT License](LICENSE). Project-page content and assets: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

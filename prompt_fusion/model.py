from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn


OPERATORS = ("add", "affine", "concat", "cross")
OPERATOR_COSTS = (0.0, 0.06, 0.30, 1.0)


def prompt_summary(prompt: Tensor) -> Tensor:
    return F.layer_norm(prompt.mean(dim=1), (prompt.shape[-1],))


class AddFusion(nn.Module):
    def forward(self, x: Tensor, prompt: Tensor) -> Tensor:
        return x + prompt_summary(prompt).unsqueeze(1)


class AffineFusion(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, dim // reduction)
        self.gamma = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        self.beta = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))
        for projection in (self.gamma[-1], self.beta[-1]):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, x: Tensor, prompt: Tensor) -> Tensor:
        summary = prompt_summary(prompt)
        gamma = 2 * torch.sigmoid(self.gamma(summary))
        return gamma.unsqueeze(1) * x + self.beta(summary).unsqueeze(1)


class ConcatFusion(nn.Module):
    def __init__(self, num_tokens: int, prompt_length: int, identity_strength: float = 10.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.prompt_length = prompt_length
        self.logits = nn.Parameter(torch.zeros(prompt_length + num_tokens, num_tokens))
        with torch.no_grad():
            self.logits[:prompt_length].normal_(std=0.01)
            self.logits[prompt_length:].diagonal().fill_(identity_strength)
        self.sparse_topk: int | None = None

    def reduction_weights(self) -> Tensor:
        logits = self.logits
        if self.sparse_topk is not None and self.sparse_topk < logits.shape[0]:
            indices = logits.topk(self.sparse_topk, dim=0).indices
            mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(0, indices, True)
            logits = logits.masked_fill(~mask, float("-inf"))
        return logits.softmax(dim=0)

    def forward(self, x: Tensor, prompt: Tensor) -> Tensor:
        if x.shape[1] != self.num_tokens or prompt.shape[1] != self.prompt_length:
            raise ValueError("Concat token counts must match the configured image and prompt lengths.")
        tokens = torch.cat((prompt, x), dim=1)
        if self.sparse_topk is not None and self.sparse_topk < self.logits.shape[0]:
            logits, indices = self.logits.topk(self.sparse_topk, dim=0)
            selected = tokens[:, indices]
            return (selected * logits.softmax(dim=0)[None, :, :, None]).sum(dim=1)
        return torch.einsum("sk,bsd->bkd", self.reduction_weights(), tokens)


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int, reduction: int = 4):
        super().__init__()
        hidden = dim // reduction
        if hidden < num_heads or hidden % num_heads:
            raise ValueError("The reduced cross-attention dimension must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.query = nn.Linear(dim, hidden, bias=False)
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(dim, hidden, bias=False)
        self.output = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: Tensor, prompt: Tensor) -> Tensor:
        batch, count, _ = x.shape
        q = self.query(x).reshape(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(prompt).reshape(batch, prompt.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(prompt).reshape(batch, prompt.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attention = ((q * self.head_dim**-0.5) @ k.transpose(-1, -2)).softmax(dim=-1)
        output = (attention @ v).transpose(1, 2).reshape(batch, count, -1)
        return x + self.output(output)


class FusionBank(nn.Module):
    def __init__(self, dim: int, num_tokens: int, prompt_length: int, num_heads: int, affine_reduction: int, cross_reduction: int, identity_strength: float):
        super().__init__()
        self.operators = nn.ModuleDict({
            "add": AddFusion(),
            "affine": AffineFusion(dim, affine_reduction),
            "concat": ConcatFusion(num_tokens, prompt_length, identity_strength),
            "cross": CrossAttentionFusion(dim, num_heads, cross_reduction),
        })

    def forward(self, x: Tensor, prompt: Tensor, weights: Tensor | None = None, operator: str | None = None) -> Tensor:
        if operator is not None:
            return self.operators[operator](x, prompt)
        return sum(weights[i] * self.operators[name](x, prompt) for i, name in enumerate(OPERATORS))


class PromptFusion(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        prompt_length: int = 10,
        affine_reduction: int = 4,
        cross_reduction: int = 4,
        share_fusion: bool = True,
        sparse_topk: int | None = 4,
        identity_strength: float = 10.0,
        prompt_std: float = 0.02,
        blueprint: Sequence[str] | None = None,
    ):
        super().__init__()
        if min(num_classes, prompt_length, affine_reduction, cross_reduction) < 1:
            raise ValueError("Model dimensions and reduction factors must be positive.")
        if sparse_topk is not None and sparse_topk < 1:
            raise ValueError("sparse_topk must be positive or None.")
        self.backbone = backbone.requires_grad_(False)
        self.backbone.eval()
        self.is_swin = hasattr(backbone, "layers") and hasattr(backbone, "patch_embed")
        if self.is_swin:
            blocks = [block for stage in backbone.layers for block in stage.blocks]
            specs = [(block.attn.qkv.in_features, block.window_area, block.attn.num_heads) for block in blocks]
        elif hasattr(backbone, "blocks") and hasattr(backbone, "_pos_embed"):
            blocks = list(backbone.blocks)
            if getattr(backbone.patch_drop, "prob", 0) > 0:
                raise ValueError("Token dropping is incompatible with token-preserving fusion.")
            count = backbone.patch_embed.num_patches + backbone.num_prefix_tokens
            specs = [(backbone.num_features, count, block.attn.num_heads) for block in blocks]
        else:
            raise TypeError("Expected a timm VisionTransformer or SwinTransformer backbone.")
        for block in blocks:
            block.attn.fused_attn = False
        self.num_layers = len(blocks)
        self.sparse_topk = sparse_topk
        self.prompts = nn.ParameterList([nn.Parameter(torch.empty(prompt_length, dim)) for dim, _, _ in specs])
        for prompt in self.prompts:
            nn.init.normal_(prompt, std=prompt_std)
        self.architecture_logits = nn.Parameter(torch.zeros(self.num_layers, len(OPERATORS)))
        self.register_buffer("operator_costs", torch.tensor(OPERATOR_COSTS))
        self.register_buffer("selected_operators", torch.full((self.num_layers,), -1, dtype=torch.long))
        self.fusion_banks = nn.ModuleDict()
        self._bank_keys = []
        self._discrete_blueprint: tuple[str, ...] | None = None
        for layer, (dim, count, heads) in enumerate(specs):
            key = f"{dim}_{count}_{heads}" if share_fusion else str(layer)
            self._bank_keys.append(key)
            if key not in self.fusion_banks:
                self.fusion_banks[key] = FusionBank(dim, count, prompt_length, heads, affine_reduction, cross_reduction, identity_strength)
        self.head = nn.Linear(backbone.num_features, num_classes)
        if blueprint is not None:
            self.discretize(blueprint)

    @property
    def is_discrete(self) -> bool:
        return self._discrete_blueprint is not None

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def weight_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad and parameter is not self.architecture_logits)

    def named_weight_parameters(self):
        return ((name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad and parameter is not self.architecture_logits)

    def architecture_parameters(self):
        return [self.architecture_logits] if self.architecture_logits.requires_grad else []

    def parameter_counts(self) -> dict[str, int]:
        return {"trainable": sum(p.numel() for p in self.parameters() if p.requires_grad), "total": sum(p.numel() for p in self.parameters())}

    def probabilities(self, temperature: float = 1.0) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        return (self.architecture_logits / temperature).softmax(dim=-1)

    def blueprint(self) -> list[str]:
        if self._discrete_blueprint is not None:
            return list(self._discrete_blueprint)
        return [OPERATORS[index] for index in self.architecture_logits.detach().argmax(dim=-1).tolist()]

    def discretize(self, blueprint: Sequence[str] | None = None, prune: bool = True) -> list[str]:
        chosen = tuple(blueprint or self.blueprint())
        if len(chosen) != self.num_layers or any(name not in OPERATORS for name in chosen):
            raise ValueError("The blueprint must contain one valid operator per transformer layer.")
        for key, name in zip(self._bank_keys, chosen):
            if name not in self.fusion_banks[key].operators:
                raise ValueError(f"Operator {name} has already been pruned from bank {key}.")
        self._discrete_blueprint = chosen
        self.selected_operators.copy_(torch.tensor([OPERATORS.index(name) for name in chosen], device=self.selected_operators.device))
        self.architecture_logits.requires_grad_(False)
        for key, bank in self.fusion_banks.items():
            active = {chosen[layer] for layer, candidate in enumerate(self._bank_keys) if candidate == key}
            for name in list(bank.operators):
                if prune and name not in active:
                    del bank.operators[name]
                elif name == "concat":
                    bank.operators[name].sparse_topk = self.sparse_topk
        return list(chosen)

    def _fuse(self, x: Tensor, layer: int, weights: Tensor) -> Tensor:
        prompt = self.prompts[layer].unsqueeze(0).expand(x.shape[0], -1, -1)
        operator = self._discrete_blueprint[layer] if self.is_discrete else None
        return self.fusion_banks[self._bank_keys[layer]](x, prompt, weights[layer], operator)

    def _forward_vit(self, images: Tensor, weights: Tensor) -> Tensor:
        backbone = self.backbone
        x = backbone.norm_pre(backbone.patch_drop(backbone._pos_embed(backbone.patch_embed(images))))
        for layer, block in enumerate(backbone.blocks):
            fused = self._fuse(block.norm1(x), layer, weights)
            x = x + block.drop_path1(block.ls1(block.attn(fused)))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        x = backbone.norm(x)
        features = x[:, 0] if getattr(backbone, "global_pool", "token") == "token" else x[:, backbone.num_prefix_tokens:].mean(dim=1)
        return backbone.fc_norm(features)

    def _swin_attention(self, block: nn.Module, x: Tensor, layer: int, weights: Tensor) -> Tensor:
        from timm.models.swin_transformer import window_partition, window_reverse

        batch, h, w, dim = x.shape
        shifted = torch.roll(x, shifts=(-block.shift_size[0], -block.shift_size[1]), dims=(1, 2))
        wh, ww = block.window_size
        shifted = F.pad(shifted, (0, 0, 0, (-w) % ww, 0, (-h) % wh))
        hp, wp = shifted.shape[1:3]
        windows = window_partition(shifted, block.window_size).reshape(-1, block.window_area, dim)
        windows = self._fuse(windows, layer, weights)
        mask = block.get_attn_mask(shifted) if getattr(block, "dynamic_mask", False) else block.attn_mask
        output = block.attn(windows, mask).reshape(-1, wh, ww, dim)
        output = window_reverse(output, block.window_size, hp, wp)[:, :h, :w]
        return torch.roll(output, shifts=block.shift_size, dims=(1, 2)).reshape(batch, h, w, dim)

    def _forward_swin(self, images: Tensor, weights: Tensor) -> Tensor:
        x = self.backbone.patch_embed(images)
        layer = 0
        for stage in self.backbone.layers:
            x = stage.downsample(x)
            for block in stage.blocks:
                x = x + block.drop_path1(self._swin_attention(block, block.norm1(x), layer, weights))
                batch, h, w, dim = x.shape
                flat = x.reshape(batch, h * w, dim)
                x = (flat + block.drop_path2(block.mlp(block.norm2(flat)))).reshape(batch, h, w, dim)
                layer += 1
        return self.backbone.norm(x).mean(dim=(1, 2))

    def forward(self, images: Tensor, temperature: float = 1.0, return_features: bool = False):
        weights = self.probabilities(temperature)
        features = (self._forward_swin if self.is_swin else self._forward_vit)(images, weights)
        logits = self.head(features)
        return {"logits": logits, "features": features, "operator_probabilities": weights} if return_features else logits


def load_backbone_checkpoint(backbone: nn.Module, checkpoint_path: str, checkpoint_format: str = "auto"):
    if checkpoint_format not in {"auto", "pytorch", "mocov3"}:
        raise ValueError("backbone_checkpoint_format must be auto, pytorch, or mocov3.")
    if Path(checkpoint_path).suffix.lower() in {".npz", ".npy"}:
        if not hasattr(backbone, "load_pretrained"):
            raise ValueError("This backbone does not support NumPy checkpoints.")
        backbone.load_pretrained(checkpoint_path)
        return
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if key in state and isinstance(state[key], Mapping):
            state = state[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    if any(key.startswith("base_encoder.") for key in state):
        state = {key.removeprefix("base_encoder."): value for key, value in state.items() if key.startswith("base_encoder.")}
    state = {key: value for key, value in state.items() if not key.startswith(("head.", "head_dist.", "fc.", "predictor.", "momentum_encoder."))}
    expected = set(backbone.state_dict())
    missing, unexpected = sorted(expected - set(state)), sorted(set(state) - expected)
    if missing or unexpected:
        raise ValueError(f"Backbone checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    backbone.load_state_dict(state, strict=True)


def create_model(config: Mapping, pretrained: bool | None = None, checkpoint_path: str | None = None) -> PromptFusion:
    import timm

    options = dict(config.get("model", config))
    name = options.pop("backbone", "vit_base_patch16_224.orig_in21k")
    kwargs = options.pop("backbone_kwargs", {})
    configured_pretrained = options.pop("pretrained", True)
    load_pretrained = bool(configured_pretrained if pretrained is None else pretrained and configured_pretrained)
    backbone_checkpoint = options.pop("backbone_checkpoint", options.pop("backbone_checkpoint_path", None))
    checkpoint_format = options.pop("backbone_checkpoint_format", "auto")
    backbone = timm.create_model(name, pretrained=load_pretrained and not backbone_checkpoint, num_classes=0, **kwargs)
    if backbone_checkpoint and pretrained is not False:
        load_backbone_checkpoint(backbone, backbone_checkpoint, checkpoint_format)
    options.setdefault("num_classes", config.get("data", {}).get("num_classes", config.get("num_classes")))
    if options["num_classes"] is None:
        raise ValueError("Set model.num_classes or data.num_classes.")
    model = PromptFusion(backbone, **options)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint.get("model", checkpoint))
    return model

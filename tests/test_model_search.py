import copy
import math

import pytest
import torch
from timm.models.swin_transformer import SwinTransformer
from timm.models.vision_transformer import VisionTransformer

from prompt_fusion.model import AddFusion, AffineFusion, ConcatFusion, CrossAttentionFusion, OPERATORS, PromptFusion, create_model, load_backbone_checkpoint, prompt_summary
from prompt_fusion.search import Architect, architecture_regularizers, cosine_temperature


def tiny_model(depth=2, **kwargs):
    backbone = VisionTransformer(img_size=16, patch_size=8, embed_dim=24, depth=depth, num_heads=3, num_classes=0)
    return PromptFusion(backbone, num_classes=3, prompt_length=2, **kwargs)


def batches(dtype=torch.float32):
    return [(torch.randn(2, 3, 16, 16, dtype=dtype), torch.tensor([0, 2])), (torch.randn(2, 3, 16, 16, dtype=dtype), torch.tensor([1, 0]))]


def test_all_operators_preserve_token_count_and_dimension():
    x, prompt = torch.randn(2, 5, 24), torch.randn(2, 2, 24)
    for operator in (AddFusion(), AffineFusion(24), ConcatFusion(5, 2), CrossAttentionFusion(24, 3)):
        assert operator(x, prompt).shape == x.shape


def test_add_is_normalized_mean_and_affine_initially_identity():
    x, prompt = torch.randn(2, 5, 24), torch.randn(2, 2, 24)
    torch.testing.assert_close(AddFusion()(x, prompt), x + prompt_summary(prompt)[:, None])
    torch.testing.assert_close(AffineFusion(24)(x, prompt), x)


def test_concat_is_column_stochastic_and_retains_image_identity():
    operator = ConcatFusion(5, 2)
    weights = operator.reduction_weights()
    torch.testing.assert_close(weights.sum(dim=0), torch.ones(5))
    assert torch.all(weights >= 0)
    assert torch.all(weights[2:].diagonal() > 0.999)
    x, prompt = torch.randn(2, 5, 24), torch.randn(2, 2, 24)
    expected = torch.einsum("sk,bsd->bkd", weights, torch.cat((prompt, x), dim=1))
    torch.testing.assert_close(operator(x, prompt), expected)


def test_sparse_concat_keeps_top_four_sources_with_exact_sparse_output():
    operator = ConcatFusion(5, 2)
    operator.sparse_topk = 4
    weights = operator.reduction_weights()
    assert (weights > 0).sum(dim=0).tolist() == [4] * 5
    torch.testing.assert_close(weights.sum(dim=0), torch.ones(5))
    x, prompt = torch.randn(2, 5, 24), torch.randn(2, 2, 24)
    expected = torch.einsum("sk,bsd->bkd", weights, torch.cat((prompt, x), dim=1))
    torch.testing.assert_close(operator(x, prompt), expected)
    operator(x, prompt).square().sum().backward()
    assert operator.logits.grad is not None


def test_cross_attention_uses_token_memory_and_residual():
    operator = CrossAttentionFusion(24, 3)
    x, prompt = torch.randn(2, 5, 24), torch.randn(2, 2, 24)
    assert not torch.allclose(operator(x, prompt), operator(x, prompt * 2))
    with torch.no_grad():
        operator.value.weight.zero_()
    torch.testing.assert_close(operator(x, prompt), x)


def test_fusion_occurs_after_norm_before_attention_with_fixed_token_interface():
    model = tiny_model(depth=1).eval()
    model.discretize(["affine"])
    block = model.backbone.blocks[0]
    captured_norm, captured_attention = [], []
    hooks = [block.norm1.register_forward_hook(lambda module, args, output: captured_norm.append(output)), block.attn.register_forward_pre_hook(lambda module, args: captured_attention.append(args[0]))]
    model(batches()[0][0])
    for hook in hooks:
        hook.remove()
    assert len(captured_norm) == 1
    assert captured_attention[0].shape == (2, 5, 24)
    torch.testing.assert_close(captured_attention[0], captured_norm[0])


def test_model_gradients_are_partitioned_and_backbone_remains_frozen():
    model = tiny_model().train()
    assert not model.backbone.training
    assert not model.backbone.blocks[0].attn.fused_attn
    images, labels = batches()[0]
    torch.nn.functional.cross_entropy(model(images), labels).backward()
    assert model.architecture_logits.grad is not None
    assert not any(parameter.grad is not None for parameter in model.backbone.parameters())
    assert all(parameter.grad is not None for parameter in model.weight_parameters())
    assert all(parameter is not model.architecture_logits for parameter in model.weight_parameters())
    assert len(model.fusion_banks) == 1


def test_discretization_selects_argmax_and_discards_inactive_modules():
    model = tiny_model(sparse_topk=None).eval()
    with torch.no_grad():
        model.architecture_logits[:] = torch.tensor([[1.0, 9.0, 2.0, 3.0], [7.0, 3.0, 0.0, 1.0]])
    images = batches()[0][0]
    near_discrete = model(images, temperature=0.001)
    assert model.discretize() == ["affine", "add"]
    assert model.is_discrete
    assert not model.architecture_logits.requires_grad
    assert set(next(iter(model.fusion_banks.values())).operators) == {"affine", "add"}
    torch.testing.assert_close(model(images), near_discrete)
    assert model.architecture_parameters() == []


def test_regularizers_use_correct_sign_and_operator_cost_order():
    model = tiny_model()
    entropy, cost = architecture_regularizers(model, 1.0)
    torch.testing.assert_close(entropy, torch.tensor(-2 * math.log(4)))
    torch.testing.assert_close(cost, torch.tensor(2 * (0 + 0.06 + 0.30 + 1.0) / 4))
    assert OPERATORS == ("add", "affine", "concat", "cross")


def test_cosine_temperature_boundaries_and_midpoint():
    assert cosine_temperature(0, 90) == 5.0
    assert cosine_temperature(90, 90) == pytest.approx(0.001)
    assert cosine_temperature(45, 90) == pytest.approx((5.0 + 0.001) / 2)
    with pytest.raises(ValueError):
        cosine_temperature(0, 0)


def test_second_order_architecture_gradient_matches_finite_difference():
    torch.manual_seed(31)
    model = tiny_model(depth=1).double().eval()
    architect = Architect(model, torch.optim.SGD(model.architecture_parameters(), lr=0.01))
    train_batch, val_batch = batches(torch.float64)
    loss, _ = architect.unrolled_objective(train_batch, val_batch, 0.07, 0.9)
    gradient = torch.autograd.grad(loss, model.architecture_logits)[0]
    epsilon = 1e-4
    finite = torch.zeros_like(gradient)
    for operator in range(4):
        with torch.no_grad():
            model.architecture_logits[0, operator] += epsilon
        positive, _ = architect.unrolled_objective(train_batch, val_batch, 0.07, 0.9)
        with torch.no_grad():
            model.architecture_logits[0, operator] -= 2 * epsilon
        negative, _ = architect.unrolled_objective(train_batch, val_batch, 0.07, 0.9)
        finite[0, operator] = (positive.detach() - negative.detach()) / (2 * epsilon)
        with torch.no_grad():
            model.architecture_logits[0, operator] += epsilon
    torch.testing.assert_close(gradient, finite, atol=1e-6, rtol=1e-4)


def test_architect_updates_only_architecture_and_clips_gradient():
    torch.manual_seed(23)
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.architecture_parameters(), lr=0.01)
    architect = Architect(model, optimizer, grad_clip=0.001)
    before = {name: parameter.clone() for name, parameter in model.named_weight_parameters()}
    before_alpha = model.architecture_logits.detach().clone()
    train_batch, val_batch = batches()
    metrics = architect.step({"images": train_batch[0], "labels": train_batch[1]}, val_batch, 0.01)
    assert math.isfinite(metrics["outer_loss"])
    assert not torch.equal(before_alpha, model.architecture_logits)
    assert model.architecture_logits.grad.norm() <= 0.00101
    for name, parameter in model.named_weight_parameters():
        torch.testing.assert_close(before[name], parameter)
        assert parameter.grad is None


def test_unrolled_gradient_includes_inner_response_term():
    torch.manual_seed(42)
    model = tiny_model(depth=1)
    architect = Architect(model, torch.optim.SGD(model.architecture_parameters(), lr=0.01))
    train_batch, val_batch = batches()
    full, _ = architect.unrolled_objective(train_batch, val_batch, 0.2, 1.0)
    full_gradient = torch.autograd.grad(full, model.architecture_logits)[0]
    direct, _ = architect.unrolled_objective(train_batch, val_batch, 0.0, 1.0)
    direct_gradient = torch.autograd.grad(direct, model.architecture_logits)[0]
    assert not torch.allclose(full_gradient, direct_gradient, atol=1e-7, rtol=1e-5)


def test_pruned_checkpoint_roundtrip_with_blueprint(tmp_path):
    config = {"model": {"backbone": "vit_tiny_patch16_224", "pretrained": False, "num_classes": 3, "prompt_length": 2, "backbone_kwargs": {"img_size": 16, "patch_size": 8, "embed_dim": 24, "depth": 2, "num_heads": 3}}}
    model = create_model(config).eval()
    config["model"]["blueprint"] = model.discretize(["concat", "cross"])
    checkpoint = tmp_path / "discrete.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    restored = create_model(config, pretrained=False, checkpoint_path=str(checkpoint)).eval()
    images = batches()[0][0]
    assert restored.blueprint() == ["concat", "cross"]
    torch.testing.assert_close(restored(images), model(images))


def test_swin_fusion_search_and_stage_sharing():
    backbone = SwinTransformer(img_size=16, patch_size=4, embed_dim=16, depths=(2, 2), num_heads=(2, 4), window_size=2, num_classes=0)
    model = PromptFusion(backbone, num_classes=3, prompt_length=2)
    assert len(model.fusion_banks) == 2
    images, labels = batches()[0]
    loss = torch.nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    assert model.architecture_logits.grad is not None
    assert not any(parameter.grad is not None for parameter in model.backbone.parameters())
    architect = Architect(model, torch.optim.SGD(model.architecture_parameters(), lr=0.01))
    metrics = architect.step(*batches(), inner_lr=0.01)
    assert math.isfinite(metrics["outer_loss"])
    model.discretize(["add", "concat", "affine", "cross"])
    assert model(images).shape == (2, 3)


def test_official_moco_checkpoint_loads_encoder_and_discards_pretraining_heads(tmp_path):
    original = tiny_model().backbone
    state = {f"module.base_encoder.{name}": value for name, value in original.state_dict().items()}
    state["module.base_encoder.head.0.weight"] = torch.randn(8, 24)
    state["module.momentum_encoder.cls_token"] = torch.randn(1, 1, 24)
    state["module.predictor.0.weight"] = torch.randn(8, 8)
    path = tmp_path / "moco.pth.tar"
    torch.save({"state_dict": state}, path)
    restored = tiny_model().backbone
    load_backbone_checkpoint(restored, str(path), "mocov3")
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name])
    state["module.base_encoder.unknown_layer.weight"] = torch.randn(2, 2)
    torch.save({"state_dict": state}, path)
    with pytest.raises(ValueError, match="unexpected"):
        load_backbone_checkpoint(restored, str(path), "mocov3")


def test_moco_linear_evaluation_checkpoint_accepts_module_prefix(tmp_path):
    original = tiny_model().backbone
    state = {f"module.{name}": value for name, value in original.state_dict().items()}
    state["module.head.weight"] = torch.randn(1000, 24)
    state["module.head.bias"] = torch.randn(1000)
    path = tmp_path / "linear-eval.pth.tar"
    torch.save({"state_dict": state}, path)
    restored = tiny_model().backbone
    load_backbone_checkpoint(restored, str(path), "mocov3")
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, original.state_dict()[name])

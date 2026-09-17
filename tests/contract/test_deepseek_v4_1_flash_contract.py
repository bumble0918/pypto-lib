# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for the DeepSeek-V4.1 Flash decode layer composition."""

import ast
import inspect
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.deepseek_v4_1_flash._golden_smoke import make_decode_layer_golden_inputs
from models.deepseek_v4_1_flash.decode_layer import (
    REPRESENTATIVE_LAYER_IDS,
    DecodeLayerKind,
    decode_layer_attention_inputs,
    decode_layer_kernel_skip_reason,
    golden_decode_layer,
    resolve_decode_layer_plan,
)
from models.deepseek_v4_1_flash.mhc import golden_mhc_mixes, golden_mhc_pre
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.decode_attention import (
    attention_half_skip_reason, build_specs, compare_normalized, compare_unchanged, make_attention_program,
)


MODEL_DIR = Path(__file__).parents[2] / "models" / "deepseek_v4_1_flash"


def _tree(name: str) -> ast.Module:
    return ast.parse((MODEL_DIR / name).read_text())


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def test_decode_layer_schedule_covers_all_40_layers():
    plans = [resolve_decode_layer_plan(layer_id) for layer_id in range(40)]
    assert Counter(plan.kind for plan in plans) == {
        DecodeLayerKind.SWA: 2,
        DecodeLayerKind.C2A_FULL: 3,
        DecodeLayerKind.C2A_REUSE: 15,
        DecodeLayerKind.C1A_FULL: 1,
        DecodeLayerKind.C1A_REINDEX: 4,
        DecodeLayerKind.C1A_REUSE: 15,
    }
    assert {kind: resolve_decode_layer_plan(layer_id).kind for kind, layer_id in REPRESENTATIVE_LAYER_IDS.items()} == {
        kind: kind for kind in DecodeLayerKind
    }


def test_decode_layer_schedule_resolves_cache_sources():
    assert resolve_decode_layer_plan(2).kv_source_layer_id == 2
    assert resolve_decode_layer_plan(7).index_source_layer_id == 2
    assert resolve_decode_layer_plan(8).kv_source_layer_id == 8
    assert resolve_decode_layer_plan(19).index_source_layer_id == 14
    assert resolve_decode_layer_plan(20).is_candidate_source
    assert resolve_decode_layer_plan(23).index_source_layer_id == 20
    assert resolve_decode_layer_plan(24).index_source_layer_id == 24
    assert resolve_decode_layer_plan(39).index_source_layer_id == 36
    with pytest.raises(ValueError, match="layer_id must be"):
        resolve_decode_layer_plan(40)


def test_decode_layer_has_all_six_attention_branches_and_block_order():
    function = _function(_tree("decode_layer.py"), "decode_layer")
    condition_names = {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assert "layer_kind" in condition_names
    assert "layer_id" not in condition_names
    calls = sorted(
        (
            node.lineno,
            node.func.id,
        )
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    )
    names = [name for _, name in calls]
    for attention_name in (
        "decode_swa",
        "decode_c2a_full",
        "decode_c2a_reuse",
        "decode_c1a_full",
        "decode_c1a_reindex",
        "decode_c1a_reuse",
    ):
        assert names.count(attention_name) == 1
    assert names.count("mhc_mixes") == 2
    assert names.count("mhc_pre") == 2
    assert names.count("mhc_post") == 2
    first_mix = names.index("mhc_mixes")
    first_pre = names.index("mhc_pre")
    first_post = names.index("mhc_post")
    second_mix = names.index("mhc_mixes", first_mix + 1)
    second_pre = names.index("mhc_pre", first_pre + 1)
    moe_call = names.index("moe")
    second_post = names.index("mhc_post", first_post + 1)
    assert first_mix < first_pre < first_post < second_mix < second_pre < moe_call < second_post


def test_decode_layer_union_abi_covers_every_attention_entry():
    decode_args = {arg.arg for arg in _function(_tree("decode_layer.py"), "decode_layer").args.args}
    aliases = {
        "x": "normalized_attention",
        "compressed_indices": "topk_indices",
        "output_window": "attention_output_window",
        "output_arrived": "attention_output_arrived",
        "output": "attention_output",
    }
    ignored = {"group_base", "tp_rank", "num_tokens", "attention_epoch"}
    for name in (
        "decode_swa",
        "decode_c2a_full",
        "decode_c2a_reuse",
        "decode_c1a_full",
        "decode_c1a_reindex",
        "decode_c1a_reuse",
    ):
        leaf_args = [arg.arg for arg in _function(_tree(f"{name}.py"), name).args.args]
        for argument in leaf_args:
            if argument in ignored:
                continue
            assert aliases.get(argument, argument) in decode_args or aliases.get(argument) in {
                "normalized_attention",
                "attention_output",
            }, (name, argument)


def test_decode_layer_calls_match_leaf_positional_contracts():
    function = _function(_tree("decode_layer.py"), "decode_layer")
    aliases = {
        "x": "normalized_attention",
        "compressed_indices": "topk_indices",
        "output_window": "attention_output_window",
        "output_arrived": "attention_output_arrived",
        "output": "attention_output",
    }
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if not name.startswith("decode_"):
            continue
        leaf = _function(_tree(f"{name}.py"), name)
        assert [ast.unparse(arg) for arg in node.args] == [
            aliases.get(arg.arg, arg.arg) for arg in leaf.args.args
        ], name


def test_c1a_decode_cache_abi_is_preserved_until_kernel_delivery():
    expected = {
        "compressed_cache": "pl.Tensor[[C.CMP_BLOCKS_DYN, 128, 1, C.HEAD_DIM], pl.FP4]",
        "index_cache": "pl.Tensor[[C.INDEX_BLOCKS_DYN, 128, 1, C.INDEX_DIM], pl.FP4]",
    }
    for filename, function_name in (
        ("decode_c1a_full.py", "decode_c1a_full"),
        ("decode_c1a_reindex.py", "decode_c1a_reindex"),
        ("decode_c1a_reuse.py", "decode_c1a_reuse"),
    ):
        arguments = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in _function(_tree(filename), function_name).args.args
        }
        assert arguments["compressed_cache"] == expected["compressed_cache"]
        if "index_cache" in arguments:
            assert arguments["index_cache"] == expected["index_cache"]
    for entry in ("decode_layer", "l3_decode_layer"):
        arguments = {
            argument.arg: ast.unparse(argument.annotation)
            for argument in _function(_tree("decode_layer.py"), entry).args.args
        }
        assert arguments["compressed_cache"] == "pl.InOut[pl.Tensor]"
        assert arguments["index_cache"] == "pl.InOut[pl.Tensor]"
        assert arguments["compressor_wkv"] == "pl.Tensor"


def test_l3_decode_layer_declares_mutable_state_and_outputs():
    host = _function(_tree("decode_layer.py"), "l3_decode_layer")
    inout_names = {
        arg.arg
        for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.InOut[")
    }
    out_names = {
        arg.arg
        for arg in host.args.args
        if arg.annotation is not None and ast.unparse(arg.annotation).startswith("pl.Out[")
    }
    assert inout_names == {
        "window_cache",
        "window_cache_scale",
        "compressed_cache",
        "compressed_cache_scale",
        "index_cache",
        "index_cache_scale",
        "compressor_state",
        "topk_indices",
        "candidate_mask",
    }
    assert out_names == {"x_next", "next_pre_mix"}


@pytest.mark.parametrize("layer_id", REPRESENTATIVE_LAYER_IDS.values())
def test_decode_layer_unfinished_device_dependencies_are_explicit(layer_id):
    reason = decode_layer_kernel_skip_reason(layer_id)
    assert reason is not None and "EP8 MoE kernel" in reason
    if layer_id >= 20:
        assert "attention kernel" in reason
        assert "cache ABI agreement" in reason


@pytest.mark.parametrize("layer_id", REPRESENTATIVE_LAYER_IDS.values())
def test_decode_layer_golden_runs_every_mode(layer_id):
    inputs = make_decode_layer_golden_inputs(layer_id)
    original_compressed = inputs["attention_inputs"]["compressed_cache"].clone()
    original_index = inputs["attention_inputs"]["index_cache"].clone()
    result = golden_decode_layer(**inputs)
    expected_next_pre_mix = golden_mhc_mixes(
        result.attention_hidden,
        inputs["hc_ffn_fn"],
        inputs["hc_ffn_scale"],
        inputs["hc_ffn_base"],
    )[0]
    assert result.output.shape == inputs["x_hc"].shape
    assert result.output.dtype is torch.float32
    assert result.next_pre_mix.shape == inputs["incoming_pre_mix"].shape
    assert torch.isfinite(result.output).all()
    assert torch.equal(result.next_pre_mix, expected_next_pre_mix)
    attn_pre = golden_mhc_mixes(
        inputs["x_hc"], inputs["hc_attn_fn"], inputs["hc_attn_scale"], inputs["hc_attn_base"]
    )[0]
    assert torch.equal(result.attention_input, golden_mhc_pre(inputs["x_hc"], inputs["incoming_pre_mix"]))
    assert torch.equal(result.ffn_input, golden_mhc_pre(result.attention_hidden, attn_pre))
    assert not torch.equal(attn_pre, inputs["incoming_pre_mix"])
    assert torch.isfinite(result.next_pre_mix).all()
    assert set(decode_layer_attention_inputs(layer_id)) <= {"x", *inputs["attention_inputs"]}
    if layer_id in (3, 21, 24):
        assert torch.equal(result.attention.compressed_cache, original_compressed)
    if layer_id == 24:
        assert torch.equal(result.attention.index_cache, original_index)


@pytest.mark.parametrize("layer_id", REPRESENTATIVE_LAYER_IDS.values())
def test_attention_half_readiness_is_independent_of_moe(layer_id):
    reason = attention_half_skip_reason(layer_id)
    assert "MoE" not in (reason or "")
    if layer_id < 20:
        assert reason is None
    else:
        assert "cache ABI agreement" in reason
        with pytest.raises(NotImplementedError, match="attention kernel"):
            make_attention_program(layer_id, C.TP_SIZE, 1, [])


@pytest.mark.parametrize("adapter,leaf", (("_swa", "decode_swa"), ("_c2a_full", "decode_c2a_full"),
                                         ("_c2a_reuse", "decode_c2a_reuse")))
def test_attention_half_adapters_keep_leaf_call_contracts(adapter, leaf):
    function = _function(_tree("decode_attention.py"), adapter)
    call = next(node for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == leaf)
    parameters = _function(_tree(f"{leaf}.py"), leaf).args.args
    assert [ast.unparse(arg) for arg in call.args] == [
        "topk_indices" if arg.arg == "compressed_indices" else arg.arg for arg in parameters
    ]
    annotations = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
    for name in ("wq_a_scale", "wq_b_scale", "wkv_scale", "wo_b_scale", "index_wq_b_scale"):
        assert "pl.MX_B_NN" in annotations[name]


@pytest.mark.parametrize("layer_id", (0, 2, 3))
def test_attention_half_specs_match_host_and_do_not_generate_weights(layer_id, monkeypatch):
    # Full's existing spec builder is eager; the SWA/Reuse builders stay lazy.
    if layer_id != 2:
        def unexpected(*args, **kwargs):
            raise AssertionError("spec creation generated weights")
        monkeypatch.setattr("models.deepseek_v4_1_flash.decode_swa.make_inputs", unexpected)
        monkeypatch.setattr("models.deepseek_v4_1_flash.decode_c2a_reuse.make_c2a_reuse_inputs", unexpected)
    args = SimpleNamespace(tp=C.TP_SIZE, dp=1, tokens=2, active_tokens=2, requests=2,
                           epochs=2, seed=17, case="mixed", bench=False)
    specs = build_specs(args, resolve_decode_layer_plan(layer_id).kind, {})
    program = make_attention_program(layer_id, C.TP_SIZE, 2, specs)
    assert [spec.name for spec in specs] == list(inspect.signature(program._func).parameters)
    parameters = inspect.signature(program._func).parameters
    for spec in specs:
        if hasattr(spec, "shape"):
            assert list(parameters[spec.name].annotation.shape) == spec.shape
    assert "compressed_indices" not in {spec.name for spec in specs}
    assert "x" not in {spec.name for spec in specs}


def test_attention_half_non_owner_scale_comparison_is_byte_exact():
    initial = torch.ones(4).to(torch.float8_e4m3fn)
    compare = compare_unchanged("scale")
    assert compare(initial.clone(), initial, inputs={"scale": initial})[0]
    changed = initial.clone()
    changed.view(torch.uint8)[0] = 0
    assert not compare(changed, initial, inputs={"scale": initial})[0]


def test_attention_half_normalized_suffix_is_byte_exact():
    expected = torch.full((1, 3, 32), 13.0, dtype=torch.bfloat16)
    kwargs = dict(inputs={"num_tokens": 2}, actual_outputs={}, expected_outputs={}, rtol=1e-3, atol=1e-3)
    assert compare_normalized(expected.clone(), expected, **kwargs)[0]
    changed = expected.clone()
    changed[:, 2:] = 13.0625
    assert not compare_normalized(changed, expected, **kwargs)[0]


def test_attention_half_partial_outputs_are_inout():
    tree = _tree("decode_attention.py")
    for name in ("attention_rank", "attention_group"):
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
        annotations = {arg.arg: ast.unparse(arg.annotation) for arg in function.args.args}
        for boundary in ("normalized_attention", "attention_output"):
            assert annotations[boundary].startswith("pl.InOut[")
        for boundary in ("attention_input", "attention_hidden", "attention_pre_mix"):
            assert annotations[boundary].startswith("pl.Out[")

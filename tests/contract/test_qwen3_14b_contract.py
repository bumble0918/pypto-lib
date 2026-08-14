# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from contract.registry import find_contract_for_model_config, get_contract


def _tiny_model_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=17,
    )


def _runtime_config() -> SimpleNamespace:
    return SimpleNamespace(
        max_batch_size=16,
        max_seq_len=2,
        page_size=128,
        vocab_pad_multiple=512,
        total_kv_pages=16,
    )


def _qwen3_14b_model_config(*, architectures: tuple[str, ...] | None = ("Qwen3ForCausalLM",)) -> SimpleNamespace:
    return SimpleNamespace(
        model_id="local-served-name",
        architecture="Qwen3ForCausalLM",
        architectures=architectures,
        model_type="qwen3",
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=40,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
    )


_TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "int32": torch.int32,
}


def _signature_params(fn: object) -> tuple[str, ...]:
    raw_fn = getattr(fn, "_func", fn)
    return tuple(inspect.signature(raw_fn).parameters)


def _assert_compile_args_match_contract(
    contract: object,
    stage_name: str,
    compile_args: tuple[torch.Tensor, ...],
    model_config: SimpleNamespace,
    runtime_config: SimpleNamespace,
) -> None:
    dims = _contract_dims(contract, model_config, runtime_config)
    stage = contract.kernels[stage_name]
    for index, (spec, tensor) in enumerate(zip(stage.args, compile_args)):
        assert tensor.dtype == _TORCH_DTYPES[spec.dtype], f"{stage_name}[{index}] {spec.name} dtype"
        assert tuple(tensor.shape) == _resolve_shape(spec.shape, dims), f"{stage_name}[{index}] {spec.name} shape"


def _contract_dims(contract: object, model_config: SimpleNamespace, runtime_config: SimpleNamespace) -> dict[str, int]:
    page = int(runtime_config.page_size)
    max_seq = int(runtime_config.max_seq_len)
    batch = int(runtime_config.max_batch_size)
    layers = int(model_config.num_hidden_layers)
    hidden = int(model_config.hidden_size)
    intermediate = int(model_config.intermediate_size)
    head_dim = int(model_config.head_dim)
    kv_heads = int(model_config.num_key_value_heads)
    blocks = (max_seq + page - 1) // page
    return {
        "BATCH": batch,
        "PREFILL_TOKENS": batch * max_seq,
        "H": hidden,
        "I": intermediate,
        "L": layers,
        "D": head_dim,
        "KVH": kv_heads * head_dim,
        "MAX_SEQ": max_seq,
        "BLOCK_TABLE_FLAT": batch * blocks,
        "KV_CACHE_ROWS": layers * batch * blocks * kv_heads * page,
        "VOCAB": _round_up(int(model_config.vocab_size), int(contract.limits["vocab_pad_multiple"])),
        "SAMPLED_IDS_PAD": int(contract.limits["sampled_ids_pad"]),
        "TOPK": int(contract.limits["topk"]),
        "SAMPLING_CONTROL_FIELDS": int(contract.limits["sampling_control_fields"]),
        "L*H": layers * hidden,
        "L*I": layers * intermediate,
    }


def _resolve_shape(shape: tuple[str | int, ...], dims: dict[str, int]) -> tuple[int, ...]:
    return tuple(dims[dim] if isinstance(dim, str) else dim for dim in shape)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def test_registry_resolves_explicit_qwen3_14b_contract() -> None:
    contract = get_contract("qwen3", "14b")

    assert contract.model.family == "qwen3"
    assert contract.model.variant == "14b"
    assert sorted(contract.kernels) == ["decode", "greedy_sample", "prefill", "topk_select"]
    assert contract.execution == {
        "prefill": ("prefill",),
        "decode": ("decode",),
        "topk_select": ("topk_select",),
    }
    assert "device_topk_sampling" in contract.capabilities
    assert contract.abi_fingerprint()


@pytest.mark.parametrize("architectures", [("Qwen3ForCausalLM",), None])
def test_registry_matches_qwen3_14b_model_config(architectures: tuple[str, ...] | None) -> None:
    model_config = _qwen3_14b_model_config(architectures=architectures)

    contract = find_contract_for_model_config(model_config)

    assert contract.model.family == "qwen3"
    assert contract.model.variant == "14b"


def test_loaded_kernel_modules_match_current_qwen3_files() -> None:
    contract = get_contract("qwen3", "14b")
    loaded = contract.load_kernels()
    model = _qwen3_14b_model()

    assert sorted(loaded.functions) == ["decode_fwd", "greedy_sample_fwd", "prefill_fwd", "topk_select_fwd"]
    assert sorted(contract.kernels) == ["decode", "greedy_sample", "prefill", "topk_select"]
    assert set(contract.kernels) <= {name.removesuffix("_fwd") for name in loaded.functions}
    contract.validate_kernels(contract, loaded, model)


def test_loaded_kernel_signatures_match_contract_args_exactly() -> None:
    contract = get_contract("qwen3", "14b")
    loaded = contract.load_kernels()

    for stage_name, stage in contract.kernels.items():
        kernel_fn = loaded.functions[f"{stage_name}_fwd"]
        contract_params = tuple(arg.name for arg in stage.args)
        host_params = _signature_params(stage.host_jit_fn)
        kernel_params = _signature_params(kernel_fn)
        assert host_params == contract_params
        assert kernel_params == contract_params


def test_compile_arg_builders_follow_loaded_stage_specs() -> None:
    contract = get_contract("qwen3", "14b")
    loaded = contract.load_kernels()
    model_config = _tiny_model_config()
    runtime_config = _runtime_config()

    for stage_name, stage in contract.kernels.items():
        compile_args = stage.compile_args_builder(model_config, runtime_config)
        assert len(compile_args) == len(stage.args)
        assert len(compile_args) == len(_signature_params(loaded.functions[f"{stage_name}_fwd"]))
        _assert_compile_args_match_contract(contract, stage_name, compile_args, model_config, runtime_config)


def test_decode_contract_uses_dynamic_batch() -> None:
    """decode serves any public batch >= 1 from one compiled program.

    Every host-visible batch axis is the same ``BATCH`` dim (prefill uses it too,
    so the two stages no longer spell the same concept differently), and
    ``limits["batch"]`` is the padded pipeline WIDTH -- one decode row window --
    not a required exact value and no longer decode's ceiling.
    """
    contract = get_contract("qwen3", "14b")
    decode_args = {arg.name: arg.shape for arg in contract.kernels["decode"].args}

    assert contract.limits["batch"] == 16
    assert decode_args["seq_lens"] == ("BATCH",)
    assert decode_args["slot_mapping"] == ("BATCH",)
    assert decode_args["out"] == ("BATCH", "VOCAB")
    assert decode_args["sampled_ids_in"] == ("BATCH", "SAMPLED_IDS_PAD")
    assert decode_args["sampled_ids_out"] == ("BATCH", "SAMPLED_IDS_PAD")
    assert decode_args["next_hidden"] == ("BATCH", "H")

    # prefill already used a dynamic batch axis; decode must now name it the same.
    prefill_args = {arg.name: arg.shape for arg in contract.kernels["prefill"].args}
    assert prefill_args["seq_lens"] == ("BATCH",)

    # Compile-time dummies stay sized at the padded width -- they bound buffer
    # capacity, not the runtime shape.
    compile_args = contract.kernels["decode"].compile_args_builder(
        _tiny_model_config(),
        _runtime_config(),
    )
    assert compile_args[6].shape == (16,)
    assert compile_args[-1].shape == (16, 8)


@pytest.mark.parametrize("batch", [1, 16, 17, 33])
def test_decode_contract_accepts_any_batch(batch: int) -> None:
    runtime = _runtime_config()
    runtime.max_batch_size = batch
    contract = get_contract("qwen3", "14b")
    # Must not raise at ANY batch: decode_fwd runs a batch above the padded width
    # as ceil(batch / batch_pad) row windows, so limits["batch"] bounds one window,
    # not the public batch.
    contract.kernels["decode"].compile_args_builder(_tiny_model_config(), runtime)


def test_decode_contract_rejects_empty_batch() -> None:
    runtime = _runtime_config()
    runtime.max_batch_size = 0
    contract = get_contract("qwen3", "14b")
    with pytest.raises(ValueError, match="max_batch_size"):
        contract.kernels["decode"].compile_args_builder(_tiny_model_config(), runtime)


def test_prefill_and_greedy_sample_stay_capped_at_pad() -> None:
    """Only decode chunks. The fixed-batch stages still bound the batch by the pad."""
    runtime = _runtime_config()
    runtime.max_batch_size = 17
    contract = get_contract("qwen3", "14b")
    for stage in ("prefill", "greedy_sample", "topk_select"):
        with pytest.raises(ValueError, match="max_batch_size"):
            contract.kernels[stage].compile_args_builder(_tiny_model_config(), runtime)


def test_runtime_arg_builders_follow_host_order() -> None:
    contract = get_contract("qwen3", "14b")
    static = SimpleNamespace(
        decode_weights={
            "decode_input_rms_weight": "input_rms_weight",
            "decode_wq": "wq",
            "decode_wk": "wk",
            "decode_wv": "wv",
            "decode_q_norm_weight": "q_norm_weight",
            "decode_k_norm_weight": "k_norm_weight",
            "decode_wo": "wo",
            "decode_w_gate": "w_gate",
            "decode_w_up": "w_up",
            "decode_w_down": "w_down",
            "decode_post_rms_weight": "post_rms_weight",
        },
        rope_cos="rope_cos",
        rope_sin="rope_sin",
        final_norm_weight="final_norm_weight",
        padded_lm_head_weight="lm_head",
        padded_embed_weight="embed",
    )
    prefill_inputs = SimpleNamespace(
        token_ids="token_ids",
        seq_lens="seq_lens",
        chunk_lens="chunk_lens",
        chunk_offsets="chunk_offsets",
        block_table="block_table",
        slot_mapping="slot_mapping",
    )
    decode_inputs = SimpleNamespace(
        seq_lens="seq_lens",
        block_table="block_table",
        slot_mapping="slot_mapping",
        logits="logits",
        token_ids="token_ids",
    )
    topk_inputs = SimpleNamespace(logits="logits", sampling_control="sampling_control")

    prefill_args = contract.kernels["prefill"].runtime_args_builder(
        prefill_inputs,
        static,
        k_cache="k_cache",
        v_cache="v_cache",
        logits="logits",
    )
    decode_args = contract.kernels["decode"].runtime_args_builder(
        decode_inputs,
        static,
        k_cache="k_cache",
        v_cache="v_cache",
        sampled_ids_buffer="sampled_ids",
        next_hidden_buffer="next_hidden",
    )
    topk_args = contract.kernels["topk_select"].runtime_args_builder(
        topk_inputs,
        static,
        topk_values_buffer="topk_values",
        topk_indices_buffer="topk_indices",
    )

    assert prefill_args == (
        "token_ids",
        "seq_lens",
        "chunk_lens",
        "chunk_offsets",
        "input_rms_weight",
        "wq",
        "wk",
        "wv",
        "q_norm_weight",
        "k_norm_weight",
        "rope_cos",
        "rope_sin",
        "block_table",
        "slot_mapping",
        "k_cache",
        "v_cache",
        "wo",
        "post_rms_weight",
        "w_gate",
        "w_up",
        "w_down",
        "final_norm_weight",
        "lm_head",
        "embed",
        "logits",
    )
    assert decode_args == (
        "input_rms_weight",
        "wq",
        "wk",
        "wv",
        "q_norm_weight",
        "k_norm_weight",
        "seq_lens",
        "block_table",
        "slot_mapping",
        "rope_cos",
        "rope_sin",
        "k_cache",
        "v_cache",
        "wo",
        "w_gate",
        "w_up",
        "w_down",
        "post_rms_weight",
        "final_norm_weight",
        "lm_head",
        "logits",
        "embed",
        "token_ids",
        "sampled_ids",
        "next_hidden",
    )
    assert topk_args == ("logits", "sampling_control", "topk_values", "topk_indices")


def test_prepare_weights_rejects_oversized_lm_head_vocab() -> None:
    contract = get_contract("qwen3", "14b")
    model = SimpleNamespace(
        lm_head=torch.zeros((5, 3)),
        embed_tokens=torch.zeros((4, 3)),
        layers=(),
        final_norm_weight=torch.ones(3),
    )

    with pytest.raises(ValueError, match=r"Model vocabulary size 5 exceeds"):
        contract.prepare_weights(model, lambda tensor: tensor, padded_vocab=4)


def test_prepare_weights_rejects_oversized_embedding_vocab() -> None:
    contract = get_contract("qwen3", "14b")
    model = SimpleNamespace(
        lm_head=torch.zeros((4, 3)),
        embed_tokens=torch.zeros((5, 3)),
        layers=(),
        final_norm_weight=torch.ones(3),
    )

    with pytest.raises(ValueError, match=r"Model embedding vocabulary size 5 exceeds"):
        contract.prepare_weights(model, lambda tensor: tensor, padded_vocab=4)


def test_prepare_weights_exports_stacked_decode_weights_once() -> None:
    contract = get_contract("qwen3", "14b")
    layer = SimpleNamespace(
        input_rms_weight=torch.ones(3),
        wq=torch.ones((3, 3)),
        wk=torch.ones((2, 3)),
        wv=torch.ones((2, 3)),
        q_norm_weight=torch.ones(2),
        k_norm_weight=torch.ones(2),
        wo=torch.ones((3, 3)),
        post_rms_weight=torch.ones(3),
        w_gate=torch.ones((4, 3)),
        w_up=torch.ones((4, 3)),
        w_down=torch.ones((3, 4)),
    )
    model = SimpleNamespace(
        lm_head=torch.zeros((4, 3)),
        embed_tokens=torch.zeros((4, 3)),
        layers=(layer,),
        final_norm_weight=torch.ones(3),
    )
    exported = []

    def export(tensor: torch.Tensor) -> torch.Tensor:
        exported.append(tensor)
        return tensor

    prepared = contract.prepare_weights(model, export, padded_vocab=5, release_layers=False)

    assert len(exported) == 3 + len(prepared.decode_weights)
    assert tuple(prepared.decode_weights) == (
        "decode_input_rms_weight",
        "decode_wq",
        "decode_wk",
        "decode_wv",
        "decode_q_norm_weight",
        "decode_k_norm_weight",
        "decode_wo",
        "decode_post_rms_weight",
        "decode_w_gate",
        "decode_w_up",
        "decode_w_down",
    )
    assert prepared.final_norm_weight.shape == (1, 3)
    assert prepared.padded_lm_head_weight.shape == (5, 3)
    assert prepared.padded_embed_weight.shape == (5, 3)
    assert prepared.final_norm_weight.dtype == torch.float32
    assert prepared.padded_lm_head_weight.dtype == torch.bfloat16
    assert prepared.padded_embed_weight.dtype == torch.bfloat16
    assert {name: tuple(tensor.shape) for name, tensor in prepared.decode_weights.items()} == {
        "decode_input_rms_weight": (1, 3),
        "decode_wq": (3, 3),
        "decode_wk": (3, 2),
        "decode_wv": (3, 2),
        "decode_q_norm_weight": (1, 2),
        "decode_k_norm_weight": (1, 2),
        "decode_wo": (3, 3),
        "decode_post_rms_weight": (1, 3),
        "decode_w_gate": (3, 4),
        "decode_w_up": (3, 4),
        "decode_w_down": (4, 3),
    }


def _qwen3_14b_model() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=5120,
            intermediate_size=17408,
            num_hidden_layers=40,
            num_attention_heads=40,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
        ),
        runtime=SimpleNamespace(
            max_batch_size=16,
            max_seq_len=4096,
            page_size=128,
            vocab_pad_multiple=512,
            total_kv_pages=16,
        ),
    )

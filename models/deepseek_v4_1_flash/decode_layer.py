# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 decode Block composition and standalone distributed entry."""

import inspect
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult
from models.deepseek_v4_1_flash.config import AttentionMode
from models.deepseek_v4_1_flash.decode_c1a_full import decode_c1a_full, golden_decode_c1a_full
from models.deepseek_v4_1_flash.decode_c1a_reindex import decode_c1a_reindex, golden_decode_c1a_reindex
from models.deepseek_v4_1_flash.decode_c1a_reuse import decode_c1a_reuse, golden_decode_c1a_reuse
from models.deepseek_v4_1_flash.decode_c2a_full import decode_c2a_full, golden_decode_c2a_full
from models.deepseek_v4_1_flash.decode_c2a_reuse import decode_c2a_reuse, golden_decode_c2a_reuse
from models.deepseek_v4_1_flash.decode_swa import decode_swa, golden_decode_swa, make_norm
from models.deepseek_v4_1_flash.golden import rms_norm
from models.deepseek_v4_1_flash.mhc import (
    golden_mhc_mixes,
    golden_mhc_post,
    golden_mhc_pre,
    mhc_mixes,
    mhc_post,
    mhc_pre,
)
from models.deepseek_v4_1_flash.moe import golden_moe, moe


class DecodeLayerKind(IntEnum):
    """Static decode implementation selected for one backbone layer."""

    SWA = 0
    C2A_FULL = 1
    C2A_REUSE = 2
    C1A_FULL = 3
    C1A_REINDEX = 4
    C1A_REUSE = 5


@dataclass(frozen=True)
class DecodeLayerPlan:
    """Resolved attention implementation and source ownership for one layer."""

    layer_id: int
    kind: DecodeLayerKind
    compression_ratio: int
    kv_source_layer_id: int | None
    index_source_layer_id: int | None
    is_candidate_source: bool


@dataclass(frozen=True)
class DecodeLayerGoldenResult:
    """Block outputs, intermediate boundaries, and attention state updates."""

    output: torch.Tensor
    next_pre_mix: torch.Tensor
    attention_input: torch.Tensor
    attention_output: torch.Tensor
    attention_hidden: torch.Tensor
    ffn_input: torch.Tensor
    ffn_output: torch.Tensor
    attention: AttentionGoldenResult


REPRESENTATIVE_LAYER_IDS = {
    DecodeLayerKind.SWA: 0,
    DecodeLayerKind.C2A_FULL: 2,
    DecodeLayerKind.C2A_REUSE: 3,
    DecodeLayerKind.C1A_FULL: 20,
    DecodeLayerKind.C1A_REINDEX: 24,
    DecodeLayerKind.C1A_REUSE: 21,
}

_ATTENTION_GOLDENS = {
    DecodeLayerKind.SWA: golden_decode_swa,
    DecodeLayerKind.C2A_FULL: golden_decode_c2a_full,
    DecodeLayerKind.C2A_REUSE: golden_decode_c2a_reuse,
    DecodeLayerKind.C1A_FULL: golden_decode_c1a_full,
    DecodeLayerKind.C1A_REINDEX: golden_decode_c1a_reindex,
    DecodeLayerKind.C1A_REUSE: golden_decode_c1a_reuse,
}

_ATTENTION_KERNEL_READY = {
    DecodeLayerKind.SWA: True,
    DecodeLayerKind.C2A_FULL: True,
    DecodeLayerKind.C2A_REUSE: True,
    DecodeLayerKind.C1A_FULL: False,
    DecodeLayerKind.C1A_REINDEX: False,
    DecodeLayerKind.C1A_REUSE: False,
}

_MOE_KERNEL_READY = False

SWA_KIND = int(DecodeLayerKind.SWA)
C2A_FULL_KIND = int(DecodeLayerKind.C2A_FULL)
C2A_REUSE_KIND = int(DecodeLayerKind.C2A_REUSE)
C1A_FULL_KIND = int(DecodeLayerKind.C1A_FULL)
C1A_REINDEX_KIND = int(DecodeLayerKind.C1A_REINDEX)
C1A_REUSE_KIND = int(DecodeLayerKind.C1A_REUSE)

normalize_attention = make_norm(C.D)


def resolve_decode_layer_plan(layer_id: int) -> DecodeLayerPlan:
    """Resolve one layer through the checkpoint-backed model configuration."""
    layer = C.FLASH.layer_config(layer_id)
    if layer.mode == AttentionMode.SWA:
        kind = DecodeLayerKind.SWA
    elif layer.compression_ratio == 2 and layer.mode == AttentionMode.FULL:
        kind = DecodeLayerKind.C2A_FULL
    elif layer.compression_ratio == 2 and layer.mode == AttentionMode.REUSE:
        kind = DecodeLayerKind.C2A_REUSE
    elif layer.compression_ratio == 1 and layer.mode == AttentionMode.FULL:
        kind = DecodeLayerKind.C1A_FULL
    elif layer.compression_ratio == 1 and layer.mode == AttentionMode.REINDEX:
        kind = DecodeLayerKind.C1A_REINDEX
    elif layer.compression_ratio == 1 and layer.mode == AttentionMode.REUSE:
        kind = DecodeLayerKind.C1A_REUSE
    else:
        raise ValueError(
            f"unsupported decode layer {layer_id}: ratio={layer.compression_ratio}, mode={layer.mode.value}"
        )
    return DecodeLayerPlan(
        layer_id=layer_id,
        kind=kind,
        compression_ratio=layer.compression_ratio,
        kv_source_layer_id=layer.kv_source_layer_id,
        index_source_layer_id=layer.index_source_layer_id,
        is_candidate_source=layer.is_candidate_source,
    )


def decode_layer_kernel_skip_reason(layer_id: int) -> str | None:
    """Return the D1/D2 dependency that prevents device compilation."""
    plan = resolve_decode_layer_plan(layer_id)
    missing = []
    if not _ATTENTION_KERNEL_READY[plan.kind]:
        missing.append(f"{plan.kind.name} attention kernel")
        missing.append("C1A cache ABI agreement")
    if not _MOE_KERNEL_READY:
        missing.append("EP8 MoE kernel")
    return None if not missing else "decode_layer requires " + " and ".join(missing)


def decode_layer_attention_inputs(layer_id: int) -> tuple[str, ...]:
    """Return the exact golden inputs consumed by the resolved attention mode."""
    golden_fn = _ATTENTION_GOLDENS[resolve_decode_layer_plan(layer_id).kind]
    return tuple(inspect.signature(golden_fn).parameters)


def _select_inputs(function, values: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    selected = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in overrides:
            selected[name] = overrides[name]
        elif name in values:
            selected[name] = values[name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise KeyError(f"missing {function.__name__} input {name}")
    return selected


def golden_decode_layer(
    layer_id: int,
    x_hc: torch.Tensor,
    incoming_pre_mix: torch.Tensor,
    hc_attn_fn: torch.Tensor,
    hc_attn_scale: torch.Tensor,
    hc_attn_base: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    hc_ffn_fn: torch.Tensor,
    hc_ffn_scale: torch.Tensor,
    hc_ffn_base: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    attention_inputs: Mapping[str, Any],
    moe_inputs: Mapping[str, Any],
    num_tokens: int | None = None,
) -> DecodeLayerGoldenResult:
    """Evaluate the official mHC-attention-mHC-MoE-mHC Block order."""
    plan = resolve_decode_layer_plan(layer_id)
    attn_pre, attn_post, attn_residual = golden_mhc_mixes(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base
    )
    attention_input = golden_mhc_pre(x_hc, incoming_pre_mix)
    normalized_attention = rms_norm(attention_input, attn_norm_weight)
    attention_fn = _ATTENTION_GOLDENS[plan.kind]
    attention_kwargs = _select_inputs(attention_fn, attention_inputs, {"x": normalized_attention})
    attention = attention_fn(**attention_kwargs)
    attention_hidden = golden_mhc_post(
        attention.output, x_hc, attn_post, attn_residual
    )

    next_pre_mix, ffn_post, ffn_residual = golden_mhc_mixes(
        attention_hidden, hc_ffn_fn, hc_ffn_scale, hc_ffn_base
    )
    ffn_input = golden_mhc_pre(attention_hidden, attn_pre)
    moe_overrides = {"x": ffn_input, "norm_weight": ffn_norm_weight}
    if num_tokens is not None:
        moe_overrides["num_tokens"] = num_tokens
    moe_kwargs = _select_inputs(golden_moe, moe_inputs, moe_overrides)
    ffn_output = golden_moe(**moe_kwargs)
    output = golden_mhc_post(ffn_output, attention_hidden, ffn_post, ffn_residual)
    return DecodeLayerGoldenResult(
        output=output,
        next_pre_mix=next_pre_mix,
        attention_input=attention_input,
        attention_output=attention.output,
        attention_hidden=attention_hidden,
        ffn_input=ffn_input,
        ffn_output=ffn_output,
        attention=attention,
    )


@pl.jit
def decode_layer(
    x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
    wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.T_DYN, C.BLOCK_SIZE], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[
        [C.ORI_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0
    ]],
    compressed_cache: pl.InOut[pl.Tensor],
    compressed_cache_scale: pl.InOut[pl.Tensor[
        [C.CMP_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ]],
    request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor],
    index_cache_scale: pl.InOut[pl.Tensor[
        [C.INDEX_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
    ]],
    index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[C.T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor,
    compressor_wgate: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
    compressor_state_rows: pl.Tensor[[C.T_DYN], pl.INT64],
    compressor_state: pl.InOut[pl.Tensor[[C.MAX_BATCH_PER_DP, C.STATE_HEADS, C.HEAD_DIM], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
    index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
    candidate_mask: pl.InOut[pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.BOOL]],
    hc_ffn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[C.D], pl.BF16],
    gate_weight: pl.Tensor[[C.N_EXPERTS, C.D], pl.FP32],
    correction_bias: pl.Tensor[[C.N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
    routed_w1_scale: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0],
    routed_w2: pl.Tensor[[C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER], pl.FP4],
    routed_w2_scale: pl.Tensor[[C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER // C.MX_GROUP], pl.FP8E8M0],
    routed_w3: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
    routed_w3_scale: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0],
    shared_w1: pl.Tensor[[C.D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[C.MOE_INTER, C.D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.MOE_INTER // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[C.D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    token_owners: pl.Tensor[[C.T_DYN], pl.INT32],
    attention_output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    attention_output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D], pl.FP8E4M3FN],
    recv_scale: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D // C.MX_GROUP], pl.UINT8],
    recv_weights: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[C.EP_SIZE, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[C.EP_SIZE, 1], pl.INT32],
    routed_output: pld.DistributedTensor[[C.T_DYN * C.TOPK, C.D], pl.FP32],
    combine_arrived: pld.DistributedTensor[[C.EP_SIZE, 1], pl.INT32],
    x_next: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32]],
    layer_kind: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    ep_rank: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    # Leaf kernels own cache storage types and compressor weight dtype.
    x_hc.bind_dynamic(0, C.T_DYN)
    incoming_pre_mix.bind_dynamic(0, C.T_DYN)
    x_next.bind_dynamic(0, C.T_DYN)
    next_pre_mix.bind_dynamic(0, C.T_DYN)

    tokens = pl.tensor.dim(x_hc, 0)
    attn_pre = pl.create_tensor([tokens, C.HC_MULT], dtype=pl.FP32)
    attn_post = pl.create_tensor([tokens, C.HC_MULT], dtype=pl.FP32)
    attn_residual = pl.create_tensor([tokens, C.HC_MULT, C.HC_MULT], dtype=pl.FP32)
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_pre, attn_post, attn_residual)
    attention_input = pl.create_tensor([tokens, C.D], dtype=pl.BF16)
    mhc_pre(x_hc, incoming_pre_mix, attention_input)
    normalized_attention = pl.create_tensor([tokens, C.D], dtype=pl.BF16)
    normalize_attention(attention_input, attn_norm_weight, normalized_attention, num_tokens)
    attention_output = pl.create_tensor([tokens, C.D], dtype=pl.BF16)

    if layer_kind == SWA_KIND:
        decode_swa(
            normalized_attention,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            attention_output_window, attention_output_arrived, attention_output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif layer_kind == C2A_FULL_KIND:
        decode_c2a_full(
            normalized_attention,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
            index_cache, index_cache_scale, index_block_table, position_ids,
            compressed_rope_cos, compressed_rope_sin,
            compressor_wkv, compressor_wgate, compressor_state_rows, compressor_state,
            compressor_norm_weight, compressed_slots,
            index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
            topk_indices, attention_output_window, attention_output_arrived, attention_output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif layer_kind == C2A_REUSE_KIND:
        decode_c2a_reuse(
            normalized_attention,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            compressed_cache, compressed_cache_scale, topk_indices,
            attention_output_window, attention_output_arrived, attention_output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif layer_kind == C1A_FULL_KIND:
        decode_c1a_full(
            normalized_attention,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
            index_cache, index_cache_scale, index_block_table,
            compressed_rope_cos, compressed_rope_sin,
            compressor_wkv, compressor_norm_weight, compressed_slots,
            index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
            topk_indices, candidate_mask,
            attention_output_window, attention_output_arrived, attention_output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif layer_kind == C1A_REINDEX_KIND:
        decode_c1a_reindex(
            normalized_attention,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            compressed_cache, compressed_cache_scale, request_ids, compressed_lens,
            index_cache, index_cache_scale, index_block_table, candidate_mask,
            index_wq_b, index_wq_b_scale, index_weights_proj, topk_indices,
            attention_output_window, attention_output_arrived, attention_output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    elif layer_kind == C1A_REUSE_KIND:
        decode_c1a_reuse(
            normalized_attention,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
            attn_sink, wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            compressed_cache, compressed_cache_scale, topk_indices,
            attention_output_window, attention_output_arrived, attention_output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )

    attention_hidden = pl.create_tensor([tokens, C.HC_MULT, C.D], dtype=pl.FP32)
    mhc_post(attention_output, x_hc, attn_post, attn_residual, attention_hidden)
    ffn_post = pl.create_tensor([tokens, C.HC_MULT], dtype=pl.FP32)
    ffn_residual = pl.create_tensor([tokens, C.HC_MULT, C.HC_MULT], dtype=pl.FP32)
    mhc_mixes(
        attention_hidden, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        next_pre_mix, ffn_post, ffn_residual,
    )
    ffn_input = pl.create_tensor([tokens, C.D], dtype=pl.BF16)
    mhc_pre(attention_hidden, attn_pre, ffn_input)
    ffn_output = pl.create_tensor([tokens, C.D], dtype=pl.BF16)
    moe(
        ffn_input,
        ffn_norm_weight, gate_weight, correction_bias,
        routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
        shared_w1, shared_w1_scale, shared_w2, shared_w2_scale, shared_w3, shared_w3_scale,
        token_owners,
        recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
        arrived, data_arrived, routed_output, combine_arrived,
        ffn_output,
        num_tokens, ep_rank, group_base, tp_rank, moe_epoch,
    )
    mhc_post(ffn_output, attention_hidden, ffn_post, ffn_residual, x_next)
    return x_next, next_pre_mix


@pl.jit.host
def l3_decode_layer(
    x_hc: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.HC_MULT, C.D], pl.FP32],
    incoming_pre_mix: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[C.EP_SIZE, C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[C.EP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[C.EP_SIZE, C.MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[C.EP_SIZE, C.D], pl.BF16],
    wq_a: pl.Tensor[[C.EP_SIZE, C.D, C.Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[C.EP_SIZE, C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[C.EP_SIZE, C.Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[C.EP_SIZE, C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[C.EP_SIZE, C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[C.EP_SIZE, C.D, C.HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[C.EP_SIZE, C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[C.EP_SIZE, C.HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[C.EP_SIZE, C.LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[C.EP_SIZE, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[C.EP_SIZE, C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[C.EP_SIZE, C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT64],
    window_indices: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.BLOCK_SIZE], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[C.EP_SIZE, C.ORI_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[
        [C.EP_SIZE, C.ORI_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0
    ]],
    compressed_cache: pl.InOut[pl.Tensor],
    compressed_cache_scale: pl.InOut[pl.Tensor[
        [C.EP_SIZE, C.CMP_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
    ]],
    request_ids: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor],
    index_cache_scale: pl.InOut[pl.Tensor[
        [C.EP_SIZE, C.INDEX_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0
    ]],
    index_block_table: pl.Tensor[[C.EP_SIZE, C.B_DYN, C.TABLE_DYN], pl.INT32],
    position_ids: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor,
    compressor_wgate: pl.Tensor[[C.EP_SIZE, C.D, C.HEAD_DIM], pl.FP32],
    compressor_state_rows: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT64],
    compressor_state: pl.InOut[pl.Tensor[[C.EP_SIZE, C.MAX_BATCH_PER_DP, C.STATE_HEADS, C.HEAD_DIM], pl.FP32]],
    compressor_norm_weight: pl.Tensor[[C.EP_SIZE, C.HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT64],
    index_wk: pl.Tensor[[C.EP_SIZE, C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[C.EP_SIZE, C.INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[C.EP_SIZE, C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[C.EP_SIZE, C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[C.EP_SIZE, C.D, C.INDEX_H], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[C.EP_SIZE, C.T_DYN, C.INDEX_TOPK], pl.INT32]],
    candidate_mask: pl.InOut[pl.Tensor[[C.EP_SIZE, C.T_DYN, C.CMP_POSITIONS_DYN], pl.BOOL]],
    hc_ffn_fn: pl.Tensor[[C.EP_SIZE, C.MIX_HC, C.HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[C.EP_SIZE, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[C.EP_SIZE, C.MIX_HC], pl.FP32],
    ffn_norm_weight: pl.Tensor[[C.EP_SIZE, C.D], pl.BF16],
    gate_weight: pl.Tensor[[C.EP_SIZE, C.N_EXPERTS, C.D], pl.FP32],
    correction_bias: pl.Tensor[[C.EP_SIZE, C.N_EXPERTS], pl.FP32],
    routed_w1: pl.Tensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
    routed_w1_scale: pl.Tensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0],
    routed_w2: pl.Tensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER], pl.FP4],
    routed_w2_scale: pl.Tensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER // C.MX_GROUP], pl.FP8E8M0],
    routed_w3: pl.Tensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
    routed_w3_scale: pl.Tensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0],
    shared_w1: pl.Tensor[[C.EP_SIZE, C.D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[C.EP_SIZE, C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0],
    shared_w2: pl.Tensor[[C.EP_SIZE, C.MOE_INTER, C.D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[C.EP_SIZE, C.MOE_INTER // C.MX_GROUP, C.D], pl.FP8E8M0],
    shared_w3: pl.Tensor[[C.EP_SIZE, C.D, C.MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[C.EP_SIZE, C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0],
    token_owners: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT32],
    x_next: pl.Out[pl.Tensor[[C.EP_SIZE, C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
    next_pre_mix: pl.Out[pl.Tensor[[C.EP_SIZE, C.T_DYN, C.HC_MULT], pl.FP32]],
    layer_kind: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    attention_output_buf = pld.alloc_window_buffer([C.DECODE_MAX_TOKENS, C.D], dtype=pl.FP32)
    attention_arrived_buf = pld.alloc_window_buffer([C.TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([C.EP_SIZE, C.N_LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D], dtype=pl.FP8E4M3FN)
    recv_scale_buf = pld.alloc_window_buffer([C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D // C.MX_GROUP], dtype=pl.UINT8)
    recv_weights_buf = pld.alloc_window_buffer([C.N_LOCAL_EXPERTS * C.RECV_MAX, C.AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buf = pld.alloc_window_buffer([C.N_LOCAL_EXPERTS * C.RECV_MAX, C.ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([C.EP_SIZE, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([C.EP_SIZE, 1], dtype=pl.INT32)
    routed_output_buf = pld.alloc_window_buffer([C.DECODE_MAX_TOKENS * C.TOPK, C.D], dtype=pl.FP32)
    combine_arrived_buf = pld.alloc_window_buffer([C.EP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        attention_output_window = pld.window(attention_output_buf, [C.DECODE_MAX_TOKENS, C.D], dtype=pl.FP32)
        attention_output_arrived = pld.window(attention_arrived_buf, [C.TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [C.EP_SIZE, C.N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D], dtype=pl.FP8E4M3FN)
        recv_scale = pld.window(recv_scale_buf, [C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D // C.MX_GROUP], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buf, [C.N_LOCAL_EXPERTS * C.RECV_MAX, C.AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buf, [C.N_LOCAL_EXPERTS * C.RECV_MAX, C.ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [C.EP_SIZE, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [C.EP_SIZE, 1], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buf, [C.DECODE_MAX_TOKENS * C.TOPK, C.D], dtype=pl.FP32)
        combine_arrived = pld.window(combine_arrived_buf, [C.EP_SIZE, 1], dtype=pl.INT32)
        group_base = (rank // C.TP_SIZE) * C.TP_SIZE
        tp_rank = rank % C.TP_SIZE
        decode_layer(
            x_hc[rank], incoming_pre_mix[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank], attn_norm_weight[rank],
            wq_a[rank], wq_a_scale[rank], q_norm_weight[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], wkv_scale[rank], kv_norm_weight[rank], attn_sink[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank], rope_cos[rank], rope_sin[rank],
            window_slots[rank], window_indices[rank], window_cache[rank], window_cache_scale[rank],
            compressed_cache[rank], compressed_cache_scale[rank], request_ids[rank], compressed_lens[rank],
            index_cache[rank], index_cache_scale[rank], index_block_table[rank], position_ids[rank],
            compressed_rope_cos[rank], compressed_rope_sin[rank],
            compressor_wkv[rank], compressor_wgate[rank], compressor_state_rows[rank], compressor_state[rank],
            compressor_norm_weight[rank], compressed_slots[rank],
            index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale[rank],
            index_weights_proj[rank], topk_indices[rank], candidate_mask[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank], ffn_norm_weight[rank],
            gate_weight[rank], correction_bias[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w2[rank], routed_w2_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w2[rank], shared_w2_scale[rank],
            shared_w3[rank], shared_w3_scale[rank], token_owners[rank],
            attention_output_window, attention_output_arrived,
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes,
            arrived, data_arrived, routed_output, combine_arrived,
            x_next[rank], next_pre_mix[rank],
            layer_kind, num_tokens, rank, group_base, tp_rank, attention_epoch, moe_epoch,
            device=rank,
        )


__all__ = [
    "DecodeLayerGoldenResult",
    "DecodeLayerKind",
    "DecodeLayerPlan",
    "REPRESENTATIVE_LAYER_IDS",
    "decode_layer",
    "decode_layer_attention_inputs",
    "decode_layer_kernel_skip_reason",
    "golden_decode_layer",
    "l3_decode_layer",
    "resolve_decode_layer_plan",
]


if __name__ == "__main__":
    from models.deepseek_v4_1_flash._golden_smoke import run_decode_layer_goldens

    run_decode_layer_goldens(golden_decode_layer, REPRESENTATIVE_LAYER_IDS.values())

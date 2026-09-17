# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Static decode Attention half-layer orchestration and hardware validation."""

# ci: no-sim
# ci: a5

import argparse
import sys
from dataclasses import replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import pypto.language.distributed as pld
import torch
from pypto.ir import DistributedConfig

from golden import TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash import decode_c2a_full as full
from models.deepseek_v4_1_flash import decode_c2a_reuse as reuse
from models.deepseek_v4_1_flash import decode_swa as swa
from models.deepseek_v4_1_flash.decode_c2a_full import decode_c2a_full
from models.deepseek_v4_1_flash.decode_c2a_reuse import decode_c2a_reuse
from models.deepseek_v4_1_flash.decode_swa import decode_swa
from models.deepseek_v4_1_flash.decode_layer import (
    DecodeLayerKind,
    normalize_attention,
    resolve_decode_layer_plan,
)
from models.deepseek_v4_1_flash.config import D, DECODE_MAX_TOKENS, HC_MULT, TP_SIZE
from models.deepseek_v4_1_flash.golden import rms_norm
from models.deepseek_v4_1_flash.mhc import (
    golden_mhc_mixes,
    golden_mhc_post,
    golden_mhc_pre,
    mhc_mixes,
    mhc_post,
    mhc_pre,
)


def attention_half_skip_reason(layer_id):
    """Check only half-layer dependencies; MoE does not gate this entry."""
    kind = resolve_decode_layer_plan(layer_id).kind
    if kind not in (DecodeLayerKind.SWA, DecodeLayerKind.C2A_FULL, DecodeLayerKind.C2A_REUSE):
        return f"{kind.name} attention kernel and C1A cache ABI agreement are pending"
    return None


# The union adapters keep selection outside the JIT dependency graph.
@pl.jit.inline(auto_scope=False)
def _swa(
    x: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor,
    attn_sink: pl.Tensor,
    wo_a: pl.Tensor,
    wo_b: pl.Tensor,
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_slots: pl.Tensor,
    window_indices: pl.Tensor,
    window_cache: pl.Tensor,
    window_cache_scale: pl.Tensor,
    compressed_cache: pl.Tensor,
    compressed_cache_scale: pl.Tensor,
    request_ids: pl.Tensor,
    compressed_lens: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    index_block_table: pl.Tensor,
    position_ids: pl.Tensor,
    compressed_rope_cos: pl.Tensor,
    compressed_rope_sin: pl.Tensor,
    compressor_wkv: pl.Tensor,
    compressor_wgate: pl.Tensor,
    compressor_state_rows: pl.Tensor,
    compressor_state: pl.Tensor,
    compressor_norm_weight: pl.Tensor,
    compressed_slots: pl.Tensor,
    index_wk: pl.Tensor,
    index_norm_weight: pl.Tensor,
    index_wq_b: pl.Tensor,
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor,
    topk_indices: pl.Tensor,
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    decode_swa(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        output_window,
        output_arrived,
        output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    return output


@pl.jit.inline(auto_scope=False)
def _c2a_full(
    x: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor,
    attn_sink: pl.Tensor,
    wo_a: pl.Tensor,
    wo_b: pl.Tensor,
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_slots: pl.Tensor,
    window_indices: pl.Tensor,
    window_cache: pl.Tensor,
    window_cache_scale: pl.Tensor,
    compressed_cache: pl.Tensor,
    compressed_cache_scale: pl.Tensor,
    request_ids: pl.Tensor,
    compressed_lens: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    index_block_table: pl.Tensor,
    position_ids: pl.Tensor,
    compressed_rope_cos: pl.Tensor,
    compressed_rope_sin: pl.Tensor,
    compressor_wkv: pl.Tensor,
    compressor_wgate: pl.Tensor,
    compressor_state_rows: pl.Tensor,
    compressor_state: pl.Tensor,
    compressor_norm_weight: pl.Tensor,
    compressed_slots: pl.Tensor,
    index_wk: pl.Tensor,
    index_norm_weight: pl.Tensor,
    index_wq_b: pl.Tensor,
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor,
    topk_indices: pl.Tensor,
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    decode_c2a_full(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        compressed_cache,
        compressed_cache_scale,
        request_ids,
        compressed_lens,
        index_cache,
        index_cache_scale,
        index_block_table,
        position_ids,
        compressed_rope_cos,
        compressed_rope_sin,
        compressor_wkv,
        compressor_wgate,
        compressor_state_rows,
        compressor_state,
        compressor_norm_weight,
        compressed_slots,
        index_wk,
        index_norm_weight,
        index_wq_b,
        index_wq_b_scale,
        index_weights_proj,
        topk_indices,
        output_window,
        output_arrived,
        output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    return output


@pl.jit.inline(auto_scope=False)
def _c2a_reuse(
    x: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor,
    attn_sink: pl.Tensor,
    wo_a: pl.Tensor,
    wo_b: pl.Tensor,
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_slots: pl.Tensor,
    window_indices: pl.Tensor,
    window_cache: pl.Tensor,
    window_cache_scale: pl.Tensor,
    compressed_cache: pl.Tensor,
    compressed_cache_scale: pl.Tensor,
    request_ids: pl.Tensor,
    compressed_lens: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    index_block_table: pl.Tensor,
    position_ids: pl.Tensor,
    compressed_rope_cos: pl.Tensor,
    compressed_rope_sin: pl.Tensor,
    compressor_wkv: pl.Tensor,
    compressor_wgate: pl.Tensor,
    compressor_state_rows: pl.Tensor,
    compressor_state: pl.Tensor,
    compressor_norm_weight: pl.Tensor,
    compressed_slots: pl.Tensor,
    index_wk: pl.Tensor,
    index_norm_weight: pl.Tensor,
    index_wq_b: pl.Tensor,
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor,
    topk_indices: pl.Tensor,
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    decode_c2a_reuse(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    return output


def make_attention_program(layer_id, world_size, epochs, specs):
    """Build one mode-specific half-layer with persistent communication epochs."""
    reason = attention_half_skip_reason(layer_id)
    if reason:
        raise NotImplementedError(reason)
    kind = resolve_decode_layer_plan(layer_id).kind
    attention = {
        DecodeLayerKind.SWA: _swa,
        DecodeLayerKind.C2A_FULL: _c2a_full,
        DecodeLayerKind.C2A_REUSE: _c2a_reuse,
    }[kind]
    dtypes = {
        torch.bfloat16: pl.BF16,
        torch.float32: pl.FP32,
        torch.float8_e4m3fn: pl.FP8E4M3FN,
        torch.float8_e8m0fnu: pl.FP8E8M0,
        torch.int32: pl.INT32,
        torch.int64: pl.INT64,
        torch.uint8: pl.UINT8,
    }
    shapes = [spec.shape for spec in specs if isinstance(spec, TensorSpec)]
    tensor_dtypes = [dtypes[spec.dtype] for spec in specs if isinstance(spec, TensorSpec)]

    @pl.jit
    def attention_rank(
        x_hc: pl.Tensor,
        incoming_pre_mix: pl.Tensor,
        hc_attn_fn: pl.Tensor,
        hc_attn_scale: pl.Tensor,
        hc_attn_base: pl.Tensor,
        attn_norm_weight: pl.Tensor,
        wq_a: pl.Tensor,
        wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor,
        wq_b: pl.Tensor,
        wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor,
        wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor,
        attn_sink: pl.Tensor,
        wo_a: pl.Tensor,
        wo_b: pl.Tensor,
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor,
        rope_sin: pl.Tensor,
        window_slots: pl.Tensor,
        window_indices: pl.Tensor,
        window_cache: pl.InOut[pl.Tensor],
        window_cache_scale: pl.InOut[pl.Tensor],
        compressed_cache: pl.InOut[pl.Tensor],
        compressed_cache_scale: pl.InOut[pl.Tensor],
        request_ids: pl.Tensor,
        compressed_lens: pl.Tensor,
        index_cache: pl.InOut[pl.Tensor],
        index_cache_scale: pl.InOut[pl.Tensor],
        index_block_table: pl.Tensor,
        position_ids: pl.Tensor,
        compressed_rope_cos: pl.Tensor,
        compressed_rope_sin: pl.Tensor,
        compressor_wkv: pl.Tensor,
        compressor_wgate: pl.Tensor,
        compressor_state_rows: pl.Tensor,
        compressor_state: pl.InOut[pl.Tensor],
        compressor_norm_weight: pl.Tensor,
        compressed_slots: pl.Tensor,
        index_wk: pl.Tensor,
        index_norm_weight: pl.Tensor,
        index_wq_b: pl.Tensor,
        index_wq_b_scale: pl.Tensor[
            [C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ],
        index_weights_proj: pl.Tensor,
        topk_indices: pl.InOut[pl.Tensor],
        attention_input: pl.Out[pl.Tensor],
        normalized_attention: pl.InOut[pl.Tensor],
        attention_output: pl.InOut[pl.Tensor],
        attention_hidden: pl.Out[pl.Tensor],
        attention_pre_mix: pl.Out[pl.Tensor],
        output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
        output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x_hc, 0)
        post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
        residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
        mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attention_pre_mix, post_mix, residual_mix)
        mhc_pre(x_hc, incoming_pre_mix, attention_input)
        normalize_attention(attention_input, attn_norm_weight, normalized_attention, num_tokens)
        for step in pl.range(epochs):
            attention(
                normalized_attention,
                wq_a,
                wq_a_scale,
                q_norm_weight,
                wq_b,
                wq_b_scale,
                wkv,
                wkv_scale,
                kv_norm_weight,
                attn_sink,
                wo_a,
                wo_b,
                wo_b_scale,
                rope_cos,
                rope_sin,
                window_slots,
                window_indices,
                window_cache,
                window_cache_scale,
                compressed_cache,
                compressed_cache_scale,
                request_ids,
                compressed_lens,
                index_cache,
                index_cache_scale,
                index_block_table,
                position_ids,
                compressed_rope_cos,
                compressed_rope_sin,
                compressor_wkv,
                compressor_wgate,
                compressor_state_rows,
                compressor_state,
                compressor_norm_weight,
                compressed_slots,
                index_wk,
                index_norm_weight,
                index_wq_b,
                index_wq_b_scale,
                index_weights_proj,
                topk_indices,
                output_window,
                output_arrived,
                attention_output,
                rank // TP_SIZE * TP_SIZE,
                rank % TP_SIZE,
                num_tokens,
                attention_epoch + step,
            )
        mhc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden)
        return attention_hidden, attention_pre_mix

    @pl.jit.host
    def attention_group(
        x_hc: pl.Tensor[shapes[0], tensor_dtypes[0]],
        incoming_pre_mix: pl.Tensor[shapes[1], tensor_dtypes[1]],
        hc_attn_fn: pl.Tensor[shapes[2], tensor_dtypes[2]],
        hc_attn_scale: pl.Tensor[shapes[3], tensor_dtypes[3]],
        hc_attn_base: pl.Tensor[shapes[4], tensor_dtypes[4]],
        attn_norm_weight: pl.Tensor[shapes[5], tensor_dtypes[5]],
        wq_a: pl.Tensor[shapes[6], tensor_dtypes[6]],
        wq_a_scale: pl.Tensor[shapes[7], tensor_dtypes[7]],
        q_norm_weight: pl.Tensor[shapes[8], tensor_dtypes[8]],
        wq_b: pl.Tensor[shapes[9], tensor_dtypes[9]],
        wq_b_scale: pl.Tensor[shapes[10], tensor_dtypes[10]],
        wkv: pl.Tensor[shapes[11], tensor_dtypes[11]],
        wkv_scale: pl.Tensor[shapes[12], tensor_dtypes[12]],
        kv_norm_weight: pl.Tensor[shapes[13], tensor_dtypes[13]],
        attn_sink: pl.Tensor[shapes[14], tensor_dtypes[14]],
        wo_a: pl.Tensor[shapes[15], tensor_dtypes[15]],
        wo_b: pl.Tensor[shapes[16], tensor_dtypes[16]],
        wo_b_scale: pl.Tensor[shapes[17], tensor_dtypes[17]],
        rope_cos: pl.Tensor[shapes[18], tensor_dtypes[18]],
        rope_sin: pl.Tensor[shapes[19], tensor_dtypes[19]],
        window_slots: pl.Tensor[shapes[20], tensor_dtypes[20]],
        window_indices: pl.Tensor[shapes[21], tensor_dtypes[21]],
        window_cache: pl.InOut[pl.Tensor[shapes[22], tensor_dtypes[22]]],
        window_cache_scale: pl.InOut[
            pl.Tensor[shapes[23], tensor_dtypes[23]]
        ],
        compressed_cache: pl.InOut[pl.Tensor[shapes[24], tensor_dtypes[24]]],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[shapes[25], tensor_dtypes[25]]
        ],
        request_ids: pl.Tensor[shapes[26], tensor_dtypes[26]],
        compressed_lens: pl.Tensor[shapes[27], tensor_dtypes[27]],
        index_cache: pl.InOut[pl.Tensor[shapes[28], tensor_dtypes[28]]],
        index_cache_scale: pl.InOut[
            pl.Tensor[shapes[29], tensor_dtypes[29]]
        ],
        index_block_table: pl.Tensor[shapes[30], tensor_dtypes[30]],
        position_ids: pl.Tensor[shapes[31], tensor_dtypes[31]],
        compressed_rope_cos: pl.Tensor[shapes[32], tensor_dtypes[32]],
        compressed_rope_sin: pl.Tensor[shapes[33], tensor_dtypes[33]],
        compressor_wkv: pl.Tensor[shapes[34], tensor_dtypes[34]],
        compressor_wgate: pl.Tensor[shapes[35], tensor_dtypes[35]],
        compressor_state_rows: pl.Tensor[
            shapes[36], tensor_dtypes[36]
        ],
        compressor_state: pl.InOut[pl.Tensor[shapes[37], tensor_dtypes[37]]],
        compressor_norm_weight: pl.Tensor[
            shapes[38], tensor_dtypes[38]
        ],
        compressed_slots: pl.Tensor[shapes[39], tensor_dtypes[39]],
        index_wk: pl.Tensor[shapes[40], tensor_dtypes[40]],
        index_norm_weight: pl.Tensor[shapes[41], tensor_dtypes[41]],
        index_wq_b: pl.Tensor[shapes[42], tensor_dtypes[42]],
        index_wq_b_scale: pl.Tensor[shapes[43], tensor_dtypes[43]],
        index_weights_proj: pl.Tensor[shapes[44], tensor_dtypes[44]],
        topk_indices: pl.InOut[pl.Tensor[shapes[45], tensor_dtypes[45]]],
        attention_input: pl.Out[pl.Tensor[shapes[46], tensor_dtypes[46]]],
        normalized_attention: pl.InOut[
            pl.Tensor[shapes[47], tensor_dtypes[47]]
        ],
        attention_output: pl.InOut[pl.Tensor[shapes[48], tensor_dtypes[48]]],
        attention_hidden: pl.Out[pl.Tensor[shapes[49], tensor_dtypes[49]]],
        attention_pre_mix: pl.Out[pl.Tensor[shapes[50], tensor_dtypes[50]]],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        data_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(world_size):
            data = pld.window(data_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
            arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            attention_rank(
                x_hc[rank],
                incoming_pre_mix[rank],
                hc_attn_fn[rank],
                hc_attn_scale[rank],
                hc_attn_base[rank],
                attn_norm_weight[rank],
                wq_a[rank],
                wq_a_scale[rank],
                q_norm_weight[rank],
                wq_b[rank],
                wq_b_scale[rank],
                wkv[rank],
                wkv_scale[rank],
                kv_norm_weight[rank],
                attn_sink[rank],
                wo_a[rank],
                wo_b[rank],
                wo_b_scale[rank],
                rope_cos[rank],
                rope_sin[rank],
                window_slots[rank],
                window_indices[rank],
                window_cache[rank],
                window_cache_scale[rank],
                compressed_cache[rank],
                compressed_cache_scale[rank],
                request_ids[rank],
                compressed_lens[rank],
                index_cache[rank],
                index_cache_scale[rank],
                index_block_table[rank],
                position_ids[rank],
                compressed_rope_cos[rank],
                compressed_rope_sin[rank],
                compressor_wkv[rank],
                compressor_wgate[rank],
                compressor_state_rows[rank],
                compressor_state[rank],
                compressor_norm_weight[rank],
                compressed_slots[rank],
                index_wk[rank],
                index_norm_weight[rank],
                index_wq_b[rank],
                index_wq_b_scale[rank],
                index_weights_proj[rank],
                topk_indices[rank],
                attention_input[rank],
                normalized_attention[rank],
                attention_output[rank],
                attention_hidden[rank],
                attention_pre_mix[rank],
                data,
                arrived,
                rank,
                num_tokens,
                attention_epoch,
                device=rank,
            )

    return attention_group


UNION_NAMES = (
    "wq_a",
    "wq_a_scale",
    "q_norm_weight",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "wkv_scale",
    "kv_norm_weight",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "rope_cos",
    "rope_sin",
    "window_slots",
    "window_indices",
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "request_ids",
    "compressed_lens",
    "index_cache",
    "index_cache_scale",
    "index_block_table",
    "position_ids",
    "compressed_rope_cos",
    "compressed_rope_sin",
    "compressor_wkv",
    "compressor_wgate",
    "compressor_state_rows",
    "compressor_state",
    "compressor_norm_weight",
    "compressed_slots",
    "index_wk",
    "index_norm_weight",
    "index_wq_b",
    "index_wq_b_scale",
    "index_weights_proj",
    "topk_indices",
)


def build_specs(args, kind, initial_cache):
    """Reuse leaf fixtures and add replicated mHC inputs and visible boundaries."""
    if kind == DecodeLayerKind.SWA:
        leaf_specs = swa.build_specs(args, "decode")
    elif kind == DecodeLayerKind.C2A_FULL:
        leaf_specs = full.build_specs(args, "decode")
    else:
        leaf_specs = reuse.build_specs(args, "decode", initial_cache)
        leaf_specs = [
            replace(spec, name="topk_indices") if spec.name == "compressed_indices" else spec
            for spec in leaf_specs
        ]
    specs = [spec for spec in leaf_specs if spec.name not in ("x", "output")]
    present = {spec.name for spec in specs}
    for name in UNION_NAMES:
        if name not in present:
            if name == "index_wq_b_scale":
                shape = [args.tp, C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM]
                specs.append(TensorSpec(name, shape, torch.float8_e8m0fnu, resident="stacked"))
            else:
                specs.append(TensorSpec(name, [args.tp, 1], torch.float32, resident="stacked"))
    generator = torch.Generator().manual_seed(args.seed + 1000)
    shapes = {
        "x_hc": [args.tokens, C.HC_MULT, C.D],
        "incoming_pre_mix": [args.tokens, C.HC_MULT],
        "hc_attn_fn": [C.MIX_HC, C.HC_DIM],
        "hc_attn_scale": [3],
        "hc_attn_base": [C.MIX_HC],
        "attn_norm_weight": [C.D],
    }
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator)
        if name == "hc_attn_fn":
            value /= C.HC_DIM**0.5
        elif name == "attn_norm_weight":
            value = torch.ones(shape, dtype=torch.bfloat16)
        elif name == "incoming_pre_mix":
            value = torch.softmax(value, dim=-1)
        specs.append(
            TensorSpec(
                name,
                [args.tp, *shape],
                value.dtype,
                init_value=value.unsqueeze(0).repeat(args.tp, *([1] * len(shape))),
                resident="stacked",
            )
        )
    for name, shape, dtype in (
        ("attention_input", [args.tokens, C.D], torch.bfloat16),
        ("normalized_attention", [args.tokens, C.D], torch.bfloat16),
        ("attention_output", [args.tokens, C.D], torch.bfloat16),
        ("attention_hidden", [args.tokens, C.HC_MULT, C.D], torch.float32),
        ("attention_pre_mix", [args.tokens, C.HC_MULT], torch.float32),
    ):
        sentinel = 13.0 if name in ("normalized_attention", "attention_output") else 0.0
        specs.append(TensorSpec(name, [args.tp, *shape], dtype, init_value=sentinel, resident="stacked"))
    by_name = {spec.name: spec for spec in specs}
    order = (
        "x_hc",
        "incoming_pre_mix",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "attn_norm_weight",
        *UNION_NAMES,
        "attention_input",
        "normalized_attention",
        "attention_output",
        "attention_hidden",
        "attention_pre_mix",
        "num_tokens",
        "attention_epoch",
    )
    return [by_name[name] for name in order]


def make_golden(kind, epochs):
    """Check mHC boundaries around the leaf file's TP-reduced reference."""

    def golden_half(tensors):
        world = tensors["x_hc"].shape[0]
        post = []
        residual = []
        active = int(tensors["num_tokens"])
        for rank in range(world):
            pre, post_mix, residual_mix = golden_mhc_mixes(
                tensors["x_hc"][rank],
                tensors["hc_attn_fn"][rank],
                tensors["hc_attn_scale"][rank],
                tensors["hc_attn_base"][rank],
            )
            tensors["attention_pre_mix"][rank].copy_(pre)
            collapsed = golden_mhc_pre(tensors["x_hc"][rank], tensors["incoming_pre_mix"][rank])
            tensors["attention_input"][rank].copy_(collapsed)
            tensors["normalized_attention"][rank, :active].copy_(
                rms_norm(collapsed[:active], tensors["attn_norm_weight"][rank])
            )
            post.append(post_mix)
            residual.append(residual_mix)
        leaf = dict(tensors, x=tensors["normalized_attention"], output=tensors["attention_output"])
        if kind == DecodeLayerKind.SWA:
            for _ in range(epochs):
                swa.golden_swa(leaf)
        elif kind == DecodeLayerKind.C2A_FULL:
            full.make_golden(epochs)(leaf)
        else:
            leaf["compressed_indices"] = tensors["topk_indices"]
            for _ in range(epochs):
                reuse.golden_c2a_reuse(leaf)
        for rank in range(world):
            tensors["attention_hidden"][rank].copy_(
                golden_mhc_post(
                    tensors["attention_output"][rank], tensors["x_hc"][rank], post[rank], residual[rank]
                )
            )

    return golden_half


def compare_unchanged(name):
    """Compare non-owner storage byte-for-byte, including CPU FP8 scales."""

    def compare(actual, expected, **kwargs):
        return torch.equal(
            actual.view(torch.uint8), expected.view(torch.uint8)
        ), f"{name}: non-owner state must stay exact"

    return compare


def compare_normalized(actual, expected, *, inputs, **kwargs):
    passed, detail = ratio_allclose(atol=1e-4, rtol=1 / 128)(actual, expected, inputs=inputs, **kwargs)
    active = int(inputs["num_tokens"])
    return passed and torch.equal(actual[:, active:], expected[:, active:]), detail


def main():
    parser = argparse.ArgumentParser(description="DeepSeek V4.1 decode Attention half-layer")
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--tp", type=int, choices=(1, 4), default=4)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--active-tokens", type=int)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=("mixed", "shuffle", "prefix"), default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.tokens <= C.DECODE_MAX_TOKENS or not 1 <= args.epochs <= 1000:
        parser.error("tokens or epochs out of range")
    args.active_tokens = args.tokens if args.active_tokens is None else args.active_tokens
    if not 1 <= args.active_tokens <= args.tokens or not 1 <= args.requests <= args.active_tokens:
        parser.error("require 1 <= requests <= active tokens <= tokens")
    args.dp, args.bench = 1, False
    if args.tp != C.TP_SIZE:
        parser.error("--tp must match the import-time tensor parallel configuration")
    devices = (
        list(range(args.tp))
        if args.device is None or args.compile_only
        else [int(device) for device in args.device.split(",")]
    )
    if len(devices) != args.tp or len(set(devices)) != args.tp or min(devices) < 0:
        parser.error("--device must name exactly TP distinct nonnegative device IDs")
    kind = resolve_decode_layer_plan(args.layer_id).kind
    reason = attention_half_skip_reason(args.layer_id)
    if reason:
        parser.error(reason)
    if args.active_tokens != args.tokens and kind != DecodeLayerKind.C2A_REUSE:
        parser.error("inactive suffix validation currently requires C2A Reuse")
    initial_cache = {}
    comparisons = {
        "attention_input": ratio_allclose(atol=1e-4, rtol=1 / 128),
        "normalized_attention": compare_normalized,
        "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
        "attention_hidden": full.compare_per_rank(full.compare_output),
    }
    if kind == DecodeLayerKind.SWA:
        comparisons.update(
            attention_output=swa.compare_reduced,
            window_cache=swa.compare_distributed_cache,
            window_cache_scale=swa.compare_scales,
        )
    elif kind == DecodeLayerKind.C2A_FULL:
        comparisons.update(
            attention_output=full.compare_replicated(full.compare_output),
            compressor_state=full.compare_per_rank(full.compare_state),
            topk_indices=full.compare_per_rank(full.compare_topk),
        )
        for name in ("window_cache", "compressed_cache", "index_cache"):
            comparisons[name] = full.compare_per_rank(full.compare_cache(name), full.CACHE_SLOTS[name])
            comparisons[name + "_scale"] = swa.compare_scales
    else:
        comparisons["attention_output"] = full.compare_replicated(reuse.compare_active_output)
        for name in reuse.REUSE_MUTABLE_NAMES:
            comparisons[name] = reuse.compare_owned_cache(name, initial_cache)
        for name in ("compressed_cache", "compressed_cache_scale", "topk_indices"):
            comparisons[name] = compare_unchanged(name)
    specs = build_specs(args, kind, initial_cache)
    result = run(
        fn=make_attention_program(args.layer_id, args.tp, args.epochs, specs),
        specs=specs,
        golden_fn=make_golden(kind, args.epochs),
        compare_fn=comparisons,
        config={
            "platform": args.platform,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
        },
        compile_only=args.compile_only,
        save_data=args.save_data,
    )
    print(f"[HALF] layer={args.layer_id} kind={kind.name} work_dir={result.work_dir}")
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()

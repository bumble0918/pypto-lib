# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3 Scope 3 MLP: gate/up projections + SiLU activation.

This test isolates the MLP operations from Qwen3 scope 3:
  - Stage 3: Gate projection (normed x w_gate)
  - Stage 4: Up projection (normed x w_up)
  - Stage 5: SiLU activation and elementwise multiply (silu(gate) * up)

These stages can be:
  - Fused (mix): single pl.at block with auto_chunk optimization
  - Split: separate pl.at blocks for each stage

Input normed_tile is BF16; w_gate/w_up are BF16; output is BF16.
"""
from __future__ import annotations

import pypto.language as pl

# ---------------------------------------------------------------------------
# MLP parameters — edit these to change problem size and tiling
# ---------------------------------------------------------------------------
BATCH = 16
HIDDEN = 8192
INTERMEDIATE = 25600

K_CHUNK = 128        # K dimension tile size for gate/up matmul
MLP_OUT_CHUNK = 256  # N dimension tile size for MLP output
BATCH_TILE = 16      # Batch dimension tile size


def build_scope3_mlp_mix_program(
    batch: int = BATCH,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
    k_chunk: int = K_CHUNK,
    mlp_out_chunk: int = MLP_OUT_CHUNK,
    batch_tile: int = BATCH_TILE,
    chunk: int = 4,
):
    """Build fused MLP program with auto_chunk optimization."""
    hidden_blocks = hidden // k_chunk
    mlp_out_blocks = intermediate // mlp_out_chunk

    @pl.program
    class Scope3MlpMixProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3_mlp(
            self,
            normed_tile: pl.Tensor[[batch, hidden], pl.BF16],
            w_gate: pl.Tensor[[hidden, intermediate], pl.BF16],
            w_up: pl.Tensor[[hidden, intermediate], pl.BF16],
            mlp_out: pl.Out[pl.Tensor[[batch, intermediate], pl.BF16]],
        ) -> pl.Tensor[[batch, intermediate], pl.BF16]:
            for ob in pl.range(mlp_out_blocks):
                o0 = ob * mlp_out_chunk
                post_chunk_0 = pl.slice(normed_tile, [batch_tile, k_chunk], [0, 0])
                with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk, pl.split(pl.SplitMode.UP_DOWN)]):
                    # Stage 3: Gate projection)
                    wg_0 = pl.slice(w_gate, [k_chunk, mlp_out_chunk], [0, o0])
                    gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)

                    for kb in pl.range(1, hidden_blocks):
                        k0 = kb * k_chunk
                        post_chunk = pl.slice(normed_tile, [batch_tile, k_chunk], [0, k0])
                        wg = pl.slice(w_gate, [k_chunk, mlp_out_chunk], [k0, o0])
                        gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                    # Stage 4: Up projection
                    wu_0 = pl.slice(w_up, [k_chunk, mlp_out_chunk], [0, o0])
                    up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
                    for kb in pl.range(1, hidden_blocks):
                        k0 = kb * k_chunk
                        post_chunk = pl.slice(normed_tile, [batch_tile, k_chunk], [0, k0])
                        wu = pl.slice(w_up, [k_chunk, mlp_out_chunk], [k0, o0])
                        up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                    # Stage 5: SiLU activation and elementwise multiply
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                    mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                    mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                mlp_out = pl.assemble(mlp_out, mlp_chunk_bf16, [0, o0])

            return mlp_out

    return Scope3MlpMixProgram


def build_scope3_mlp_split_program(
    batch: int = BATCH,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
    k_chunk: int = K_CHUNK,
    mlp_out_chunk: int = MLP_OUT_CHUNK,
    batch_tile: int = BATCH_TILE,
):
    """Build unfused MLP program with separate pl.at blocks."""
    hidden_blocks = hidden // k_chunk
    mlp_out_blocks = intermediate // mlp_out_chunk

    @pl.program
    class Scope3MlpSplitProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def scope3_mlp(
            self,
            normed_tile: pl.Tensor[[batch, hidden], pl.BF16],
            w_gate: pl.Tensor[[hidden, intermediate], pl.BF16],
            w_up: pl.Tensor[[hidden, intermediate], pl.BF16],
            mlp_out: pl.Out[pl.Tensor[[batch, intermediate], pl.BF16]],
        ) -> pl.Tensor[[batch, intermediate], pl.BF16]:
            for ob in pl.range(mlp_out_blocks):
                o0 = ob * mlp_out_chunk
                post_chunk_0 = pl.slice(normed_tile, [batch_tile, k_chunk], [0, 0])
                wg_0 = pl.slice(w_gate, [k_chunk, mlp_out_chunk], [0, o0])
                wu_0 = pl.slice(w_up, [k_chunk, mlp_out_chunk], [0, o0])

                # Stage 3: Gate projection
                with pl.at(level=pl.Level.CORE_GROUP):
                    gate_acc = pl.matmul(post_chunk_0, wg_0, out_dtype=pl.FP32)
                    for kb in pl.range(1, hidden_blocks):
                        k0 = kb * k_chunk
                        post_chunk = pl.slice(normed_tile, [batch_tile, k_chunk], [0, k0])
                        wg = pl.slice(w_gate, [k_chunk, mlp_out_chunk], [k0, o0])
                        gate_acc = pl.matmul_acc(gate_acc, post_chunk, wg)

                # Stage 4: Up projection
                with pl.at(level=pl.Level.CORE_GROUP):
                    up_acc = pl.matmul(post_chunk_0, wu_0, out_dtype=pl.FP32)
                    for kb in pl.range(1, hidden_blocks):
                        k0 = kb * k_chunk
                        post_chunk = pl.slice(normed_tile, [batch_tile, k_chunk], [0, k0])
                        wu = pl.slice(w_up, [k_chunk, mlp_out_chunk], [k0, o0])
                        up_acc = pl.matmul_acc(up_acc, post_chunk, wu)

                # Stage 5: SiLU activation and elementwise multiply
                with pl.at(level=pl.Level.CORE_GROUP):
                    sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                    mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                    mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                mlp_out = pl.assemble(mlp_out, mlp_chunk_bf16, [0, o0])

            return mlp_out

    return Scope3MlpSplitProgram


def build_tensor_specs(
    batch: int = BATCH,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
):
    import torch
    from golden import TensorSpec

    def init_normed_tile():
        return torch.rand(batch, hidden) - 0.5

    def init_w_gate():
        return (torch.rand(hidden, intermediate) - 0.5) / hidden ** 0.5

    def init_w_up():
        return (torch.rand(hidden, intermediate) - 0.5) / hidden ** 0.5

    return [
        TensorSpec("normed_tile", [batch, hidden], torch.bfloat16,
                   init_value=init_normed_tile),
        TensorSpec("w_gate", [hidden, intermediate], torch.bfloat16,
                   init_value=init_w_gate),
        TensorSpec("w_up", [hidden, intermediate], torch.bfloat16,
                   init_value=init_w_up),
        TensorSpec("mlp_out", [batch, intermediate], torch.bfloat16, is_output=True),
    ]


def golden_scope3_mlp(tensors):
    import torch

    normed_tile = tensors["normed_tile"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]

    # Gate projection
    gate = torch.matmul(normed_tile.float(), w_gate.float())
    # Up projection
    up = torch.matmul(normed_tile.float(), w_up.float())
    # SiLU activation and elementwise multiply
    mlp_out = (gate * torch.sigmoid(gate) * up).bfloat16()

    tensors["mlp_out"][:] = mlp_out


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=4,
                        help="Chunk size for parallel loop (smaller = more parallel tasks)")
    parser.add_argument("--mix", action="store_true",
                        help="Use fused mix version (default: split version)")
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    if args.mix:
        program = build_scope3_mlp_mix_program(chunk=args.chunk)
    else:
        program = build_scope3_mlp_split_program()

    result = run(
        program=program,
        specs=build_tensor_specs(),
        golden_fn=golden_scope3_mlp,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

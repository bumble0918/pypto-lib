# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FLAG_RE = re.compile(r"(set_flag|wait_flag)\((\w+),\s*(\w+),\s*(\w+)\)")


def test_fused_attention_declares_real_output_first() -> None:
    source = _REPO_ROOT / "models" / "qwen3_14b" / "paged_attention_cce.py"
    tree = ast.parse(source.read_text())
    func = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "paged_attention_rope_cce"
    )
    output_like = [
        arg.arg
        for arg in func.args.args
        if isinstance(arg.annotation, ast.Subscript)
        and isinstance(arg.annotation.value, ast.Attribute)
        and arg.annotation.value.attr in {"Out", "InOut"}
    ]

    assert output_like[0] == "out", (
        "single-result extern binds its return to the first Out/InOut parameter"
    )


def test_fused_attention_uses_standalone_rope_worker_count() -> None:
    decode_source = _REPO_ROOT / "models" / "qwen3_14b" / "decode_fwd.py"
    decode_tree = ast.parse(decode_source.read_text())
    rope_cores = next(
        node.value.value
        for node in decode_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "ROPE_CORES"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
    )

    kernel_dir = (
        _REPO_ROOT
        / "models"
        / "qwen3_14b"
        / "kernels"
        / "paged_attention_cce"
        / "kernel"
    )
    fai_body = (kernel_dir / "fai_body.hpp").read_text()
    assert f"constexpr uint32_t kQwenRopeCores = {rope_cores};" in fai_body
    # Match the lane guard and the guarded call, but NOT the call's trailing
    # arguments: regenerating the RoPE body can change the parameter list (e.g.
    # adding a dynamic-dim scalar), and that is a legitimate change this test
    # should not block. The arg mapping itself is covered by the static_assert
    # on the function-pointer type in fai_body.hpp.
    guarded_call = re.search(
        r"uint32_t rope_lane = block_idx \* 2 \+ sub_block_idx;\s*"
        r"if \(rope_lane < kQwenRopeCores\) \{\s*"
        r"qwen_rope_gen::rope_qkv\(",
        fai_body,
        flags=re.DOTALL,
    )
    assert guarded_call is not None

    # Anchor on the hand-written provenance banner, not on a generated
    # `const int64_t vNN = 32;` line -- SSA constants are renumbered by every
    # regeneration, so pinning one makes any regen look like a real failure.
    generated_rope = (kernel_dir / "rope_qkv_generated.hpp").read_text()
    assert f"// ROPE_CORES: {rope_cores}" in generated_rope, (
        "update the ROPE_CORES provenance banner in rope_qkv_generated.hpp's "
        "hand-written preamble when regenerating the specialized RoPE body"
    )


def test_generated_rope_every_feasible_path_is_sync_safe() -> None:
    """Every executable path through the guarded RoPE items must not deadlock."""
    body = _rope_qkv_function_body()
    lines = body.split("\n")

    blocks: list[tuple[int, int]] = []
    for n, line in enumerate(lines):
        if not re.match(r"\s*if \(v\d+ < v\d+\) \{", line):
            continue
        depth = 0
        for end in range(n, len(lines)):
            depth += lines[end].count("{") - lines[end].count("}")
            if depth == 0 and end > n:
                blocks.append((n, end))
                break
    assert len(blocks) == 2, f"expected 2 guarded item blocks, found {len(blocks)}"

    def line_block(index: int) -> int | None:
        for b, (start, end) in enumerate(blocks):
            if start <= index <= end:
                return b
        return None

    # Reachable prefixes only: item L+ROPE_CORES cannot pass a guard that item L
    # failed, so {block 1 alone} is not a reachable state.
    for taken in ((), (0,), (0, 1)):
        credits: Counter = Counter()
        for index, line in enumerate(lines):
            owner = line_block(index)
            if owner is not None and owner not in taken:
                continue
            for kind, src_pipe, dst_pipe, event in _FLAG_RE.findall(line):
                key = (src_pipe, dst_pipe, event)
                if kind == "set_flag":
                    credits[key] += 1
                else:
                    assert credits[key] > 0, (
                        f"path {taken or '(no guarded block)'}: wait_flag{key} at "
                        f"generated line {index} has no outstanding set_flag -- "
                        f"this path would hang the AIV at a runtime batch < BATCH_PAD"
                    )
                    credits[key] -= 1
        outstanding = {k: v for k, v in credits.items() if v}
        assert not outstanding, (
            f"path {taken or '(no guarded block)'} leaves undrained sync credits "
            f"{outstanding}; the epilogue must consume exactly what the prologue set"
        )


def _rope_qkv_function_body() -> str:
    path = (
        _REPO_ROOT / "models" / "qwen3_14b" / "kernels" / "paged_attention_cce"
        / "kernel" / "rope_qkv_generated.hpp"
    )
    src = path.read_text()
    start = src.index("static __aicore__ void rope_qkv(")
    depth, idx = 0, src.index("{", start)
    while True:
        if src[idx] == "{":
            depth += 1
        elif src[idx] == "}":
            depth -= 1
            if depth == 0:
                return src[start:idx + 1]
        idx += 1

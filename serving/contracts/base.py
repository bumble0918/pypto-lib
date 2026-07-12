# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Phase-1 serving contract dataclasses."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal


ArgDirection = Literal["in", "out", "inout"]


@dataclass(frozen=True)
class ModelId:
    """Stable identity for a model-serving contract."""

    family: str
    variant: str
    size: str | None = None
    quant: str | None = None


@dataclass(frozen=True)
class TensorArgSpec:
    """Phase-1 tensor argument ABI metadata."""

    name: str
    dtype: str
    shape: tuple[str | int, ...]
    direction: ArgDirection = "in"


@dataclass(frozen=True)
class KernelSpec:
    """Phase-1 metadata and hooks for one logical serving kernel stage."""

    name: str
    public_name: str
    args: tuple[TensorArgSpec, ...]
    host_jit_fn: Callable[..., Any]
    compile_args_builder: Callable[..., tuple[Any, ...]]
    runtime_args_builder: Callable[..., tuple[Any, ...]]


@dataclass(frozen=True)
class LoadedKernelModules:
    """Kernel functions and constants loaded from lib-owned kernel modules."""

    functions: Mapping[str, object]
    constants: Mapping[str, int]


@dataclass(frozen=True)
class ModelServingContract:
    """Top-level serving ABI contract owned by pypto-lib."""

    schema_version: str
    model: ModelId
    capabilities: tuple[str, ...]
    limits: Mapping[str, int | str]
    execution: Mapping[str, tuple[str, ...]]
    kernels: Mapping[str, KernelSpec]
    kernel_binder: Callable[..., None]
    prepare_weights: Callable[..., Any]
    load_kernels: Callable[[], LoadedKernelModules]
    validate_kernels: Callable[["ModelServingContract", LoadedKernelModules, Any], None]

    def abi_fingerprint(self) -> str:
        """Return a stable hash of public phase-1 ABI metadata."""
        payload = {
            "schema_version": self.schema_version,
            "model": asdict(self.model),
            "capabilities": sorted(self.capabilities),
            "limits": dict(self.limits),
            "execution": {name: stages for name, stages in sorted(self.execution.items())},
            "kernels": {
                name: {
                    "public_name": spec.public_name,
                    "args": [asdict(arg) for arg in spec.args],
                }
                for name, spec in sorted(self.kernels.items())
            },
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ContractRegistration:
    """Model-owned serving contract registration."""

    family: str
    variant: str
    factory: Callable[[], ModelServingContract]
    matcher: Callable[[object], bool] | None = None
    implemented: bool = True

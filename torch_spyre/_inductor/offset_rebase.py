# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bind a graph input's storage_offset at launch instead of baking it in.

A compiled kernel addresses each input from its storage base, so an input view
with a nonzero ``storage_offset`` gets the offset baked into the kernel, and the
``_spyre_TENSOR_MATCH`` guard keys on it (#4449). Views that differ only by
offset, such as one row per layer of a packed tensor, then compile once each.

When the offset is a whole-stick shift of the allocation
(``is_rebasable_offset_view``), Inductor lowers that input at offset 0 instead
(``zero_offset``, read by the placeholder patch in ``patches.py``), and every
call passes an offset-0 alias whose storage starts at the view
(``rebase_offset_view``). Nothing is baked, so the guard can accept any
rebasable offset for that input and every such view shares one compiled graph.
AOTAutograd and the shape env still see the original tensor.

The guard is built after the backend compile of the same frame, and has to
agree with it exactly: an input that was not rebased keeps its baked offset and
its exact guard. The compile records the inputs it rebased under its compile id
for the guard to look up (``was_rebased``).
"""

import contextlib
import threading
import weakref
from typing import Any, Callable, Sequence

import torch
from torch._guards import CompileContext
from torch._inductor.virtualized import V

from . import config
from .constants import DEVICE_NAME

# Ops whose arguments carry an explicit storage offset (dynamo bakes
# ``x.storage_offset()`` into the call), which a rebased input would apply twice.
_OFFSET_AWARE_TARGETS = (
    "copy_from_d2d",
    "to_dtype_d2d",
    "as_strided",
    "storage_offset",
)

# compile id -> {id(input): weakref(input)} for the inputs that compile rebased;
# trimmed to the latest entries. Keyed by id: tensors compare elementwise.
_rebased_by_compile: dict[Any, dict[int, "weakref.ref[torch.Tensor]"]] = {}
_MAX_TRACKED_COMPILES = 256
_lock = threading.Lock()


def _graph_reads_offsets(gm: torch.fx.GraphModule) -> bool:
    for node in gm.graph.nodes:
        name = getattr(node.target, "__name__", None) or str(node.target)
        if any(t in name for t in _OFFSET_AWARE_TARGETS):
            return True
    return False


def _has_symbolic_inputs(gm: torch.fx.GraphModule) -> bool:
    # A symbolic offset or stride cannot be checked for rebasing at compile time.
    for node in gm.graph.find_nodes(op="placeholder"):
        val = node.meta.get("example_value", node.meta.get("val"))
        if isinstance(val, torch.SymInt):
            return True
        if isinstance(val, torch.Tensor) and any(
            not isinstance(s, int) for s in (*val.shape, *val.stride())
        ):
            return True
    return False


def _output_storages(gm: torch.fx.GraphModule) -> set[int]:
    # A graph output that aliases a rebased input would be rebuilt over the
    # alias storage and escape the call; such inputs are left alone.
    out: set[int] = set()
    nodes: list[torch.fx.Node] = []
    torch.fx.node.map_arg(gm.graph.output_node().args, nodes.append)
    for node in nodes:
        val = node.meta.get("example_value", node.meta.get("val"))
        if isinstance(val, torch.Tensor):
            out.add(val.untyped_storage()._cdata)
    return out


def _placeholder_storages(gm: torch.fx.GraphModule) -> list[int | None]:
    out: list[int | None] = []
    for node in gm.graph.find_nodes(op="placeholder"):
        val = node.meta.get("example_value", node.meta.get("val"))
        out.append(
            val.untyped_storage()._cdata if isinstance(val, torch.Tensor) else None
        )
    return out


def _is_candidate(t: Any) -> bool:
    from torch_spyre._C import is_rebasable_offset_view

    if not isinstance(t, torch.Tensor) or t.device.type != DEVICE_NAME:
        return False
    if isinstance(t, torch._subclasses.FakeTensor):
        return False
    if t.storage_offset() == 0:
        # E.g. the first row of a packed tensor: the later rows should reuse
        # its graph, so it is compiled rebased too.
        return t._base is not None
    return is_rebasable_offset_view(t)


def select_inputs(gm: torch.fx.GraphModule, example_inputs: Sequence[Any]) -> list[int]:
    """Indices of the example inputs to rebase for this compile."""
    if not config.rebase_offset_inputs:
        return []
    if _graph_reads_offsets(gm) or _has_symbolic_inputs(gm):
        return []
    candidates = [i for i, t in enumerate(example_inputs) if _is_candidate(t)]
    if not candidates:
        return []
    # Rebasing gives an input its own storage, which would hide aliasing
    # between inputs from AOTAutograd; leave shared storages alone.
    storages: dict[int, int] = {}
    for t in example_inputs:
        if isinstance(t, torch.Tensor) and t.device.type == DEVICE_NAME:
            key = t.untyped_storage()._cdata
            storages[key] = storages.get(key, 0) + 1
    placeholders = _placeholder_storages(gm)
    if len(placeholders) != len(example_inputs):
        return []
    outputs = _output_storages(gm)
    return [
        i
        for i in candidates
        if storages[example_inputs[i].untyped_storage()._cdata] == 1
        and placeholders[i] not in outputs
    ]


def record(example_inputs: Sequence[Any], indices: Sequence[int]) -> None:
    compile_id = CompileContext.current_compile_id()
    if compile_id is None or not indices:
        return
    with _lock:
        rebased = _rebased_by_compile.setdefault(compile_id, {})
        for i in indices:
            rebased[id(example_inputs[i])] = weakref.ref(example_inputs[i])
        while len(_rebased_by_compile) > _MAX_TRACKED_COMPILES:
            _rebased_by_compile.pop(next(iter(_rebased_by_compile)))


def was_rebased(t: torch.Tensor) -> bool:
    """True if the current frame's compile rebased ``t`` (read by the guard)."""
    compile_id = CompileContext.current_compile_id()
    if compile_id is None:
        return False
    with _lock:
        ref = _rebased_by_compile.get(compile_id, {}).get(id(t))
    return ref is not None and ref() is t


def rebase(args: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    from torch_spyre._C import rebase_offset_view

    out = list(args)
    for i in indices:
        out[i] = rebase_offset_view(args[i])
    return out


_active = threading.local()


@contextlib.contextmanager
def lowering_at_offset_zero(indices: Sequence[int]):
    """Make Inductor lower the ``indices`` graph inputs at storage_offset 0."""
    prev = getattr(_active, "indices", None)
    _active.indices = frozenset(indices)
    try:
        yield
    finally:
        _active.indices = prev


def zero_offset(graph: Any, index: int, example: Any) -> bool:
    """True if placeholder ``index`` of ``graph`` is lowered at offset 0.

    ``propagate_layouts`` pairs graph inputs with the real inputs by position,
    so ``example`` must describe the same tensor as real input ``index``.
    """
    indices = getattr(_active, "indices", None)
    if not indices or index not in indices:
        return False
    if getattr(graph, "parent", None) is not None:
        return False
    real = V.get_real_inputs()[index]
    if not isinstance(example, torch.Tensor) or (
        tuple(example.shape) != tuple(real.shape)
        or tuple(example.stride()) != tuple(real.stride())
    ):
        raise AssertionError(
            f"graph input {index} does not match the real input it rebases: "
            f"{tuple(getattr(example, 'shape', ()))} vs {tuple(real.shape)}"
        )
    return True


def wrap_compiled(fn: Callable, indices: Sequence[int]) -> Callable:
    """Rebase the ``indices`` arguments before every call of ``fn``.

    ``compile_fx`` returns a function in the unboxed calling convention.
    """
    from torch_spyre._C import rebase_offset_view

    indices = tuple(indices)

    def rebased(*args):
        args = list(args)
        for i in indices:
            args[i] = rebase_offset_view(args[i])
        return fn(*args)

    return rebased

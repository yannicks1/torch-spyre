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


"""Graph inputs whose storage_offset is a whole-stick shift share one graph.

Such a view is passed to the compiled kernel as an offset-0 alias of its
allocation (``torch_spyre/_inductor/offset_rebase.py``), so the offset is bound
at launch instead of baked into the kernel and guarded. Each case checks the
values for every offset against CPU and that the offsets compiled one graph.
"""

import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

import torch_spyre  # noqa: F401
from torch_spyre._C import is_rebasable_offset_view, rebase_offset_view
from torch_spyre._inductor import config

DEVICE = "spyre"
DTYPE = torch.float16


def _run(fn, device_args, cpu_args, atol):
    """Compile ``fn`` once, call it per offset, return the frame count."""
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, dynamic=False)
    for dev, cpu in zip(device_args, cpu_args):
        got = compiled(*dev).cpu().float()
        torch.testing.assert_close(got, fn(*cpu).float(), atol=atol, rtol=0)
    return counters["stats"]["unique_graphs"]


@pytest.mark.parametrize("T", [1, 4, 16])
def test_per_layer_rows_share_one_graph(T):
    """gemma-4 per-layer embeddings: row ``[:, i, :]`` of ``[T, L, D]`` per layer."""
    L, D, H = 6, 256, 512
    torch.manual_seed(0)
    ple = torch.randn(T, L, D, dtype=DTYPE)
    x = torch.randn(T, H, dtype=DTYPE)
    w = torch.randn(D, H, dtype=DTYPE) * 0.05
    ple_d, x_d, w_d = ple.to(DEVICE), x.to(DEVICE), w.to(DEVICE)
    layers = list(range(L)) + [2, 0]

    frames = _run(
        lambda x, row, w: x + row @ w,
        [(x_d, ple_d[:, i, :], w_d) for i in layers],
        [(x, ple[:, i, :], w) for i in layers],
        atol=0.05,
    )
    assert frames == 1


@pytest.mark.parametrize(
    "shape, rows",
    [
        ((8, 128), 1),  # one row of a stick-multiple width
        ((8, 100), 3),  # rows of a stick-padded width
    ],
)
def test_row_chunks_share_one_graph(shape, rows):
    torch.manual_seed(0)
    base = torch.randn(*shape, dtype=DTYPE)
    base_d = base.to(DEVICE)
    starts = range(0, shape[0] - rows + 1, 2)

    frames = _run(
        lambda x: x + 1.0,
        [(base_d[s : s + rows],) for s in starts],
        [(base[s : s + rows],) for s in starts],
        atol=0.01,
    )
    assert frames == 1


def test_expert_chunks_share_one_graph():
    """MoE: a chunk of experts out of a 3-D weight, through bmm."""
    E, H, F, N, chunk = 8, 32, 64, 32, 2
    torch.manual_seed(0)
    w = torch.randn(E, H, F, dtype=DTYPE)
    x = torch.randn(chunk, F, N, dtype=DTYPE)
    w_d, x_d = w.to(DEVICE), x.to(DEVICE)
    starts = range(0, E, chunk)

    frames = _run(
        torch.bmm,
        [(w_d[s : s + chunk], x_d) for s in starts],
        [(w[s : s + chunk], x) for s in starts],
        atol=0.1,
    )
    assert frames == 1


def test_in_place_write_lands_in_the_base():
    def fn(row):
        row.add_(1.0)
        return row.sum()

    base_d = torch.zeros(4, 3, 128, dtype=DTYPE).to(DEVICE)
    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(fn, dynamic=False)
    for i in range(3):
        compiled(base_d[:, i, :])

    assert bool((base_d.cpu() == 1).all())
    assert counters["stats"]["unique_graphs"] == 1


def test_disabled_keeps_one_graph_per_offset(monkeypatch):
    monkeypatch.setattr(config, "rebase_offset_inputs", False)
    torch.manual_seed(0)
    base = torch.randn(8, 128, dtype=DTYPE)
    base_d = base.to(DEVICE)

    frames = _run(
        lambda x: x + 1.0,
        [(base_d[s : s + 1],) for s in (0, 3, 5)],
        [(base[s : s + 1],) for s in (0, 3, 5)],
        atol=0.01,
    )
    assert frames == 3


def test_rebase_aliases_the_view():
    base = torch.arange(4 * 3 * 64, dtype=DTYPE).reshape(4, 3, 64).to(DEVICE)
    view = base[:, 1, :]
    alias = rebase_offset_view(view)

    assert is_rebasable_offset_view(view)
    assert alias.storage_offset() == 0
    assert alias.shape == view.shape and alias.stride() == view.stride()
    assert alias.device_tensor_layout() == view.device_tensor_layout()
    torch.testing.assert_close(alias.cpu(), view.cpu(), atol=0, rtol=0)


def test_offset_inside_a_stick_is_not_rebasable():
    base = torch.randn(4, 256, dtype=DTYPE).to(DEVICE)
    for view in (base[:, 32:96], base.view(-1)[16:80]):
        assert not is_rebasable_offset_view(view)
        assert rebase_offset_view(view) is view

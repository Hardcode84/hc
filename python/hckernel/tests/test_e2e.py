# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from hckernel.kernel import *
import pytest

try:
    import torch
    from torch.testing import assert_close

    _has_torch = True
except ImportError:
    _has_torch = False

require_e2e = pytest.mark.skipif(
    not _has_torch and torch.cuda.is_available(), reason="e2e tests are disabled"
)


def get_backend():
    device = torch.device("cuda")
    # TODO: Stupid way to check for rocm torch
    if hasattr(torch.cuda.get_device_properties(device), "gcnArchName"):
        return "rocm"

    return "nvvm"


def to_device(array):
    return array.to("cuda")


def get_device(backend):
    if backend == "rocm":
        return "hip:0"

    if backend == "nvvm":
        return "cuda:0"

    assert False, f"Invalid backend: {backend}"


W = sym.W
H = sym.H
DT = typename.DT


@require_e2e
def test_copy():
    backend = get_backend()

    @kernel(work_shape=(W, H), backend=backend, device=get_device(backend))
    def copy_kernel(group: CurrentGroup, src: Buffer[W, H, DT], dst: Buffer[W, H, DT]):
        x, y = group.work_offset
        val = group.load(src[x:, y:], shape=group.shape)
        group.store(dst[x:, y:], val)

    a = to_device(torch.arange(200, dtype=torch.int32).reshape(10, 20))
    b = torch.zeros_like(a)
    copy_kernel(a, b)
    assert_close(b, a)

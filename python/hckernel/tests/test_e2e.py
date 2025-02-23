# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from hckernel.kernel import *
import pytest
import sympy

try:
    import torch
    from torch.testing import assert_close

    _has_torch = True
except ImportError:
    _has_torch = False

require_e2e = pytest.mark.skipif(
    not (_has_torch and torch.cuda.is_available()), reason="e2e tests are disabled"
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


_copy_shapes = [
    (1, 27),
    (1, 72),
    (1, 128),
    (10, 20),
    (111, 813),
    (256, 64),
    (256, 1024),
]


@require_e2e
@pytest.mark.parametrize("shape", _copy_shapes)
def test_copy(shape):
    backend = get_backend()

    W = sym.W
    H = sym.H
    DT = typename.DT

    @kernel(work_shape=(W, H), backend=backend, device=get_device(backend))
    def copy_kernel(group: CurrentGroup, src: Buffer[W, H, DT], dst: Buffer[W, H, DT]):
        x, y = group.work_offset
        val = group.load(src[x:, y:], shape=group.shape)
        group.store(dst[x:, y:], val)

    a = to_device(torch.arange(shape[0] * shape[1], dtype=torch.int32).reshape(shape))
    b = torch.zeros_like(a)
    copy_kernel(a, b)
    assert_close(b, a)


@require_e2e
@pytest.mark.parametrize("shape", _copy_shapes)
@pytest.mark.parametrize("unroll", [1, 2, 4])
def test_copy_unroll(shape, unroll):
    backend = get_backend()

    W = sym.W
    H = sym.H
    DT = typename.DT
    V = sym.V
    H1 = sympy.ceiling(H / V)

    @kernel(
        work_shape=(W, H1),
        backend=backend,
        device=get_device(backend),
    )
    def copy_kernel(
        group: CurrentGroup, src: Buffer[W, H, DT], dst: Buffer[W, H, DT], v: V
    ):
        x, y = group.work_offset
        val = group.load(src[x:, y * V :], shape=(group.shape[0], group.shape[1] * V))
        group.store(dst[x:, y * V :], val)

    a = to_device(torch.arange(shape[0] * shape[1], dtype=torch.int32).reshape(shape))
    b = torch.zeros_like(a)
    copy_kernel(a, b, unroll)
    assert_close(b, a)


@require_e2e
@pytest.mark.parametrize("shape", _copy_shapes)
@pytest.mark.parametrize("unroll", [1, 2, 4])
def test_copy_vec(shape, unroll):
    backend = get_backend()

    W = sym.W
    H = sym.H
    DT = typename.DT
    V = sym.V
    H1 = sympy.ceiling(H / V)

    @kernel(
        work_shape=(W, H1),
        backend=backend,
        device=get_device(backend),
        literals=[V],
    )
    def copy_kernel(
        group: CurrentGroup, src: Buffer[W, H, DT], dst: Buffer[W, H, DT], v: V
    ):
        x, y = group.work_offset
        val = group.vload(src[x:, y * V :], shape=(group.shape[0], group.shape[1] * V))
        group.store(dst[x:, y * V :], val)

    a = to_device(torch.arange(shape[0] * shape[1], dtype=torch.int32).reshape(shape))
    b = torch.zeros_like(a)
    copy_kernel(a, b, unroll)
    assert_close(b, a)

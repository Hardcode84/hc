# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from itertools import product
from numpy.testing import assert_equal, assert_allclose
import numpy as np
import pytest

from hckernel.kernel_sim import (
    Buffer,
    CurrentGroup,
    CurrentSubGroup,
    CurrentWorkitem,
    TunableParam,
    atomic_ref,
    create_mapping,
    kernel,
    sym,
    typename,
)

DT = typename.DT


def ceil_div(a, b):
    return (a + b - 1) // b


def test_reduction_1():
    WS, GS, N = sym.WS, sym.GS, sym.N
    TN = TunableParam(N, 2, range(1, 64))

    @kernel(work_shape=ceil_div(WS, N), group_shape=GS, tunables=TN)
    def reduction(
        gr: CurrentGroup, a: Buffer[WS, DT], result: Buffer[1, DT], gshape: GS
    ):
        a_view = gr.load(a[gr.work_offset[0] * N :], shape=gr.size * N)
        atomic_ref(result)[0] += a_view.sum()

    a = np.arange(0, 20)
    result = np.zeros(1)

    reduction(a, result, 8)

    assert_equal(result[0], a.sum())


def test_reduction_2():
    WS, GS, N = sym.WS, sym.GS, sym.N
    TN = TunableParam(N, 2, range(1, 64))

    @kernel(work_shape=ceil_div(WS, N), group_shape=GS, tunables=TN)
    def reduction(
        group: CurrentGroup, a: Buffer[WS, DT], result: Buffer[1, DT], gshape: GS
    ):
        temp_result = group.zeros(group.size, dtype=a.dtype)
        for i in range(N):
            work_offset = group.work_offset[0] * N + i * group.size
            a_view = group.load(a[work_offset:], shape=group.size)
            temp_result += a_view

        atomic_ref(result)[0] += temp_result.sum()

    a = np.arange(0, 20)
    result = np.zeros(1)

    reduction(a, result, 8)

    assert_equal(result[0], a.sum())


def test_dot_1():
    M, N, K = sym.M, sym.N, sym.K
    GM, GN = sym.GM, sym.GN
    MI, NI, KB = sym.MI, sym.NI, sym.KB

    WORK_SHAPE = (ceil_div(M, MI), ceil_div(N, NI))
    GROUP_SHAPE = (GM, GN)

    KI = ceil_div(K, KB)

    TMI = TunableParam(MI, 1, range(1, 8))
    TNI = TunableParam(NI, 1, range(1, 8))
    TKB = TunableParam(KB, 16, range(8, 128))

    @kernel(work_shape=WORK_SHAPE, group_shape=GROUP_SHAPE, tunables=(TMI, TNI, TKB))
    def dot(
        gr: CurrentGroup,
        a: Buffer[M, K, DT],
        b: Buffer[K, N, DT],
        out: Buffer[M, N, DT],
        gshape: GROUP_SHAPE,
    ):
        m_start = gr.work_offset[0] * MI
        n_start = gr.work_offset[1] * NI

        m_block = gr.shape[0] * MI
        n_block = gr.shape[1] * NI
        k_block = KB

        r = gr.zeros(shape=(m_block, n_block), dtype=out.dtype)
        for k in range(KI):
            k_start = k * k_block
            a_view = gr.load(a[m_start:, k_start:], shape=(m_block, k_block))
            b_view = gr.load(b[k_start:, n_start:], shape=(k_block, n_block))

            r += np.dot(a_view, b_view)

        gr.store(out[m_start:, n_start:], r)

    m, n, k = 16, 32, 64

    a = 2 * np.arange(m * k).reshape(m, k) / (m * k)
    b = np.arange(n * k).reshape(k, n) / (n * k)
    result = np.zeros(shape=(m, n))

    dot(a, b, result, (8, 8))

    assert_equal(result, np.dot(a, b))


def test_dot_2():
    M, N, K = sym.M, sym.N, sym.K
    GM, GN = sym.GM, sym.GN
    MI, NI = sym.MI, sym.NI

    WORK_SHAPE = (ceil_div(M, MI), ceil_div(N, NI))
    GROUP_SHAPE = (GM, GN)

    TMI = TunableParam(MI, 1, range(1, 8))
    TNI = TunableParam(NI, 1, range(1, 8))

    @kernel(work_shape=WORK_SHAPE, group_shape=GROUP_SHAPE, tunables=(TMI, TNI))
    def dot(
        gr: CurrentGroup,
        a: Buffer[M, K, DT],
        b: Buffer[K, N, DT],
        out: Buffer[M, N, DT],
        gshape: GROUP_SHAPE,
    ):
        m_start = gr.work_offset[0] * MI
        n_start = gr.work_offset[1] * NI

        m_block = gr.shape[0] * MI
        n_block = gr.shape[1] * NI

        r = gr.zeros(shape=(m_block, n_block), dtype=out.dtype)
        a_view = gr.load(a[m_start:, :], shape=(m_block, K))
        b_view = gr.load(b[:, n_start:], shape=(K, n_block))

        r += np.dot(a_view, b_view)

        gr.store(out[m_start:, n_start:], r)

    m, n, k = 16, 32, 64

    a = 2 * np.arange(m * k).reshape(m, k) / (m * k)
    b = np.arange(n * k).reshape(k, n) / (n * k)
    result = np.zeros(shape=(m, n))

    dot(a, b, result, (8, 8))

    assert_equal(result, np.dot(a, b))


@pytest.mark.parametrize("n", [1, 2, 4])
@pytest.mark.parametrize("c", [1, 3])
@pytest.mark.parametrize("nf", [1, 2, 8])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_implicit_gemm(n, c, nf, stride):
    h, w = 4, 4  # Image.
    cf, hf, wf = c, 2, 2  # Filters.
    x = np.random.randn(n, c, h, w)
    we = np.random.randn(nf, cf, hf, wf)
    padding = 0

    def conv_ref(X, W):
        m, n_C_prev, n_H_prev, n_W_prev = X.shape

        n_H = int((n_H_prev + 2 * padding - hf) / stride) + 1
        n_W = int((n_W_prev + 2 * padding - hf) / stride) + 1

        out = np.zeros((m, nf, n_H, n_W))
        for i in range(m):  # For each image.
            for c in range(nf):  # For each channel.
                for h in range(n_H):  # Slide the filter vertically.
                    h_start = h * stride
                    h_end = h_start + hf

                    for w in range(n_W):  # Slide the filter horizontally.
                        w_start = w * stride
                        w_end = w_start + hf

                        out[i, c, h, w] = np.sum(
                            X[i, :, h_start:h_end, w_start:w_end] * W[c, ...]
                        )
        return out

    out_ref = conv_ref(x, we)

    N, C, H, W = sym.N, sym.C, sym.H, sym.W
    NF, HF, WF = sym.NF, sym.HF, sym.WF
    KB = sym.KB
    H_OUT = (H + 2 * padding - HF) / stride + 1
    W_OUT = (W + 2 * padding - WF) / stride + 1
    SZ = HF * WF * C
    KI = ceil_div(SZ, KB)

    TN, TNF = sym.TN, sym.TNF

    WORK_SHAPE = (N, NF, H_OUT * W_OUT)
    GROUP_SHAPE = (TN, TNF, 1)

    TTN = TunableParam(TN, 8, range(1, 64))
    TTNF = TunableParam(TNF, 8, range(1, 64))
    TKB = TunableParam(KB, 16, range(8, 128))

    x_map = create_mapping(
        lambda i, j: (i, j // (HF * WF), (j % (HF * WF)) % WF, (j % (HF * WF)) // WF)
    )
    f_map = create_mapping(
        lambda i, j: (j, i // (HF * WF), i % WF, (i % (HF * WF)) // WF)
    )
    out_map = create_mapping(lambda i, j: (i, j, 0, 0))

    @kernel(work_shape=WORK_SHAPE, group_shape=GROUP_SHAPE, tunables=(TTN, TTNF, TKB))
    def conv(
        gr: CurrentGroup,
        x: Buffer[N, C, H, W, DT],
        f: Buffer[NF, C, HF, WF, DT],
        out: Buffer[N, NF, H_OUT, W_OUT, DT],
    ):
        n, nf, w_idx = gr.work_offset

        i = w_idx % W_OUT
        j = w_idx // W_OUT
        x_view = gr.load(
            x[n:, :, i * stride :, j * stride :], shape=(TN, SZ), mapping=x_map
        )

        f_view = gr.load(f[nf:, :, :, :], shape=(SZ, TNF), mapping=f_map)

        r = gr.zeros(shape=(x_view.shape[0], f_view.shape[1]), dtype=out.dtype)
        for k in range(KI):
            k_start = k * KB
            x_slice = gr.vload(x_view[:, k_start:], shape=(TN, KB))
            f_slice = gr.vload(f_view[k_start:, :], shape=(KB, TNF))
            r += np.dot(x_slice, f_slice)

        gr.store(out[n:, nf:, i:, j:], r, mapping=out_map)

    result = np.zeros_like(out_ref)

    conv(x, we, result)

    assert_allclose(result, out_ref)


def test_pairwise_distance():
    M, N, D = sym.M, sym.N, sym.D
    GM, GN = sym.GM, sym.GN
    MI, NI = sym.MI, sym.NI

    WORK_SHAPE = (ceil_div(M, MI), ceil_div(N, NI))
    GROUP_SHAPE = (GM, GN)

    TMI = TunableParam(MI, 1, range(1, 8))
    TNI = TunableParam(NI, 1, range(1, 8))

    @kernel(work_shape=WORK_SHAPE, group_shape=GROUP_SHAPE, tunables=(TMI, TNI))
    def pw_distance(
        gr: CurrentGroup,
        a: Buffer[M, D, DT],
        b: Buffer[N, D, DT],
        out: Buffer[M, N, DT],
        gshape: GROUP_SHAPE,
    ):
        m_start = gr.work_offset[0] * MI
        n_start = gr.work_offset[1] * NI

        m_block = gr.shape[0] * MI
        n_block = gr.shape[1] * NI

        a_view = gr.load(a[m_start:, :], shape=(m_block, D))
        b_view = gr.load(b[n_start:, :], shape=(n_block, D))

        diff = a_view[:, None, :] - b_view[None, :, :]

        gr.store(out[m_start:, n_start:], np.linalg.norm(diff, axis=2))

    m, n, d = 16, 32, 8

    a = 2 * np.arange(m * d).reshape(m, d) / (m * d)
    b = np.arange(n * d).reshape(n, d) / (n * d)
    result = np.zeros(shape=(m, n))

    pw_distance(a, b, result, (8, 8))

    assert_equal(result, np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2))


def test_softmax_over_axis():
    M, N = sym.M, sym.N
    GM = sym.GM
    MI = sym.MI

    TMI = TunableParam(MI, 1, range(1, 8))

    @kernel(work_shape=ceil_div(M, MI), group_shape=GM, tunables=TMI)
    def softmax(
        gr: CurrentGroup, a: Buffer[M, N, DT], out: Buffer[M, N, DT], gshape: GM
    ):
        m_start = gr.work_offset[0] * MI
        m_block = gr.shape[0] * MI

        a_view = gr.load(a[m_start:, :], shape=(m_block, N))
        exp = np.exp(a_view - a_view.max(axis=1, keepdims=True))

        gr.store(out[m_start:, :], exp / (exp.sum(axis=1, keepdims=True)))

    m, n = 30, 25

    a = 2 * np.arange(m * n).reshape(m, n) / (m * n)
    result = np.zeros(shape=(m, n))

    softmax(a, result, 8)

    def np_softmax(a):
        exp = np.exp(a - a.max(axis=1, keepdims=True))

        return exp / exp.sum(axis=1, keepdims=True)

    assert_equal(result, np_softmax(a))

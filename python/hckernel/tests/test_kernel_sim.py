# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from itertools import product
from numpy.testing import assert_equal
import numpy as np
import pytest

from hckernel.kernel_sim import (
    Buffer,
    CurrentGroup,
    CurrentSubGroup,
    CurrentWorkitem,
    TunableParam,
    atomic_ref,
    kernel,
    sym,
    typename,
)

DT = typename.DT


@pytest.mark.parametrize(
    "gsize", [(512, 1, 1), (511, 1, 1), (1, 16, 1), (1, 1, 16), (1, 1, 1)]
)
@pytest.mark.parametrize("lsize", [(64, 1, 1), (1, 1, 1)])
def test_group_iteration(gsize, lsize):
    def get_range(i):
        return range((gsize[i] + lsize[i] - 1) // lsize[i])

    def get_group_ranges():
        return product(get_range(0), get_range(1), get_range(2))

    res_ids = []
    res_offsets = []

    G1, G2, G3 = sym.G1, sym.G2, sym.G3
    L1, L2, L3 = sym.L1, sym.L2, sym.L3

    @kernel(work_shape=(G1, G2, G3), group_shape=(L1, L2, L3))
    def test(gr: CurrentGroup, gsize: tuple[G1, G2, G3], lsize: tuple[L1, L2, L3]):
        res_ids.append(gr.id)
        res_offsets.append(gr.work_offset)

    test(gsize, lsize)
    exp_res_ids = [(i, j, k) for i, j, k in get_group_ranges()]
    assert res_ids == exp_res_ids
    exp_res_offsets = [
        (i * lsize[0], j * lsize[1], k * lsize[2]) for i, j, k in get_group_ranges()
    ]
    assert_equal(res_offsets, exp_res_offsets)


def test_group_load_small():
    gsize = 8
    lsize = 8

    res = []

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr: CurrentGroup, gsize: sym.G, lsize: sym.L, arr):
        a = gr.load(arr[gr.work_offset[0] :], shape=gr.shape[0])
        res.append(a.compressed())

    src = np.arange(12)
    test(gsize, lsize, src)
    expected = [np.array([0, 1, 2, 3, 4, 5, 6, 7])]
    assert_equal(res, expected)


def test_group_load():
    gsize = 16
    lsize = 8

    res = []

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr: CurrentGroup, gsize: sym.G, lsize: sym.L, arr):
        a = gr.load(arr[gr.work_offset[0] :], shape=gr.shape[0])
        res.append(a.compressed())

    src = np.arange(12)
    test(gsize, lsize, src)
    expected = [np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([8, 9, 10, 11])]
    assert_equal(res, expected)


def test_group_store():
    gsize = 16
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L, arr1, arr2):
        gid = gr.work_offset
        a = gr.load(arr1[gid[0] :], shape=gr.shape[0])
        gr.store(arr2[gid[0] :], a)

    src = np.arange(12)
    dst = np.full(16, fill_value=-1)
    test(gsize, lsize, src, dst)
    assert_equal(dst, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, -1, -1, -1])


def test_group_empty():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.empty((3, 7), dtype=np.int32)
        assert_equal(a.shape, (3, 7))

    test(gsize, lsize)


def test_group_zeros():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.zeros((2, 3), dtype=np.int32)
        assert_equal(a, [[0, 0, 0], [0, 0, 0]])

    test(gsize, lsize)


def test_group_ones():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.ones((2, 3), dtype=np.int32)
        assert_equal(a, [[1, 1, 1], [1, 1, 1]])

    test(gsize, lsize)


def test_group_full():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.full((2, 3), dtype=np.int32, fill_value=42)
        assert_equal(a, [[42, 42, 42], [42, 42, 42]])

    test(gsize, lsize)


def test_group_vempty():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.vempty((3, 7), dtype=np.int32)
        assert_equal(a.shape, (3, 7))

    test(gsize, lsize)


def test_group_vzeros():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.vzeros((2, 3), dtype=np.int32)
        assert_equal(a, [[0, 0, 0], [0, 0, 0]])

    test(gsize, lsize)


def test_group_vones():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.vones((2, 3), dtype=np.int32)
        assert_equal(a, [[1, 1, 1], [1, 1, 1]])

    test(gsize, lsize)


def test_group_vfull():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.vfull((2, 3), dtype=np.int32, fill_value=42)
        assert_equal(a, [[42, 42, 42], [42, 42, 42]])

    test(gsize, lsize)


def test_group_vec1():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.full((2, 3), dtype=np.int32, fill_value=42)
        v = gr.vec(a)
        assert_equal(v, [[42, 42, 42], [42, 42, 42]])

    test(gsize, lsize)


def test_group_vec2():
    gsize = 8
    lsize = 8

    @kernel(work_shape=sym.G, group_shape=sym.L)
    def test(gr, gsize: sym.G, lsize: sym.L):
        a = gr.full((2, 3), dtype=np.int32, fill_value=42)
        v = gr.vec(a, shape=(1, 2))
        assert_equal(v, [[42, 42]])

    test(gsize, lsize)


def test_subgroup_iteration1():
    gsize = (1, 1, 16)
    lsize = (1, 1, 8)
    sgsize = 4

    res_ids = []
    res_sizes = []

    G1, G2, G3 = sym.G1, sym.G2, sym.G3
    L1, L2, L3 = sym.L1, sym.L2, sym.L3

    @kernel(work_shape=(G1, G2, G3), group_shape=(L1, L2, L3), subgroup_size=sgsize)
    def test(gr: CurrentGroup, gsize: tuple[G1, G2, G3], lsize: tuple[L1, L2, L3]):
        @gr.subgroups
        def inner(sg: CurrentSubGroup):
            res_ids.append(sg.subgroup_id())
            res_sizes.append(sg.size())

        inner()

    test(gsize, lsize)
    assert_equal(res_ids, [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 1)])
    assert_equal(res_sizes, [4, 4, 4, 4])


def test_subgroup_iteration2():
    gsize = (1, 1, 16)
    lsize = (1, 1, 8)
    sgsize = 4

    res_ids = []
    res_sizes = []

    G1, G2, G3 = sym.G1, sym.G2, sym.G3
    L1, L2, L3 = sym.L1, sym.L2, sym.L3
    SG = sym.SG

    @kernel(
        work_shape=(G1, G2, G3),
        group_shape=(L1, L2, L3),
        subgroup_size=SG,
        literals={SG},
    )
    def test(
        gr: CurrentGroup, gsize: tuple[G1, G2, G3], lsize: tuple[L1, L2, L3], sgsize: SG
    ):
        @gr.subgroups
        def inner(sg: CurrentSubGroup):
            res_ids.append(sg.subgroup_id())
            res_sizes.append(sg.size())

        inner()

    test(gsize, lsize, sgsize)
    assert_equal(res_ids, [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 1)])
    assert_equal(res_sizes, [4, 4, 4, 4])


def test_subgroup_barrier():
    gsize = (1, 1, 16)
    lsize = (1, 1, 8)
    sgsize = 4

    res = []

    G1, G2, G3 = sym.G1, sym.G2, sym.G3
    L1, L2, L3 = sym.L1, sym.L2, sym.L3

    @kernel(work_shape=(G1, G2, G3), group_shape=(L1, L2, L3), subgroup_size=sgsize)
    def test(gr, gsize: tuple[G1, G2, G3], lsize: tuple[L1, L2, L3]):
        @gr.subgroups
        def inner(sg):
            res.append((1, sg.subgroup_id()))
            gr.barrier()
            res.append((2, sg.subgroup_id()))

        inner()

    test(gsize, lsize)
    assert_equal(
        res,
        [
            (1, (0, 0, 0)),
            (1, (0, 0, 1)),
            (2, (0, 0, 0)),
            (2, (0, 0, 1)),
            (1, (0, 0, 0)),
            (1, (0, 0, 1)),
            (2, (0, 0, 0)),
            (2, (0, 0, 1)),
        ],
    )


def test_workitems_iteration():
    gsize = (1, 1, 4)
    lsize = (1, 1, 2)

    res_gids = []
    res_lids = []

    G1, G2, G3 = sym.G1, sym.G2, sym.G3
    L1, L2, L3 = sym.L1, sym.L2, sym.L3

    @kernel(work_shape=(G1, G2, G3), group_shape=(L1, L2, L3))
    def test(gr: CurrentGroup, gsize: tuple[G1, G2, G3], lsize: tuple[L1, L2, L3]):
        @gr.workitems
        def inner(wi: CurrentWorkitem):
            res_lids.append(wi.local_id())
            res_gids.append(wi.global_id())

        inner()

    test(gsize, lsize)
    assert_equal(res_lids, [(0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 1)])
    assert_equal(res_gids, [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)])


def test_workitem_barrier():
    gsize = (1, 1, 4)
    lsize = (1, 1, 2)

    res = []

    G1, G2, G3 = sym.G1, sym.G2, sym.G3
    L1, L2, L3 = sym.L1, sym.L2, sym.L3

    @kernel(work_shape=(G1, G2, G3), group_shape=(L1, L2, L3))
    def test(gr: CurrentGroup, gsize: tuple[G1, G2, G3], lsize: tuple[L1, L2, L3]):
        @gr.workitems
        def inner(wi: CurrentWorkitem):
            res.append((1, wi.global_id()))
            gr.barrier()
            res.append((2, wi.global_id()))

        inner()

    test(gsize, lsize)
    assert_equal(
        res,
        [
            (1, (0, 0, 0)),
            (1, (0, 0, 1)),
            (2, (0, 0, 0)),
            (2, (0, 0, 1)),
            (1, (0, 0, 2)),
            (1, (0, 0, 3)),
            (2, (0, 0, 2)),
            (2, (0, 0, 3)),
        ],
    )


def test_buffer_dims1():
    G = sym.G

    @kernel(work_shape=G)
    def test(gr: CurrentGroup, arr: Buffer[G, DT]):
        @gr.workitems
        def inner(wi: CurrentWorkitem):
            gid = wi.global_id()[0]
            if gid < arr.shape[0]:
                arr[gid] = gid

        inner()

    src = np.zeros(12)
    test(src)
    assert_equal(src, np.arange(12))


def test_buffer_dims2():
    G1 = sym.G1
    G2 = sym.G2

    @kernel(work_shape=(G1, G2))
    def test(gr: CurrentGroup, arr: Buffer[G1, G2, DT]):
        @gr.workitems
        def inner(wi: CurrentWorkitem):
            gid = wi.global_id()[:2]
            if gid[0] < arr.shape[0] and gid[1] < arr.shape[1]:
                arr[gid] = gid[0] * arr.shape[1] + gid[1]

        inner()

    src = np.zeros(12 * 5).reshape(12, 5)
    test(src)
    assert_equal(src, np.arange(12 * 5).reshape(12, 5))


def test_symbol_resolving_freevar():
    G = sym.G

    @kernel(work_shape=G)
    def test(gr: CurrentGroup, arr: Buffer[G, DT]):
        arr[:] = G

    src = np.zeros(12)
    test(src)
    assert_equal(src, np.full(12, 12))


GG = sym.GG


def test_symbol_resolving_global():
    @kernel(work_shape=GG)
    def test(gr: CurrentGroup, arr: Buffer[GG, DT]):
        arr[:] = GG

    src = np.zeros(12)
    test(src)
    assert_equal(src, np.full(12, 12))


def test_symbol_tuning_param_freevar():
    G = sym.G
    TG = TunableParam(G, 5, range(0, 10))

    @kernel(work_shape=10, tunables=TG)
    def test(gr: CurrentGroup, arr: Buffer[10, DT]):
        arr[:] = G

    src = np.zeros(10)
    test(src)
    assert_equal(src, np.full(10, 5))
    test.parametrize({G: 10})(src)
    assert_equal(src, np.full(10, 10))


def test_symbol_tuning_param_global():
    TG = TunableParam(GG, 5, range(0, 10))

    @kernel(work_shape=10, tunables=TG)
    def test(gr: CurrentGroup, arr: Buffer[10, DT]):
        arr[:] = GG

    src = np.zeros(10)
    test(src)
    assert_equal(src, np.full(10, 5))
    test.parametrize({GG: 10})(src)
    assert_equal(src, np.full(10, 10))

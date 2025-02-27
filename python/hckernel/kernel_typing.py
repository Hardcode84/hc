# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .typing import *
from .mlir import ir, typing
from .compiler import EnableDumpIR
from .settings import DUMP_TYPING_IR


_registry = TypingRegistry()


def get_typing_module():
    global _registry
    with EnableDumpIR(DUMP_TYPING_IR):
        _registry.compile_type_resolvers()

    return _registry.module


NoneTyp = ir.NoneType.get()
Index = ir.IndexType.get()
ValueType = typing.ValueType.get()
HCKernelMod = typing.IdentType.get("hckernel")
HCKernelAPI = typing.IdentType.get("hckernel.kernel_api")
Indexing = typing.IdentType.get("hckernel.indexing")
Typename = typing.IdentType.get("hckernel.typename")
BufferBase = typing.IdentType.get("hckernel.kernel_api.BufferBase")
CurrentGroup1 = typing.IdentType.get("hckernel.kernel_api.CurrentGroup1")
CurrentGroup2 = typing.IdentType.get("hckernel.kernel_api.CurrentGroup2")
CurrentGroup3 = typing.IdentType.get("hckernel.kernel_api.CurrentGroup3")

GroupLoad = typing.IdentType.get("hckernel.kernel_api.CurrentGroup.load")
GroupVLoad = typing.IdentType.get("hckernel.kernel_api.CurrentGroup.vload")
GroupStore = typing.IdentType.get("hckernel.kernel_api.CurrentGroup.store")


@func
def check_type(a: ValueType, b: ValueType):
    check(is_same(a, b))


@func
def check_is_ident(t: ValueType):
    check(is_same(get_metatype_name(t), "typing.ident"))


@func
def check_is_symbolic(t: ValueType):
    check(
        is_same(get_metatype_name(t), "typing.expr")
        or is_same(get_metatype_name(t), "typing.literal")
        or is_same(get_metatype_name(t), "typing.symbol")
    )


@func
def is_tuple(t: ValueType):
    return is_same(get_type_name(t), "Tuple")


@func
def check_is_tuple(t: ValueType):
    check_is_ident(t)
    check(is_tuple(t))


@func
def check_is_buffer(t: ValueType):
    check_is_ident(t)
    check(is_same(get_type_name(t), "Buffer"))


@func
def check_is_tensor(t: ValueType):
    check_is_ident(t)
    check(is_same(get_type_name(t), "Tensor"))


@func
def check_is_vector(t: ValueType):
    check_is_ident(t)
    check(is_same(get_type_name(t), "Vector"))


@func
def check_is_shaped(t: ValueType):
    check_is_ident(t)
    check(
        is_same(get_type_name(t), "Buffer")
        or is_same(get_type_name(t), "Tensor")
        or is_same(get_type_name(t), "Vector")
    )


@func
def check_is_current_group(t: ValueType):
    check(
        is_same(t, CurrentGroup1)
        or is_same(t, CurrentGroup2)
        or is_same(t, CurrentGroup3)
    )


@func
def make_tuple2(a: ValueType, b: ValueType):
    seq = create_seq()
    seq = append_seq(seq, a)
    seq = append_seq(seq, b)
    return make_type("Tuple", elements=seq)


@func
def make_tuple3(a: ValueType, b: ValueType, c: ValueType):
    seq = create_seq()
    seq = append_seq(seq, a)
    seq = append_seq(seq, b)
    seq = append_seq(seq, c)
    return make_type("Tuple", elements=seq)


@type_resolver(_registry, ["py_ir.load_module", "hckernel"])
def resolver():
    return HCKernelMod


@type_resolver(_registry, ["py_ir.getattr", "kernel_api"])
def resolver(a: ValueType):
    check_type(a, HCKernelMod)
    return HCKernelAPI


@type_resolver(_registry, ["py_ir.getattr", "indexing"])
def resolver(a: ValueType):
    check_type(a, HCKernelMod)
    return Indexing


@type_resolver(_registry, ["py_ir.getattr", "typename"])
def resolver(a: ValueType):
    check_type(a, HCKernelMod)
    return Typename


@type_resolver(_registry, ["py_ir.getattr"])
def resolver(a: ValueType):
    check_type(a, Indexing)
    return make_symbol(get_attr("name"))


@type_resolver(_registry, ["py_ir.getattr"])
def resolver(a: ValueType):
    check_type(a, Typename)
    return make_type("Typename", name=get_attr("name"))


@type_resolver(_registry, ["py_ir.tuple_pack"])
def resolver(a: ValueType):
    count = get_num_args()
    i = 0
    seq = create_seq()
    while i < count:
        seq = append_seq(seq, get_arg(i))
        i += 1

    return make_type("Tuple", elements=seq)


@type_resolver(_registry, ["py_ir.tuple_unpack"])
def resolver(a: ValueType):
    check_is_tuple(a)
    return unpack_seq(get_type_param(a, "elements"))


@type_resolver(_registry, ["py_ir.getitem"])
def resolver(target: ValueType, index: ValueType):
    check_type(target, BufferBase)
    check_is_tuple(index)
    elements = get_type_param(index, "elements")
    dims = create_seq()
    dims_len = get_seq_size(elements) - 1
    i = 0
    while i < dims_len:
        dims = append_seq(dims, get_seq_element(elements, i))
        i += 1

    dtype = get_seq_element(elements, i)
    return make_type("Buffer", dims=dims, dtype=dtype)


@type_resolver(_registry, ["py_ir.getattr", "Buffer"])
def resolver(a: ValueType):
    check_type(a, HCKernelAPI)
    return BufferBase


@type_resolver(_registry, ["py_ir.getattr", "CurrentGroup1"])
def resolver(a: ValueType):
    check_type(a, HCKernelAPI)
    return CurrentGroup1


@type_resolver(_registry, ["py_ir.getattr", "CurrentGroup2"])
def resolver(a: ValueType):
    check_type(a, HCKernelAPI)
    return CurrentGroup2


@type_resolver(_registry, ["py_ir.getattr", "CurrentGroup3"])
def resolver(a: ValueType):
    check_type(a, HCKernelAPI)
    return CurrentGroup3


@type_resolver(_registry, ["py_ir.getattr", "work_offset"])
def resolver(a: ValueType):
    check_is_current_group(a)
    off = get_global_attr("kernel.work_offset")
    return make_type("Tuple", elements=off)


@type_resolver(_registry, ["py_ir.getattr", "shape"])
def resolver(a: ValueType):
    check_is_current_group(a)
    shape = get_global_attr("kernel.group_shape")
    return make_type("Tuple", elements=shape)


@type_resolver(_registry, ["py_ir.getitem"])
def resolver(target: ValueType, index: ValueType):
    check_is_tuple(target)
    idx = to_int(index)
    elements = get_type_param(target, "elements")
    # TODO: check(idx >= 0 and idx < size)
    return get_seq_element(elements, idx)


@type_resolver(_registry, ["py_ir.slice"])
def resolver():
    lower = get_named_arg("lower")
    upper = get_named_arg("upper")
    step = get_named_arg("step")
    return make_type("Slice", lower=lower, upper=upper, step=step)


@func
def getitem_typing(dims: ValueType, index: ValueType):
    if is_tuple(index):
        indices = get_type_param(index, "elements")
    else:
        indices = create_seq()
        indices = append_seq(indices, index)

    res_dims = create_seq()
    indices_len = get_seq_size(indices)
    i = 0
    dim_idx = 0
    while i < indices_len:
        elem = get_seq_element(indices, i)
        if is_same(elem, NoneTyp):
            res_dims = append_seq(res_dims, 1)
        elif is_same(get_type_name(elem), "Slice"):
            lower = get_type_param(elem, "lower")
            # TODO: check for literals
            if is_same(lower, NoneTyp):
                lower = 0

            upper = get_type_param(elem, "upper")
            if is_same(upper, NoneTyp):
                upper = get_seq_element(dims, dim_idx)

            size = upper - lower
            res_dims = append_seq(res_dims, size)
            dim_idx += 1
        else:
            dim_idx += 1

        i += 1

    return res_dims


@type_resolver(_registry, ["py_ir.getitem"])
def resolver(target: ValueType, index: ValueType):
    check_is_buffer(target)
    return make_type(
        "Buffer",
        dims=getitem_typing(get_type_param(target, "dims"), index),
        dtype=get_type_param(target, "dtype"),
    )


@type_resolver(_registry, ["py_ir.getitem"])
def resolver(target: ValueType, index: ValueType):
    check_is_tensor(target)
    return make_type(
        "Tensor",
        dims=getitem_typing(get_type_param(target, "dims"), index),
        dtype=get_type_param(target, "dtype"),
    )


@type_resolver(_registry, ["py_ir.getattr", "shape"])
def resolver(target: ValueType):
    check_is_shaped(target)
    dims = get_type_param(target, "dims")
    return make_type("Tuple", elements=dims)


@type_resolver(_registry, ["py_ir.binop"])
def resolver(lhs: ValueType, rhs: ValueType):
    check_is_symbolic(lhs)
    check_is_symbolic(rhs)
    op = get_attr("op")
    if is_same(op, 2):
        return lhs * rhs
    check(False)
    return lhs


@type_resolver(_registry, ["py_ir.binop"])
def resolver(lhs: ValueType, rhs: ValueType):
    check_is_tensor(lhs)
    # TODO: broadcasting
    check(is_same(lhs, rhs))
    return lhs


@type_resolver(_registry, ["py_ir.getattr", "load"])
def resolver(target: ValueType):
    check_is_current_group(target)
    return GroupLoad


@type_resolver(_registry, ["py_ir.getattr", "vload"])
def resolver(target: ValueType):
    check_is_current_group(target)
    return GroupVLoad


@type_resolver(_registry, ["py_ir.call"])
def resolver(func: ValueType):
    check_type(func, GroupLoad)
    buffer = get_named_arg("array")
    if is_same(buffer, NoneTyp):
        buffer = get_arg(1)

    shape = get_named_arg("shape")
    if is_same(shape, NoneTyp):
        shape = get_arg(2)
    check_is_tuple(shape)
    elements = get_type_param(shape, "elements")
    return make_type("Tensor", dims=elements, dtype=get_type_param(buffer, "dtype"))


@type_resolver(_registry, ["py_ir.call"])
def resolver(func: ValueType):
    check_type(func, GroupVLoad)
    buffer = get_named_arg("array")
    if is_same(buffer, NoneTyp):
        buffer = get_arg(1)

    shape = get_named_arg("shape")
    if is_same(shape, NoneTyp):
        shape = get_arg(2)
    check_is_tuple(shape)
    elements = get_type_param(shape, "elements")
    return make_type("Vector", dims=elements, dtype=get_type_param(buffer, "dtype"))


@type_resolver(_registry, ["py_ir.getattr", "store"])
def resolver(target: ValueType):
    check_is_current_group(target)
    return GroupStore


@type_resolver(_registry, ["py_ir.call"])
def resolver(func: ValueType):
    check_type(func, GroupStore)
    dst = get_named_arg("dst")
    if is_same(dst, NoneTyp):
        dst = get_arg(1)

    check_is_shaped(dst)

    src = get_named_arg("src")
    if is_same(src, NoneTyp):
        src = get_arg(2)

    check_is_shaped(src)
    return NoneTyp

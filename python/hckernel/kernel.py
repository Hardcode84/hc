# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sympy
import inspect

from .convert_expr import convert_sympy_expr
from .dispatcher import create_dispatcher
from .indexing import _index_symbol_internal
from .kernel_api import *
from .kernel_api import _verify_kernel_params
from .mlir import ir
from .mlir import typing


def _get_caller_locals():
    frame = inspect.currentframe()
    if frame is None:
        return {}

    res = frame.f_back.f_back.f_locals
    if res is None:
        ret = {}

    return res


def _get_num_dims(arg):
    if not isinstance(arg, Iterable):
        return 1

    return len(arg)


def _resolve_globals(func, mapping):
    old_closure = func.__closure__
    new_closure = None

    def resolve_mapping(val):
        try:
            return mapping[val]
        except:
            pass

        return None

    if old_closure is not None:
        cell_cls = type(old_closure[0])

        def resolve_cell(cell):
            new_val = resolve_mapping(cell.cell_contents)
            if new_val is not None:
                return cell_cls(new_val)

            return cell

        new_closure = tuple([resolve_cell(cell) for cell in old_closure])

    def resolve_global(val):
        new_val = resolve_mapping(val)
        if new_val is not None:
            return new_val

        return val

    new_globals = {key: resolve_global(val) for key, val in func.__globals__.items()}

    g = types.FunctionType(
        func.__code__,
        new_globals,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=new_closure,
    )
    g = functools.update_wrapper(g, func)
    g.__kwdefaults__ = func.__kwdefaults__
    return g


def _get_typing_module():
    from .kernel_typing import get_typing_module

    return get_typing_module()


def _get_shape_attr(src):
    seq = typing.SequenceType.get([convert_sympy_expr(s) for s in src])
    return typing.TypeAttr.get(seq)


def _get_symbol_attr(sym):
    return typing.TypeAttr.get(convert_sympy_expr(sym))


def _get_global_attrs(
    work_shape, group_shape, subgroup_size, literals, backend, device
):
    ret = {}
    n = len(work_shape)
    if group_shape is None:
        group_shape = tuple(_index_symbol_internal(f"GROUP_SHAPE{i}") for i in range(n))
    elif not isinstance(group_shape, (list, tuple)):
        group_shape = (group_shape,)

    ret["kernel.work_shape"] = _get_shape_attr(work_shape)

    group_id = tuple(_index_symbol_internal(f"GROUP_ID{i}") for i in range(n))
    group_count = tuple(sympy.ceiling(w / g) for w, g in zip(work_shape, group_shape))
    work_offset = tuple(i * s for i, s in zip(group_id, group_shape))

    ret["kernel.group_shape"] = _get_shape_attr(group_shape)
    ret["kernel.group_count"] = _get_shape_attr(group_count)
    ret["kernel.group_id"] = _get_shape_attr(group_id)
    ret["kernel.work_offset"] = _get_shape_attr(work_offset)

    local_id = tuple(_index_symbol_internal(f"LOCAL_ID{i}") for i in range(n))

    if subgroup_size is None:
        subgroup_size = _index_symbol_internal("SUBGROUP_SIZE")

    ret["kernel.subgroup_size"] = _get_symbol_attr(subgroup_size)
    ret["kernel.subgroup_id"] = _get_symbol_attr(local_id[-1] // subgroup_size)

    ret["kernel.local_id"] = _get_shape_attr(local_id)

    ret["kernel.backend"] = backend
    ret["kernel.device"] = device

    return ret


def kernel(
    work_shape,
    group_shape=None,
    subgroup_size=None,
    literals=(),
    tunables=(),
    backend="rocm",
    device="hip:0",
):
    _verify_kernel_params(work_shape, group_shape, subgroup_size, literals, tunables)

    def _kernel_impl(func):
        caller_vars = _get_caller_locals() | func.__globals__

        gr_index = _get_num_dims(work_shape) - 1
        new_current_group = [CurrentGroup1, CurrentGroup2, CurrentGroup3][gr_index]
        mapping = {CurrentGroup: new_current_group}
        caller_vars = {
            k: (new_current_group if v is CurrentGroup else v)
            for k, v in caller_vars.items()
        }

        new_func = _resolve_globals(func, mapping)
        attrs = _get_global_attrs(
            work_shape=work_shape,
            group_shape=group_shape,
            subgroup_size=subgroup_size,
            literals=literals,
            backend=backend,
            device=device,
        )
        return create_dispatcher(
            new_func,
            prelink_module=_get_typing_module,
            global_attrs=attrs,
            caller_vars=caller_vars,
        )

    return _kernel_impl

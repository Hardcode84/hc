# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sympy
import inspect
from collections import namedtuple
from .kernel_api import _verify_kernel_params
from .kernel_api import *
from .dispatcher import create_dispatcher
from .indexing import _index_symbol_internal
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


def _get_literal(value):
    attr = ir.IntegerAttr.get(ir.IndexType.get(), int(value))
    return typing.LiteralType.get(attr)


def _get_typing_module():
    from .kernel_typing import get_typing_module

    return get_typing_module()


_Rational = namedtuple("_Rational", ["numerator", "denominator"])


def _get_symbol_type(expr):
    stack = []

    def addi(lhs, rhs):
        return lhs + rhs

    def muli(lhs, rhs):
        return lhs * rhs

    def _add(lhs, rhs):
        is_rational_lhs = isinstance(lhs, _Rational)
        is_rational_rhs = isinstance(rhs, _Rational)
        if is_rational_lhs and not is_rational_rhs:
            numerator = muli(lhs.denominator, rhs)
            numerator = addi(numerator, lhs.numerator)
            return _Rational(numerator, lhs.denominator)
        elif not is_rational_lhs and is_rational_rhs:
            numerator = muli(lhs, rhs.denominator)
            numerator = addi(numerator, rhs.numerator)
            return _Rational(numerator, rhs.denominator)
        elif is_rational_lhs and is_rational_rhs:
            lhs_numerator = muli(lhs.numerator, rhs.denominator)
            rhs_numerator = muli(rhs.numerator, lhs.denominator)
            numerator = addi(lhs_numerator, rhs_numerator)
            denominator = muli(lhs.denominator, rhs.denominator)
            return _Rational(numerator, denominator)
        else:
            return addi(lhs, rhs)

    # `x * (a/b)` transformed into `(x * a) / b`
    def _mul(lhs, rhs):
        is_rational_lhs = isinstance(lhs, _Rational)
        is_rational_rhs = isinstance(rhs, _Rational)
        if is_rational_lhs and not is_rational_rhs:
            numerator = muli(lhs.numerator, rhs)
            return _Rational(numerator, lhs.denominator)
        elif not is_rational_lhs and is_rational_rhs:
            numerator = muli(lhs, rhs.numerator)
            return _Rational(numerator, rhs.denominator)
        elif is_rational_lhs and is_rational_rhs:
            numerator = muli(lhs.numerator, rhs.numerator)
            denominator = muli(lhs.denominator, rhs.denominator)
            return _Rational(numerator, denominator)
        else:
            return muli(lhs, rhs)

    def _floor(value):
        if isinstance(value, _Rational):
            value = value.numerator // value.denominator

        return value

    def _ceiling(value):
        if isinstance(value, _Rational):
            value = value.numerator / value.denominator

        return value

    def _group_rationals(stack, count):
        """Group rationals and non-rationals args into 2 contiguous sets.

        This allows to mul/add all non-rationals first, reducing total number of ops.
        """
        rationals = []
        non_rationals = []
        for _ in range(count):
            val = stack.pop()
            if isinstance(val, _Rational):
                rationals.append(val)
            else:
                non_rationals.append(val)

        return non_rationals + rationals

    def _apply(args, func):
        assert len(args) > 0
        value = args[0]
        for val in args[1:]:
            value = func(value, val)

        return value

    def _enforce_non_rational(val, term):
        if isinstance(val, _Rational):
            raise ValueError(f"Rational is not supported yet in '{type(term)}'")

    if not isinstance(expr, sympy.Expr):
        expr = sympy.sympify(expr)

    for term in sympy.postorder_traversal(expr):
        match term:
            case sympy.Symbol():
                stack.append(typing.SymbolType.get(term.name))
            case sympy.Integer():
                stack.append(_get_literal(int(term)))
            case sympy.Mul():
                args = _group_rationals(stack, len(term.args))
                stack.append(_apply(args, _mul))
            case sympy.Add():
                args = _group_rationals(stack, len(term.args))
                stack.append(_apply(args, _add))
            case sympy.Mod():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                mod = lhs % rhs
                stack.append(mod)
            case sympy.Pow():
                assert term.args[1] == -1, f"Only -1 power is supported, got {p}"
                p = stack.pop()
                val = stack.pop()
                stack.append(_Rational(_get_literal(1), val))
            case sympy.floor():
                stack.append(_floor(stack.pop()))
            case sympy.ceiling():
                stack.append(_ceiling(stack.pop()))
            case sympy.Rational():
                numerator = _get_literal(term.p)
                denominator = _get_literal(term.q)
                stack.append(_Rational(numerator, denominator))
            case sympy.UnevaluatedExpr():
                continue
            case _:
                raise ValueError(f"Can not handle {type(term)} : {term}")

    if len(stack) != 1 or isinstance(stack[0], _Rational):
        raise ValueError(f"Expected single result, got {len(stack)}: {stack}")

    return stack[0]


def _get_shape_attr(src):
    seq = typing.SequenceType.get([_get_symbol_type(s) for s in src])
    return typing.TypeAttr.get(seq)


def _get_symbol_attr(sym):
    sym = _get_symbol_type(sym)
    return typing.TypeAttr.get(sym)


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

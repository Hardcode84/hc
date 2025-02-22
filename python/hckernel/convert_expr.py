# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sympy
from collections import namedtuple

from .mlir import ir
from .mlir import typing


_Rational = namedtuple("_Rational", ["numerator", "denominator"])


def _get_literal(value):
    attr = ir.IntegerAttr.get(ir.IndexType.get(), int(value))
    return typing.LiteralType.get(attr)


def convert_sympy_expr(expr):
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

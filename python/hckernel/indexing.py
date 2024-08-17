# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, ClassVar, Optional, Type, TypeVar, Union, cast
from .symbol_registry import register_symbol as _reg_symbol_impl
import sympy

IndexSymbol = sympy.core.Symbol
IndexExpr = sympy.core.Expr


def index_symbol(name: str) -> IndexSymbol:
    sym = sympy.Symbol(name, integer=True)
    _reg_symbol_impl(sym, name, __name__, overwrite=True)
    return sym


def _index_symbol_internal(name: str) -> IndexSymbol:
    name = "$" + name
    sym = sympy.Symbol(name, integer=True)
    _reg_symbol_impl(sym, name, __name__, overwrite=True)
    return sym


def index_expr(value: Any) -> IndexExpr:
    expr = sympy.sympify(value)
    return expr


class _IndexSymbolExpando:
    def __getattr__(self, n):
        s = index_symbol(n)
        return s


sym = _IndexSymbolExpando()

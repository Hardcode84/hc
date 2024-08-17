# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .symbol_registry import register_symbol as _reg_symbol_impl


_typenames = {}


class Typename:
    def __init__(self, name):
        self.name = name


class _TypenameExpando:
    def __getattr__(self, name):
        res = _typenames.get(name, None)
        if res:
            return res

        res = Typename(name)
        _typenames[name] = res
        _reg_symbol_impl(res, name, __name__, overwrite=True)
        return res


typename = _TypenameExpando()

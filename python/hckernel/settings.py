# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .utils import readenv as _readenv_orig

settings = {}


def _readenv(name, ctor, default):
    res = _readenv_orig(name, ctor, default)
    settings[name[len("HC_") :]] = res
    return res


def _split_str(src):
    return list(filter(None, str(src).split(",")))


DUMP_AST = _readenv("HC_DUMP_AST", int, 0)
DUMP_IR = _readenv("HC_DUMP_IR", int, 0)
DEBUG_TYPE = _readenv("HC_DEBUG_TYPE", _split_str, [])

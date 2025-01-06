# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .utils import readenv as _readenv_orig
from os import path
from os import environ as env
import ctypes

settings = {}


def _readenv(name, ctor, default):
    res = _readenv_orig(name, ctor, default)
    settings[name[len("HC_") :]] = res
    return res


def _split_str(src):
    return list(filter(None, str(src).split(",")))


DUMP_AST = _readenv("HC_DUMP_AST", int, 0)
DUMP_IR = _readenv("HC_DUMP_IR", int, 0)
DUMP_TYPING_IR = _readenv("HC_DUMP_TYPING_IR", int, 0)
DUMP_LLVM = _readenv("HC_DUMP_LLVM", int, 0)
DUMP_OPTIMIZED = _readenv("HC_DUMP_OPTIMIZED", int, 0)
DUMP_ASSEMBLY = _readenv("HC_DUMP_ASSEMBLY", int, 0)
DEBUG_TYPE = _readenv("HC_DEBUG_TYPE", _split_str, [])
settings["JIT_SYMBOLS"] = {}


def register_cfunc(lib, name):
    cfunc = getattr(lib, name)
    ptr = ctypes.cast(cfunc, ctypes.c_void_p)
    settings["JIT_SYMBOLS"][name] = int(ptr.value)


_llvm_bin_path = path.join(path.dirname(__file__), "_native")
settings["LLVM_BIN_PATH"] = _llvm_bin_path

if "ROCM_PATH" not in env:
    env["ROCM_PATH"] = _llvm_bin_path

# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType

from ._native.compiler import create_context, Dispatcher
from ._native.compiler import enable_dump_ast as _enable_dump_ast
from ._native.compiler import enable_dump_ir as _enable_dump_ir
from ._native.compiler import enable_dump_llvm as _enable_dump_llvm
from ._native.compiler import enable_dump_opt_llvm as _enable_dump_opt_llvm
from ._native.compiler import enable_dump_asm as _enable_dump_asm
from .settings import DUMP_AST, DUMP_IR, DUMP_LLVM, DUMP_OPTIMIZED, DUMP_ASSEMBLY
from .settings import settings as _settings
from . import py_runtime

mlir_context = create_context(_settings)


def enable_dump_ast(enable):
    return _enable_dump_ast(mlir_context, bool(enable))


def enable_dump_ir(enable):
    return _enable_dump_ir(mlir_context, bool(enable))


def enable_dump_llvm(enable):
    return _enable_dump_llvm(mlir_context, bool(enable))


def enable_dump_opt_llvm(enable):
    return _enable_dump_opt_llvm(mlir_context, bool(enable))


def enable_dump_asm(enable):
    return _enable_dump_asm(mlir_context, bool(enable))


enable_dump_ast(DUMP_AST)
enable_dump_ir(DUMP_IR)
enable_dump_llvm(DUMP_LLVM)
enable_dump_opt_llvm(DUMP_OPTIMIZED)
enable_dump_asm(DUMP_ASSEMBLY)


class EnableDumpIR:
    def __init__(self, enable):
        self._prev = enable_dump_ir(enable)

    def __enter__(self):
        return None

    def __exit__(self, type, value, traceback):
        enable_dump_ir(self._prev)

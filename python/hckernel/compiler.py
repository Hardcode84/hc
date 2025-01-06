# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType

from ._native.compiler import create_context, Dispatcher
from ._native.compiler import enable_dump_ast as _enable_dump_ast
from ._native.compiler import enable_dump_ir as _enable_dump_ir
from .settings import DUMP_AST, DUMP_IR
from .settings import settings as _settings
from . import py_runtime

mlir_context = create_context(_settings)


def enable_dump_ast(enable):
    return _enable_dump_ast(mlir_context, bool(enable))


def enable_dump_ir(enable):
    return _enable_dump_ir(mlir_context, bool(enable))


enable_dump_ast(DUMP_AST)
enable_dump_ir(DUMP_IR)

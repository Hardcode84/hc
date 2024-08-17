# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from types import FunctionType

from ._native.compiler import create_context, Dispatcher
from .settings import settings as _settings

mlir_context = create_context(_settings)

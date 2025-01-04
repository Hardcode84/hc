# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
from .utils import load_lib
from .settings import register_cfunc

runtime_lib = load_lib("hc-python-runtime")

register_cfunc(runtime_lib, "hcgpuConvertPyArray")

register_cfunc(runtime_lib, "hcgpuGetKernel")
register_cfunc(runtime_lib, "hcgpuSuggestBlockSize")
register_cfunc(runtime_lib, "hcgpuLaunchKernel")

_enable_tracing = runtime_lib.hcgpuEnableTraceFunctions
_enable_tracing.argtypes = [ctypes.c_int]


def enable_func_tracing(val):
    _enable_tracing(ctypes.c_int(val))

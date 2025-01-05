# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
from .utils import load_lib, get_runtime_search_paths
from .settings import register_cfunc

runtime_lib = load_lib("hc-python-runtime")

register_cfunc(runtime_lib, "hcgpuConvertPyArray")

register_cfunc(runtime_lib, "hcgpuGetKernel")
register_cfunc(runtime_lib, "hcgpuSuggestBlockSize")
register_cfunc(runtime_lib, "hcgpuLaunchKernel")

_enable_tracing = runtime_lib.hcgpuEnableTraceFunctions
_enable_tracing.argtypes = [ctypes.c_int]

_set_runtime_search_paths = runtime_lib.hcgpuSetRuntimeSearchPaths
_set_runtime_search_paths.argtypes = [ctypes.py_object]

_search_paths = [str.encode(s) for s in get_runtime_search_paths()]
_set_runtime_search_paths(ctypes.py_object(_search_paths))


def enable_func_tracing(val):
    _enable_tracing(ctypes.c_int(val))

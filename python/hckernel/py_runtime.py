# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .utils import load_lib
from .settings import register_cfunc

runtime_lib = load_lib("hc-python-runtime")

register_cfunc(runtime_lib, "hcgpuGetPyArg")
register_cfunc(runtime_lib, "hcgpuConvertPyArray")

register_cfunc(runtime_lib, "hcgpuGetKernel")
register_cfunc(runtime_lib, "hcgpuSuggestBlockSize")
register_cfunc(runtime_lib, "hcgpuLaunchKernel")

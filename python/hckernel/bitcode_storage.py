# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from ._native import compiler as _compiler

_bitcode_path = os.path.dirname(_compiler.__file__)


def get_bitcode_file(name):
    return os.path.join(_bitcode_path, f"{name}.mlirbc")

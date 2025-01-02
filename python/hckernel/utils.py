# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import warnings


def readenv(name, ctor, default):
    value = os.environ.get(name, None)
    if value is None:
        return default() if callable(default) else default
    try:
        return ctor(value)
    except Exception:
        warnings.warn(
            f"environ {name} defined but failed to parse '{value}'",
            RuntimeWarning,
        )
        return default

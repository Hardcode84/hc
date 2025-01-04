# SPDX-FileCopyrightText: 2024 The HC Authors
# SPDX-FileCopyrightText: 2025 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
import os
import sys
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


def get_runtime_search_paths():
    path = os.path
    runtime_search_paths = [path.join(path.dirname(__file__), "_native")]

    try:
        runtime_search_paths += os.environ["PYTHONPATH"].split(os.pathsep)
    except KeyError:
        pass

    return runtime_search_paths


def load_lib(name):
    runtime_search_paths = get_runtime_search_paths()

    platform = sys.platform
    if platform.startswith("linux"):
        lib_name = f"lib{name}.so"
    elif platform.startswith("darwin"):
        lib_name = f"lib{name}.dylib"
    elif platform.startswith("win"):
        lib_name = f"{name}.dll"
    else:
        assert False, f"Unsupported platform: {platform}"

    saved_errors = []
    for path in runtime_search_paths:
        lib_path = lib_name if len(path) == 0 else os.path.join(path, lib_name)
        try:
            return ctypes.CDLL(lib_path)
        except Exception as e:
            saved_errors.append(f'CDLL("{lib_path}"): {str(e)}')

    raise ValueError(f'load_lib("{name}") failed:\n' + "\n".join(saved_errors))

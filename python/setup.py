# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess
from setuptools import setup, find_packages
import shutil

root_dir = os.path.dirname(os.path.abspath(__file__))
cmake_build_dir = os.path.join(root_dir, "build")
cmake_dir = os.path.join(root_dir, "..")
install_dir = os.path.join(root_dir, "hckernel", "_native")

env = os.environ


def get_env(env_name):
    env_val = env.get(env_name)

    if env_val is None:
        raise RuntimeError(f"Enviroment variable '{env_name}' is not set")

    return env_val


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def rmdir(path):
    shutil.rmtree(path, ignore_errors=True)


def clear_dir(path):
    rmdir(path)
    mkdir(path)


LLVM_PATH = get_env("LLVM_PATH")
LLVM_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "llvm")
MLIR_DIR = os.path.join(LLVM_PATH, "lib", "cmake", "mlir")

mkdir(cmake_build_dir)
clear_dir(install_dir)


def invoke_cmake(args):
    subprocess.check_call(
        ["cmake"] + args,
        stderr=subprocess.STDOUT,
        shell=False,
        cwd=cmake_build_dir,
        env=env,
    )


invoke_cmake(
    [
        cmake_dir,
        "-GNinja",
        "-DCMAKE_INSTALL_PREFIX=" + install_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLVM_DIR=" + LLVM_DIR,
        "-DMLIR_DIR=" + MLIR_DIR,
        "-DHC_ENABLE_PYTHON=ON",
        "-DHC_ENABLE_TOOLS=ON",  # need hc-opt to convert mlir files to bytecode
        "-DHC_ENABLE_TESTS=OFF",
    ]
)
invoke_cmake(["--build", ".", "--config", "Release"])
invoke_cmake(["--install", ".", "--config", "Release"])

setup(
    name="hckernel",
    version="0.1",
    packages=find_packages(where=root_dir, include=["hckernel", "hckernel.*"]),
)

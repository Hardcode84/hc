# SPDX-FileCopyrightText: 2024 The HC Authors
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include_directories(${MLIR_INCLUDE_DIRS})

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
mlir_tablegen(Transforms.capi.h.inc -gen-pass-capi-header --prefix Transforms)
mlir_tablegen(Transforms.capi.cpp.inc -gen-pass-capi-impl --prefix Transforms)
add_public_tablegen_target(HCTransformsPassIncGen)

add_mlir_doc(Passes HCGeneralPasses ./ -gen-pass-doc)

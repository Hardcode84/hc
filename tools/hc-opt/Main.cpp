// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <hc/InitHCDialects.hpp>
#include <hc/InitHCPasses.hpp>

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  hc::registerAllPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  hc::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  hc::registerAllExtensions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "HC modular optimizer driver\n", registry));
}

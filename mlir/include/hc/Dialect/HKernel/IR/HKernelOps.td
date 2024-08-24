// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HKERNEL_OPS
#define HKERNEL_OPS

include "mlir/IR/OpAsmInterface.td"

def HKernel_Dialect : Dialect {
  let name = "hkernel";
  let cppNamespace = "::hc::hk";

  let summary = "TBD.";
  let description = [{
    TBD
  }];

//  let useDefaultAttributePrinterParser = 1;
//  let useDefaultTypePrinterParser = 1;
}

#endif // HKERNEL_OPS

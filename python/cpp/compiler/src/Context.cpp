// SPDX-FileCopyrightText: 2024 The HC Authors
// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "Context.hpp"

#include "hc/InitHCDialects.hpp"

#include "PyWrappers.hpp"

#include <mlir/InitAllExtensions.h>
#include <mlir/Target/LLVM/NVVM/Target.h>
#include <mlir/Target/LLVM/ROCDL/Target.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>

#include <llvm/ExecutionEngine/Orc/Mangling.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/StringSaver.h>
#include <llvm/Support/TargetSelect.h>

namespace py = nanobind;

static mlir::DialectRegistry createRegistry() {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  hc::registerAllExtensions(registry);
  mlir::ROCDL::registerROCDLTargetInterfaceExternalModels(registry);
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  return registry;
}

static hc::ExecutionEngineOptions
getExecutionEngineOpts(const py::dict &settings) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  llvm::SmallVector<std::pair<std::string, void *>, 0> symMap;
  for (auto &&[key, val] : py::cast<py::dict>(settings["JIT_SYMBOLS"])) {
    auto name = py::str(key).c_str();
    auto ptr = reinterpret_cast<void *>(py::cast<intptr_t>(val));
    symMap.emplace_back(name, ptr);
  }

  hc::ExecutionEngineOptions opts;
  opts.symbolMap = [syms = std::move(symMap)](
                       llvm::orc::MangleAndInterner m) -> llvm::orc::SymbolMap {
    llvm::orc::SymbolMap ret;
    for (auto &&[name, ptr] : syms) {
      llvm::orc::ExecutorSymbolDef jitPtr{llvm::orc::ExecutorAddr::fromPtr(ptr),
                                          llvm::JITSymbolFlags::Exported};
      ret.insert({m(name), jitPtr});
    }
    return ret;
  };
  opts.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;

  opts.lateTransformer = [](llvm::Module &m) -> llvm::Error {
    m.dump();
    return llvm::Error::success();
  };
  //    auto llvmPrinter = settings["llvm_printer"];
  //    if (!llvmPrinter.is_none())
  //      opts.transformer = getLLModulePrinter(llvmPrinter);

  //    auto optimizedPrinter = settings["optimized_printer"];
  //    if (!optimizedPrinter.is_none())
  //      opts.lateTransformer = getLLModulePrinter(optimizedPrinter);

  //    auto asmPrinter = settings["asm_printer"];
  //    if (!asmPrinter.is_none())
  //      opts.asmPrinter = getPrinter(asmPrinter);

  return opts;
}

static void readSettings(Settings &ret, py::dict &settings) {
  ret.dumpAST = py::cast<int>(settings["DUMP_AST"]);
  ret.dumpIR = py::cast<int>(settings["DUMP_IR"]);
}

Context::Context(nanobind::dict settings_)
    : context(createRegistry()),
      executionEngine(getExecutionEngineOpts(settings_)) {
  context.loadDialect<hc::py_ir::PyIRDialect, hc::typing::TypingDialect>();

  readSettings(settings, settings_);
  //  llvmBinPath = toString(settings_["LLVM_BIN_PATH"]);
  pushContext(&context);
}

Context::~Context() { popContext(&context); }

static void readDebugTypes(py::dict &settings) {
  auto debugType = py::cast<py::list>(settings["DEBUG_TYPE"]);
  auto debugTypeSize = debugType.size();
  if (debugTypeSize != 0) {
    llvm::DebugFlag = true;
    llvm::BumpPtrAllocator alloc;
    auto types = alloc.Allocate<const char *>(debugTypeSize);
    llvm::StringSaver strSaver(alloc);
    for (auto i : llvm::seq<size_t>(0, debugTypeSize))
      types[i] = strSaver.save(toString(debugType[i])).data();

    llvm::setCurrentDebugTypes(types, static_cast<unsigned>(debugTypeSize));
  }
}

py::capsule createContext(py::dict settings) {
  auto ctx = std::make_unique<Context>(settings);
  readDebugTypes(settings);
  auto dtor = [](void *ptr) noexcept { delete static_cast<Context *>(ptr); };
  py::capsule ret(ctx.get(), dtor);
  ctx.release();
  return ret;
}

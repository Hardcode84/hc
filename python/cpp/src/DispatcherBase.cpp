// SPDX-FileCopyrightText: 2024 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "DispatcherBase.hpp"

#include <llvm/ADT/Twine.h>
#include <mlir/CAPI/IR.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

#include "hc/Dialect/HKernel/IR/HKernelOps.hpp"
#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Transforms/ModuleLinker.hpp"

#include "CompilerFront.hpp"
#include "Context.hpp"
#include "Utils.hpp"

#include "IRModule.h"

namespace py = nanobind;

template <> struct std::iterator_traits<nanobind::iterator> {
  using value_type = nanobind::handle;
  using reference = const value_type;
  using pointer = void;
  using difference_type = std::ptrdiff_t;
  using iterator_category = std::forward_iterator_tag;
};

DispatcherBase::DispatcherBase(py::capsule ctx, py::object getDesc)
    : context(*static_cast<Context *>(ctx.data())), contextRef(std::move(ctx)),
      getFuncDesc(std::move(getDesc)),
      argsHandlerBuilder(
          std::make_unique<ArgsHandlerBuilder>(context.context)) {}

DispatcherBase::~DispatcherBase() {}

void DispatcherBase::definePyClass(py::module_ &m) {
  py::class_<DispatcherBase>(m, "DispatcherBase");
}

static std::pair<std::string, std::string> getSource(py::handle desc) {
  return {py::cast<std::string>(desc.attr("source")),
          py::cast<std::string>(desc.attr("name"))};
}

static mlir::Attribute translateLiteral(mlir::MLIRContext *ctx,
                                        py::handle obj) {
  mlir::OpBuilder builder(ctx);
  if (py::isinstance<py::int_>(obj))
    return builder.getI64IntegerAttr(py::cast<int64_t>(obj));

  if (py::isinstance<py::float_>(obj))
    return builder.getF64FloatAttr(py::cast<double>(obj));

  if (py::isinstance<mlir::python::PyType>(obj)) {
    auto t = py::cast<mlir::python::PyType>(obj);
    return hc::typing::TypeAttr::get(unwrap(t.get()));
  }

  reportError(llvm::Twine("Unsupported literal type: ") +
              py::cast<std::string>(py::str(obj)));
}

static mlir::OwningOpRef<mlir::Operation *> importImpl(Context &context,
                                                       py::handle desc) {
  auto [src, funcName] = getSource(desc);

  llvm::SmallVector<ImportedSym> symbols;
  for (auto &&[name, val] : py::cast<py::dict>(desc.attr("imported_symbols"))) {
    ImportedSym sym;
    sym.name = py::cast<std::string>(name);
    for (auto path : val)
      sym.modulePath.emplace_back(py::cast<std::string>(path));

    symbols.emplace_back(std::move(sym));
  }

  auto *mlirContext = &context.context;
  llvm::SmallVector<Literal> literals;
  for (auto &&[name, val] : py::cast<py::dict>(desc.attr("literals"))) {
    Literal lit;
    lit.name = py::cast<std::string>(name);
    lit.attr = translateLiteral(mlirContext, val);
    literals.emplace_back(std::move(lit));
  }

  auto res = compileAST(context, src, funcName, symbols, literals);
  if (mlir::failed(res))
    reportError("AST import failed");

  mlir::OwningOpRef newMod = res->release();
  auto prelink = desc.attr("prelink_module");
  if (!prelink.is_none()) {
    auto prelinkMod = unwrap(py::cast<MlirModule>(prelink));
    mlir::OwningOpRef preMod = prelinkMod->clone();
    if (mlir::failed(hc::linkModules(preMod.get(), newMod.get())))
      reportError("Module linking failed");

    newMod = std::move(preMod);
  }
  auto globalAttrs = desc.attr("global_attrs");
  if (!globalAttrs.is_none()) {
    for (auto &&[key, val] : py::cast<py::dict>(globalAttrs)) {
      auto keyAttr =
          mlir::StringAttr::get(mlirContext, py::cast<std::string>(key));
      auto attr = unwrap(py::cast<mlir::python::PyAttribute>(val));
      newMod->setAttr(keyAttr, attr);
    }
  }
  return newMod;
}

static hc::py_ir::PyModuleOp getIRModImpl(mlir::Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;

  mlir::Region &reg = op->getRegion(0);
  if (!llvm::hasSingleElement(reg))
    return nullptr;

  mlir::Block &block = reg.front();
  auto ops = block.getOps<hc::py_ir::PyModuleOp>();
  if (ops.empty())
    return nullptr;

  return *ops.begin();
}

static hc::py_ir::PyModuleOp getIRMod(mlir::Operation *op) {
  auto ret = getIRModImpl(op);
  if (!ret)
    reportError("no python IR module");

  return ret;
}

static mlir::Value getModResult(hc::py_ir::PyModuleOp mod) {
  auto term =
      mlir::cast<hc::py_ir::PyModuleEndOp>(mod.getBody()->getTerminator());
  mlir::ValueRange results = term.getResults();
  if (results.size() != 1)
    reportError("Invalid results count");

  return results.front();
}

static void getModuleDeps(
    hc::py_ir::PyModuleOp irMod, const py::dict &deps,
    llvm::SmallVectorImpl<std::pair<DispatcherBase *, mlir::Operation *>>
        &unresolved) {
  for (mlir::Operation &op : irMod.getBody()->without_terminator()) {
    auto loadVar = mlir::dyn_cast<hc::py_ir::LoadVarOp>(op);
    if (!loadVar)
      continue;

    auto name = loadVar.getName();
    py::str pyName(name.data(), name.size());
    if (deps.contains(pyName)) {
      auto &disp = py::cast<DispatcherBase &>(deps[pyName]);
      unresolved.emplace_back(&disp, &op);
    }
  }
}

void DispatcherBase::linkModules(mlir::Operation *rootModule,
                                 const py::dict &currentDeps) {
  auto irMod = getIRMod(rootModule);

  llvm::SmallVector<mlir::OwningOpRef<mlir::Operation *>> modules;
  llvm::SmallDenseMap<DispatcherBase *, mlir::Value> modMap;
  modMap.try_emplace(this, getModResult(irMod));

  llvm::SmallVector<std::pair<DispatcherBase *, mlir::Operation *>> deps;
  getModuleDeps(irMod, currentDeps, deps);
  size_t currentDep = 0;
  while (currentDep < deps.size()) {
    auto &&[dispatcher, op] = deps[currentDep++];
    if (!modMap.contains(dispatcher)) {
      auto mod = dispatcher->importFuncForLinking(deps);
      modMap.try_emplace(dispatcher, getModResult(getIRMod(mod.get())));
      modules.emplace_back(std::move(mod));
    }
  }

  mlir::IRRewriter builder(rootModule->getContext());
  mlir::Block *dstBlock = irMod.getBody();
  for (auto &mod : modules) {
    auto pyIr = getIRMod(mod.get());
    mlir::Block *srcBlock = pyIr.getBody();
    builder.eraseOp(srcBlock->getTerminator());
    builder.inlineBlockBefore(srcBlock, dstBlock, dstBlock->begin());
  }

  mlir::DominanceInfo dom;
  for (auto &&[disp, op] : deps) {
    auto it = modMap.find(disp);
    assert(it != modMap.end());
    mlir::Value resolvedSym = it->second;
    if (!dom.properlyDominates(resolvedSym, op))
      reportError("Circular module dependency");

    assert(op->getNumResults() == 1);
    op->replaceAllUsesWith(mlir::ValueRange(resolvedSym));
  }
}

mlir::Operation *DispatcherBase::runFrontend() {
  if (!mod) {
    assert(getFuncDesc);
    py::object desc = getFuncDesc();
    auto newMod = importImpl(context, desc);
    runPipeline(context, newMod.get(),
                [this](mlir::PassManager &pm) { populateImportPipeline(pm); });

    linkModules(newMod.get(), py::cast<py::dict>(desc.attr("dispatcher_deps")));

    runPipeline(context, newMod.get(), [this](mlir::PassManager &pm) {
      populateFrontendPipeline(pm);
    });

    mod = std::move(newMod);

    populateArgsHandlers(desc.attr("args"));
  }
  return mod.get();
}

void DispatcherBase::invokeFunc(const py::args &args,
                                const py::kwargs &kwargs) {
  llvm::SmallVector<PyObject *, 16> funcArgs;
  mlir::Attribute key = processArgs(args, kwargs, funcArgs);
  const void *keyPtr = key.getAsOpaquePointer();
  auto it = funcsCache.find(keyPtr);
  if (it == funcsCache.end()) {
    OpRef newMod = mod->clone();
    newMod->setAttr(hc::hk::getKernelMetadataAttrName(), key);
    runPipeline(context, newMod.get(),
                [this](mlir::PassManager &pm) { populateInvokePipeline(pm); });
    newMod.get()->dump();
    reportError("TODO: compile");

    // TODO: codegen
    it = funcsCache.insert({keyPtr, nullptr}).first;
  }

  auto func = it->second;
  ExceptionDesc exc;
  if (func(&exc, funcArgs.data()) != 0)
    reportError(exc.message);
}

using HandlerT =
    std::function<void(mlir::MLIRContext &, py::handle,
                       llvm::SmallMapVector<mlir::Type, mlir::Type, 8> &,
                       llvm::SmallVectorImpl<PyObject *> &)>;

template <typename T> static std::string toStr(T &&val) {
  std::string ret;
  llvm::raw_string_ostream os(ret);
  os << val;
  os.flush();
  return ret;
}

static void updateRetMap(llvm::SmallMapVector<mlir::Type, mlir::Type, 8> &ret,
                         mlir::Type key, mlir::Type val) {
  auto it = ret.find(key);
  if (it == ret.end()) {
    ret.insert({key, val});
    return;
  }

  auto oldVal = it->second;
  if (oldVal != val)
    reportError(llvm::Twine("Metadata conflict for ") + toStr(key) + ": " +
                toStr(oldVal) + " and " + toStr(val));
}

// static bool issubclass(py::handle cls, py::handle parent) {
//   const auto result = PyObject_IsSubclass(cls.ptr(), parent.ptr());
//   if (result == -1)
//     throw py::python_error();
//   return result != 0;
// }

template <unsigned Width, bool Signed>
static mlir::Type getIntType(mlir::MLIRContext *ctx) {
  return mlir::IntegerType::get(ctx, Width,
                                Signed ? mlir::IntegerType::Signed
                                       : mlir::IntegerType::Unsigned);
}

template <unsigned Width>
static mlir::Type getFloatType(mlir::MLIRContext *ctx) {
  switch (Width) {
  case 16:
    return mlir::FloatType::getF16(ctx);
  case 32:
    return mlir::FloatType::getF32(ctx);
  case 64:
    return mlir::FloatType::getF64(ctx);
  }
  llvm_unreachable("Invalid float width");
}

static void reportWrongDtype(py::handle dt) {
  reportError(llvm::Twine("Invalid dtype: ") +
              py::cast<std::string>(py::str(dt)));
};

struct DispatcherBase::ArgsHandlerBuilder {
  ArgsHandlerBuilder(mlir::MLIRContext &c) : ctx(c) {
    auto mod = py::module_::import_("hckernel.kernel_api");
    symbolType = mod.attr("Symbol");
    typenameType = mod.attr("Typename");
    bufferType = mod.attr("Buffer");

    auto np = py::module_::import_("numpy");
    for (auto &&[src, dst] : llvm::zip_equal(NumpyDTypes, numpyDTypes)) {
      auto &&[name, func] = src;

      dst = std::pair(py::cast<py::object>(np.attr(name.data())), func(&ctx));
    }
  }

  HandlerT getArgHandler(py::handle arg) {
    auto getSym = [&](py::handle a) -> mlir::Type {
      return hc::typing::SymbolType::get(&ctx,
                                         py::cast<std::string>(a.attr("name")));
    };

    if (py::isinstance(arg, symbolType)) {
      auto sym = getSym(arg);
      return [sym](mlir::MLIRContext &ctx, py::handle obj,
                   llvm::SmallMapVector<mlir::Type, mlir::Type, 8> &ret,
                   llvm::SmallVectorImpl<PyObject *> &args) {
        if (py::isinstance<py::int_>(obj)) {
          updateRetMap(ret, sym, mlir::IntegerType::get(&ctx, 64));
        } else if (py::isinstance<py::float_>(obj)) {
          updateRetMap(ret, sym, mlir::Float64Type::get(&ctx));
        } else {
          reportError(llvm::Twine("Unsupported type: ") +
                      py::cast<std::string>(py::str(obj)));
        }
        args.emplace_back(obj.ptr());
      };
    }
    if (py::isinstance<py::tuple>(arg)) {
      auto count = py::len(arg);
      llvm::SmallVector<HandlerT, 0> handlers;
      handlers.reserve(count);
      for (auto elem : arg)
        handlers.emplace_back(getArgHandler(elem));

      return [handlersCopy = std::move(handlers)](
                 mlir::MLIRContext &ctx, py::handle obj,
                 llvm::SmallMapVector<mlir::Type, mlir::Type, 8> &ret,
                 llvm::SmallVectorImpl<PyObject *> &args) {
        for (auto &&[h, elem] : llvm::zip_equal(handlersCopy, obj))
          h(ctx, elem, ret, args);
      };
    }
    if (issubclass(arg, bufferType)) {
      auto shape = arg.attr("shape");
      auto dtype = arg.attr("dtype");
      llvm::SmallVector<mlir::Type> srcShape(
          py::len(py::cast<py::handle>(shape)));
      for (auto &&[i, s] : llvm::enumerate(shape)) {
        if (py::isinstance<py::int_>(s)) {
          // Nothing
        } else if (py::isinstance(s, symbolType)) {
          // srcShape[i] = getSym(s);
          // TODO: Handle literals
        } else {
          reportError(llvm::Twine("Unsupported dim type: ") +
                      py::cast<std::string>(py::str(s)));
        }
      }

      mlir::Type dtypeSym;
      if (py::isinstance(dtype, typenameType)) {
        dtypeSym = getSym(dtype);
      } else {
        dtypeSym = translateDtype(dtype);
      }

      if (!dtypeSym)
        reportWrongDtype(dtype);

      return [this, argShape = std::move(srcShape),
              dtypeSym](mlir::MLIRContext &ctx, py::handle obj,
                        llvm::SmallMapVector<mlir::Type, mlir::Type, 8> &ret,
                        llvm::SmallVectorImpl<PyObject *> &args) {
        auto arrInterface = py::cast<py::dict>(obj.attr("__array_interface__"));
        auto shape = py::cast<py::tuple>(arrInterface["shape"]);
        if (argShape.size() != shape.size())
          reportError("Invalid buffer rank: " + llvm::Twine(argShape.size()) +
                      " vs " + llvm::Twine(shape.size()));

        // TODO: Handle literals
        //        for (auto &&[i, s] : llvm::enumerate(argShape)) {
        //          if (s == kDynamic) {
        //            resShape[i] = kDynamic;
        //          } else {
        //            resShape[i] = s;
        //          }
        //        }

        auto dtypeObj = obj.attr("dtype");
        auto dtype = translateDtype(dtypeObj);
        if (!dtype)
          reportWrongDtype(dtypeObj);

        if (mlir::isa<hc::typing::SymbolType>(dtypeSym)) {
          updateRetMap(ret, dtypeSym, dtype);
        } else {
          if (dtypeSym != dtype)
            reportError(llvm::Twine("dtype mismatch: ") + toStr(dtypeSym) +
                        " vs " + toStr(dtype));
        }
        // TODO: layout
        args.emplace_back(obj.ptr());
      };
    }

    reportError(llvm::Twine("Unsupported arg type: ") +
                py::cast<std::string>(py::str(arg)));
  }

private:
  using fptr_t = mlir::Type (*)(mlir::MLIRContext *);
  constexpr static const std::pair<llvm::StringRef, fptr_t> NumpyDTypes[] = {
      {"int8", &getIntType<8, true>},   {"uint8", &getIntType<8, false>},
      {"int16", &getIntType<16, true>}, {"uint16", &getIntType<16, false>},
      {"int32", &getIntType<32, true>}, {"uint32", &getIntType<32, false>},
      {"int64", &getIntType<64, true>}, {"uint64", &getIntType<64, false>},
      {"float16", &getFloatType<16>},   {"float32", &getFloatType<32>},
      {"float64", &getFloatType<64>},
  };

  mlir::MLIRContext &ctx;
  py::object symbolType;
  py::object typenameType;
  py::object bufferType;
  std::array<std::pair<py::object, mlir::Type>, std::size(NumpyDTypes)>
      numpyDTypes;

  mlir::Type translateDtype(py::handle obj) const {
    for (auto &&[dtype, type] : numpyDTypes) {
      if (obj.equal(dtype))
        return type;
    }
    return nullptr;
  }
};

void DispatcherBase::populateArgsHandlers(py::handle args) {
  auto &ctx = context.context;
  assert(argsHandlers.empty());
  argsHandlers.reserve(py::len(args));
  for (auto [name, elem] : py::cast<py::dict>(args)) {
    auto nameAttr = mlir::StringAttr::get(&ctx, py::cast<std::string>(name));
    auto handler = argsHandlerBuilder->getArgHandler(elem);
    argsHandlers.emplace_back(ArgDesc{nameAttr.getValue(), std::move(handler)});
  }
}

mlir::Attribute
DispatcherBase::processArgs(const py::args &args, const py::kwargs &kwargs,
                            llvm::SmallVectorImpl<PyObject *> &retArgs) const {
  auto srcNumArgs = args.size();
  bool hasKWArgs = kwargs.size() > 0;
  auto getKWArg = [&](llvm::StringRef name) -> py::handle {
    if (!hasKWArgs)
      return nullptr;

    py::str n(name.data(), name.size());
    if (kwargs.contains(n))
      return kwargs[n];

    return nullptr;
  };

  llvm::SmallMapVector<mlir::Type, mlir::Type, 8> metadata;

  auto &ctx = context.context;
  size_t idx = 0;
  for (auto &arg : argsHandlers) {
    auto name = arg.name;
    if (auto kwarg = getKWArg(name)) {
      arg.handler(ctx, kwarg, metadata, retArgs);
      continue;
    }
    if (idx >= srcNumArgs)
      reportError("Insufficient args");

    auto srcArg = args[idx++];
    arg.handler(context.context, srcArg, metadata, retArgs);
  }

  llvm::SmallVector<mlir::Attribute> array;
  array.reserve(metadata.size() * 2);
  for (auto &[key, val] : metadata) {
    array.emplace_back(hc::typing::TypeAttr::get(key));
    array.emplace_back(hc::typing::TypeAttr::get(val));
  }

  return mlir::ArrayAttr::get(&ctx, array);
}

DispatcherBase::OpRef DispatcherBase::importFuncForLinking(
    llvm::SmallVectorImpl<std::pair<DispatcherBase *, mlir::Operation *>>
        &unresolved) {
  assert(getFuncDesc);
  py::object desc = getFuncDesc();
  auto ret = importImpl(context, desc);

  runPipeline(context, ret.get(),
              [this](mlir::PassManager &pm) { populateImportPipeline(pm); });

  auto deps = py::cast<py::dict>(desc.attr("dispatcher_deps"));
  auto irMod = getIRMod(ret.get());
  getModuleDeps(irMod, deps, unresolved);
  return ret;
}

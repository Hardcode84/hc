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

#include "hc/Dialect/PyIR/IR/PyIROps.hpp"
#include "hc/Dialect/Typing/IR/TypingOps.hpp"
#include "hc/Transforms/ModuleLinker.hpp"

#include "CompilerFront.hpp"
#include "Context.hpp"
#include "Utils.hpp"

#include "IRModule.h"

namespace py = pybind11;

DispatcherBase::DispatcherBase(py::capsule ctx, py::object getDesc)
    : context(*ctx.get_pointer<Context>()), contextRef(std::move(ctx)),
      getFuncDesc(std::move(getDesc)) {}

DispatcherBase::~DispatcherBase() {}

void DispatcherBase::definePyClass(py::module_ &m) {
  py::class_<DispatcherBase>(m, "DispatcherBase");
}

static std::pair<std::string, std::string> getSource(py::handle desc) {
  return {desc.attr("source").cast<std::string>(),
          desc.attr("name").cast<std::string>()};
}

static mlir::Attribute translateLiteral(mlir::MLIRContext *ctx,
                                        py::handle obj) {
  mlir::OpBuilder builder(ctx);
  if (py::isinstance<py::int_>(obj))
    return builder.getI64IntegerAttr(obj.cast<int64_t>());

  if (py::isinstance<py::float_>(obj))
    return builder.getF64FloatAttr(obj.cast<double>());

  if (py::isinstance<mlir::python::PyType>(obj)) {
    auto t = py::cast<mlir::python::PyType>(obj);
    return hc::typing::TypeAttr::get(unwrap(t.get()));
  }

  reportError(llvm::Twine("Unsupported literal type: ") +
              py::str(obj).cast<std::string>());
}

static mlir::OwningOpRef<mlir::Operation *> importImpl(Context &context,
                                                       py::handle desc) {
  auto [src, funcName] = getSource(desc);

  llvm::SmallVector<ImportedSym> symbols;
  for (auto &&[name, val] : desc.attr("imported_symbols").cast<py::dict>()) {
    ImportedSym sym;
    sym.name = name.cast<std::string>();
    for (auto path : val)
      sym.modulePath.emplace_back(path.cast<std::string>());

    symbols.emplace_back(std::move(sym));
  }

  auto *mlirContext = &context.context;
  llvm::SmallVector<Literal> literals;
  for (auto &&[name, val] : desc.attr("literals").cast<py::dict>()) {
    Literal lit;
    lit.name = name.cast<std::string>();
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
    for (auto &&[key, val] : globalAttrs.cast<py::dict>()) {
      auto keyAttr =
          mlir::StringAttr::get(mlirContext, key.cast<std::string>());
      auto attr = unwrap(val.cast<mlir::python::PyAttribute>());
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
      auto &disp = deps[pyName].cast<DispatcherBase &>();
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

    linkModules(newMod.get(), desc.attr("dispatcher_deps").cast<py::dict>());

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
  mlir::Type key = processArgs(args, kwargs, funcArgs);
  auto it = funcsCache.find(key);
  if (it == funcsCache.end()) {
    OpRef newMod = mod->clone();
    runPipeline(context, newMod.get(),
                [this](mlir::PassManager &pm) { populateInvokePipeline(pm); });

    // TODO: codegen
    it = funcsCache.insert({key, nullptr}).first;
  }

  auto func = it->second;
  ExceptionDesc exc;
  if (func(&exc, funcArgs.data()) != 0)
    reportError(exc.message);
}

using HandlerT = std::function<void(mlir::MLIRContext &, pybind11::handle,
                                    llvm::SmallVectorImpl<mlir::Type> &,
                                    llvm::SmallVectorImpl<PyObject *> &)>;

static const constexpr int64_t kDynamic = mlir::ShapedType::kDynamic;
static const constexpr int64_t kLiteral = kDynamic + 1;

static HandlerT getArgHandler(py::handle arg) {
  py::str sym("sym");
  if (arg.equal(sym)) {
    return [](mlir::MLIRContext &ctx, py::handle obj,
              llvm::SmallVectorImpl<mlir::Type> &ret,
              llvm::SmallVectorImpl<PyObject *> &args) {
      if (py::isinstance<py::int_>(obj)) {
        ret.emplace_back(mlir::IntegerType::get(&ctx, 64));
      } else if (py::isinstance<py::float_>(obj)) {
        ret.emplace_back(mlir::Float64Type::get(&ctx));
      } else {
        reportError(llvm::Twine("Unsupported type") +
                    py::str(obj).cast<std::string>());
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

    return [handlersCopy =
                std::move(handlers)](mlir::MLIRContext &ctx, py::handle obj,
                                     llvm::SmallVectorImpl<mlir::Type> &ret,
                                     llvm::SmallVectorImpl<PyObject *> &args) {
      for (auto &&[h, elem] : llvm::zip_equal(handlersCopy, obj))
        h(ctx, elem, ret, args);
    };
  }
  if (py::isinstance<py::list>(arg)) {
    py::str lit("lit");
    llvm::SmallVector<int64_t> srcShape(py::len(arg) - 1);
    for (auto &&[i, s] : llvm::enumerate(arg)) {
      if (i == srcShape.size()) {
        // TODO: get dtype
      } else if (py::isinstance<py::int_>(s)) {
        srcShape[i] = s.cast<int64_t>();
      } else if (s.equal(sym)) {
        srcShape[i] = kDynamic;
      } else if (s.equal(lit)) {
        srcShape[i] = kLiteral;
      } else {
        reportError(llvm::Twine("Unsupported dim type: ") +
                    py::str(s).cast<std::string>());
      }
    }

    return [argShape =
                std::move(srcShape)](mlir::MLIRContext &ctx, py::handle obj,
                                     llvm::SmallVectorImpl<mlir::Type> &ret,
                                     llvm::SmallVectorImpl<PyObject *> &args) {
      auto arrInterface = obj.attr("__array_interface__").cast<py::dict>();
      auto shape = arrInterface["shape"].cast<py::tuple>();
      if (argShape.size() != shape.size())
        reportError("Invalid buffer rank");

      llvm::SmallVector<int64_t> resShape(argShape.size());
      for (auto &&[i, s] : llvm::enumerate(argShape)) {
        if (s == kDynamic) {
          resShape[i] = kDynamic;
        } else if (s == kLiteral) {
          resShape[i] = shape[i].cast<int64_t>();
        } else {
          resShape[i] = s;
        }
      }

      // TODO: dtype
      // TODO: layout
      auto dtype = mlir::Float64Type::get(&ctx);
      ret.emplace_back(mlir::MemRefType::get(resShape, dtype));
      args.emplace_back(obj.ptr());
    };
  }

  reportError(llvm::Twine("Unsupported arg type: ") +
              py::str(arg).cast<std::string>());
}

void DispatcherBase::populateArgsHandlers(pybind11::handle args) {
  auto &ctx = context.context;
  assert(argsHandlers.empty());
  argsHandlers.reserve(py::len(args));
  for (auto [name, elem] : args.cast<py::dict>()) {
    auto nameAttr = mlir::StringAttr::get(&ctx, name.cast<std::string>());
    auto handler = getArgHandler(elem);
    argsHandlers.emplace_back(ArgDesc{nameAttr.getValue(), std::move(handler)});
  }
}

mlir::Type
DispatcherBase::processArgs(const pybind11::args &args,
                            const pybind11::kwargs &kwargs,
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

  llvm::SmallVector<mlir::Type> types;
  types.reserve(argsHandlers.size());

  auto &ctx = context.context;
  size_t idx = 0;
  for (auto &arg : argsHandlers) {
    auto name = arg.name;
    if (auto kwarg = getKWArg(name)) {
      arg.handler(ctx, kwarg, types, retArgs);
      continue;
    }
    if (idx >= srcNumArgs)
      reportError("Insufficient args");

    auto srcArg = args[idx++];
    arg.handler(context.context, srcArg, types, retArgs);
  }

  if (types.empty())
    return nullptr;

  if (types.size() == 1)
    return types.front();

  return mlir::TupleType::get(&ctx);
}

DispatcherBase::OpRef DispatcherBase::importFuncForLinking(
    llvm::SmallVectorImpl<std::pair<DispatcherBase *, mlir::Operation *>>
        &unresolved) {
  assert(getFuncDesc);
  py::object desc = getFuncDesc();
  auto ret = importImpl(context, desc);

  runPipeline(context, ret.get(),
              [this](mlir::PassManager &pm) { populateImportPipeline(pm); });

  auto deps = desc.attr("dispatcher_deps").cast<py::dict>();
  auto irMod = getIRMod(ret.get());
  getModuleDeps(irMod, deps, unresolved);
  return ret;
}

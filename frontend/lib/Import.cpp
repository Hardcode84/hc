// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/PyFront/Import.hpp"

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>

#include "hc/Dialect/PyAST/IR/PyASTOps.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

[[noreturn]] static void reportError(const llvm::Twine &msg) {
  throw std::runtime_error(msg.str());
}

static void initPython() {
  static int i = []() {
    if (Py_IsInitialized())
      return 0;

#ifdef _WIN32
    auto condaEnv = _wgetenv(L"CONDA_PREFIX");
    if (condaEnv && condaEnv[0] != 0)
      Py_SetPythonHome(condaEnv);

#endif
    Py_InitializeEx(0);
    return 0;
  }();
}

static py::str toStr(llvm::StringRef str) {
  return py::str(str.data(), str.size());
}

namespace {
struct ParserState;
using NodeHandlerPtr = void (*)(ParserState &, py::handle);
using HandlerPair = std::pair<py::object, NodeHandlerPtr>;
static void fillHandlers(py::handle astMod,
                         llvm::SmallVectorImpl<HandlerPair> &handlers);

struct ParserState {
  ParserState(mlir::MLIRContext *ctx, py::handle astMod)
      : astModule(astMod.cast<py::object>()), builder(ctx) {
    fillHandlers(astMod, handlersList);
  }

  py::object astModule;
  mlir::OpBuilder builder;

  llvm::SmallVector<HandlerPair, 0> handlersList;

  llvm::SmallVector<mlir::Value> argsStack;
  llvm::SmallVector<mlir::OpBuilder::InsertionGuard> guardsStack;
  llvm::SmallVector<std::pair<py::object, NodeHandlerPtr>> handlersStack;

  hc::py_ast::PyModuleOp rootModule;

  mlir::Location getLoc(py::handle /*node*/) {
    // TODO: get loc
    return builder.getUnknownLoc();
  }

  NodeHandlerPtr getHandler(py::handle node) {
    for (auto &&[cls, handler] : handlersList)
      if (py::isinstance(node, cls))
        return handler;

    auto nodeTypeStr = py::str(node.get_type()).cast<std::string>();
    reportError(llvm::Twine("Unsupported ast node: ") + nodeTypeStr);
  }

  void pushHandler(py::handle node, NodeHandlerPtr handler) {
    handlersStack.emplace_back(node.cast<py::object>(), handler);
  }

  void pushHandler(py::handle node) { pushHandler(node, getHandler(node)); }

  void pushHandlerOptional(py::handle node) {
    if (!node.is_none())
      pushHandler(node);
  }

  void reverseLastNHandlers(int64_t n) {
    assert(handlersStack.size() > = n);
    std::reverse(handlersStack.end() - n, handlersStack.end());
  }

  template <typename C> void pushHandlers(C &&c) {
    int64_t count = 0;
    for (auto node : c) {
      pushHandler(node, getHandler(node));
      ++count;
    }

    reverseLastNHandlers(count);
  }

  void pushGuard() { guardsStack.emplace_back(builder); }
  void popGuard() {
    assert(!guardsStack.empty());
    guardsStack.pop_back();
  }

  void parse(mlir::Block *block, py::handle rootNode) {
    assert(block && "Invalid block");
    builder.setInsertionPointToStart(block);
    pushHandler(rootNode, getHandler(rootNode));

    while (!handlersStack.empty()) {
      auto &&[node, handler] = handlersStack.pop_back_val();
      handler(*this, node);
    }
    assert(argsStack.empty());
    assert(guardsStack.empty());
  }
};

struct ArgHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("arg"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArg);
    auto annot = node.attr("annotation");
    if (!annot.is_none())
      state.pushHandler(annot);
  }

  static void processArg(ParserState &state, py::handle node) {
    auto annot = node.attr("annotation");
    mlir::Value annotVal;
    if (!annot.is_none())
      annotVal = state.argsStack.pop_back_val();

    auto name = node.attr("arg").cast<std::string>();
    auto &builder = state.builder;
    mlir::Value val =
        builder.create<hc::py_ast::ArgOp>(state.getLoc(node), name, annotVal);
    state.argsStack.push_back(val);
  }
};

struct AssignHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("Assign");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandler(node.attr("value"));
    state.pushHandlers(node.attr("targets"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto value = state.argsStack.pop_back_val();

    auto nArgs = py::len(node.attr("targets"));
    mlir::ValueRange args(state.argsStack);
    args = args.take_back(nArgs);

    auto &builder = state.builder;
    builder.create<hc::py_ast::AssignOp>(state.getLoc(node), args, value);
    state.argsStack.pop_back_n(nArgs);
  }
};

struct AttributeHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("Attribute");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArg);
    state.pushHandler(node.attr("value"));
  }

  static void processArg(ParserState &state, py::handle node) {
    auto value = state.argsStack.pop_back_val();
    auto attr = node.attr("attr").cast<std::string>();
    auto &builder = state.builder;
    mlir::Value res = builder.create<hc::py_ast::AttributeOp>(
        state.getLoc(node), value, attr);
    state.argsStack.push_back(res);
  }
};

static hc::py_ast::BinOpVal getBinOpVal(ParserState &state, py::handle obj) {
  using BinOpVal = hc::py_ast::BinOpVal;
  py::handle astMod = state.astModule;
  std::pair<py::handle, BinOpVal> handlers[] = {
      {astMod.attr("Add"), BinOpVal::add},
      {astMod.attr("Sub"), BinOpVal::sub},
      {astMod.attr("Mult"), BinOpVal::mul},
      {astMod.attr("Div"), BinOpVal::div},
      {astMod.attr("FloorDiv"), BinOpVal::floor_div},
      {astMod.attr("Mod"), BinOpVal::mod},
      {astMod.attr("Pow"), BinOpVal::pow},
      {astMod.attr("LShift"), BinOpVal::lshift},
      {astMod.attr("RShift"), BinOpVal::rshift},
      {astMod.attr("BitOr"), BinOpVal::bit_or},
      {astMod.attr("BitXor"), BinOpVal::bit_xor},
      {astMod.attr("BitAnd"), BinOpVal::bit_and},
      {astMod.attr("MatMult"), BinOpVal::matmul},
  };

  for (auto &&[t, op] : handlers)
    if (py::isinstance(obj, t))
      return op;

  reportError(llvm::Twine("unhandled BinOp type \"") +
              py::str(obj.get_type()).cast<std::string>() + "\"");
}

struct AugAssignHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("AugAssign");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandler(node.attr("target"));
    state.pushHandler(node.attr("value"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto target = state.argsStack.pop_back_val();
    auto value = state.argsStack.pop_back_val();

    auto &builder = state.builder;
    builder.create<hc::py_ast::AugAssignOp>(
        state.getLoc(node), target, getBinOpVal(state, node.attr("op")), value);
  }
};

struct BinOpHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("BinOp"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandler(node.attr("right"));
    state.pushHandler(node.attr("left"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto right = state.argsStack.pop_back_val();
    auto left = state.argsStack.pop_back_val();

    auto &builder = state.builder;
    mlir::Value res = builder.create<hc::py_ast::BinOp>(
        state.getLoc(node), left, getBinOpVal(state, node.attr("op")), right);

    state.argsStack.push_back(res);
  }
};

struct BoolOpHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("BoolOp");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandlers(node.attr("values"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto nArgs = py::len(node.attr("values"));
    mlir::ValueRange args(state.argsStack);
    args = args.take_back(nArgs);

    py::handle astMod = state.astModule;
    auto pyOp = node.attr("op");

    using BT = hc::py_ast::BoolOpType;
    BT Op;
    if (py::isinstance(pyOp, astMod.attr("Or"))) {
      Op = BT::or_;
    } else if (py::isinstance(pyOp, astMod.attr("And"))) {
      Op = BT::and_;
    } else {
      reportError(llvm::Twine("unhandled BoolOp op \"") +
                  py::str(pyOp.get_type()).cast<std::string>() + "\"");
    }

    auto &builder = state.builder;
    mlir::Value res =
        builder.create<hc::py_ast::BoolOp>(state.getLoc(node), Op, args);
    state.argsStack.pop_back_n(nArgs);

    state.argsStack.push_back(res);
  }
};

struct BreakHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("Break"); }

  static void parse(ParserState &state, py::handle node) {
    auto &builder = state.builder;
    builder.create<hc::py_ast::BreakOp>(state.getLoc(node));
  }
};

struct CallHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("Call"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandler(node.attr("func"));
    state.pushHandlers(node.attr("keywords"));
    state.pushHandlers(node.attr("args"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto func = state.argsStack.pop_back_val();

    auto nArgs = py::len(node.attr("args"));
    auto nKeywods = py::len(node.attr("keywords"));
    mlir::ValueRange args(state.argsStack);
    auto kwArgs = args.take_back(nKeywods);
    auto posArgs = args.drop_back(nKeywods).take_back(nArgs);

    auto &builder = state.builder;
    mlir::Value res = builder.create<hc::py_ast::CallOp>(state.getLoc(node),
                                                         func, posArgs, kwArgs);
    state.argsStack.pop_back_n(nArgs + nKeywods);

    state.argsStack.push_back(res);
  }
};

struct CompareHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("Compare");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandlers(node.attr("comparators"));
    state.pushHandler(node.attr("left"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto nArgs = py::len(node.attr("comparators"));
    mlir::ValueRange args(state.argsStack);
    args = args.take_back(nArgs);
    auto left = args.drop_back(nArgs).back();

    using CmpOp = hc::py_ast::CmpOp;
    py::handle astMod = state.astModule;
    std::pair<py::handle, CmpOp> cmpHandlers[] = {
        {astMod.attr("Eq"), CmpOp::eq}, {astMod.attr("NotEq"), CmpOp::ne},
        {astMod.attr("Lt"), CmpOp::lt}, {astMod.attr("LtE"), CmpOp::le},
        {astMod.attr("Gt"), CmpOp::gt}, {astMod.attr("GtE"), CmpOp::ge},
        {astMod.attr("Is"), CmpOp::is}, {astMod.attr("IsNot"), CmpOp::isn},
        {astMod.attr("In"), CmpOp::in}, {astMod.attr("NotIn"), CmpOp::nin},
    };

    auto getCmpOp = [&](py::handle obj) -> CmpOp {
      for (auto &&[t, op] : cmpHandlers)
        if (py::isinstance(obj, t))
          return op;

      reportError(llvm::Twine("unhandled cmp type \"") +
                  py::str(obj.get_type()).cast<std::string>() + "\"");
    };

    llvm::SmallVector<CmpOp> ops;
    for (auto op : node.attr("ops"))
      ops.emplace_back(getCmpOp(op));

    auto &builder = state.builder;
    mlir::Value res = builder.create<hc::py_ast::CompareOp>(state.getLoc(node),
                                                            left, ops, args);
    state.argsStack.pop_back_n(nArgs + 1);

    state.argsStack.push_back(res);
  }
};

class dummy_complex : public py::object {
public:
  static bool check_(py::handle h) {
    return h.ptr() != nullptr && PyComplex_Check(h.ptr());
  }
};

struct ConstantHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("Constant");
  }

  static void parse(ParserState &state, py::handle node) {
    auto &builder = state.builder;
    auto val = node.attr("value");
    mlir::Attribute attr;
    if (py::isinstance<py::int_>(val)) {
      attr = builder.getI64IntegerAttr(val.cast<int64_t>());
    } else if (py::isinstance<py::float_>(val)) {
      attr = builder.getF64FloatAttr(val.cast<double>());
    } else if (py::isinstance<dummy_complex>(val)) {
      auto c = val.cast<std::complex<double>>();
      auto type = mlir::ComplexType::get(builder.getF64Type());
      attr = mlir::complex::NumberAttr::get(type, c.real(), c.imag());
    } else if (py::isinstance<py::str>(val)) {
      auto str = val.cast<std::string>();
      attr = builder.getStringAttr(str);
    } else if (py::isinstance<py::none>(val)) {
      attr = hc::py_ast::NoneAttr::get(builder.getContext());
    } else {
      reportError(llvm::Twine("unhandled const type \"") +
                  py::str(val.get_type()).cast<std::string>() + "\"");
    }

    mlir::Value res =
        builder.create<hc::py_ast::ConstantOp>(state.getLoc(node), attr);
    state.argsStack.push_back(res);
  }
};

struct ContinueHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("Continue");
  }

  static void parse(ParserState &state, py::handle node) {
    auto &builder = state.builder;
    builder.create<hc::py_ast::ContinueOp>(state.getLoc(node));
  }
};

struct ExprHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("Expr"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandler(node.attr("value"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto val = state.argsStack.pop_back_val();
    auto &builder = state.builder;
    builder.create<hc::py_ast::ExprOp>(state.getLoc(node), val);
  }
};

struct ForHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("For"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &parseBody);
    auto target = node.attr("target");
    auto iter = node.attr("iter");
    state.pushHandler(target);
    state.pushHandler(iter);
  }

  static void parseBody(ParserState &state, py::handle node) {
    auto hasElse = py::len(node.attr("orelse")) > 0;
    if (hasElse) {
      reportError(llvm::Twine("else statement is not supported for \"for\""
                              " loop"));
    }

    auto &builder = state.builder;

    auto target = state.argsStack.pop_back_val();
    auto iter = state.argsStack.pop_back_val();

    auto op =
        builder.create<hc::py_ast::ForOp>(state.getLoc(node), target, iter);

    state.pushGuard();

    builder.setInsertionPointToStart(op.getBody(0));
    state.pushHandler(node, &popGuard);
    state.pushHandlers(node.attr("body"));
  }

  static void popGuard(ParserState &state, py::handle /*node*/) {
    state.popGuard();
  }
};

struct FuncHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("FunctionDef");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &parseBody);
    state.pushHandlers(node.attr("decorator_list"));
    auto args = node.attr("args").attr("args");
    state.pushHandlers(args);
  }

  static void parseBody(ParserState &state, py::handle node) {
    state.pushGuard();
    auto &builder = state.builder;

    auto nArgs = py::len(node.attr("args").attr("args"));
    auto nDecors = py::len(node.attr("decorator_list"));
    mlir::ValueRange args(state.argsStack);
    auto posArgs = args.drop_back(nDecors).take_back(nArgs);
    auto decors = args.take_back(nDecors);

    auto name = node.attr("name").cast<std::string>();
    auto mod = builder.create<hc::py_ast::PyFuncOp>(state.getLoc(node), name,
                                                    posArgs, decors);
    state.argsStack.pop_back_n(nArgs + nDecors);

    builder.setInsertionPointToStart(mod.getBody());
    state.pushHandler(node, &popGuard);
    state.pushHandlers(node.attr("body"));
  }

  static void popGuard(ParserState &state, py::handle /*node*/) {
    state.popGuard();
  }
};

struct IfExpOpHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("IfExp"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandler(node.attr("test"));
    state.pushHandler(node.attr("body"));
    state.pushHandler(node.attr("orelse"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto test = state.argsStack.pop_back_val();
    auto body = state.argsStack.pop_back_val();
    auto orelse = state.argsStack.pop_back_val();

    auto &builder = state.builder;
    mlir::Value res = builder.create<hc::py_ast::IfExpOp>(state.getLoc(node),
                                                          test, body, orelse);

    state.argsStack.push_back(res);
  }
};

struct IfHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("If"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &parseThenBody);
    auto test = node.attr("test");
    state.pushHandler(test);
  }

  static void parseThenBody(ParserState &state, py::handle node) {
    auto &builder = state.builder;
    auto test = state.argsStack.pop_back_val();
    auto hasElse = py::len(node.attr("orelse")) > 0;
    auto op =
        builder.create<hc::py_ast::IfOp>(state.getLoc(node), test, hasElse);

    state.pushGuard();

    builder.setInsertionPointToStart(op.getBody(0));

    if (hasElse) {
      state.pushHandler(node, &parseElseBody);
      state.pushHandler(py::capsule(static_cast<mlir::Operation *>(op)),
                        &setupElseBody);
    } else {
      state.pushHandler(node, &popGuard);
    }

    state.pushHandlers(node.attr("body"));
  }

  static void setupElseBody(ParserState &state, py::handle node) {
    state.popGuard();
    state.pushGuard();

    auto op = static_cast<mlir::Operation *>(node.cast<py::capsule>());

    auto &builder = state.builder;
    auto &reg = op->getRegion(1);
    assert(!reg.empty());
    builder.setInsertionPointToStart(&reg.front());
  }

  static void parseElseBody(ParserState &state, py::handle node) {
    state.pushHandler(node, &popGuard);
    state.pushHandlers(node.attr("orelse"));
  }

  static void popGuard(ParserState &state, py::handle /*node*/) {
    state.popGuard();
  }
};

struct KeywordHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("keyword");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandler(node.attr("value"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto val = state.argsStack.pop_back_val();
    auto &builder = state.builder;

    auto arg = node.attr("arg").cast<std::string>();
    mlir::Value res =
        builder.create<hc::py_ast::KeywordOp>(state.getLoc(node), arg, val);

    state.argsStack.push_back(res);
  }
};

struct ListHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("List"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandlers(node.attr("elts"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto &builder = state.builder;
    auto nArgs = py::len(node.attr("elts"));
    mlir::ValueRange args(state.argsStack);
    args = args.take_back(nArgs);
    mlir::Value res =
        builder.create<hc::py_ast::ListOp>(state.getLoc(node), args);
    state.argsStack.pop_back_n(nArgs);
    state.argsStack.push_back(res);
  }
};

struct ModuleHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("Module");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushGuard();
    auto &builder = state.builder;
    auto mod = builder.create<hc::py_ast::PyModuleOp>(state.getLoc(node));
    state.rootModule = mod;
    builder.setInsertionPointToStart(mod.getBody());
    state.pushHandler(node, &popGuard);
    state.pushHandlers(node.attr("body"));
  }

  static void popGuard(ParserState &state, py::handle /*node*/) {
    state.popGuard();
  }
};

struct NameHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("Name"); }

  static void parse(ParserState &state, py::handle node) {
    auto &builder = state.builder;
    auto id = node.attr("id").cast<std::string>();
    mlir::Value val =
        builder.create<hc::py_ast::NameOp>(state.getLoc(node), id);
    state.argsStack.push_back(val);
  }
};

struct PassHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("Pass"); }

  static void parse(ParserState &state, py::handle node) {
    auto &builder = state.builder;
    builder.create<hc::py_ast::PassOp>(state.getLoc(node));
  }
};

struct ReturnHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("Return");
  }

  static void parse(ParserState &state, py::handle node) {
    auto value = node.attr("value");
    if (value.is_none()) {
      auto &builder = state.builder;
      builder.create<hc::py_ast::ReturnOp>(state.getLoc(node), nullptr);
      return;
    }

    state.pushHandler(node, &processArgs);
    state.pushHandler(value);
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto val = state.argsStack.pop_back_val();
    auto &builder = state.builder;
    builder.create<hc::py_ast::ReturnOp>(state.getLoc(node), val);
  }
};

struct SliceHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("Slice"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandlerOptional(node.attr("step"));
    state.pushHandlerOptional(node.attr("upper"));
    state.pushHandlerOptional(node.attr("lower"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto popArg = [&](const char *name) -> mlir::Value {
      if (node.attr(name).is_none())
        return nullptr;

      return state.argsStack.pop_back_val();
    };

    auto step = popArg("step");
    auto upper = popArg("upper");
    auto lower = popArg("lower");
    auto &builder = state.builder;
    mlir::Value res = builder.create<hc::py_ast::SliceOp>(state.getLoc(node),
                                                          lower, upper, step);
    state.argsStack.push_back(res);
  }
};

struct SubscriptHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("Subscript");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    py::object args[] = {node.attr("value"), node.attr("slice")};
    state.pushHandlers(args);
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto slice = state.argsStack.pop_back_val();
    auto val = state.argsStack.pop_back_val();
    auto &builder = state.builder;
    mlir::Value res =
        builder.create<hc::py_ast::SubscriptOp>(state.getLoc(node), val, slice);
    state.argsStack.push_back(res);
  }
};

struct TupleHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("Tuple"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandlers(node.attr("elts"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto &builder = state.builder;
    auto nArgs = py::len(node.attr("elts"));
    mlir::ValueRange args(state.argsStack);
    args = args.take_back(nArgs);
    mlir::Value res =
        builder.create<hc::py_ast::TupleOp>(state.getLoc(node), args);
    state.argsStack.pop_back_n(nArgs);
    state.argsStack.push_back(res);
  }
};

struct UnaryOpHandler {
  static py::object getClass(py::handle astMod) {
    return astMod.attr("UnaryOp");
  }

  static hc::py_ast::UnaryOpVal getUnaryOpVal(ParserState &state,
                                              py::handle obj) {
    using UnaryOpVal = hc::py_ast::UnaryOpVal;
    py::handle astMod = state.astModule;
    std::pair<py::handle, UnaryOpVal> handlers[] = {
        {astMod.attr("Invert"), UnaryOpVal::invert},
        {astMod.attr("Not"), UnaryOpVal::unot},
        {astMod.attr("UAdd"), UnaryOpVal::uadd},
        {astMod.attr("USub"), UnaryOpVal::usub},
    };

    for (auto &&[t, op] : handlers)
      if (py::isinstance(obj, t))
        return op;

    reportError(llvm::Twine("unhandled UnaryOp type \"") +
                py::str(obj.get_type()).cast<std::string>() + "\"");
  }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &processArgs);
    state.pushHandler(node.attr("operand"));
  }

  static void processArgs(ParserState &state, py::handle node) {
    auto operand = state.argsStack.pop_back_val();

    auto &builder = state.builder;
    mlir::Value res = builder.create<hc::py_ast::UnaryOp>(
        state.getLoc(node), getUnaryOpVal(state, node.attr("op")), operand);

    state.argsStack.push_back(res);
  }
};

struct WhileHandler {
  static py::object getClass(py::handle astMod) { return astMod.attr("While"); }

  static void parse(ParserState &state, py::handle node) {
    state.pushHandler(node, &parseBody);
    auto test = node.attr("test");
    state.pushHandler(test);
  }

  static void parseBody(ParserState &state, py::handle node) {
    auto hasElse = py::len(node.attr("orelse")) > 0;
    if (hasElse) {
      reportError(llvm::Twine("else statement is not supported for \"while\""
                              " loop"));
    }

    auto &builder = state.builder;

    auto test = state.argsStack.pop_back_val();

    auto op = builder.create<hc::py_ast::WhileOp>(state.getLoc(node), test);

    state.pushGuard();

    builder.setInsertionPointToStart(op.getBody(0));
    state.pushHandler(node, &popGuard);
    state.pushHandlers(node.attr("body"));
  }

  static void popGuard(ParserState &state, py::handle /*node*/) {
    state.popGuard();
  }
};

template <typename T> static HandlerPair getHandler(py::handle astMod) {
  return {T::getClass(astMod), &T::parse};
}

void fillHandlers(
    py::handle astMod,
    llvm::SmallVectorImpl<std::pair<py::object, NodeHandlerPtr>> &handlers) {
  handlers.emplace_back(getHandler<ArgHandler>(astMod));
  handlers.emplace_back(getHandler<AssignHandler>(astMod));
  handlers.emplace_back(getHandler<AttributeHandler>(astMod));
  handlers.emplace_back(getHandler<AugAssignHandler>(astMod));
  handlers.emplace_back(getHandler<BinOpHandler>(astMod));
  handlers.emplace_back(getHandler<BoolOpHandler>(astMod));
  handlers.emplace_back(getHandler<BreakHandler>(astMod));
  handlers.emplace_back(getHandler<CallHandler>(astMod));
  handlers.emplace_back(getHandler<CompareHandler>(astMod));
  handlers.emplace_back(getHandler<ConstantHandler>(astMod));
  handlers.emplace_back(getHandler<ContinueHandler>(astMod));
  handlers.emplace_back(getHandler<ExprHandler>(astMod));
  handlers.emplace_back(getHandler<ForHandler>(astMod));
  handlers.emplace_back(getHandler<FuncHandler>(astMod));
  handlers.emplace_back(getHandler<IfExpOpHandler>(astMod));
  handlers.emplace_back(getHandler<IfHandler>(astMod));
  handlers.emplace_back(getHandler<KeywordHandler>(astMod));
  handlers.emplace_back(getHandler<ListHandler>(astMod));
  handlers.emplace_back(getHandler<ModuleHandler>(astMod));
  handlers.emplace_back(getHandler<NameHandler>(astMod));
  handlers.emplace_back(getHandler<PassHandler>(astMod));
  handlers.emplace_back(getHandler<ReturnHandler>(astMod));
  handlers.emplace_back(getHandler<SliceHandler>(astMod));
  handlers.emplace_back(getHandler<SubscriptHandler>(astMod));
  handlers.emplace_back(getHandler<TupleHandler>(astMod));
  handlers.emplace_back(getHandler<UnaryOpHandler>(astMod));
  handlers.emplace_back(getHandler<WhileHandler>(astMod));
}
} // namespace

static hc::py_ast::PyModuleOp parseModule(py::handle astMod, py::handle ast,
                                          mlir::ModuleOp module) {
  auto ctx = module->getContext();
  ctx->loadDialect<mlir::complex::ComplexDialect>();
  ctx->loadDialect<hc::py_ast::PyASTDialect>();
  ParserState parser(ctx, astMod);

  parser.parse(module.getBody(), ast);
  return parser.rootModule;
}

static mlir::Operation *
importPyModuleImpl(llvm::StringRef str, mlir::ModuleOp module, bool dumpAST) {
  initPython();
  auto mod = py::module::import("ast");
  auto parse = mod.attr("parse");
  auto ast = parse(toStr(str));
  if (dumpAST) {
    llvm::outs() << mod.attr("dump")(ast, "indent"_a = 1).cast<std::string>()
                 << "\n";
  }

  return parseModule(mod, ast, module);
}

mlir::FailureOr<mlir::Operation *>
hc::importPyModule(llvm::StringRef str, mlir::Operation *module, bool dumpAST) {
  try {
    return importPyModuleImpl(str, mlir::cast<mlir::ModuleOp>(module), dumpAST);
  } catch (std::exception &e) {
    module->emitError(e.what());
    return mlir::failure();
  }
}

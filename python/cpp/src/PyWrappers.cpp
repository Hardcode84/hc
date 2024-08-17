// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PyWrappers.hpp"

#include <mlir/CAPI/AffineExpr.h>
#include <mlir/CAPI/IR.h>
#include <mlir/IR/Builders.h>

#include "Globals.h"
#include "IRModule.h"
#include "Pass.h"

#include "hc/Dialect/Typing/IR/TypingOps.hpp"

namespace py = pybind11;
static mlir::python::PyMlirContextRef translateContext(mlir::MLIRContext *ctx) {
  return mlir::python::PyMlirContext::forContext(MlirContext{ctx});
}

void pushContext(mlir::MLIRContext *ctx) {
  translateContext(ctx)->contextEnter();
}

void popContext(mlir::MLIRContext *ctx) {
  translateContext(ctx)->contextExit(py::none(), py::none(), py::none());
}

template <typename T> static bool genTypeIsA(MlirType type) {
  return mlir::isa<T>(unwrap(type));
}

template <typename T> static bool genAttrIsA(MlirAttribute attr) {
  return mlir::isa<T>(unwrap(attr));
}

template <typename T> static MlirTypeID genGetTypeID() {
  return wrap(T::getTypeID());
}

using namespace mlir;
using namespace py::literals;
using namespace mlir::python;

namespace {
class PyValueType : public PyConcreteType<PyValueType> {
public:
  static constexpr IsAFunctionTy isaFunction =
      genTypeIsA<hc::typing::ValueType>;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      genGetTypeID<hc::typing::ValueType>;
  static constexpr const char *pyClassName = "ValueType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](DefaultingPyMlirContext context) {
          MlirType t = wrap(hc::typing::ValueType::get(unwrap(context->get())));
          return PyValueType(context->getRef(), t);
        },
        py::arg("context") = py::none(), "Create typing.value type");
  }
};

class PyIdentType : public PyConcreteType<PyIdentType> {
public:
  static constexpr IsAFunctionTy isaFunction =
      genTypeIsA<hc::typing::IdentType>;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      genGetTypeID<hc::typing::IdentType>;
  static constexpr const char *pyClassName = "IdentType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](py::str name, py::dict params, DefaultingPyMlirContext context) {
          llvm::SmallVector<mlir::StringAttr> paramNames;
          llvm::SmallVector<mlir::Type> paramTypes;

          auto ctx = unwrap(context->get());
          mlir::OpBuilder builder(ctx);
          auto nameAttr = builder.getStringAttr(name.cast<std::string>());
          for (auto &&[key, value] : params) {
            paramNames.emplace_back(
                builder.getStringAttr(key.cast<std::string>()));
            paramTypes.emplace_back(unwrap(key.cast<PyType>()));
          }

          MlirType t = wrap(hc::typing::IdentType::get(ctx, nameAttr,
                                                       paramNames, paramTypes));
          return PyValueType(context->getRef(), t);
        },
        py::arg("name"), py::arg("params") = py::dict(),
        py::arg("context") = py::none(), "Create ident type");
  }
};

class PySequenceType : public PyConcreteType<PySequenceType> {
public:
  static constexpr IsAFunctionTy isaFunction =
      genTypeIsA<hc::typing::SequenceType>;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      genGetTypeID<hc::typing::SequenceType>;
  static constexpr const char *pyClassName = "SequenceType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](py::iterable params, DefaultingPyMlirContext context) {
          llvm::SmallVector<mlir::Type> paramsArr;
          for (auto param : params)
            paramsArr.emplace_back(unwrap(param.cast<PyType>()));

          auto ctx = unwrap(context->get());
          MlirType t = wrap(hc::typing::SequenceType::get(ctx, paramsArr));
          return PySequenceType(context->getRef(), t);
        },
        py::arg("params"), py::arg("context") = py::none(),
        "Create sequence type");
  }
};

template <typename ClassTy> static void defExprOpearators(ClassTy &c);

class PySymbolType : public PyConcreteType<PySymbolType> {
public:
  static constexpr IsAFunctionTy isaFunction =
      genTypeIsA<hc::typing::SymbolType>;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      genGetTypeID<hc::typing::SymbolType>;
  static constexpr const char *pyClassName = "SymbolType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](py::str name, DefaultingPyMlirContext context) {
          auto ctx = unwrap(context->get());
          MlirType t =
              wrap(hc::typing::SymbolType::get(ctx, name.cast<std::string>()));
          return PySymbolType(context->getRef(), t);
        },
        py::arg("params"), py::arg("context") = py::none(),
        "Create symbol type");

    defExprOpearators(c);
  }
};

class PyExprType : public PyConcreteType<PyExprType> {
public:
  static constexpr IsAFunctionTy isaFunction = genTypeIsA<hc::typing::ExprType>;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      genGetTypeID<hc::typing::ExprType>;
  static constexpr const char *pyClassName = "ExprType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](py::list params, PyAffineExpr expr) {
          auto ctx = unwrap(expr.get()).getContext();
          llvm::SmallVector<mlir::Type> paramsArr;
          for (auto p : params)
            paramsArr.emplace_back(unwrap(p.cast<mlir::python::PyType>()));

          auto type =
              hc::typing::ExprType::get(ctx, paramsArr, unwrap(expr.get()));
          return PyExprType(expr.getContext(), wrap(type));
        },
        py::arg("params"), py::arg("expr"), "Create expr type");

    defExprOpearators(c);
  }
};

template <typename Op>
static PyExprType makeExpr(mlir::python::PyType lhs, mlir::python::PyType rhs) {
  auto ulhs = unwrap(lhs.get());
  auto urhs = unwrap(rhs.get());
  auto ctx = ulhs.getContext();
  auto op = Op()(mlir::getAffineSymbolExpr(0, ctx),
                 mlir::getAffineSymbolExpr(1, ctx));
  auto expr = hc::typing::ExprType::get(ctx, {ulhs, urhs}, op);
  return PyExprType(lhs.getContext(), wrap(expr));
}

template <typename ClassTy> static void defExprOpearators(ClassTy &c) {
  c.def("__mul__", &makeExpr<std::multiplies<void>>, py::arg("rhs"), "mul op");
}

class PyTypeAttr : public PyConcreteAttribute<PyTypeAttr> {
public:
  static constexpr IsAFunctionTy isaFunction = genAttrIsA<hc::typing::TypeAttr>;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      genGetTypeID<hc::typing::TypeAttr>;
  static constexpr const char *pyClassName = "TypeAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType type) {
          mlir::Type t = unwrap(type);
          MlirAttribute typeAttr = wrap(hc::typing::TypeAttr::get(t));
          return PyTypeAttr(
              mlir::python::PyMlirContext::forContext(wrap(t.getContext())),
              typeAttr);
        },
        "Create TypeAttr value");
  }
};
} // namespace

static void populateTypingTypes(py::module &m) {
  PyValueType::bind(m);
  PyIdentType::bind(m);
  PySequenceType::bind(m);
  PySymbolType::bind(m);
  PyExprType::bind(m);

  PyTypeAttr::bind(m);
}

void populateMlirModule(py::module &m) {
  // TODO: refactor upstream
  m.doc() = "MLIR Python Native Extension";

  py::class_<PyGlobals>(m, "_Globals", py::module_local())
      .def_property("dialect_search_modules",
                    &PyGlobals::getDialectSearchPrefixes,
                    &PyGlobals::setDialectSearchPrefixes)
      .def(
          "append_dialect_search_prefix",
          [](PyGlobals &self, std::string moduleName) {
            self.getDialectSearchPrefixes().push_back(std::move(moduleName));
          },
          "module_name"_a)
      .def(
          "_check_dialect_module_loaded",
          [](PyGlobals &self, const std::string &dialectNamespace) {
            return self.loadDialectModule(dialectNamespace);
          },
          "dialect_namespace"_a)
      .def("_register_dialect_impl", &PyGlobals::registerDialectImpl,
           "dialect_namespace"_a, "dialect_class"_a,
           "Testing hook for directly registering a dialect")
      .def("_register_operation_impl", &PyGlobals::registerOperationImpl,
           "operation_name"_a, "operation_class"_a, py::kw_only(),
           "replace"_a = false,
           "Testing hook for directly registering an operation");

  // Aside from making the globals accessible to python, having python manage
  // it is necessary to make sure it is destroyed (and releases its python
  // resources) properly.
  m.attr("globals") =
      py::cast(new PyGlobals, py::return_value_policy::take_ownership);

  // Registration decorators.
  m.def(
      "register_dialect",
      [](py::object pyClass) {
        std::string dialectNamespace =
            pyClass.attr("DIALECT_NAMESPACE").cast<std::string>();
        PyGlobals::get().registerDialectImpl(dialectNamespace, pyClass);
        return pyClass;
      },
      "dialect_class"_a,
      "Class decorator for registering a custom Dialect wrapper");
  m.def(
      "register_operation",
      [](const py::object &dialectClass, bool replace) -> py::cpp_function {
        return py::cpp_function(
            [dialectClass, replace](py::object opClass) -> py::object {
              std::string operationName =
                  opClass.attr("OPERATION_NAME").cast<std::string>();
              PyGlobals::get().registerOperationImpl(operationName, opClass,
                                                     replace);

              // Dict-stuff the new opClass by name onto the dialect class.
              py::object opClassName = opClass.attr("__name__");
              dialectClass.attr(opClassName) = opClass;
              return opClass;
            });
      },
      "dialect_class"_a, py::kw_only(), "replace"_a = false,
      "Produce a class decorator for registering an Operation class as part of "
      "a dialect");
  m.def(
      MLIR_PYTHON_CAPI_TYPE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> py::cpp_function {
        return py::cpp_function([mlirTypeID,
                                 replace](py::object typeCaster) -> py::object {
          PyGlobals::get().registerTypeCaster(mlirTypeID, typeCaster, replace);
          return typeCaster;
        });
      },
      "typeid"_a, py::kw_only(), "replace"_a = false,
      "Register a type caster for casting MLIR types to custom user types.");
  m.def(
      MLIR_PYTHON_CAPI_VALUE_CASTER_REGISTER_ATTR,
      [](MlirTypeID mlirTypeID, bool replace) -> py::cpp_function {
        return py::cpp_function(
            [mlirTypeID, replace](py::object valueCaster) -> py::object {
              PyGlobals::get().registerValueCaster(mlirTypeID, valueCaster,
                                                   replace);
              return valueCaster;
            });
      },
      "typeid"_a, py::kw_only(), "replace"_a = false,
      "Register a value caster for casting MLIR values to custom user values.");

  // Define and populate IR submodule.
  auto irModule = m.def_submodule("ir", "MLIR IR Bindings");
  populateIRCore(irModule);
  populateIRAffine(irModule);
  populateIRAttributes(irModule);
  populateIRInterfaces(irModule);
  populateIRTypes(irModule);

  // Define and populate PassManager submodule.
  auto passModule =
      m.def_submodule("passmanager", "MLIR Pass Management Bindings");
  populatePassManagerSubmodule(passModule);

  auto typingModule = m.def_submodule("typing", "hc typing module");
  populateTypingTypes(typingModule);
}

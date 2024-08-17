# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(pybind11 REQUIRED)
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# TODO: something better
set(_python_sources ${LLVM_MAIN_INCLUDE_DIR}/../src/python/)
if(NOT EXISTS ${_python_sources})
  message(FATAL_ERROR "MLIR python bindings not found")
endif()

set(BINDINGS_COMPONENTS
  MLIRPythonCAPI.HeaderSources
  MLIRPythonExtension.Core
  )

set(MLIR_BINDINGS_SOURCES)
foreach(_component IN LISTS BINDINGS_COMPONENTS)
   file(GLOB_RECURSE _tmp_src ${_python_sources}/${_component}/*.cpp ${_python_sources}/*.h )
   list(APPEND MLIR_BINDINGS_SOURCES ${_tmp_src})
endforeach()


set(SOURCES_LIST
    src/CompilerFront.cpp
    src/CompilerFront.hpp
    src/Context.cpp
    src/Context.hpp
    src/Dispatcher.cpp
    src/Dispatcher.hpp
    src/DispatcherBase.cpp
    src/DispatcherBase.hpp
    src/PyModule.cpp
    src/PyWrappers.cpp
    src/PyWrappers.hpp
    src/TypingDispatcher.cpp
    src/TypingDispatcher.hpp
    src/TypingPipeline.cpp
    src/TypingPipeline.hpp
    src/Utils.cpp
    src/Utils.hpp
  )

pybind11_add_module(${python_module_name} MODULE
  ${SOURCES_LIST}
  ${MLIR_BINDINGS_SOURCES}
)

target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${python_module_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(${python_module_name} PRIVATE
    hc_core
    hc_frontend
    MLIRCAPIDebug
    MLIRCAPIIR
    MLIRCAPIInterfaces
    MLIRPass
  )

apply_llvm_compile_flags(${python_module_name})

target_include_directories(${python_module_name} PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${_python_sources}/MLIRPythonExtension.Core
  )

install(TARGETS ${python_module_name} DESTINATION ${CMAKE_INSTALL_PREFIX})

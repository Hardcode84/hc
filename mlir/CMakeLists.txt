# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

project(hc_core LANGUAGES CXX C)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

#file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../llvm-sha.txt EXPECTED_LLVM_SHA)
#message(STATUS "Expected llvm sha: \"${EXPECTED_LLVM_SHA}\"")
#
#file(STRINGS ${LLVM_INCLUDE_DIR}/llvm/Support/VCSRevision.h REVISION_FILE_DATA)
#message(DEBUG "VCSRevision: ${REVISION_FILE_DATA}")
#string(REGEX MATCH "\"([^\"]*)\"" LLVM_SHA ${REVISION_FILE_DATA})
#string(REPLACE "\"" "" LLVM_SHA ${LLVM_SHA})
#message(STATUS "llvm sha: \"${LLVM_SHA}\"")
#
#if (NOT EXPECTED_LLVM_SHA STREQUAL LLVM_SHA)
#    message(FATAL_ERROR "Invalid llvm version")
#endif()

file(REAL_PATH "./include" HC_INCLUDE_DIR)
set(HC_TABLEGEN_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS} ${HC_INCLUDE_DIR})

add_subdirectory(include/hc/Dialect)
add_subdirectory(include/hc/Transforms)

set(SOURCES_LIST
    lib/Dialect/PyAST/IR/PyASTOps.cpp
    lib/Dialect/PyIR/IR/PyIROps.cpp
    lib/Dialect/Typing/IR/TypingOps.cpp
    lib/Dialect/Typing/Transforms/Interpreter.cpp
    lib/Pipelines/FrontendPipeline.cpp
    lib/Transforms/ClenaupPySetVar.cpp
    lib/Transforms/ConvertPyASTToIR.cpp
    lib/Transforms/ConvertPyFuncToFunc.cpp
    lib/Transforms/DropTypeResolvers.cpp
    lib/Transforms/ModuleLinker.cpp
    lib/Transforms/PyIRPromoteFuncsToStatic.cpp
    lib/Transforms/PyTypeInference.cpp
    lib/Transforms/ReconstructPySSA.cpp
    lib/Transforms/SimplifyAST.cpp
  )
set(HEADERS_LIST
    include/hc/Dialect/PyAST/IR/PyASTOps.hpp
    include/hc/Dialect/PyIR/IR/PyIROps.hpp
    include/hc/Dialect/Typing/IR/TypingOps.hpp
    include/hc/Dialect/Typing/IR/TypingOpsInterfaces.hpp
    include/hc/Dialect/Typing/Transforms/Interpreter.hpp
    include/hc/InitHCDialects.hpp
    include/hc/InitHCPasses.hpp
    include/hc/Pipelines/FrontendPipeline.hpp
    include/hc/Transforms/ModuleLinker.hpp
    include/hc/Transforms/Passes.hpp
    include/hc/Utils.hpp
  )

add_library(${PROJECT_NAME} STATIC ${SOURCES_LIST} ${HEADERS_LIST})

apply_llvm_compile_flags(${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} PRIVATE
    MLIRComplexDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRIR
    MLIRTransformUtils
    MLIRTransforms
    )

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${MLIR_INCLUDE_DIRS}
    PRIVATE
    ./lib
    )

target_include_directories(${PROJECT_NAME} PUBLIC
    ./include
    ${PROJECT_BINARY_DIR}/include
    )

add_dependencies(${PROJECT_NAME}
    HCTransformsPassIncGen
    MLIRPyASTOpsIncGen
    MLIRPyIROpsIncGen
    MLIRTypingOpsIncGen
    MLIRTypingOpsInterfacesIncGen
    )

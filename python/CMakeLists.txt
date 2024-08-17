# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(python_module_name compiler)

add_subdirectory(cpp)

add_custom_target(python-all ALL
    DEPENDS ${python_module_name}
  )

set(_opt_path ${CMAKE_BINARY_DIR}/tools/hc-opt/hc-opt)

function(process_mlir_file filename)
  set(_src_file ${CMAKE_CURRENT_SOURCE_DIR}/ir/${filename}.mlir)
  set(_res_file ${filename}.mlirbc)
  add_custom_command(
    OUTPUT ${_res_file}
    COMMAND ${_opt_path} --emit-bytecode ${_src_file} > ${_res_file}
    DEPENDS ${_src_file}
    DEPENDS hc-opt
    VERBATIM)

  set(_tgt_name "generate_mlir_${filename}")
  add_custom_target(${_tgt_name} DEPENDS ${_res_file})
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${_res_file} DESTINATION ${CMAKE_INSTALL_PREFIX})

  add_dependencies(python-all ${_tgt_name})
endfunction()

process_mlir_file("typing")

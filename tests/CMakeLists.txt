# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Needed for llvm lit to work.
find_package (Python3 COMPONENTS Interpreter REQUIRED)

find_package(LLVM REQUIRED CONFIG)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TESTS_DEPENDS
    Python3::Interpreter
    FileCheck
    pyfront
    hc-opt
    )

add_lit_testsuite(check-hc "Running the regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TESTS_DEPENDS}
        )
set_target_properties(check-hc PROPERTIES FOLDER "Tests")

add_lit_testsuites(check-hc ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TESTS_DEPENDS})

// SPDX-FileCopyrightText: 2025 The HC Authors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cstdint>

void convertPyArrayImpl(void *obj, int rank, void *ret);
void convertPyInt64(void *obj, int64_t *ret);

void setRuntimeSearchPathsImpl(void *obj);

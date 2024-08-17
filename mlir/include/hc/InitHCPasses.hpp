// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "hc/Transforms/Passes.hpp"

namespace hc {
inline void registerAllPasses() { registerTransformsPasses(); }
} // namespace hc

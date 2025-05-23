#pragma once

#include "sip/types.hpp"
#include "sip_optimal_control/types.hpp"

namespace sip::optimal_control {

auto solve(const Input &input, const ::sip::Settings &settings,
           Workspace &workspace) -> ::sip::Output;

} // namespace sip::optimal_control

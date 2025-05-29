#pragma once

#include "sip_optimal_control/types.hpp"

namespace sip::optimal_control {

class CallbackProvider {
public:
  CallbackProvider(const Input &input, Workspace &workspace);

  void factor(const double *w, const double r1, const double r2,
              const double r3);
  void solve(const double *b, double *sol);
  void add_Kx_to_y(const double *w, const double r1, const double r2,
                   const double r3, const double *x_x, const double *x_y,
                   const double *x_z, double *y_x, double *y_y, double *y_z);
  void add_Hx_to_y(const double *x, double *y);
  void add_Cx_to_y(const double *x, double *y);
  void add_CTx_to_y(const double *x, double *y);
  void add_Gx_to_y(const double *x, double *y);
  void add_GTx_to_y(const double *x, double *y);

private:
  const Input &input_;
  Workspace &workspace_;
  LQR::Input lqr_input_;
};

} // namespace sip::optimal_control

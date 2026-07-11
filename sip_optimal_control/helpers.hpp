#pragma once

#include "sip_optimal_control/types.hpp"

namespace sip::optimal_control {

class CallbackProvider {
public:
  CallbackProvider(const Input &input, Workspace &workspace);

  bool factor(const double *w, const double r1, const double *r2,
              const double *r3);
  void solve(const double *b, double *sol);
  void add_Kx_to_y(const double *w, const double r1, const double *r2,
                   const double *r3, const double *x_x, const double *x_y,
                   const double *x_z, double *y_x, double *y_y, double *y_z);
  void add_Hx_to_y(const double *x, double *y);
  void add_Cx_to_y(const double *x, double *y);
  void add_CTx_to_y(const double *x, double *y);
  void add_Gx_to_y(const double *x, double *y);
  void add_GTx_to_y(const double *x, double *y);

private:
  void form_theta_jacobian();
  void solve_stagewise_kkt(const double *b, double *sol);
  void solve_stagewise_kkt_matrix(const double *b, double *sol, int num_rhs);

  const Input &input_;
  Workspace &workspace_;
  LQR::Input lqr_input_;
  bool input_is_valid_;
};

} // namespace sip::optimal_control

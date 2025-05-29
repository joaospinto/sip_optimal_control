#pragma once

#include "sip/types.hpp"
#include "sip_optimal_control/lqr.hpp"

#include <Eigen/Core>

namespace sip::optimal_control {

struct ModelCallbackInput {
  // The states.
  double **states;
  // The controls.
  double **controls;
  // The co-states.
  double **costates;
  // The (non-dynamics) equality constraint multipliers.
  double **equality_constraint_multipliers;
  // The inequality constraint multipliers.
  double **inequality_constraint_multipliers;

  // To dynamically allocate the required memory.
  void reserve(int num_stages);
  void free();

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int num_stages, unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int num_stages) -> int {
    return (5 * num_stages + 4) * sizeof(double *);
  }
};

struct ModelCallbackOutput {
  // The cost.
  double f;
  // The first derivative of the cost with respect to states.
  double **df_dx;
  // The first derivative of the cost with respect to controls.
  double **df_du;

  // The dynamics residuals (x_init - x_0 and dyn_i(x_i, u_i) - x_{i+1}).
  double **dyn_res;
  // The first derivative of the dyn_i with respect to the states.
  double **ddyn_dx;
  // The first derivative of the dyn_i with respect to the controls.
  double **ddyn_du;

  // The equality constraint values (c(x) = 0); excludes the dynamics.
  double **c;
  // The first derivative of the c(x) with respect to the states.
  double **dc_dx;
  // The first derivative of the c(x) with respect to the controls.
  double **dc_du;

  // The inequality constraint values (g(x) <= 0).
  double **g;
  // The first derivative of the g(x) with respect to the states.
  double **dg_dx;
  // The first derivative of the g(x) with respect to the controls.
  double **dg_du;

  // The second derivative of the Lagrangian with respect to states.
  double **d2L_dx2;
  // The second derivative of the Lagrangian with respect to states and
  // controls.
  double **d2L_dxdu;
  // The second derivative of the Lagrangian with respect to controls.
  double **d2L_du2;
  // NOTE: d2L should be positive semi-definite and
  // d2L_du2 should be positive definite.

  // To dynamically allocate the required memory.
  void reserve(int state_dim, int control_dim, int num_stages, int c_dim,
               int g_dim);
  void free(int num_stages);

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int state_dim, int control_dim, int num_stages, int c_dim,
                  int g_dim, unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int state_dim, int control_dim,
                                  int num_stages, int c_dim, int g_dim) -> int {
    const int n = state_dim;
    const int m = control_dim;
    const int T = num_stages;

    const int df_dx_size =
        (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
    const int df_du_size = T * sizeof(double *) + T * m * sizeof(double);

    const int dyn_res_size =
        (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
    const int ddyn_dx_size = T * sizeof(double *) + T * n * n * sizeof(double);
    const int ddyn_du_size = T * sizeof(double *) + T * n * m * sizeof(double);

    const int c_size =
        (T + 1) * sizeof(double *) + (T + 1) * c_dim * sizeof(double);
    const int dc_dx_size =
        (T + 1) * sizeof(double *) + (T + 1) * c_dim * n * sizeof(double);
    const int dc_du_size =
        T * sizeof(double *) + T * c_dim * m * sizeof(double);

    const int g_size =
        (T + 1) * sizeof(double *) + (T + 1) * g_dim * sizeof(double);
    const int dg_dx_size =
        (T + 1) * sizeof(double *) + (T + 1) * g_dim * n * sizeof(double);
    const int dg_du_size =
        T * sizeof(double *) + T * g_dim * m * sizeof(double);

    const int d2L_dx2_size =
        (T + 1) * sizeof(double *) + (T + 1) * n * n * sizeof(double);
    const int d2L_dxdu_size = T * sizeof(double *) + T * n * m * sizeof(double);
    const int d2L_du2_size = T * sizeof(double *) + T * m * m * sizeof(double);

    return df_dx_size + df_du_size + dyn_res_size + ddyn_dx_size +
           ddyn_du_size + c_size + dc_dx_size + dc_du_size + g_size +
           dg_dx_size + dg_du_size + d2L_dx2_size + d2L_dxdu_size +
           d2L_du2_size;
  }
};

struct Input {
  struct Dimensions {
    int num_stages;
    int state_dim;
    int control_dim;
    int c_dim;
    int g_dim;

    int get_x_dim() const {
      return num_stages * (state_dim + control_dim) + state_dim;
    }

    int get_y_dim() const { return (c_dim + state_dim) * (num_stages + 1); }

    int get_z_dim() const { return g_dim * (num_stages + 1); }
  };
  using ModelCallback = std::function<void(const ModelCallbackInput &)>;

  // Callback for filling the ModelCallbackOutput object.
  ModelCallback model_callback;
  // Callback for (optionally) declaring a timeout. Return true for timeout.
  ::sip::Input::TimeoutCallback timeout_callback;
  // The problem dimensions.
  Dimensions dimensions;
};

struct Workspace {
  struct RegularizedLQRData {

    double **mod_w_inv;
    double **Q_mod;
    double **M_mod;
    double **R_mod;
    double **q_mod;
    double **r_mod;
    double **c_mod;
    double r2;
    double r2_inv;

    // To dynamically allocate the required memory.
    void reserve(int state_dim, int control_dim, int num_stages, int g_dim);
    void free(int num_stages);

    // For using pre-allocated (possibly statically allocated) memory.
    auto mem_assign(int state_dim, int control_dim, int num_stages, int g_dim,
                    unsigned char *mem_ptr) -> int;

    // For knowing how much memory to pre-allocate.
    static constexpr auto num_bytes(int state_dim, int control_dim,
                                    int num_stages, int g_dim) -> int {
      const int T = num_stages;
      const int n = state_dim;
      const int m = control_dim;

      const int mod_w_inv_size =
          (T + 1) * sizeof(double *) + (T + 1) * g_dim * sizeof(double);
      const int Q_mod_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * n * sizeof(double);
      const int M_mod_size = T * sizeof(double *) + T * n * m * sizeof(double);
      const int R_mod_size = T * sizeof(double *) + T * m * m * sizeof(double);
      const int q_mod_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
      const int r_mod_size = T * sizeof(double *) + T * m * sizeof(double);
      const int c_mod_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);

      return mod_w_inv_size + Q_mod_size + M_mod_size + R_mod_size +
             q_mod_size + r_mod_size + c_mod_size;
    }
  };

  // To dynamically allocate the required memory.
  void reserve(int state_dim, int control_dim, int num_stages, int c_dim,
               int g_dim);
  void free(int num_stages);

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int state_dim, int control_dim, int num_stages, int c_dim,
                  int g_dim, unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int state_dim, int control_dim,
                                  int num_stages, int c_dim, int g_dim) -> int {
    const int x_dim = num_stages * (state_dim + control_dim) + state_dim;
    const int y_dim = (c_dim + state_dim) * (num_stages + 1);
    const int z_dim = g_dim * (num_stages + 1);
    return ModelCallbackOutput::num_bytes(state_dim, control_dim, num_stages,
                                          c_dim, g_dim) +
           ModelCallbackInput::num_bytes(num_stages) +
           (x_dim + y_dim + z_dim) * sizeof(double) +
           LQR::Workspace::num_bytes(state_dim, control_dim, num_stages) +
           LQR::Output::num_bytes(num_stages) +
           RegularizedLQRData::num_bytes(state_dim, control_dim, num_stages,
                                         g_dim) +
           sip::Workspace::num_bytes(x_dim, z_dim, y_dim);
  }

  ModelCallbackOutput model_callback_output;

  ModelCallbackInput model_callback_input;

  double *gradient_f;
  double *c;
  double *g;

  LQR::Workspace lqr_workspace;

  LQR::Output lqr_output;

  RegularizedLQRData regularized_lqr_data;

  sip::Workspace sip_workspace;
};

} // namespace sip::optimal_control

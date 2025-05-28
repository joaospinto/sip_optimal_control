#pragma once

#include "sip/types.hpp"

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

  // The inequality constraint values (g(x) < 0).
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
    return (14 * num_stages + 7) * sizeof(double *) +
           ((num_stages + 1) *
                (2 * state_dim + c_dim * (1 + state_dim) +
                 g_dim * (1 + state_dim) + state_dim * state_dim) +
            num_stages * (control_dim + state_dim * (state_dim + control_dim) +
                          (c_dim + g_dim) * control_dim +
                          control_dim * (state_dim + control_dim))) *
               sizeof(double);
  }
};

struct Input {
  struct Dimensions {
    int num_stages;
    int state_dim;
    int control_dim;
    int c_dim;
    int g_dim;
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
    // NOTE: we need to store these for ALL stages.
    double **W;
    double **K;
    double **V;
    double **G_inv;
    double **k;
    double **v;

    // NOTE: we only need to store these for one stage at a time.
    double *G;
    double *g;
    double *H;
    double *h;
    double *F;
    double *F_inv;
    double *f;

    // Other helpers.
    double **mod_w_inv;
    double *Q_mod;
    double *M_mod;
    double *R_mod;
    double *q_mod;
    double *r_mod;
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
      return (7 * num_stages + 3) * sizeof(double *) +
             ((num_stages + 1) * (state_dim * (state_dim + 1) + g_dim) +
              (num_stages * (state_dim * (state_dim + control_dim) +
                             control_dim * (control_dim + 1))) +
              control_dim * (2 * control_dim + 2 * state_dim + 3) +
              state_dim * (3 * state_dim + 2)) *
                 sizeof(double);
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
           ModelCallbackInput::num_bytes(num_stages) + (x_dim + y_dim + z_dim) * sizeof(double) +
           RegularizedLQRData::num_bytes(state_dim, control_dim, num_stages,
                                         g_dim) +
           sip::Workspace::num_bytes(x_dim, z_dim, y_dim);
  }

  ModelCallbackOutput model_callback_output;

  ModelCallbackInput model_callback_input;

  double *gradient_f;
  double *c;
  double *g;

  RegularizedLQRData regularized_lqr_data;

  sip::Workspace sip_workspace;
};

} // namespace sip::optimal_control

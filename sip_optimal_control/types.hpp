#pragma once

#include "sip/types.hpp"
#include "sip_optimal_control/lqr.hpp"

#include <Eigen/Core>

#include <cstddef>

namespace sip::optimal_control {

struct Dimensions {
  int num_stages;
  int state_dim;
  int control_dim;
  int c_dim;
  int g_dim;
  // Number of dense global variables eliminated through the Schur complement.
  // Non-tree DAG separator variables should be represented in this block.
  int theta_dim = 0;
  const int *state_dims = nullptr;
  const int *control_dims = nullptr;
  const int *c_dims = nullptr;
  const int *g_dims = nullptr;

  int num_nodes() const { return num_stages + 1; }
  int num_edges() const { return num_stages; }
  int get_schur_dim() const { return theta_dim; }
  int get_state_dim(const int node) const {
    return state_dims == nullptr ? state_dim : state_dims[node];
  }
  int get_control_dim(const int edge) const {
    return control_dims == nullptr ? control_dim : control_dims[edge];
  }
  int get_c_dim(const int node) const {
    return c_dims == nullptr ? c_dim : c_dims[node];
  }
  int get_g_dim(const int node) const {
    return g_dims == nullptr ? g_dim : g_dims[node];
  }
  int max_state_dim() const {
    int result = 0;
    for (int node = 0; node < num_nodes(); ++node) {
      const int dim = get_state_dim(node);
      result = result < dim ? dim : result;
    }
    return result;
  }
  int max_control_dim() const {
    int result = 0;
    for (int edge = 0; edge < num_edges(); ++edge) {
      const int dim = get_control_dim(edge);
      result = result < dim ? dim : result;
    }
    return result;
  }
  int max_c_dim() const {
    int result = 0;
    for (int node = 0; node < num_nodes(); ++node) {
      const int dim = get_c_dim(node);
      result = result < dim ? dim : result;
    }
    return result;
  }
  int max_g_dim() const {
    int result = 0;
    for (int node = 0; node < num_nodes(); ++node) {
      const int dim = get_g_dim(node);
      result = result < dim ? dim : result;
    }
    return result;
  }

  int get_stagewise_x_dim() const {
    int result = get_state_dim(num_stages);
    for (int edge = 0; edge < num_edges(); ++edge) {
      result += get_state_dim(edge) + get_control_dim(edge);
    }
    return result;
  }

  int get_x_dim() const { return get_stagewise_x_dim() + theta_dim; }

  int get_y_dim() const {
    int result = 0;
    for (int node = 0; node < num_nodes(); ++node) {
      result += get_state_dim(node) + get_c_dim(node);
    }
    return result;
  }

  int get_z_dim() const {
    int result = 0;
    for (int node = 0; node < num_nodes(); ++node) {
      result += get_g_dim(node);
    }
    return result;
  }

  int get_stagewise_kkt_dim() const {
    return get_stagewise_x_dim() + get_y_dim() + get_z_dim();
  }

  int get_x_state_offset(const int node) const {
    int offset = 0;
    for (int edge = 0; edge < node; ++edge) {
      offset += get_state_dim(edge) + get_control_dim(edge);
    }
    return offset;
  }
  int get_x_control_offset(const int edge) const {
    return get_x_state_offset(edge) + get_state_dim(edge);
  }
  int get_y_dyn_offset(const int node) const {
    int offset = 0;
    for (int i = 0; i < node; ++i) {
      offset += get_state_dim(i) + get_c_dim(i);
    }
    return offset;
  }
  int get_y_c_offset(const int node) const {
    return get_y_dyn_offset(node) + get_state_dim(node);
  }
  int get_z_offset(const int node) const {
    int offset = 0;
    for (int i = 0; i < node; ++i) {
      offset += get_g_dim(i);
    }
    return offset;
  }
};

struct ModelCallbackInput {
  // Dense global/separator variables shared by all stages.
  double *theta;
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
  // The first derivative of the cost with respect to global/separator
  // variables.
  double *df_dtheta;

  // The dynamics residuals (x_init - x_0 and dyn_i(x_i, u_i) - x_{i+1}).
  double **dyn_res;
  // The first derivative of the dyn_i with respect to the states.
  double **ddyn_dx;
  // The first derivative of the dyn_i with respect to the controls.
  double **ddyn_du;
  // The first derivative of dyn_i with respect to global/separator variables.
  double **ddyn_dtheta;

  // The equality constraint values (c(x) = 0); excludes the dynamics.
  double **c;
  // The first derivative of the c(x) with respect to the states.
  double **dc_dx;
  // The first derivative of the c(x) with respect to the controls.
  double **dc_du;
  // The first derivative of c(x) with respect to global/separator variables.
  double **dc_dtheta;

  // The inequality constraint values (g(x) <= 0).
  double **g;
  // The first derivative of the g(x) with respect to the states.
  double **dg_dx;
  // The first derivative of the g(x) with respect to the controls.
  double **dg_du;
  // The first derivative of g(x) with respect to global/separator variables.
  double **dg_dtheta;

  // The second derivative of the Lagrangian with respect to states.
  double **d2L_dx2;
  // The second derivative of the Lagrangian with respect to states and
  // controls.
  double **d2L_dxdu;
  // The second derivative of the Lagrangian with respect to controls.
  double **d2L_du2;
  // The second derivative of the Lagrangian with respect to states and
  // global/separator variables.
  double **d2L_dxdtheta;
  // The second derivative of the Lagrangian with respect to controls and
  // global/separator variables.
  double **d2L_dudtheta;
  // The second derivative of the Lagrangian with respect to global/separator
  // variables.
  double *d2L_dtheta2;
  // The user should provide the true Lagrangian Hessian blocks. SIP applies
  // primal regularization and retries factorization when the Riccati pivots do
  // not certify the desired Newton-KKT inertia.

  // To dynamically allocate the required memory.
  void reserve(int state_dim, int control_dim, int num_stages, int c_dim,
               int g_dim, int theta_dim = 0);
  void reserve(const Dimensions &dimensions);
  void free(int num_stages);

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int state_dim, int control_dim, int num_stages, int c_dim,
                  int g_dim, unsigned char *mem_ptr, int theta_dim = 0) -> int;
  auto mem_assign(const Dimensions &dimensions, unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int state_dim, int control_dim,
                                  int num_stages, int c_dim, int g_dim,
                                  int theta_dim = 0) -> int {
    const int n = state_dim;
    const int m = control_dim;
    const int T = num_stages;
    const int p = theta_dim;

    const int df_dx_size =
        (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
    const int df_du_size = T * sizeof(double *) + T * m * sizeof(double);
    const int df_dtheta_size = p * sizeof(double);

    const int dyn_res_size =
        (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
    const int ddyn_dx_size = T * sizeof(double *) + T * n * n * sizeof(double);
    const int ddyn_du_size = T * sizeof(double *) + T * n * m * sizeof(double);
    const int ddyn_dtheta_size =
        T * sizeof(double *) + T * n * p * sizeof(double);

    const int c_size =
        (T + 1) * sizeof(double *) + (T + 1) * c_dim * sizeof(double);
    const int dc_dx_size =
        (T + 1) * sizeof(double *) + (T + 1) * c_dim * n * sizeof(double);
    const int dc_du_size =
        T * sizeof(double *) + T * c_dim * m * sizeof(double);
    const int dc_dtheta_size =
        (T + 1) * sizeof(double *) + (T + 1) * c_dim * p * sizeof(double);

    const int g_size =
        (T + 1) * sizeof(double *) + (T + 1) * g_dim * sizeof(double);
    const int dg_dx_size =
        (T + 1) * sizeof(double *) + (T + 1) * g_dim * n * sizeof(double);
    const int dg_du_size =
        T * sizeof(double *) + T * g_dim * m * sizeof(double);
    const int dg_dtheta_size =
        (T + 1) * sizeof(double *) + (T + 1) * g_dim * p * sizeof(double);

    const int d2L_dx2_size =
        (T + 1) * sizeof(double *) + (T + 1) * n * n * sizeof(double);
    const int d2L_dxdu_size = T * sizeof(double *) + T * n * m * sizeof(double);
    const int d2L_du2_size = T * sizeof(double *) + T * m * m * sizeof(double);
    const int d2L_dxdtheta_size =
        (T + 1) * sizeof(double *) + (T + 1) * n * p * sizeof(double);
    const int d2L_dudtheta_size =
        T * sizeof(double *) + T * m * p * sizeof(double);
    const int d2L_dtheta2_size = p * p * sizeof(double);

    return df_dx_size + df_du_size + df_dtheta_size + dyn_res_size +
           ddyn_dx_size + ddyn_du_size + ddyn_dtheta_size + c_size +
           dc_dx_size + dc_du_size + dc_dtheta_size + g_size + dg_dx_size +
           dg_du_size + dg_dtheta_size + d2L_dx2_size + d2L_dxdu_size +
           d2L_du2_size + d2L_dxdtheta_size + d2L_dudtheta_size +
           d2L_dtheta2_size;
  }
  static auto num_bytes(const Dimensions &dimensions) -> int;
};

struct Input {
  using Dimensions = ::sip::optimal_control::Dimensions;
  using Topology = LQR::Input::Topology;
  using ModelCallback = std::function<void(const ModelCallbackInput &)>;

  // Callback for filling the ModelCallbackOutput object.
  ModelCallback model_callback;
  // Callback for (optionally) declaring a timeout. Return true for timeout.
  ::sip::Input::TimeoutCallback timeout_callback;
  // The problem dimensions.
  Dimensions dimensions;
  // Optional rooted-tree topology for direct Riccati elimination. When omitted,
  // edges use the chain topology edge -> (edge, edge + 1). Non-tree DAG
  // linkages should be condensed into theta and handled through the dense
  // Schur complement blocks above.
  Topology topology = {};
};

enum class InputValidationStatus {
  SUCCESS = 0,
  INVALID_DIMENSIONS = 1,
  INVALID_TOPOLOGY = 2,
};

auto validate_input(const Input &input) -> InputValidationStatus;

struct Workspace {
  struct RegularizedLQRData {

    double **mod_w_inv;
    double **Q_mod;
    double **M_mod;
    double **R_mod;
    double **q_mod;
    double **r_mod;
    double **c_mod;
    double **dyn_r2;
    double **c_r2_inv;
    double *theta_jacobian;
    double *theta_solution;
    double *theta_schur;
    double *theta_schur_factor;
    double *theta_rhs;
    double *theta_stagewise_rhs;
    double *stagewise_scratch;

    // To dynamically allocate the required memory.
    void reserve(int state_dim, int control_dim, int num_stages, int c_dim,
                 int g_dim, int theta_dim = 0);
    void reserve(const Dimensions &dimensions);
    void free(int num_stages);

    // For using pre-allocated (possibly statically allocated) memory.
    auto mem_assign(int state_dim, int control_dim, int num_stages, int c_dim,
                    int g_dim, unsigned char *mem_ptr, int theta_dim = 0)
        -> int;
    auto mem_assign(const Dimensions &dimensions, unsigned char *mem_ptr) -> int;

    // For knowing how much memory to pre-allocate.
    static constexpr auto num_bytes(int state_dim, int control_dim,
                                    int num_stages, int c_dim, int g_dim,
                                    int theta_dim = 0)
        -> int {
      const int T = num_stages;
      const int n = state_dim;
      const int m = control_dim;
      const int p = theta_dim;
      const int num_rhs = p > 0 ? p : 1;
      const int stagewise_x_dim = T * (n + m) + n;
      const int y_dim = (c_dim + n) * (T + 1);
      const int z_dim = g_dim * (T + 1);
      const int stagewise_kkt_dim = stagewise_x_dim + y_dim + z_dim;

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
      const int dyn_r2_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
      const int c_r2_inv_size =
          (T + 1) * sizeof(double *) + (T + 1) * c_dim * sizeof(double);
      const int theta_data_size =
          p > 0 ? (2 * stagewise_kkt_dim * p + 2 * p * p + p +
                   stagewise_kkt_dim) *
                      static_cast<int>(sizeof(double))
                : 0;
      const int stagewise_scratch_size =
          2 * n * num_rhs * static_cast<int>(sizeof(double));

      return mod_w_inv_size + Q_mod_size + M_mod_size + R_mod_size +
             q_mod_size + r_mod_size + c_mod_size + dyn_r2_size +
             c_r2_inv_size + theta_data_size + stagewise_scratch_size;
    }
    static auto num_bytes(const Dimensions &dimensions) -> int;
  };

  // To dynamically allocate the required memory.
  void reserve(int state_dim, int control_dim, int num_stages, int c_dim,
               int g_dim, int theta_dim = 0);
  void reserve(const Dimensions &dimensions);
  void free(int num_stages);

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(int state_dim, int control_dim, int num_stages, int c_dim,
                  int g_dim, unsigned char *mem_ptr, int theta_dim = 0) -> int;
  auto mem_assign(const Dimensions &dimensions, unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int state_dim, int control_dim,
                                  int num_stages, int c_dim, int g_dim,
                                  int theta_dim = 0) -> int {
    const int x_dim =
        num_stages * (state_dim + control_dim) + state_dim + theta_dim;
    const int y_dim = (c_dim + state_dim) * (num_stages + 1);
    const int z_dim = g_dim * (num_stages + 1);
    const int metadata_size = (9 * num_stages + 7) * sizeof(int);
    int total = ModelCallbackOutput::num_bytes(state_dim, control_dim,
                                               num_stages, c_dim, g_dim,
                                               theta_dim) +
                ModelCallbackInput::num_bytes(num_stages) +
                (x_dim + y_dim + z_dim) * sizeof(double) + metadata_size;
    total = ((total + alignof(std::max_align_t) - 1) /
             alignof(std::max_align_t)) *
            alignof(std::max_align_t);
    total += LQR::Workspace::num_bytes(state_dim, control_dim, num_stages);
    total = ((total + alignof(std::max_align_t) - 1) /
             alignof(std::max_align_t)) *
            alignof(std::max_align_t);
    total += LQR::Output::num_bytes(num_stages);
    total = ((total + alignof(std::max_align_t) - 1) /
             alignof(std::max_align_t)) *
            alignof(std::max_align_t);
    total += RegularizedLQRData::num_bytes(state_dim, control_dim, num_stages,
                                           c_dim, g_dim, theta_dim);
    total = ((total + alignof(std::max_align_t) - 1) /
             alignof(std::max_align_t)) *
            alignof(std::max_align_t);
    total += sip::Workspace::num_bytes(x_dim, z_dim, y_dim);
    return total;
  }
  static auto num_bytes(const Dimensions &dimensions) -> int;

  ModelCallbackOutput model_callback_output;

  ModelCallbackInput model_callback_input;

  double *gradient_f;
  double *c;
  double *g;
  int stagewise_x_dim;
  int x_dim;
  int y_dim;
  int z_dim;
  int stagewise_kkt_dim;
  int *state_dims;
  int *control_dims;
  int *c_dims;
  int *g_dims;
  int *x_state_offsets;
  int *x_control_offsets;
  int *y_dyn_offsets;
  int *y_c_offsets;
  int *z_offsets;

  LQR::Workspace lqr_workspace;

  LQR::Output lqr_output;

  RegularizedLQRData regularized_lqr_data;

  sip::Workspace sip_workspace;
};

} // namespace sip::optimal_control

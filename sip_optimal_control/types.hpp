#pragma once

#include "sip/types.hpp"
#include "sip_optimal_control/lqr.hpp"

#include <Eigen/Core>

#include <cstddef>

namespace sip::optimal_control {

struct NodeModelCallbackInput {
  int node;
  const double *state;
  const double *equality_constraint_multipliers;
  const double *inequality_constraint_multipliers;
};

struct EdgeModelCallbackInput {
  int edge;
  int parent;
  int child;
  const double *parent_state;
  const double *control;
  const double *child_state;
  const double *costate;
  const double *equality_constraint_multipliers;
  const double *inequality_constraint_multipliers;
};

struct ModelCallbackInput {
  const double *theta;
  NodeModelCallbackInput *nodes;
  EdgeModelCallbackInput *edges;

  void reserve(const Topology &topology);
  void free();
  auto mem_assign(const Topology &topology, unsigned char *mem_ptr) -> int;

  static constexpr auto num_bytes(const int num_edges) -> int {
    return (num_edges + 1) * sizeof(NodeModelCallbackInput) +
           num_edges * sizeof(EdgeModelCallbackInput);
  }
};

// A node term depends only on the node state and the global/separator
// variables. In particular, it cannot depend on any edge control.
struct NodeModelCallbackOutput {
  double f;
  double *df_dx;
  double *df_dtheta;
  double *c;
  double *dc_dx;
  double *dc_dtheta;
  double *g;
  double *dg_dx;
  double *dg_dtheta;
  double *d2L_dx2;
  double *d2L_dxdtheta;
  double *d2L_dtheta2;
};

// An edge term depends only on its parent state, its own control, and the
// global/separator variables. The dynamics residual additionally depends on
// the child state through the fixed -I Jacobian.
struct EdgeModelCallbackOutput {
  double f;
  double *df_dx;
  double *df_du;
  double *df_dtheta;
  double *dyn_res;
  double *ddyn_dx;
  double *ddyn_du;
  double *ddyn_dtheta;
  double *c;
  double *dc_dx;
  double *dc_du;
  double *dc_dtheta;
  double *g;
  double *dg_dx;
  double *dg_du;
  double *dg_dtheta;
  double *d2L_dx2;
  double *d2L_dxdu;
  double *d2L_du2;
  double *d2L_dxdtheta;
  double *d2L_dudtheta;
  double *d2L_dtheta2;
};

struct ModelCallbackOutput {
  NodeModelCallbackOutput *nodes;
  EdgeModelCallbackOutput *edges;

  // To dynamically allocate the required memory.
  void reserve(const Dimensions &dimensions, const Topology &topology);
  void free(const Topology &topology);

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(const Dimensions &dimensions, const Topology &topology,
                  unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int state_dim, int control_dim, int num_edges,
                                  int node_c_dim, int node_g_dim,
                                  int edge_c_dim, int edge_g_dim,
                                  int theta_dim = 0) -> int {
    const int n = state_dim;
    const int m = control_dim;
    const int T = num_edges;
    const int p = theta_dim;
    const int node_values = n + p + node_c_dim + node_c_dim * n +
                            node_c_dim * p + node_g_dim + node_g_dim * n +
                            node_g_dim * p + n * n + n * p + p * p;
    const int edge_values = n + m + p + n + n * n + n * m + n * p + edge_c_dim +
                            edge_c_dim * n + edge_c_dim * m + edge_c_dim * p +
                            edge_g_dim + edge_g_dim * n + edge_g_dim * m +
                            edge_g_dim * p + n * n + n * m + m * m + n * p +
                            m * p + p * p;
    return (T + 1) * sizeof(NodeModelCallbackOutput) +
           T * sizeof(EdgeModelCallbackOutput) +
           ((T + 1) * node_values + T * edge_values) * sizeof(double);
  }
  static auto num_bytes(const Dimensions &dimensions, const Topology &topology)
      -> int;
};

struct Input {
  using ModelCallback =
      std::function<void(const ModelCallbackInput &, ModelCallbackOutput &)>;

  Dimensions dimensions;
  Topology topology;
  // The fixed root state, with dimensions.get_state_dim(topology.root)
  // entries. Its residual is initial_state - root_state.
  const double *initial_state;
  // Called once per model evaluation with separate node and edge views.
  ModelCallback model_callback;
  // Callback for (optionally) declaring a timeout. Return true for timeout.
  ::sip::Input::TimeoutCallback timeout_callback;
  // Bounds on variables in SIP's flattened primal ordering:
  // [x_0, u_0, ..., x_{E-1}, u_{E-1}, x_E, theta].
  const double *lower_bounds;
  const double *upper_bounds;
  // Multipliers applied to flattened model residuals. Equality residuals are
  // [dyn_0, node_c_0, ..., dyn_E, node_c_E, edge_c_0, ..., edge_c_{E-1}];
  // inequalities are [node_g_0, ..., node_g_E, edge_g_0, ..., edge_g_{E-1}].
  ::sip::Input::ResidualScaling residual_scaling;

  auto num_bound_sides() const -> int;
};

enum class InputValidationStatus {
  SUCCESS = 0,
  INVALID_DIMENSIONS = 1,
  INVALID_TOPOLOGY = 2,
};

auto validate_input(const Dimensions &dimensions, const Topology &topology)
    -> InputValidationStatus;

struct Workspace {
  struct RegularizedLQRData {

    double **node_mod_w_inv;
    double **edge_mod_w_inv;
    double **Q_mod;
    double **M_mod;
    double **R_mod;
    double **q_mod;
    double **r_mod;
    double **c_mod;
    double **dyn_r2;
    double **node_c_r2_inv;
    double **edge_c_r2_inv;
    double *theta_jacobian;
    double *theta_solution;
    double *theta_schur;
    double *theta_schur_factor;
    double *theta_rhs;
    double *theta_stagewise_rhs;
    double *stagewise_scratch;

    // To dynamically allocate the required memory.
    void reserve(const Dimensions &dimensions, int num_edges);
    void free(int num_edges);

    // For using pre-allocated (possibly statically allocated) memory.
    auto mem_assign(const Dimensions &dimensions, int num_edges,
                    unsigned char *mem_ptr) -> int;

    // For knowing how much memory to pre-allocate.
    static constexpr auto num_bytes(int state_dim, int control_dim,
                                    int num_edges, int node_c_dim,
                                    int node_g_dim, int edge_c_dim,
                                    int edge_g_dim, int theta_dim = 0) -> int {
      const int T = num_edges;
      const int n = state_dim;
      const int m = control_dim;
      const int p = theta_dim;
      const int num_rhs = p > 0 ? p : 1;
      const int stagewise_x_dim = T * (n + m) + n;
      const int y_dim = (node_c_dim + n) * (T + 1) + edge_c_dim * T;
      const int z_dim = node_g_dim * (T + 1) + edge_g_dim * T;
      const int stagewise_kkt_dim = stagewise_x_dim + y_dim + z_dim;

      const int mod_w_inv_size =
          (2 * T + 1) * sizeof(double *) +
          ((T + 1) * node_g_dim + T * edge_g_dim) * sizeof(double);
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
          (2 * T + 1) * sizeof(double *) +
          ((T + 1) * node_c_dim + T * edge_c_dim) * sizeof(double);
      const int theta_data_size = p > 0 ? (2 * stagewise_kkt_dim * p +
                                           2 * p * p + p + stagewise_kkt_dim) *
                                              static_cast<int>(sizeof(double))
                                        : 0;
      const int stagewise_scratch_size =
          2 * n * num_rhs * static_cast<int>(sizeof(double));

      return mod_w_inv_size + Q_mod_size + M_mod_size + R_mod_size +
             q_mod_size + r_mod_size + c_mod_size + dyn_r2_size +
             c_r2_inv_size + theta_data_size + stagewise_scratch_size;
    }
    static auto num_bytes(const Dimensions &dimensions, int num_edges) -> int;
  };

  // To dynamically allocate the required memory.
  void reserve(const Dimensions &dimensions, const Topology &topology,
               int num_bound_sides, const sip::Settings &settings);
  void free(const Topology &topology);

  // For using pre-allocated (possibly statically allocated) memory.
  auto mem_assign(const Dimensions &dimensions, const Topology &topology,
                  int num_bound_sides, const sip::Settings &settings,
                  unsigned char *mem_ptr) -> int;

  // For knowing how much memory to pre-allocate.
  static constexpr auto num_bytes(int state_dim, int control_dim, int num_edges,
                                  int node_c_dim, int node_g_dim,
                                  int edge_c_dim, int edge_g_dim, int theta_dim,
                                  int num_bound_sides,
                                  const sip::Settings &settings) -> int {
    const int x_dim =
        num_edges * (state_dim + control_dim) + state_dim + theta_dim;
    const int y_dim =
        (node_c_dim + state_dim) * (num_edges + 1) + edge_c_dim * num_edges;
    const int z_dim = node_g_dim * (num_edges + 1) + edge_g_dim * num_edges;
    const int metadata_size =
        (7 * num_edges + 4) * sizeof(int) + 2 * num_edges * sizeof(double *);
    int total = ModelCallbackInput::num_bytes(num_edges) +
                ModelCallbackOutput::num_bytes(
                    state_dim, control_dim, num_edges, node_c_dim, node_g_dim,
                    edge_c_dim, edge_g_dim, theta_dim) +
                (x_dim + y_dim + z_dim) * sizeof(double) + metadata_size;
    total =
        ((total + alignof(std::max_align_t) - 1) / alignof(std::max_align_t)) *
        alignof(std::max_align_t);
    total += LQR::Workspace::num_bytes(state_dim, control_dim, num_edges);
    total =
        ((total + alignof(std::max_align_t) - 1) / alignof(std::max_align_t)) *
        alignof(std::max_align_t);
    total += LQR::Output::num_bytes(num_edges);
    total =
        ((total + alignof(std::max_align_t) - 1) / alignof(std::max_align_t)) *
        alignof(std::max_align_t);
    total += RegularizedLQRData::num_bytes(state_dim, control_dim, num_edges,
                                           node_c_dim, node_g_dim, edge_c_dim,
                                           edge_g_dim, theta_dim);
    total =
        ((total + alignof(std::max_align_t) - 1) / alignof(std::max_align_t)) *
        alignof(std::max_align_t);
    total += sip::Workspace::num_bytes(x_dim, z_dim, y_dim, num_bound_sides,
                                       settings);
    return total;
  }
  static auto num_bytes(const Dimensions &dimensions, const Topology &topology,
                        int num_bound_sides, const sip::Settings &settings)
      -> int;

  ModelCallbackInput model_callback_input;
  ModelCallbackOutput model_callback_output;

  double f;
  double *gradient_f;
  double *c;
  double *g;
  int stagewise_x_dim;
  int x_dim;
  int y_dim;
  int z_dim;
  int stagewise_kkt_dim;
  int *x_state_offsets;
  int *x_control_offsets;
  int *y_dyn_offsets;
  int *y_node_c_offsets;
  int *y_edge_c_offsets;
  int *z_node_offsets;
  int *z_edge_offsets;
  double **ddyn_dx;
  double **ddyn_du;

  LQR::Workspace lqr_workspace;

  LQR::Output lqr_output;

  RegularizedLQRData regularized_lqr_data;

  sip::Workspace sip_workspace;
};

} // namespace sip::optimal_control

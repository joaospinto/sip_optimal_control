#include "sip_optimal_control/types.hpp"

#include <cassert>
#include <cstddef>

namespace sip::optimal_control {

namespace {

auto align_size(const int size) -> int {
  constexpr int alignment = alignof(std::max_align_t);
  return ((size + alignment - 1) / alignment) * alignment;
}

auto topology_root(const Input &input) -> int {
  if (input.topology.root == nullptr) {
    return 0;
  }
  return input.topology.root(input.topology.context);
}

auto topology_edge_parent(const Input &input, const int edge) -> int {
  if (input.topology.edge_parent == nullptr) {
    return edge;
  }
  return input.topology.edge_parent(input.topology.context, edge);
}

auto topology_edge_child(const Input &input, const int edge) -> int {
  if (input.topology.edge_child == nullptr) {
    return edge + 1;
  }
  return input.topology.edge_child(input.topology.context, edge);
}

auto incoming_parent(const Input &input, const int node) -> int {
  for (int edge = 0; edge < input.dimensions.num_edges(); ++edge) {
    if (topology_edge_child(input, edge) == node) {
      return topology_edge_parent(input, edge);
    }
  }
  return -1;
}

void populate_workspace_metadata(Workspace &workspace,
                                 const Dimensions &dimensions) {
  workspace.stagewise_x_dim = dimensions.get_stagewise_x_dim();
  workspace.x_dim = dimensions.get_x_dim();
  workspace.y_dim = dimensions.get_y_dim();
  workspace.z_dim = dimensions.get_z_dim();
  workspace.stagewise_kkt_dim = dimensions.get_stagewise_kkt_dim();

  int x_offset = 0;
  for (int node = 0; node < dimensions.num_nodes(); ++node) {
    workspace.state_dims[node] = dimensions.get_state_dim(node);
    workspace.c_dims[node] = dimensions.get_c_dim(node);
    workspace.g_dims[node] = dimensions.get_g_dim(node);
    workspace.x_state_offsets[node] = x_offset;
    if (node < dimensions.num_edges()) {
      workspace.control_dims[node] = dimensions.get_control_dim(node);
      x_offset += workspace.state_dims[node];
      workspace.x_control_offsets[node] = x_offset;
      x_offset += workspace.control_dims[node];
    }
  }

  int y_offset = 0;
  for (int node = 0; node < dimensions.num_nodes(); ++node) {
    workspace.y_dyn_offsets[node] = y_offset;
    y_offset += workspace.state_dims[node];
    workspace.y_c_offsets[node] = y_offset;
    y_offset += workspace.c_dims[node];
  }

  int z_offset = 0;
  for (int node = 0; node < dimensions.num_nodes(); ++node) {
    workspace.z_offsets[node] = z_offset;
    z_offset += workspace.g_dims[node];
  }
}

} // namespace

auto validate_input(const Input &input) -> InputValidationStatus {
  const auto &dimensions = input.dimensions;
  if (dimensions.num_stages < 0 || dimensions.theta_dim < 0) {
    return InputValidationStatus::INVALID_DIMENSIONS;
  }

  for (int node = 0; node < dimensions.num_nodes(); ++node) {
    if (dimensions.get_state_dim(node) <= 0 || dimensions.get_c_dim(node) < 0 ||
        dimensions.get_g_dim(node) < 0) {
      return InputValidationStatus::INVALID_DIMENSIONS;
    }
  }
  for (int edge = 0; edge < dimensions.num_edges(); ++edge) {
    if (dimensions.get_control_dim(edge) < 0) {
      return InputValidationStatus::INVALID_DIMENSIONS;
    }
  }

  const int root = topology_root(input);
  const int num_nodes = dimensions.num_nodes();
  const int num_edges = dimensions.num_edges();
  if (root < 0 || root >= num_nodes) {
    return InputValidationStatus::INVALID_TOPOLOGY;
  }

  for (int edge = 0; edge < num_edges; ++edge) {
    const int parent = topology_edge_parent(input, edge);
    const int child = topology_edge_child(input, edge);
    if (parent < 0 || parent >= num_nodes || child < 0 || child >= num_nodes ||
        parent == child) {
      return InputValidationStatus::INVALID_TOPOLOGY;
    }
  }

  for (int node = 0; node < num_nodes; ++node) {
    int indegree = 0;
    for (int edge = 0; edge < num_edges; ++edge) {
      if (topology_edge_child(input, edge) == node) {
        ++indegree;
      }
    }
    if ((node == root && indegree != 0) || (node != root && indegree != 1)) {
      return InputValidationStatus::INVALID_TOPOLOGY;
    }
  }

  for (int node = 0; node < num_nodes; ++node) {
    int current = node;
    for (int steps = 0; steps < num_nodes && current != root; ++steps) {
      current = incoming_parent(input, current);
      if (current < 0) {
        return InputValidationStatus::INVALID_TOPOLOGY;
      }
    }
    if (current != root) {
      return InputValidationStatus::INVALID_TOPOLOGY;
    }
  }

  return InputValidationStatus::SUCCESS;
}

void ModelCallbackInput::reserve(int num_stages) {
  theta = nullptr;
  states = new double *[num_stages + 1];
  controls = new double *[num_stages];
  costates = new double *[num_stages + 1];
  equality_constraint_multipliers = new double *[num_stages + 1];
  inequality_constraint_multipliers = new double *[num_stages + 1];
}

void ModelCallbackInput::free() {
  delete[] states;
  delete[] controls;
  delete[] costates;
  delete[] equality_constraint_multipliers;
  delete[] inequality_constraint_multipliers;
}

auto ModelCallbackInput::mem_assign(int num_stages, unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  theta = nullptr;

  states = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  controls = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  costates = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  equality_constraint_multipliers =
      reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  inequality_constraint_multipliers =
      reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  assert(cum_size == ModelCallbackInput::num_bytes(num_stages));

  return cum_size;
}

void ModelCallbackOutput::reserve(int state_dim, int control_dim,
                                  int num_stages, int c_dim, int g_dim,
                                  int theta_dim) {
  const Dimensions dimensions = {
      .num_stages = num_stages,
      .state_dim = state_dim,
      .control_dim = control_dim,
      .c_dim = c_dim,
      .g_dim = g_dim,
      .theta_dim = theta_dim,
  };
  reserve(dimensions);
}

void ModelCallbackOutput::reserve(const Dimensions &dimensions) {
  const int num_stages = dimensions.num_stages;
  const int theta_dim = dimensions.theta_dim;
  const int max_state_dim = dimensions.max_state_dim();
  const int max_c_dim = dimensions.max_c_dim();
  const int max_g_dim = dimensions.max_g_dim();

  df_dx = new double *[num_stages + 1];
  df_du = new double *[num_stages];
  df_dtheta = new double[theta_dim];
  dyn_res = new double *[num_stages + 1];
  ddyn_dx = new double *[num_stages];
  ddyn_du = new double *[num_stages];
  ddyn_dtheta = new double *[num_stages];
  c = new double *[num_stages + 1];
  dc_dx = new double *[num_stages + 1];
  dc_du = new double *[num_stages];
  dc_dtheta = new double *[num_stages + 1];
  g = new double *[num_stages + 1];
  dg_dx = new double *[num_stages + 1];
  dg_du = new double *[num_stages];
  dg_dtheta = new double *[num_stages + 1];
  d2L_dx2 = new double *[num_stages + 1];
  d2L_dxdu = new double *[num_stages];
  d2L_du2 = new double *[num_stages];
  d2L_dxdtheta = new double *[num_stages + 1];
  d2L_dudtheta = new double *[num_stages];
  d2L_dtheta2 = new double[theta_dim * theta_dim];

  for (int i = 0; i < num_stages; ++i) {
    const int n_i = dimensions.get_state_dim(i);
    const int m_i = dimensions.get_control_dim(i);
    const int c_i = dimensions.get_c_dim(i);
    const int g_i = dimensions.get_g_dim(i);

    df_dx[i] = new double[n_i];
    df_du[i] = new double[m_i];
    dyn_res[i] = new double[n_i];
    ddyn_dx[i] = new double[max_state_dim * max_state_dim];
    ddyn_du[i] = new double[max_state_dim * m_i];
    ddyn_dtheta[i] = new double[max_state_dim * theta_dim];
    c[i] = new double[c_i];
    dc_dx[i] = new double[c_i * n_i];
    dc_du[i] = new double[max_c_dim * m_i];
    dc_dtheta[i] = new double[c_i * theta_dim];
    g[i] = new double[g_i];
    dg_dx[i] = new double[g_i * n_i];
    dg_du[i] = new double[max_g_dim * m_i];
    dg_dtheta[i] = new double[g_i * theta_dim];
    d2L_dx2[i] = new double[n_i * n_i];
    d2L_dxdu[i] = new double[max_state_dim * m_i];
    d2L_du2[i] = new double[m_i * m_i];
    d2L_dxdtheta[i] = new double[n_i * theta_dim];
    d2L_dudtheta[i] = new double[m_i * theta_dim];
  }

  const int n_N = dimensions.get_state_dim(num_stages);
  const int c_N = dimensions.get_c_dim(num_stages);
  const int g_N = dimensions.get_g_dim(num_stages);

  df_dx[num_stages] = new double[n_N];
  dyn_res[num_stages] = new double[n_N];
  c[num_stages] = new double[c_N];
  dc_dx[num_stages] = new double[c_N * n_N];
  dc_dtheta[num_stages] = new double[c_N * theta_dim];
  g[num_stages] = new double[g_N];
  dg_dx[num_stages] = new double[g_N * n_N];
  dg_dtheta[num_stages] = new double[g_N * theta_dim];
  d2L_dx2[num_stages] = new double[n_N * n_N];
  d2L_dxdtheta[num_stages] = new double[n_N * theta_dim];
}

void ModelCallbackOutput::free(int num_stages) {
  for (int i = 0; i < num_stages; ++i) {
    delete[] df_dx[i];
    delete[] df_du[i];
    delete[] dyn_res[i];
    delete[] ddyn_dx[i];
    delete[] ddyn_du[i];
    delete[] ddyn_dtheta[i];
    delete[] c[i];
    delete[] dc_dx[i];
    delete[] dc_du[i];
    delete[] dc_dtheta[i];
    delete[] g[i];
    delete[] dg_dx[i];
    delete[] dg_du[i];
    delete[] dg_dtheta[i];
    delete[] d2L_dx2[i];
    delete[] d2L_dxdu[i];
    delete[] d2L_du2[i];
    delete[] d2L_dxdtheta[i];
    delete[] d2L_dudtheta[i];
  }

  delete[] df_dx[num_stages];
  delete[] dyn_res[num_stages];
  delete[] c[num_stages];
  delete[] dc_dx[num_stages];
  delete[] dc_dtheta[num_stages];
  delete[] g[num_stages];
  delete[] dg_dx[num_stages];
  delete[] dg_dtheta[num_stages];
  delete[] d2L_dx2[num_stages];
  delete[] d2L_dxdtheta[num_stages];

  delete[] df_dtheta;
  delete[] d2L_dtheta2;
  delete[] df_dx;
  delete[] df_du;
  delete[] dyn_res;
  delete[] ddyn_dx;
  delete[] ddyn_du;
  delete[] ddyn_dtheta;
  delete[] c;
  delete[] dc_dx;
  delete[] dc_du;
  delete[] dc_dtheta;
  delete[] g;
  delete[] dg_dx;
  delete[] dg_du;
  delete[] dg_dtheta;
  delete[] d2L_dx2;
  delete[] d2L_dxdu;
  delete[] d2L_du2;
  delete[] d2L_dxdtheta;
  delete[] d2L_dudtheta;
}

auto ModelCallbackOutput::mem_assign(int state_dim, int control_dim,
                                     int num_stages, int c_dim, int g_dim,
                                     unsigned char *mem_ptr, int theta_dim)
    -> int {
  const Dimensions dimensions = {
      .num_stages = num_stages,
      .state_dim = state_dim,
      .control_dim = control_dim,
      .c_dim = c_dim,
      .g_dim = g_dim,
      .theta_dim = theta_dim,
  };
  return mem_assign(dimensions, mem_ptr);
}

auto ModelCallbackOutput::mem_assign(const Dimensions &dimensions,
                                     unsigned char *mem_ptr) -> int {
  const int num_stages = dimensions.num_stages;
  const int theta_dim = dimensions.theta_dim;
  const int max_state_dim = dimensions.max_state_dim();
  const int max_c_dim = dimensions.max_c_dim();
  const int max_g_dim = dimensions.max_g_dim();
  int cum_size = 0;

  df_dx = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  df_du = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  df_dtheta = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += theta_dim * sizeof(double);
  dyn_res = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  ddyn_dx = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  ddyn_du = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  ddyn_dtheta = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  c = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  dc_dx = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  dc_du = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  dc_dtheta = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  g = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  dg_dx = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  dg_du = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  dg_dtheta = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  d2L_dx2 = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  d2L_dxdu = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  d2L_du2 = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  d2L_dxdtheta = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  d2L_dudtheta = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  d2L_dtheta2 = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += theta_dim * theta_dim * sizeof(double);

  for (int i = 0; i < num_stages; ++i) {
    const int n_i = dimensions.get_state_dim(i);
    const int m_i = dimensions.get_control_dim(i);
    const int c_i = dimensions.get_c_dim(i);
    const int g_i = dimensions.get_g_dim(i);

    df_dx[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n_i * sizeof(double);
    df_du[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += m_i * sizeof(double);
    dyn_res[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n_i * sizeof(double);
    ddyn_dx[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += max_state_dim * max_state_dim * sizeof(double);
    ddyn_du[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += max_state_dim * m_i * sizeof(double);
    ddyn_dtheta[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += max_state_dim * theta_dim * sizeof(double);
    c[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += c_i * sizeof(double);
    dc_dx[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += c_i * n_i * sizeof(double);
    dc_du[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += max_c_dim * m_i * sizeof(double);
    dc_dtheta[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += c_i * theta_dim * sizeof(double);
    g[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += g_i * sizeof(double);
    dg_dx[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += g_i * n_i * sizeof(double);
    dg_du[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += max_g_dim * m_i * sizeof(double);
    dg_dtheta[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += g_i * theta_dim * sizeof(double);
    d2L_dx2[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n_i * n_i * sizeof(double);
    d2L_dxdu[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += max_state_dim * m_i * sizeof(double);
    d2L_du2[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += m_i * m_i * sizeof(double);
    d2L_dxdtheta[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n_i * theta_dim * sizeof(double);
    d2L_dudtheta[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += m_i * theta_dim * sizeof(double);
  }

  const int n_N = dimensions.get_state_dim(num_stages);
  const int c_N = dimensions.get_c_dim(num_stages);
  const int g_N = dimensions.get_g_dim(num_stages);
  df_dx[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += n_N * sizeof(double);
  dyn_res[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += n_N * sizeof(double);
  c[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += c_N * sizeof(double);
  dc_dx[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += c_N * n_N * sizeof(double);
  dc_dtheta[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += c_N * theta_dim * sizeof(double);
  g[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += g_N * sizeof(double);
  dg_dx[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += g_N * n_N * sizeof(double);
  dg_dtheta[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += g_N * theta_dim * sizeof(double);
  d2L_dx2[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += n_N * n_N * sizeof(double);
  d2L_dxdtheta[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += n_N * theta_dim * sizeof(double);

  assert(cum_size == ModelCallbackOutput::num_bytes(dimensions));
  return cum_size;
}

auto ModelCallbackOutput::num_bytes(const Dimensions &dimensions) -> int {
  const int T = dimensions.num_stages;
  const int p = dimensions.theta_dim;
  const int max_state_dim = dimensions.max_state_dim();
  const int max_c_dim = dimensions.max_c_dim();
  const int max_g_dim = dimensions.max_g_dim();
  int total = (10 * (T + 1) + 9 * T) * static_cast<int>(sizeof(double *));
  total += p * sizeof(double) + p * p * sizeof(double);

  for (int i = 0; i < T; ++i) {
    const int n_i = dimensions.get_state_dim(i);
    const int m_i = dimensions.get_control_dim(i);
    const int c_i = dimensions.get_c_dim(i);
    const int g_i = dimensions.get_g_dim(i);
    total +=
        (n_i + m_i + n_i + max_state_dim * max_state_dim +
         max_state_dim * m_i + max_state_dim * p + c_i + c_i * n_i +
         max_c_dim * m_i + c_i * p + g_i + g_i * n_i + max_g_dim * m_i +
         g_i * p + n_i * n_i + max_state_dim * m_i + m_i * m_i + n_i * p +
         m_i * p) *
        static_cast<int>(sizeof(double));
  }

  const int n_N = dimensions.get_state_dim(T);
  const int c_N = dimensions.get_c_dim(T);
  const int g_N = dimensions.get_g_dim(T);
  total += (n_N + n_N + c_N + c_N * n_N + c_N * p + g_N + g_N * n_N +
            g_N * p + n_N * n_N + n_N * p) *
           static_cast<int>(sizeof(double));
  return total;
}

void Workspace::RegularizedLQRData::reserve(int state_dim, int control_dim,
                                            int num_stages, int c_dim,
                                            int g_dim, int theta_dim) {
  const Dimensions dimensions = {
      .num_stages = num_stages,
      .state_dim = state_dim,
      .control_dim = control_dim,
      .c_dim = c_dim,
      .g_dim = g_dim,
      .theta_dim = theta_dim,
  };
  reserve(dimensions);
}

void Workspace::RegularizedLQRData::reserve(const Dimensions &dimensions) {
  const int num_stages = dimensions.num_stages;
  const int theta_dim = dimensions.theta_dim;
  const int scratch_rhs_dim = theta_dim > 0 ? theta_dim : 1;
  const int max_state_dim = dimensions.max_state_dim();

  mod_w_inv = new double *[num_stages + 1];
  Q_mod = new double *[num_stages + 1];
  M_mod = new double *[num_stages];
  R_mod = new double *[num_stages];
  q_mod = new double *[num_stages + 1];
  r_mod = new double *[num_stages];
  c_mod = new double *[num_stages + 1];
  dyn_r2 = new double *[num_stages + 1];
  c_r2_inv = new double *[num_stages + 1];
  const int stagewise_kkt_dim = dimensions.get_stagewise_kkt_dim();
  if (theta_dim > 0) {
    theta_jacobian = new double[stagewise_kkt_dim * theta_dim];
    theta_solution = new double[stagewise_kkt_dim * theta_dim];
    theta_schur = new double[theta_dim * theta_dim];
    theta_schur_factor = new double[theta_dim * theta_dim];
    theta_rhs = new double[theta_dim];
    theta_stagewise_rhs = new double[stagewise_kkt_dim];
  } else {
    theta_jacobian = nullptr;
    theta_solution = nullptr;
    theta_schur = nullptr;
    theta_schur_factor = nullptr;
    theta_rhs = nullptr;
    theta_stagewise_rhs = nullptr;
  }
  stagewise_scratch =
      new double[2 * dimensions.max_state_dim() * scratch_rhs_dim];

  for (int i = 0; i < num_stages; ++i) {
    const int n_i = dimensions.get_state_dim(i);
    const int m_i = dimensions.get_control_dim(i);
    const int c_i = dimensions.get_c_dim(i);
    const int g_i = dimensions.get_g_dim(i);

    mod_w_inv[i] = new double[g_i];
    Q_mod[i] = new double[n_i * n_i];
    M_mod[i] = new double[max_state_dim * m_i];
    R_mod[i] = new double[m_i * m_i];
    q_mod[i] = new double[n_i];
    r_mod[i] = new double[m_i];
    c_mod[i] = new double[n_i];
    dyn_r2[i] = new double[n_i];
    c_r2_inv[i] = new double[c_i];
  }

  const int n_N = dimensions.get_state_dim(num_stages);
  const int c_N = dimensions.get_c_dim(num_stages);
  const int g_N = dimensions.get_g_dim(num_stages);
  mod_w_inv[num_stages] = new double[g_N];
  Q_mod[num_stages] = new double[n_N * n_N];
  q_mod[num_stages] = new double[n_N];
  c_mod[num_stages] = new double[n_N];
  dyn_r2[num_stages] = new double[n_N];
  c_r2_inv[num_stages] = new double[c_N];
}

void Workspace::RegularizedLQRData::free(int num_stages) {
  for (int i = 0; i < num_stages; ++i) {
    delete[] mod_w_inv[i];
    delete[] Q_mod[i];
    delete[] M_mod[i];
    delete[] R_mod[i];
    delete[] q_mod[i];
    delete[] r_mod[i];
    delete[] c_mod[i];
    delete[] dyn_r2[i];
    delete[] c_r2_inv[i];
  }

  delete[] mod_w_inv[num_stages];
  delete[] Q_mod[num_stages];
  delete[] q_mod[num_stages];
  delete[] c_mod[num_stages];
  delete[] dyn_r2[num_stages];
  delete[] c_r2_inv[num_stages];

  delete[] mod_w_inv;
  delete[] Q_mod;
  delete[] M_mod;
  delete[] R_mod;
  delete[] q_mod;
  delete[] r_mod;
  delete[] c_mod;
  delete[] dyn_r2;
  delete[] c_r2_inv;
  delete[] theta_jacobian;
  delete[] theta_solution;
  delete[] theta_schur;
  delete[] theta_schur_factor;
  delete[] theta_rhs;
  delete[] theta_stagewise_rhs;
  delete[] stagewise_scratch;
}

auto Workspace::RegularizedLQRData::mem_assign(int state_dim, int control_dim,
                                               int num_stages, int c_dim,
                                               int g_dim,
                                               unsigned char *mem_ptr,
                                               int theta_dim) -> int {
  const Dimensions dimensions = {
      .num_stages = num_stages,
      .state_dim = state_dim,
      .control_dim = control_dim,
      .c_dim = c_dim,
      .g_dim = g_dim,
      .theta_dim = theta_dim,
  };
  return mem_assign(dimensions, mem_ptr);
}

auto Workspace::RegularizedLQRData::mem_assign(const Dimensions &dimensions,
                                               unsigned char *mem_ptr) -> int {
  const int num_stages = dimensions.num_stages;
  const int theta_dim = dimensions.theta_dim;
  const int stagewise_kkt_dim = dimensions.get_stagewise_kkt_dim();
  const int scratch_rhs_dim = theta_dim > 0 ? theta_dim : 1;
  const int max_state_dim = dimensions.max_state_dim();
  int cum_size = 0;

  mod_w_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  Q_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  M_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  R_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  q_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  r_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);
  c_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  dyn_r2 = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);
  c_r2_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  for (int i = 0; i < num_stages; ++i) {
    const int n_i = dimensions.get_state_dim(i);
    const int m_i = dimensions.get_control_dim(i);
    const int c_i = dimensions.get_c_dim(i);
    const int g_i = dimensions.get_g_dim(i);

    mod_w_inv[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += g_i * sizeof(double);
    Q_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n_i * n_i * sizeof(double);
    M_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += max_state_dim * m_i * sizeof(double);
    R_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += m_i * m_i * sizeof(double);
    q_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n_i * sizeof(double);
    r_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += m_i * sizeof(double);
    c_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n_i * sizeof(double);
    dyn_r2[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n_i * sizeof(double);
    c_r2_inv[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += c_i * sizeof(double);
  }

  const int n_N = dimensions.get_state_dim(num_stages);
  const int c_N = dimensions.get_c_dim(num_stages);
  const int g_N = dimensions.get_g_dim(num_stages);
  mod_w_inv[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += g_N * sizeof(double);
  Q_mod[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += n_N * n_N * sizeof(double);
  q_mod[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += n_N * sizeof(double);
  c_mod[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += n_N * sizeof(double);
  dyn_r2[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += n_N * sizeof(double);
  c_r2_inv[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += c_N * sizeof(double);

  if (theta_dim > 0) {
    theta_jacobian = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += stagewise_kkt_dim * theta_dim * sizeof(double);
    theta_solution = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += stagewise_kkt_dim * theta_dim * sizeof(double);
    theta_schur = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += theta_dim * theta_dim * sizeof(double);
    theta_schur_factor = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += theta_dim * theta_dim * sizeof(double);
    theta_rhs = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += theta_dim * sizeof(double);
    theta_stagewise_rhs = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += stagewise_kkt_dim * sizeof(double);
  } else {
    theta_jacobian = nullptr;
    theta_solution = nullptr;
    theta_schur = nullptr;
    theta_schur_factor = nullptr;
    theta_rhs = nullptr;
    theta_stagewise_rhs = nullptr;
  }
  stagewise_scratch = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size +=
      2 * dimensions.max_state_dim() * scratch_rhs_dim * sizeof(double);

  assert(cum_size == Workspace::RegularizedLQRData::num_bytes(dimensions));
  return cum_size;
}

auto Workspace::RegularizedLQRData::num_bytes(const Dimensions &dimensions)
    -> int {
  const int T = dimensions.num_stages;
  const int p = dimensions.theta_dim;
  const int stagewise_kkt_dim = dimensions.get_stagewise_kkt_dim();
  const int scratch_rhs_dim = p > 0 ? p : 1;
  const int max_state_dim = dimensions.max_state_dim();
  int total = (6 * (T + 1) + 3 * T) * static_cast<int>(sizeof(double *));

  for (int i = 0; i < T; ++i) {
    const int n_i = dimensions.get_state_dim(i);
    const int m_i = dimensions.get_control_dim(i);
    const int c_i = dimensions.get_c_dim(i);
    const int g_i = dimensions.get_g_dim(i);
    total += (g_i + n_i * n_i + max_state_dim * m_i + m_i * m_i + n_i + m_i +
              n_i + n_i + c_i) *
             static_cast<int>(sizeof(double));
  }

  const int n_N = dimensions.get_state_dim(T);
  const int c_N = dimensions.get_c_dim(T);
  const int g_N = dimensions.get_g_dim(T);
  total += (g_N + n_N * n_N + n_N + n_N + n_N + c_N) *
           static_cast<int>(sizeof(double));

  if (p > 0) {
    total += (2 * stagewise_kkt_dim * p + 2 * p * p + p +
              stagewise_kkt_dim) *
             static_cast<int>(sizeof(double));
  }
  total += 2 * dimensions.max_state_dim() * scratch_rhs_dim *
           static_cast<int>(sizeof(double));
  return total;
}

void Workspace::reserve(int state_dim, int control_dim, int num_stages,
                        int c_dim, int g_dim, int theta_dim) {
  const Dimensions dimensions = {
      .num_stages = num_stages,
      .state_dim = state_dim,
      .control_dim = control_dim,
      .c_dim = c_dim,
      .g_dim = g_dim,
      .theta_dim = theta_dim,
  };
  reserve(dimensions);
}

void Workspace::reserve(const Dimensions &dimensions) {
  model_callback_output.reserve(dimensions);
  model_callback_input.reserve(dimensions.num_stages);

  gradient_f = new double[dimensions.get_x_dim()];
  c = new double[dimensions.get_y_dim()];
  g = new double[dimensions.get_z_dim()];
  state_dims = new int[dimensions.num_nodes()];
  control_dims = new int[dimensions.num_edges()];
  c_dims = new int[dimensions.num_nodes()];
  g_dims = new int[dimensions.num_nodes()];
  x_state_offsets = new int[dimensions.num_nodes()];
  x_control_offsets = new int[dimensions.num_edges()];
  y_dyn_offsets = new int[dimensions.num_nodes()];
  y_c_offsets = new int[dimensions.num_nodes()];
  z_offsets = new int[dimensions.num_nodes()];
  populate_workspace_metadata(*this, dimensions);

  const LQR::Input::Dimensions lqr_dimensions = {
      .state_dim = dimensions.state_dim,
      .control_dim = dimensions.control_dim,
      .num_stages = dimensions.num_stages,
      .state_dims = dimensions.state_dims,
      .control_dims = dimensions.control_dims,
  };
  lqr_workspace.reserve(lqr_dimensions);
  lqr_output.reserve(dimensions.num_stages);

  regularized_lqr_data.reserve(dimensions);

  sip_workspace.reserve(dimensions.get_x_dim(), dimensions.get_z_dim(),
                        dimensions.get_y_dim());
}

void Workspace::free(int num_stages) {
  model_callback_output.free(num_stages);
  model_callback_input.free();
  delete[] gradient_f;
  delete[] c;
  delete[] g;
  delete[] state_dims;
  delete[] control_dims;
  delete[] c_dims;
  delete[] g_dims;
  delete[] x_state_offsets;
  delete[] x_control_offsets;
  delete[] y_dyn_offsets;
  delete[] y_c_offsets;
  delete[] z_offsets;
  lqr_workspace.free(num_stages);
  lqr_output.free();
  regularized_lqr_data.free(num_stages);
  sip_workspace.free();
}

auto Workspace::mem_assign(int state_dim, int control_dim, int num_stages,
                           int c_dim, int g_dim, unsigned char *mem_ptr,
                           int theta_dim)
    -> int {
  const Dimensions dimensions = {
      .num_stages = num_stages,
      .state_dim = state_dim,
      .control_dim = control_dim,
      .c_dim = c_dim,
      .g_dim = g_dim,
      .theta_dim = theta_dim,
  };
  return mem_assign(dimensions, mem_ptr);
}

auto Workspace::mem_assign(const Dimensions &dimensions,
                           unsigned char *mem_ptr) -> int {
  const int x_dim = dimensions.get_x_dim();
  const int y_dim = dimensions.get_y_dim();
  const int z_dim = dimensions.get_z_dim();

  int cum_size = 0;

  cum_size +=
      model_callback_output.mem_assign(dimensions, mem_ptr + cum_size);
  cum_size += model_callback_input.mem_assign(dimensions.num_stages,
                                              mem_ptr + cum_size);

  gradient_f = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  c = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);

  g = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += z_dim * sizeof(double);

  state_dims = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += dimensions.num_nodes() * sizeof(int);
  control_dims = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += dimensions.num_edges() * sizeof(int);
  c_dims = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += dimensions.num_nodes() * sizeof(int);
  g_dims = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += dimensions.num_nodes() * sizeof(int);
  x_state_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += dimensions.num_nodes() * sizeof(int);
  x_control_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += dimensions.num_edges() * sizeof(int);
  y_dyn_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += dimensions.num_nodes() * sizeof(int);
  y_c_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += dimensions.num_nodes() * sizeof(int);
  z_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += dimensions.num_nodes() * sizeof(int);
  populate_workspace_metadata(*this, dimensions);
  cum_size = align_size(cum_size);

  const LQR::Input::Dimensions lqr_dimensions = {
      .state_dim = dimensions.state_dim,
      .control_dim = dimensions.control_dim,
      .num_stages = dimensions.num_stages,
      .state_dims = dimensions.state_dims,
      .control_dims = dimensions.control_dims,
  };
  cum_size += lqr_workspace.mem_assign(lqr_dimensions, mem_ptr + cum_size);

  cum_size = align_size(cum_size);
  cum_size +=
      lqr_output.mem_assign(dimensions.num_stages, mem_ptr + cum_size);

  cum_size = align_size(cum_size);
  cum_size +=
      regularized_lqr_data.mem_assign(dimensions, mem_ptr + cum_size);

  cum_size = align_size(cum_size);
  cum_size += sip_workspace.mem_assign(x_dim, z_dim, y_dim, mem_ptr + cum_size);

  assert(cum_size == Workspace::num_bytes(dimensions));

  return cum_size;
}

auto Workspace::num_bytes(const Dimensions &dimensions) -> int {
  const int x_dim = dimensions.get_x_dim();
  const int y_dim = dimensions.get_y_dim();
  const int z_dim = dimensions.get_z_dim();
  const LQR::Input::Dimensions lqr_dimensions = {
      .state_dim = dimensions.state_dim,
      .control_dim = dimensions.control_dim,
      .num_stages = dimensions.num_stages,
      .state_dims = dimensions.state_dims,
      .control_dims = dimensions.control_dims,
  };
  int total = ModelCallbackOutput::num_bytes(dimensions) +
              ModelCallbackInput::num_bytes(dimensions.num_stages) +
              (x_dim + y_dim + z_dim) * static_cast<int>(sizeof(double)) +
              (7 * dimensions.num_nodes() + 2 * dimensions.num_edges()) *
                  static_cast<int>(sizeof(int));
  total = align_size(total);
  total += LQR::Workspace::num_bytes(lqr_dimensions);
  total = align_size(total);
  total += LQR::Output::num_bytes(dimensions.num_stages);
  total = align_size(total);
  total += RegularizedLQRData::num_bytes(dimensions);
  total = align_size(total);
  total += sip::Workspace::num_bytes(x_dim, z_dim, y_dim);
  return total;
}

} // namespace sip::optimal_control

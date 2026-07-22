#include "sip_optimal_control/types.hpp"

#include <cassert>
#include <cstddef>

namespace sip::optimal_control {

namespace {

auto align_size(const int size) -> int {
  constexpr int alignment = alignof(std::max_align_t);
  return ((size + alignment - 1) / alignment) * alignment;
}

auto incoming_parent(const Topology &topology, const int node) -> int {
  for (int edge = 0; edge < topology.num_edges; ++edge) {
    if (topology.edge_children[edge] == node) {
      return topology.edge_parents[edge];
    }
  }
  return -1;
}

void populate_workspace_metadata(Workspace &workspace,
                                 const Dimensions &dimensions,
                                 const int num_edges) {
  const int num_nodes = num_edges + 1;
  workspace.stagewise_x_dim = dimensions.get_stagewise_x_dim(num_edges);
  workspace.x_dim = dimensions.get_x_dim(num_edges);
  workspace.y_dim = dimensions.get_y_dim(num_edges);
  workspace.z_dim = dimensions.get_z_dim(num_edges);
  workspace.stagewise_kkt_dim = dimensions.get_stagewise_kkt_dim(num_edges);
  int x_offset = 0;
  for (int node = 0; node < num_nodes; ++node) {
    workspace.x_state_offsets[node] = x_offset;
    if (node < num_edges) {
      x_offset += dimensions.get_state_dim(node);
      workspace.x_control_offsets[node] = x_offset;
      x_offset += dimensions.get_control_dim(node);
    }
  }

  int y_offset = 0;
  for (int node = 0; node < num_nodes; ++node) {
    workspace.y_dyn_offsets[node] = y_offset;
    y_offset += dimensions.get_state_dim(node);
    workspace.y_node_c_offsets[node] = y_offset;
    y_offset += dimensions.get_node_c_dim(node);
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    workspace.y_edge_c_offsets[edge] = y_offset;
    y_offset += dimensions.get_edge_c_dim(edge);
  }

  int z_offset = 0;
  for (int node = 0; node < num_nodes; ++node) {
    workspace.z_node_offsets[node] = z_offset;
    z_offset += dimensions.get_node_g_dim(node);
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    workspace.z_edge_offsets[edge] = z_offset;
    z_offset += dimensions.get_edge_g_dim(edge);
  }
}

} // namespace

auto validate_input(const Dimensions &dimensions, const Topology &topology)
    -> InputValidationStatus {
  const int num_edges = topology.num_edges;
  const int num_nodes = topology.num_nodes();
  if (num_edges < 0 || dimensions.theta_dim < 0) {
    return InputValidationStatus::INVALID_DIMENSIONS;
  }

  for (int node = 0; node < num_nodes; ++node) {
    if (dimensions.get_state_dim(node) < 0 ||
        dimensions.get_node_c_dim(node) < 0 ||
        dimensions.get_node_g_dim(node) < 0) {
      return InputValidationStatus::INVALID_DIMENSIONS;
    }
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    if (dimensions.get_control_dim(edge) < 0 ||
        dimensions.get_edge_c_dim(edge) < 0 ||
        dimensions.get_edge_g_dim(edge) < 0) {
      return InputValidationStatus::INVALID_DIMENSIONS;
    }
  }

  if (topology.edge_parents == nullptr || topology.edge_children == nullptr) {
    return InputValidationStatus::INVALID_TOPOLOGY;
  }
  const int root = topology.root;
  if (root < 0 || root >= num_nodes) {
    return InputValidationStatus::INVALID_TOPOLOGY;
  }

  for (int edge = 0; edge < num_edges; ++edge) {
    const int parent = topology.edge_parents[edge];
    const int child = topology.edge_children[edge];
    if (parent < 0 || parent >= num_nodes || child < 0 || child >= num_nodes ||
        parent == child) {
      return InputValidationStatus::INVALID_TOPOLOGY;
    }
  }

  for (int node = 0; node < num_nodes; ++node) {
    int indegree = 0;
    for (int edge = 0; edge < num_edges; ++edge) {
      if (topology.edge_children[edge] == node) {
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
      current = incoming_parent(topology, current);
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

void ModelCallbackInput::reserve(const Topology &topology) {
  theta = nullptr;
  nodes = new NodeModelCallbackInput[topology.num_nodes()];
  edges = new EdgeModelCallbackInput[topology.num_edges];
}

void ModelCallbackInput::free() {
  delete[] nodes;
  delete[] edges;
}

auto ModelCallbackInput::mem_assign(const Topology &topology,
                                    unsigned char *mem_ptr) -> int {
  int cum_size = 0;
  theta = nullptr;
  nodes = reinterpret_cast<NodeModelCallbackInput *>(mem_ptr + cum_size);
  cum_size += topology.num_nodes() * sizeof(NodeModelCallbackInput);
  edges = reinterpret_cast<EdgeModelCallbackInput *>(mem_ptr + cum_size);
  cum_size += topology.num_edges * sizeof(EdgeModelCallbackInput);
  assert(cum_size == ModelCallbackInput::num_bytes(topology.num_edges));
  return cum_size;
}

namespace {

void reserve_node_output(NodeModelCallbackOutput &output, const int n,
                         const int c, const int g, const int p) {
  output.df_dx = new double[n];
  output.df_dtheta = new double[p];
  output.c = new double[c];
  output.dc_dx = new double[c * n];
  output.dc_dtheta = new double[c * p];
  output.g = new double[g];
  output.dg_dx = new double[g * n];
  output.dg_dtheta = new double[g * p];
  output.d2L_dx2 = new double[n * n];
  output.d2L_dxdtheta = new double[n * p];
  output.d2L_dtheta2 = new double[p * p];
}

void free_node_output(NodeModelCallbackOutput &output) {
  delete[] output.df_dx;
  delete[] output.df_dtheta;
  delete[] output.c;
  delete[] output.dc_dx;
  delete[] output.dc_dtheta;
  delete[] output.g;
  delete[] output.dg_dx;
  delete[] output.dg_dtheta;
  delete[] output.d2L_dx2;
  delete[] output.d2L_dxdtheta;
  delete[] output.d2L_dtheta2;
}

void reserve_edge_output(EdgeModelCallbackOutput &output, const int n_parent,
                         const int n_child, const int m, const int c,
                         const int g, const int p) {
  output.df_dx = new double[n_parent];
  output.df_du = new double[m];
  output.df_dtheta = new double[p];
  output.dyn_res = new double[n_child];
  output.ddyn_dx = new double[n_child * n_parent];
  output.ddyn_du = new double[n_child * m];
  output.ddyn_dtheta = new double[n_child * p];
  output.c = new double[c];
  output.dc_dx = new double[c * n_parent];
  output.dc_du = new double[c * m];
  output.dc_dtheta = new double[c * p];
  output.g = new double[g];
  output.dg_dx = new double[g * n_parent];
  output.dg_du = new double[g * m];
  output.dg_dtheta = new double[g * p];
  output.d2L_dx2 = new double[n_parent * n_parent];
  output.d2L_dxdu = new double[n_parent * m];
  output.d2L_du2 = new double[m * m];
  output.d2L_dxdtheta = new double[n_parent * p];
  output.d2L_dudtheta = new double[m * p];
  output.d2L_dtheta2 = new double[p * p];
}

void free_edge_output(EdgeModelCallbackOutput &output) {
  delete[] output.df_dx;
  delete[] output.df_du;
  delete[] output.df_dtheta;
  delete[] output.dyn_res;
  delete[] output.ddyn_dx;
  delete[] output.ddyn_du;
  delete[] output.ddyn_dtheta;
  delete[] output.c;
  delete[] output.dc_dx;
  delete[] output.dc_du;
  delete[] output.dc_dtheta;
  delete[] output.g;
  delete[] output.dg_dx;
  delete[] output.dg_du;
  delete[] output.dg_dtheta;
  delete[] output.d2L_dx2;
  delete[] output.d2L_dxdu;
  delete[] output.d2L_du2;
  delete[] output.d2L_dxdtheta;
  delete[] output.d2L_dudtheta;
  delete[] output.d2L_dtheta2;
}

void assign(double *&target, const int count, unsigned char *mem_ptr,
            int &cum_size) {
  target = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += count * sizeof(double);
}

void assign_node_output(NodeModelCallbackOutput &output, const int n,
                        const int c, const int g, const int p,
                        unsigned char *mem_ptr, int &cum_size) {
  assign(output.df_dx, n, mem_ptr, cum_size);
  assign(output.df_dtheta, p, mem_ptr, cum_size);
  assign(output.c, c, mem_ptr, cum_size);
  assign(output.dc_dx, c * n, mem_ptr, cum_size);
  assign(output.dc_dtheta, c * p, mem_ptr, cum_size);
  assign(output.g, g, mem_ptr, cum_size);
  assign(output.dg_dx, g * n, mem_ptr, cum_size);
  assign(output.dg_dtheta, g * p, mem_ptr, cum_size);
  assign(output.d2L_dx2, n * n, mem_ptr, cum_size);
  assign(output.d2L_dxdtheta, n * p, mem_ptr, cum_size);
  assign(output.d2L_dtheta2, p * p, mem_ptr, cum_size);
}

void assign_edge_output(EdgeModelCallbackOutput &output, const int n_parent,
                        const int n_child, const int m, const int c,
                        const int g, const int p, unsigned char *mem_ptr,
                        int &cum_size) {
  assign(output.df_dx, n_parent, mem_ptr, cum_size);
  assign(output.df_du, m, mem_ptr, cum_size);
  assign(output.df_dtheta, p, mem_ptr, cum_size);
  assign(output.dyn_res, n_child, mem_ptr, cum_size);
  assign(output.ddyn_dx, n_child * n_parent, mem_ptr, cum_size);
  assign(output.ddyn_du, n_child * m, mem_ptr, cum_size);
  assign(output.ddyn_dtheta, n_child * p, mem_ptr, cum_size);
  assign(output.c, c, mem_ptr, cum_size);
  assign(output.dc_dx, c * n_parent, mem_ptr, cum_size);
  assign(output.dc_du, c * m, mem_ptr, cum_size);
  assign(output.dc_dtheta, c * p, mem_ptr, cum_size);
  assign(output.g, g, mem_ptr, cum_size);
  assign(output.dg_dx, g * n_parent, mem_ptr, cum_size);
  assign(output.dg_du, g * m, mem_ptr, cum_size);
  assign(output.dg_dtheta, g * p, mem_ptr, cum_size);
  assign(output.d2L_dx2, n_parent * n_parent, mem_ptr, cum_size);
  assign(output.d2L_dxdu, n_parent * m, mem_ptr, cum_size);
  assign(output.d2L_du2, m * m, mem_ptr, cum_size);
  assign(output.d2L_dxdtheta, n_parent * p, mem_ptr, cum_size);
  assign(output.d2L_dudtheta, m * p, mem_ptr, cum_size);
  assign(output.d2L_dtheta2, p * p, mem_ptr, cum_size);
}

} // namespace

void ModelCallbackOutput::reserve(const Dimensions &dimensions,
                                  const Topology &topology) {
  const int num_edges = topology.num_edges;
  const int num_nodes = topology.num_nodes();
  const int p = dimensions.theta_dim;
  nodes = new NodeModelCallbackOutput[num_nodes];
  edges = new EdgeModelCallbackOutput[num_edges];
  for (int node = 0; node < num_nodes; ++node) {
    reserve_node_output(nodes[node], dimensions.get_state_dim(node),
                        dimensions.get_node_c_dim(node),
                        dimensions.get_node_g_dim(node), p);
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    const int parent = topology.edge_parents[edge];
    const int child = topology.edge_children[edge];
    reserve_edge_output(
        edges[edge], dimensions.get_state_dim(parent),
        dimensions.get_state_dim(child), dimensions.get_control_dim(edge),
        dimensions.get_edge_c_dim(edge), dimensions.get_edge_g_dim(edge), p);
  }
}

void ModelCallbackOutput::free(const Topology &topology) {
  for (int node = 0; node < topology.num_nodes(); ++node) {
    free_node_output(nodes[node]);
  }
  for (int edge = 0; edge < topology.num_edges; ++edge) {
    free_edge_output(edges[edge]);
  }
  delete[] nodes;
  delete[] edges;
}

auto ModelCallbackOutput::mem_assign(const Dimensions &dimensions,
                                     const Topology &topology,
                                     unsigned char *mem_ptr) -> int {
  const int num_edges = topology.num_edges;
  const int num_nodes = topology.num_nodes();
  const int p = dimensions.theta_dim;
  int cum_size = 0;
  nodes = reinterpret_cast<NodeModelCallbackOutput *>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(NodeModelCallbackOutput);
  edges = reinterpret_cast<EdgeModelCallbackOutput *>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(EdgeModelCallbackOutput);
  for (int node = 0; node < num_nodes; ++node) {
    assign_node_output(nodes[node], dimensions.get_state_dim(node),
                       dimensions.get_node_c_dim(node),
                       dimensions.get_node_g_dim(node), p, mem_ptr, cum_size);
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    const int parent = topology.edge_parents[edge];
    const int child = topology.edge_children[edge];
    assign_edge_output(edges[edge], dimensions.get_state_dim(parent),
                       dimensions.get_state_dim(child),
                       dimensions.get_control_dim(edge),
                       dimensions.get_edge_c_dim(edge),
                       dimensions.get_edge_g_dim(edge), p, mem_ptr, cum_size);
  }
  assert(cum_size == ModelCallbackOutput::num_bytes(dimensions, topology));
  return cum_size;
}

auto ModelCallbackOutput::num_bytes(const Dimensions &dimensions,
                                    const Topology &topology) -> int {
  const int num_edges = topology.num_edges;
  const int num_nodes = topology.num_nodes();
  const int p = dimensions.theta_dim;
  int total = num_nodes * sizeof(NodeModelCallbackOutput) +
              num_edges * sizeof(EdgeModelCallbackOutput);
  for (int node = 0; node < num_nodes; ++node) {
    const int n = dimensions.get_state_dim(node);
    const int c = dimensions.get_node_c_dim(node);
    const int g = dimensions.get_node_g_dim(node);
    total += (n + p + c + c * n + c * p + g + g * n + g * p + n * n + n * p +
              p * p) *
             static_cast<int>(sizeof(double));
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    const int parent = topology.edge_parents[edge];
    const int child = topology.edge_children[edge];
    const int n_parent = dimensions.get_state_dim(parent);
    const int n_child = dimensions.get_state_dim(child);
    const int m = dimensions.get_control_dim(edge);
    const int c = dimensions.get_edge_c_dim(edge);
    const int g = dimensions.get_edge_g_dim(edge);
    total += (n_parent + m + p + n_child + n_child * n_parent + n_child * m +
              n_child * p + c + c * n_parent + c * m + c * p + g +
              g * n_parent + g * m + g * p + n_parent * n_parent +
              n_parent * m + m * m + n_parent * p + m * p + p * p) *
             static_cast<int>(sizeof(double));
  }
  return total;
}

void Workspace::RegularizedLQRData::reserve(const Dimensions &dimensions,
                                            const int num_edges) {
  const int num_nodes = num_edges + 1;
  const int theta_dim = dimensions.theta_dim;
  const int scratch_rhs_dim = theta_dim > 0 ? theta_dim : 1;
  const int max_state_dim = dimensions.max_state_dim(num_nodes);

  node_mod_w_inv = new double *[num_nodes];
  edge_mod_w_inv = new double *[num_edges];
  Q_mod = new double *[num_edges + 1];
  M_mod = new double *[num_edges];
  R_mod = new double *[num_edges];
  q_mod = new double *[num_edges + 1];
  r_mod = new double *[num_edges];
  c_mod = new double *[num_edges + 1];
  dyn_r2 = new double *[num_edges + 1];
  node_c_r2_inv = new double *[num_nodes];
  edge_c_r2_inv = new double *[num_edges];
  const int stagewise_kkt_dim = dimensions.get_stagewise_kkt_dim(num_edges);
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
      new double[2 * dimensions.max_state_dim(num_nodes) * scratch_rhs_dim];

  for (int node = 0; node < num_nodes; ++node) {
    const int n = dimensions.get_state_dim(node);
    node_mod_w_inv[node] = new double[dimensions.get_node_g_dim(node)];
    Q_mod[node] = new double[n * n];
    q_mod[node] = new double[n];
    c_mod[node] = new double[n];
    dyn_r2[node] = new double[n];
    node_c_r2_inv[node] = new double[dimensions.get_node_c_dim(node)];
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    const int m = dimensions.get_control_dim(edge);
    edge_mod_w_inv[edge] = new double[dimensions.get_edge_g_dim(edge)];
    M_mod[edge] = new double[max_state_dim * m];
    R_mod[edge] = new double[m * m];
    r_mod[edge] = new double[m];
    edge_c_r2_inv[edge] = new double[dimensions.get_edge_c_dim(edge)];
  }
}

void Workspace::RegularizedLQRData::free(int num_edges) {
  for (int node = 0; node <= num_edges; ++node) {
    delete[] node_mod_w_inv[node];
    delete[] Q_mod[node];
    delete[] q_mod[node];
    delete[] c_mod[node];
    delete[] dyn_r2[node];
    delete[] node_c_r2_inv[node];
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    delete[] edge_mod_w_inv[edge];
    delete[] M_mod[edge];
    delete[] R_mod[edge];
    delete[] r_mod[edge];
    delete[] edge_c_r2_inv[edge];
  }
  delete[] node_mod_w_inv;
  delete[] edge_mod_w_inv;
  delete[] Q_mod;
  delete[] M_mod;
  delete[] R_mod;
  delete[] q_mod;
  delete[] r_mod;
  delete[] c_mod;
  delete[] dyn_r2;
  delete[] node_c_r2_inv;
  delete[] edge_c_r2_inv;
  delete[] theta_jacobian;
  delete[] theta_solution;
  delete[] theta_schur;
  delete[] theta_schur_factor;
  delete[] theta_rhs;
  delete[] theta_stagewise_rhs;
  delete[] stagewise_scratch;
}

auto Workspace::RegularizedLQRData::mem_assign(const Dimensions &dimensions,
                                               const int num_edges,
                                               unsigned char *mem_ptr) -> int {
  const int num_nodes = num_edges + 1;
  const int theta_dim = dimensions.theta_dim;
  const int stagewise_kkt_dim = dimensions.get_stagewise_kkt_dim(num_edges);
  const int scratch_rhs_dim = theta_dim > 0 ? theta_dim : 1;
  const int max_state_dim = dimensions.max_state_dim(num_nodes);
  int cum_size = 0;

  node_mod_w_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(double *);
  edge_mod_w_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(double *);
  Q_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_edges + 1) * sizeof(double *);
  M_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(double *);
  R_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(double *);
  q_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_edges + 1) * sizeof(double *);
  r_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(double *);
  c_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_edges + 1) * sizeof(double *);
  dyn_r2 = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_edges + 1) * sizeof(double *);
  node_c_r2_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(double *);
  edge_c_r2_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(double *);

  for (int node = 0; node < num_nodes; ++node) {
    const int n = dimensions.get_state_dim(node);
    node_mod_w_inv[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += dimensions.get_node_g_dim(node) * sizeof(double);
    Q_mod[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n * n * sizeof(double);
    q_mod[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n * sizeof(double);
    c_mod[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n * sizeof(double);
    dyn_r2[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += n * sizeof(double);
    node_c_r2_inv[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += dimensions.get_node_c_dim(node) * sizeof(double);
  }
  for (int edge = 0; edge < num_edges; ++edge) {
    const int m = dimensions.get_control_dim(edge);
    edge_mod_w_inv[edge] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += dimensions.get_edge_g_dim(edge) * sizeof(double);
    M_mod[edge] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += max_state_dim * m * sizeof(double);
    R_mod[edge] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += m * m * sizeof(double);
    r_mod[edge] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += m * sizeof(double);
    edge_c_r2_inv[edge] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += dimensions.get_edge_c_dim(edge) * sizeof(double);
  }

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
  cum_size += 2 * dimensions.max_state_dim(num_nodes) * scratch_rhs_dim *
              sizeof(double);

  assert(cum_size ==
         Workspace::RegularizedLQRData::num_bytes(dimensions, num_edges));
  return cum_size;
}

auto Input::num_bound_sides() const -> int {
  return ::sip::num_bound_sides(lower_bounds, upper_bounds,
                                dimensions.get_x_dim(topology.num_edges));
}

auto Workspace::RegularizedLQRData::num_bytes(const Dimensions &dimensions,
                                              const int num_edges) -> int {
  const int T = num_edges;
  const int num_nodes = num_edges + 1;
  const int p = dimensions.theta_dim;
  const int stagewise_kkt_dim = dimensions.get_stagewise_kkt_dim(num_edges);
  const int scratch_rhs_dim = p > 0 ? p : 1;
  const int max_state_dim = dimensions.max_state_dim(num_nodes);
  int total = (6 * (T + 1) + 5 * T) * static_cast<int>(sizeof(double *));

  for (int node = 0; node < num_nodes; ++node) {
    const int n = dimensions.get_state_dim(node);
    total += (dimensions.get_node_g_dim(node) + n * n + 3 * n +
              dimensions.get_node_c_dim(node)) *
             static_cast<int>(sizeof(double));
  }
  for (int edge = 0; edge < T; ++edge) {
    const int m = dimensions.get_control_dim(edge);
    total += (dimensions.get_edge_g_dim(edge) + max_state_dim * m + m * m + m +
              dimensions.get_edge_c_dim(edge)) *
             static_cast<int>(sizeof(double));
  }

  if (p > 0) {
    total += (2 * stagewise_kkt_dim * p + 2 * p * p + p + stagewise_kkt_dim) *
             static_cast<int>(sizeof(double));
  }
  total += 2 * dimensions.max_state_dim(num_nodes) * scratch_rhs_dim *
           static_cast<int>(sizeof(double));
  return total;
}

void Workspace::reserve(const Dimensions &dimensions, const Topology &topology,
                        const int num_bound_sides,
                        const sip::Settings &settings) {
  const int num_edges = topology.num_edges;
  const int num_nodes = topology.num_nodes();
  model_callback_input.reserve(topology);
  model_callback_output.reserve(dimensions, topology);

  gradient_f = new double[dimensions.get_x_dim(num_edges)];
  c = new double[dimensions.get_y_dim(num_edges)];
  g = new double[dimensions.get_z_dim(num_edges)];
  x_state_offsets = new int[num_nodes];
  x_control_offsets = new int[num_edges];
  y_dyn_offsets = new int[num_nodes];
  y_node_c_offsets = new int[num_nodes];
  y_edge_c_offsets = new int[num_edges];
  z_node_offsets = new int[num_nodes];
  z_edge_offsets = new int[num_edges];
  ddyn_dx = new double *[num_edges];
  ddyn_du = new double *[num_edges];
  for (int edge = 0; edge < num_edges; ++edge) {
    ddyn_dx[edge] = model_callback_output.edges[edge].ddyn_dx;
    ddyn_du[edge] = model_callback_output.edges[edge].ddyn_du;
  }
  populate_workspace_metadata(*this, dimensions, num_edges);

  lqr_workspace.reserve(dimensions, topology);
  lqr_output.reserve(num_edges);

  regularized_lqr_data.reserve(dimensions, num_edges);

  sip_workspace.reserve(
      dimensions.get_x_dim(num_edges), dimensions.get_z_dim(num_edges),
      dimensions.get_y_dim(num_edges), num_bound_sides, settings);
}

void Workspace::free(const Topology &topology) {
  const int num_edges = topology.num_edges;
  model_callback_input.free();
  model_callback_output.free(topology);
  delete[] gradient_f;
  delete[] c;
  delete[] g;
  delete[] x_state_offsets;
  delete[] x_control_offsets;
  delete[] y_dyn_offsets;
  delete[] y_node_c_offsets;
  delete[] y_edge_c_offsets;
  delete[] z_node_offsets;
  delete[] z_edge_offsets;
  delete[] ddyn_dx;
  delete[] ddyn_du;
  lqr_workspace.free(num_edges);
  lqr_output.free();
  regularized_lqr_data.free(num_edges);
  sip_workspace.free();
}

auto Workspace::mem_assign(const Dimensions &dimensions,
                           const Topology &topology, const int num_bound_sides,
                           const sip::Settings &settings,
                           unsigned char *mem_ptr) -> int {
  const int num_edges = topology.num_edges;
  const int num_nodes = topology.num_nodes();
  const int x_dim = dimensions.get_x_dim(num_edges);
  const int y_dim = dimensions.get_y_dim(num_edges);
  const int z_dim = dimensions.get_z_dim(num_edges);

  int cum_size = 0;

  cum_size += model_callback_input.mem_assign(topology, mem_ptr + cum_size);
  cum_size += model_callback_output.mem_assign(dimensions, topology,
                                               mem_ptr + cum_size);

  gradient_f = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  c = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);

  g = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += z_dim * sizeof(double);

  x_state_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(int);
  x_control_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(int);
  y_dyn_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(int);
  y_node_c_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(int);
  y_edge_c_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(int);
  z_node_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(int);
  z_edge_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(int);
  ddyn_dx = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(double *);
  ddyn_du = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_edges * sizeof(double *);
  for (int edge = 0; edge < num_edges; ++edge) {
    ddyn_dx[edge] = model_callback_output.edges[edge].ddyn_dx;
    ddyn_du[edge] = model_callback_output.edges[edge].ddyn_du;
  }
  populate_workspace_metadata(*this, dimensions, num_edges);
  cum_size = align_size(cum_size);

  cum_size +=
      lqr_workspace.mem_assign(dimensions, topology, mem_ptr + cum_size);

  cum_size = align_size(cum_size);
  cum_size += lqr_output.mem_assign(num_edges, mem_ptr + cum_size);

  cum_size = align_size(cum_size);
  cum_size += regularized_lqr_data.mem_assign(dimensions, num_edges,
                                              mem_ptr + cum_size);

  cum_size = align_size(cum_size);
  cum_size += sip_workspace.mem_assign(x_dim, z_dim, y_dim, num_bound_sides,
                                       settings, mem_ptr + cum_size);

  assert(cum_size ==
         Workspace::num_bytes(dimensions, topology, num_bound_sides, settings));

  return cum_size;
}

auto Workspace::num_bytes(const Dimensions &dimensions,
                          const Topology &topology, const int num_bound_sides,
                          const sip::Settings &settings) -> int {
  const int num_edges = topology.num_edges;
  const int num_nodes = topology.num_nodes();
  const int x_dim = dimensions.get_x_dim(num_edges);
  const int y_dim = dimensions.get_y_dim(num_edges);
  const int z_dim = dimensions.get_z_dim(num_edges);
  int total = ModelCallbackInput::num_bytes(num_edges) +
              ModelCallbackOutput::num_bytes(dimensions, topology) +
              (x_dim + y_dim + z_dim) * static_cast<int>(sizeof(double)) +
              (4 * num_nodes + 3 * num_edges) * static_cast<int>(sizeof(int)) +
              2 * num_edges * static_cast<int>(sizeof(double *));
  total = align_size(total);
  total += LQR::Workspace::num_bytes(dimensions, topology);
  total = align_size(total);
  total += LQR::Output::num_bytes(num_edges);
  total = align_size(total);
  total += RegularizedLQRData::num_bytes(dimensions, num_edges);
  total = align_size(total);
  total +=
      sip::Workspace::num_bytes(x_dim, z_dim, y_dim, num_bound_sides, settings);
  return total;
}

} // namespace sip::optimal_control

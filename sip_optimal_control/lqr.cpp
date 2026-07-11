#define EIGEN_NO_MALLOC

#include "sip_optimal_control/lqr.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace sip::optimal_control {

void LQR::Output::reserve(int num_stages) {
  x = new double *[num_stages + 1];
  u = new double *[num_stages];
  y = new double *[num_stages + 1];
}

void LQR::Output::free() {
  delete[] x;
  delete[] u;
  delete[] y;
}

auto LQR::Output::mem_assign(int num_stages, unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  x = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  u = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  y = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  assert(cum_size == LQR::Output::num_bytes(num_stages));

  return cum_size;
}

void LQR::Workspace::reserve(int state_dim, int control_dim, int num_stages) {
  const Input::Dimensions dimensions = {
      .state_dim = state_dim,
      .control_dim = control_dim,
      .num_stages = num_stages,
  };
  reserve(dimensions);
}

void LQR::Workspace::reserve(const Input::Dimensions &dimensions) {
  const int num_stages = dimensions.num_edges();
  const int num_nodes = dimensions.num_nodes();
  const int max_state_dim = dimensions.max_state_dim();
  const int max_control_dim = dimensions.max_control_dim();

  W = new double *[num_stages];
  K = new double *[num_stages];
  V = new double *[num_nodes];
  G_factor = new double *[num_stages];
  F_factor = new double *[num_nodes];
  sqrt_delta = new double *[num_nodes];
  sqrt_delta_inv = new double *[num_nodes];
  k = new double *[num_stages];
  v = new double *[num_nodes];

  for (int i = 0; i < num_stages; ++i) {
    const int control_dim = dimensions.get_control_dim(i);

    W[i] = new double[max_state_dim * max_state_dim];
    K[i] = new double[control_dim * max_state_dim];
    G_factor[i] = new double[control_dim * control_dim];
    k[i] = new double[control_dim];
  }

  for (int node = 0; node < num_nodes; ++node) {
    const int state_dim = dimensions.get_state_dim(node);
    V[node] = new double[state_dim * state_dim];
    v[node] = new double[state_dim];
    F_factor[node] = new double[state_dim * state_dim];
    sqrt_delta[node] = new double[state_dim];
    sqrt_delta_inv[node] = new double[state_dim];
  }

  G = new double[max_control_dim * max_control_dim];
  g = new double[max_state_dim];
  H = new double[max_control_dim * max_state_dim];
  h = new double[max_control_dim];
  F = new double[max_state_dim * max_state_dim];
  f = new double[max_state_dim];

  child_offsets = new int[num_nodes + 1];
  child_edges = new int[num_stages];
  edge_parents = new int[num_stages];
  edge_children = new int[num_stages];
  preorder_nodes = new int[num_nodes];
  postorder_nodes = new int[num_nodes];
  node_marks = new int[num_nodes];
  topology_is_initialized = false;
  topology_status = FactorStatus::SUCCESS;
  topology_state_dim = 0;
  topology_control_dim = 0;
  topology_num_stages = 0;
  topology_state_dims = nullptr;
  topology_control_dims = nullptr;
  topology_context = nullptr;
  topology_root = nullptr;
  topology_edge_parent = nullptr;
  topology_edge_child = nullptr;
}

void LQR::Workspace::free(int num_stages) {
  (void)num_stages;

  delete[] child_offsets;
  delete[] child_edges;
  delete[] edge_parents;
  delete[] edge_children;
  delete[] preorder_nodes;
  delete[] postorder_nodes;
  delete[] node_marks;

  delete[] G;
  delete[] g;
  delete[] H;
  delete[] h;
  delete[] F;
  delete[] f;

  for (int i = 0; i < num_stages; ++i) {
    delete[] W[i];
    delete[] K[i];
    delete[] V[i];
    delete[] G_factor[i];
    delete[] F_factor[i];
    delete[] sqrt_delta[i];
    delete[] sqrt_delta_inv[i];
    delete[] k[i];
    delete[] v[i];
  }

  delete[] V[num_stages];
  delete[] v[num_stages];
  delete[] F_factor[num_stages];
  delete[] sqrt_delta[num_stages];
  delete[] sqrt_delta_inv[num_stages];

  delete[] W;
  delete[] K;
  delete[] V;
  delete[] G_factor;
  delete[] F_factor;
  delete[] sqrt_delta;
  delete[] sqrt_delta_inv;
  delete[] k;
  delete[] v;
}

auto LQR::Workspace::mem_assign(int state_dim, int control_dim, int num_stages,
                                unsigned char *mem_ptr) -> int {
  const Input::Dimensions dimensions = {
      .state_dim = state_dim,
      .control_dim = control_dim,
      .num_stages = num_stages,
  };
  return mem_assign(dimensions, mem_ptr);
}

auto LQR::Workspace::mem_assign(const Input::Dimensions &dimensions,
                                unsigned char *mem_ptr) -> int {
  const int num_stages = dimensions.num_edges();
  const int num_nodes = dimensions.num_nodes();
  const int max_state_dim = dimensions.max_state_dim();
  const int max_control_dim = dimensions.max_control_dim();

  int cum_size = 0;

  W = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  K = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  V = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  G_factor = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  F_factor = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(double *);

  sqrt_delta = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(double *);

  sqrt_delta_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(double *);

  k = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  v = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(double *);

  for (int i = 0; i < num_stages; ++i) {
    const int control_dim = dimensions.get_control_dim(i);

    W[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += max_state_dim * max_state_dim * sizeof(double);

    K[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * max_state_dim * sizeof(double);

    G_factor[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * control_dim * sizeof(double);

    k[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * sizeof(double);
  }

  for (int node = 0; node < num_nodes; ++node) {
    const int state_dim = dimensions.get_state_dim(node);

    V[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * state_dim * sizeof(double);

    F_factor[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * state_dim * sizeof(double);

    sqrt_delta[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double);

    sqrt_delta_inv[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double);

    v[node] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double);
  }

  G = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += max_control_dim * max_control_dim * sizeof(double);

  g = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += max_state_dim * sizeof(double);

  H = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += max_control_dim * max_state_dim * sizeof(double);

  h = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += max_control_dim * sizeof(double);

  F = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += max_state_dim * max_state_dim * sizeof(double);

  f = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += max_state_dim * sizeof(double);

  child_offsets = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += (num_nodes + 1) * sizeof(int);

  child_edges = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(int);

  edge_parents = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(int);

  edge_children = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(int);

  preorder_nodes = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(int);

  postorder_nodes = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(int);

  node_marks = reinterpret_cast<int *>(mem_ptr + cum_size);
  cum_size += num_nodes * sizeof(int);
  topology_is_initialized = false;
  topology_status = FactorStatus::SUCCESS;
  topology_state_dim = 0;
  topology_control_dim = 0;
  topology_num_stages = 0;
  topology_state_dims = nullptr;
  topology_control_dims = nullptr;
  topology_context = nullptr;
  topology_root = nullptr;
  topology_edge_parent = nullptr;
  topology_edge_child = nullptr;

  assert(cum_size == LQR::Workspace::num_bytes(dimensions));

  return cum_size;
}

auto LQR::Workspace::num_bytes(const Input::Dimensions &dimensions) -> int {
  const int num_edges = dimensions.num_edges();
  const int num_nodes = dimensions.num_nodes();
  const int max_state_dim = dimensions.max_state_dim();
  const int max_control_dim = dimensions.max_control_dim();

  int edge_data_size = 0;
  for (int edge = 0; edge < num_edges; ++edge) {
    const int control_dim = dimensions.get_control_dim(edge);
    edge_data_size +=
        (max_state_dim * max_state_dim + control_dim * max_state_dim +
         control_dim * control_dim + control_dim) *
        static_cast<int>(sizeof(double));
  }

  int node_data_size = 0;
  for (int node = 0; node < num_nodes; ++node) {
    const int state_dim = dimensions.get_state_dim(node);
    node_data_size +=
        (2 * state_dim * state_dim + 3 * state_dim) *
        static_cast<int>(sizeof(double));
  }

  const int pointer_size =
      (4 * num_edges + 5 * num_nodes) * static_cast<int>(sizeof(double *));
  const int scratch_size =
      (max_control_dim * max_control_dim + 2 * max_state_dim +
       max_control_dim * max_state_dim + max_control_dim +
       max_state_dim * max_state_dim) *
      static_cast<int>(sizeof(double));
  const int topology_size =
      (num_nodes + 1 + 3 * num_edges + 3 * num_nodes) *
      static_cast<int>(sizeof(int));

  return pointer_size + edge_data_size + node_data_size + scratch_size +
         topology_size;
}

namespace {

bool compute_delta_sqrt(const double *delta_data, double *sqrt_delta_data,
                        double *sqrt_delta_inv_data, const int n) {
  for (int i = 0; i < n; ++i) {
    if (delta_data[i] <= 0.0) {
      return false;
    }
    sqrt_delta_data[i] = std::sqrt(delta_data[i]);
    sqrt_delta_inv_data[i] = 1.0 / sqrt_delta_data[i];
  }
  return true;
}

auto factor_F(const double *delta_data,
              const Eigen::Ref<const Eigen::MatrixXd> &V, double *F_factor_data,
              double *sqrt_delta_data, double *sqrt_delta_inv_data, const int n)
    -> LQR::FactorStatus {
  if (!compute_delta_sqrt(delta_data, sqrt_delta_data, sqrt_delta_inv_data,
                          n)) {
    return LQR::FactorStatus::INVALID_DELTA;
  }

  auto F_factor = Eigen::Map<Eigen::MatrixXd>(F_factor_data, n, n);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      F_factor(row, col) =
          sqrt_delta_data[row] * V(row, col) * sqrt_delta_data[col];
    }
    F_factor(col, col) += 1.0;
  }

  Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(F_factor);
  return llt.info() == Eigen::Success
             ? LQR::FactorStatus::SUCCESS
             : LQR::FactorStatus::F_FACTORIZATION_FAILURE;
}

void compute_regularized_W(const double *F_factor_data,
                           Eigen::Ref<Eigen::MatrixXd> result,
                           const double *sqrt_delta_inv_data, const int n) {
  const auto F_factor = Eigen::Map<const Eigen::MatrixXd>(F_factor_data, n, n);

  result.setIdentity();
  F_factor.template triangularView<Eigen::Lower>().solveInPlace(result);
  F_factor.transpose().template triangularView<Eigen::Upper>().solveInPlace(
      result);

  result *= -1.0;
  result.diagonal().array() += 1.0;

  for (int col = 0; col < result.cols(); ++col) {
    for (int row = 0; row < n; ++row) {
      result(row, col) *= sqrt_delta_inv_data[row] * sqrt_delta_inv_data[col];
    }
  }
}

void F_inv_mult_vector(const double *F_factor_data,
                       const Eigen::Ref<const Eigen::VectorXd> &rhs,
                       Eigen::Ref<Eigen::VectorXd> result,
                       const double *sqrt_delta_data,
                       const double *sqrt_delta_inv_data, const int n) {
  const auto F_factor = Eigen::Map<const Eigen::MatrixXd>(F_factor_data, n, n);

  for (int row = 0; row < n; ++row) {
    result(row) = sqrt_delta_inv_data[row] * rhs(row);
  }

  F_factor.template triangularView<Eigen::Lower>().solveInPlace(result);
  F_factor.transpose().template triangularView<Eigen::Upper>().solveInPlace(
      result);

  for (int row = 0; row < n; ++row) {
    result(row) *= sqrt_delta_data[row];
  }
}

auto topology_root(const LQR::Input &input) -> int {
  if (input.topology.root == nullptr) {
    return 0;
  }
  return input.topology.root(input.topology.context);
}

auto topology_edge_parent(const LQR::Input &input, const int edge) -> int {
  if (input.topology.edge_parent == nullptr) {
    return edge;
  }
  return input.topology.edge_parent(input.topology.context, edge);
}

auto topology_edge_child(const LQR::Input &input, const int edge) -> int {
  if (input.topology.edge_child == nullptr) {
    return edge + 1;
  }
  return input.topology.edge_child(input.topology.context, edge);
}

auto topology_cache_matches(const LQR::Input &input,
                            const LQR::Workspace &workspace) -> bool {
  return workspace.topology_is_initialized &&
         workspace.topology_state_dim == input.dimensions.state_dim &&
         workspace.topology_control_dim == input.dimensions.control_dim &&
         workspace.topology_num_stages == input.dimensions.num_stages &&
         workspace.topology_state_dims == input.dimensions.state_dims &&
         workspace.topology_control_dims == input.dimensions.control_dims &&
         workspace.topology_context == input.topology.context &&
         workspace.topology_root == input.topology.root &&
         workspace.topology_edge_parent == input.topology.edge_parent &&
         workspace.topology_edge_child == input.topology.edge_child;
}

void record_topology_cache_signature(const LQR::Input &input,
                                     LQR::Workspace &workspace) {
  workspace.topology_is_initialized = true;
  workspace.topology_state_dim = input.dimensions.state_dim;
  workspace.topology_control_dim = input.dimensions.control_dim;
  workspace.topology_num_stages = input.dimensions.num_stages;
  workspace.topology_state_dims = input.dimensions.state_dims;
  workspace.topology_control_dims = input.dimensions.control_dims;
  workspace.topology_context = input.topology.context;
  workspace.topology_root = input.topology.root;
  workspace.topology_edge_parent = input.topology.edge_parent;
  workspace.topology_edge_child = input.topology.edge_child;
}

auto compile_topology_data(const LQR::Input &input, LQR::Workspace &workspace)
    -> LQR::FactorStatus {
  const int num_edges = input.dimensions.num_edges();
  const int num_nodes = input.dimensions.num_nodes();
  const int root = topology_root(input);
  if (root < 0 || root >= num_nodes) {
    return LQR::FactorStatus::INVALID_TOPOLOGY;
  }

  std::fill_n(workspace.child_offsets, num_nodes + 1, 0);
  for (int edge = 0; edge < num_edges; ++edge) {
    const int parent = topology_edge_parent(input, edge);
    const int child = topology_edge_child(input, edge);
    if (parent < 0 || parent >= num_nodes || child < 0 || child >= num_nodes ||
        parent == child) {
      return LQR::FactorStatus::INVALID_TOPOLOGY;
    }
    workspace.edge_parents[edge] = parent;
    workspace.edge_children[edge] = child;
    ++workspace.child_offsets[parent + 1];
  }

  for (int node = 0; node < num_nodes; ++node) {
    workspace.child_offsets[node + 1] += workspace.child_offsets[node];
  }

  std::copy_n(workspace.child_offsets, num_nodes,
              workspace.postorder_nodes);
  for (int edge = 0; edge < num_edges; ++edge) {
    const int parent = workspace.edge_parents[edge];
    const int offset = workspace.postorder_nodes[parent]++;
    workspace.child_edges[offset] = edge;
  }

  int stack_size = 0;
  workspace.postorder_nodes[stack_size++] = root;
  int preorder_size = 0;
  std::fill_n(workspace.node_marks, num_nodes, 0);
  while (stack_size > 0) {
    const int node = workspace.postorder_nodes[--stack_size];
    if (preorder_size >= num_nodes || workspace.node_marks[node] != 0) {
      return LQR::FactorStatus::INVALID_TOPOLOGY;
    }
    workspace.node_marks[node] = 1;
    workspace.preorder_nodes[preorder_size++] = node;

    const int child_begin = workspace.child_offsets[node];
    const int child_end = workspace.child_offsets[node + 1];
    for (int child_index = child_end - 1; child_index >= child_begin;
         --child_index) {
      const int edge = workspace.child_edges[child_index];
      workspace.postorder_nodes[stack_size++] = workspace.edge_children[edge];
    }
  }

  if (preorder_size != num_nodes) {
    return LQR::FactorStatus::INVALID_TOPOLOGY;
  }

  for (int order = 0; order < num_nodes; ++order) {
    workspace.postorder_nodes[order] =
        workspace.preorder_nodes[num_nodes - 1 - order];
  }

  return LQR::FactorStatus::SUCCESS;
}

} // namespace

LQR::LQR(const LQR::Input &input, LQR::Workspace &workspace)
    : input_(input), workspace_(workspace) {
  if (!topology_cache_matches(input_, workspace_)) {
    compile_topology();
  }
}

auto LQR::compile_topology() -> FactorStatus {
  workspace_.topology_status = compile_topology_data(input_, workspace_);
  record_topology_cache_signature(input_, workspace_);
  return workspace_.topology_status;
}

auto LQR::factor_with_status() -> FactorStatus {
  if (!topology_cache_matches(input_, workspace_)) {
    compile_topology();
  }
  if (workspace_.topology_status != FactorStatus::SUCCESS) {
    return workspace_.topology_status;
  }
  const int num_nodes = input_.dimensions.num_nodes();

  for (int order = 0; order < num_nodes; ++order) {
    const int node = workspace_.postorder_nodes[order];
    const int node_dim = input_.dimensions.get_state_dim(node);
    const auto Q_node =
        Eigen::Map<const Eigen::MatrixXd>(input_.Q[node], node_dim, node_dim);
    auto V_node =
        Eigen::Map<Eigen::MatrixXd>(workspace_.V[node], node_dim, node_dim);
    V_node.noalias() = Q_node;

    for (int child_index = workspace_.child_offsets[node];
         child_index < workspace_.child_offsets[node + 1]; ++child_index) {
      const int edge = workspace_.child_edges[child_index];
      const int child = workspace_.edge_children[edge];
      const int child_dim = input_.dimensions.get_state_dim(child);
      const int control_dim = input_.dimensions.get_control_dim(edge);

      const auto A_edge =
          Eigen::Map<const Eigen::MatrixXd>(input_.A[edge], child_dim,
                                            node_dim);
      const auto B_edge =
          Eigen::Map<const Eigen::MatrixXd>(input_.B[edge], child_dim,
                                            control_dim);
      const auto M_edge =
          Eigen::Map<const Eigen::MatrixXd>(input_.M[edge], node_dim,
                                            control_dim);
      const auto R_edge =
          Eigen::Map<const Eigen::MatrixXd>(input_.R[edge], control_dim,
                                            control_dim);

      auto W_edge =
          Eigen::Map<Eigen::MatrixXd>(workspace_.W[edge], child_dim,
                                      child_dim);
      auto G_edge_factor = Eigen::Map<Eigen::MatrixXd>(
          workspace_.G_factor[edge], control_dim, control_dim);
      auto H_child =
          Eigen::Map<Eigen::MatrixXd>(workspace_.H, control_dim, child_dim);
      auto H_parent =
          Eigen::Map<Eigen::MatrixXd>(workspace_.H, control_dim, node_dim);
      auto K_edge = Eigen::Map<Eigen::MatrixXd>(workspace_.K[edge],
                                                control_dim, node_dim);
      auto F_child_parent =
          Eigen::Map<Eigen::MatrixXd>(workspace_.F, child_dim, node_dim);

      compute_regularized_W(workspace_.F_factor[child], W_edge,
                            workspace_.sqrt_delta_inv[child], child_dim);

      H_child.noalias() = B_edge.transpose() * W_edge;
      G_edge_factor.noalias() = R_edge;
      G_edge_factor.noalias() += H_child * B_edge;

      {
        Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(G_edge_factor);
        if (llt.info() != Eigen::Success) {
          return FactorStatus::G_FACTORIZATION_FAILURE;
        }
      }

      F_child_parent.noalias() = W_edge * A_edge;
      H_parent.noalias() = M_edge.transpose();
      H_parent.noalias() += B_edge.transpose() * F_child_parent;

      K_edge.noalias() = H_parent;
      G_edge_factor.template triangularView<Eigen::Lower>().solveInPlace(
          K_edge);
      G_edge_factor.transpose()
          .template triangularView<Eigen::Upper>()
          .solveInPlace(K_edge);
      K_edge *= -1.0;

      V_node.noalias() += A_edge.transpose() * F_child_parent;
      auto F_parent =
          Eigen::Map<Eigen::MatrixXd>(workspace_.F, node_dim, node_dim);
      F_parent.noalias() = K_edge.transpose() * H_parent;
      V_node += F_parent;
    }

    const auto factor_status =
        factor_F(input_.delta[node], V_node, workspace_.F_factor[node],
                 workspace_.sqrt_delta[node],
                 workspace_.sqrt_delta_inv[node], node_dim);
    if (factor_status != FactorStatus::SUCCESS) {
      return factor_status;
    }
  }

  return FactorStatus::SUCCESS;
}

bool LQR::factor() { return factor_with_status() == FactorStatus::SUCCESS; }

void LQR::solve(Output &output) {
  const int num_nodes = input_.dimensions.num_nodes();

  for (int order = 0; order < num_nodes; ++order) {
    const int node = workspace_.postorder_nodes[order];
    const int node_dim = input_.dimensions.get_state_dim(node);
    const auto q_node =
        Eigen::Map<const Eigen::VectorXd>(input_.q[node], node_dim);
    auto v_node = Eigen::Map<Eigen::VectorXd>(workspace_.v[node], node_dim);
    v_node.noalias() = q_node;

    for (int child_index = workspace_.child_offsets[node];
         child_index < workspace_.child_offsets[node + 1]; ++child_index) {
      const int edge = workspace_.child_edges[child_index];
      const int child = workspace_.edge_children[edge];
      const int child_dim = input_.dimensions.get_state_dim(child);
      const int control_dim = input_.dimensions.get_control_dim(edge);

      const auto A_edge =
          Eigen::Map<const Eigen::MatrixXd>(input_.A[edge], child_dim,
                                            node_dim);
      const auto B_edge =
          Eigen::Map<const Eigen::MatrixXd>(input_.B[edge], child_dim,
                                            control_dim);
      const auto r_edge =
          Eigen::Map<const Eigen::VectorXd>(input_.r[edge], control_dim);
      const auto c_child =
          Eigen::Map<const Eigen::VectorXd>(input_.c[child], child_dim);
      const auto delta_child =
          Eigen::Map<const Eigen::VectorXd>(input_.delta[child], child_dim);
      const auto v_child =
          Eigen::Map<const Eigen::VectorXd>(workspace_.v[child], child_dim);
      const auto W_edge =
          Eigen::Map<const Eigen::MatrixXd>(workspace_.W[edge], child_dim,
                                            child_dim);
      const auto G_edge_factor = Eigen::Map<const Eigen::MatrixXd>(
          workspace_.G_factor[edge], control_dim, control_dim);
      const auto K_edge = Eigen::Map<const Eigen::MatrixXd>(
          workspace_.K[edge], control_dim, node_dim);

      auto g = Eigen::Map<Eigen::VectorXd>(workspace_.g, child_dim);
      auto h = Eigen::Map<Eigen::VectorXd>(workspace_.h, control_dim);
      auto k_edge =
          Eigen::Map<Eigen::VectorXd>(workspace_.k[edge], control_dim);
      auto f = Eigen::Map<Eigen::VectorXd>(workspace_.f, child_dim);

      f.noalias() = delta_child.cwiseProduct(v_child);
      f.noalias() -= c_child;
      g.noalias() = v_child;
      g.noalias() -= W_edge * f;

      h.noalias() = r_edge;
      h.noalias() += B_edge.transpose() * g;
      k_edge.noalias() = h;
      G_edge_factor.template triangularView<Eigen::Lower>().solveInPlace(
          k_edge);
      G_edge_factor.transpose()
          .template triangularView<Eigen::Upper>()
          .solveInPlace(k_edge);
      k_edge *= -1.0;

      v_node.noalias() += A_edge.transpose() * g;
      v_node.noalias() += K_edge.transpose() * h;
    }
  }

  const int root = workspace_.preorder_nodes[0];
  const int root_dim = input_.dimensions.get_state_dim(root);
  const auto c_root =
      Eigen::Map<const Eigen::VectorXd>(input_.c[root], root_dim);
  const auto delta_root =
      Eigen::Map<const Eigen::VectorXd>(input_.delta[root], root_dim);
  const auto V_root =
      Eigen::Map<const Eigen::MatrixXd>(workspace_.V[root], root_dim,
                                        root_dim);
  const auto v_root =
      Eigen::Map<const Eigen::VectorXd>(workspace_.v[root], root_dim);
  auto f = Eigen::Map<Eigen::VectorXd>(workspace_.f, root_dim);
  auto x_root = Eigen::Map<Eigen::VectorXd>(output.x[root], root_dim);
  auto y_root = Eigen::Map<Eigen::VectorXd>(output.y[root], root_dim);

  f.noalias() = delta_root.cwiseProduct(v_root);
  f.noalias() -= c_root;
  F_inv_mult_vector(workspace_.F_factor[root], f, x_root,
                    workspace_.sqrt_delta[root],
                    workspace_.sqrt_delta_inv[root], root_dim);
  x_root *= -1.0;
  y_root.noalias() = v_root;
  y_root.noalias() += V_root * x_root;

  for (int order = 0; order < num_nodes; ++order) {
    const int node = workspace_.preorder_nodes[order];
    const int node_dim = input_.dimensions.get_state_dim(node);
    const auto x_node =
        Eigen::Map<const Eigen::VectorXd>(output.x[node], node_dim);

    for (int child_index = workspace_.child_offsets[node];
         child_index < workspace_.child_offsets[node + 1]; ++child_index) {
      const int edge = workspace_.child_edges[child_index];
      const int child = workspace_.edge_children[edge];
      const int child_dim = input_.dimensions.get_state_dim(child);
      const int control_dim = input_.dimensions.get_control_dim(edge);

      const auto A_edge =
          Eigen::Map<const Eigen::MatrixXd>(input_.A[edge], child_dim,
                                            node_dim);
      const auto B_edge =
          Eigen::Map<const Eigen::MatrixXd>(input_.B[edge], child_dim,
                                            control_dim);
      const auto K_edge = Eigen::Map<const Eigen::MatrixXd>(
          workspace_.K[edge], control_dim, node_dim);
      const auto k_edge =
          Eigen::Map<const Eigen::VectorXd>(workspace_.k[edge], control_dim);
      const auto V_child =
          Eigen::Map<const Eigen::MatrixXd>(workspace_.V[child], child_dim,
                                            child_dim);
      const auto v_child =
          Eigen::Map<const Eigen::VectorXd>(workspace_.v[child], child_dim);
      const auto c_child =
          Eigen::Map<const Eigen::VectorXd>(input_.c[child], child_dim);
      const auto delta_child =
          Eigen::Map<const Eigen::VectorXd>(input_.delta[child], child_dim);

      auto u_edge =
          Eigen::Map<Eigen::VectorXd>(output.u[edge], control_dim);
      auto x_child = Eigen::Map<Eigen::VectorXd>(output.x[child], child_dim);
      auto y_child = Eigen::Map<Eigen::VectorXd>(output.y[child], child_dim);
      auto f_child = Eigen::Map<Eigen::VectorXd>(workspace_.f, child_dim);

      u_edge.noalias() = k_edge;
      u_edge.noalias() += K_edge * x_node;

      f_child.noalias() = c_child;
      f_child.noalias() -= delta_child.cwiseProduct(v_child);
      f_child.noalias() += A_edge * x_node;
      f_child.noalias() += B_edge * u_edge;
      F_inv_mult_vector(workspace_.F_factor[child], f_child, x_child,
                        workspace_.sqrt_delta[child],
                        workspace_.sqrt_delta_inv[child], child_dim);

      y_child.noalias() = v_child;
      y_child.noalias() += V_child * x_child;
    }
  }
}

} // namespace sip::optimal_control

#undef EIGEN_NO_MALLOC

#include "sip_optimal_control/helpers.hpp"

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace sip::optimal_control {
namespace {

void fill_sequence(double *data, const int size, const double scale) {
  for (int i = 0; i < size; ++i) {
    data[i] = scale * static_cast<double>(i + 1);
  }
}

void fill_spd(double *data, const int size, const double diagonal) {
  auto matrix = Eigen::Map<Eigen::MatrixXd>(data, size, size);
  matrix.setIdentity();
  matrix *= diagonal;
}

struct BranchTopology {
  std::array<int, 2> parent = {0, 0};
  std::array<int, 2> child = {1, 2};
  mutable int root_calls = 0;
  mutable int edge_parent_calls = 0;
  mutable int edge_child_calls = 0;

  static int root(const void *context) {
    ++static_cast<const BranchTopology *>(context)->root_calls;
    return 0;
  }
  static int edge_parent(const void *context, const int edge) {
    const auto *topology = static_cast<const BranchTopology *>(context);
    ++topology->edge_parent_calls;
    return topology->parent[edge];
  }
  static int edge_child(const void *context, const int edge) {
    const auto *topology = static_cast<const BranchTopology *>(context);
    ++topology->edge_child_calls;
    return topology->child[edge];
  }
};

struct InvalidDagTopology {
  std::array<int, 2> parent = {0, 1};
  std::array<int, 2> child = {2, 2};

  static int root(const void *) { return 0; }
  static int edge_parent(const void *context, const int edge) {
    return static_cast<const InvalidDagTopology *>(context)->parent[edge];
  }
  static int edge_child(const void *context, const int edge) {
    return static_cast<const InvalidDagTopology *>(context)->child[edge];
  }
};

TEST(InputValidation, AcceptsChainAndTreeRejectsNonTreeDag) {
  const std::array<int, 3> state_dims = {2, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> c_dims = {0, 1, 0};
  const std::array<int, 3> g_dims = {1, 0, 2};

  const Dimensions dimensions = {
      .num_stages = 2,
      .state_dim = 0,
      .control_dim = 0,
      .c_dim = 0,
      .g_dim = 0,
      .theta_dim = 2,
      .state_dims = state_dims.data(),
      .control_dims = control_dims.data(),
      .c_dims = c_dims.data(),
      .g_dims = g_dims.data(),
  };

  const Input chain_input = {
      .model_callback = [](const ModelCallbackInput &) {},
      .timeout_callback = []() { return false; },
      .dimensions = dimensions,
  };
  EXPECT_EQ(validate_input(chain_input), InputValidationStatus::SUCCESS);

  const BranchTopology tree_topology;
  const Input tree_input = {
      .model_callback = [](const ModelCallbackInput &) {},
      .timeout_callback = []() { return false; },
      .dimensions = dimensions,
      .topology =
          {
              .context = &tree_topology,
              .root = BranchTopology::root,
              .edge_parent = BranchTopology::edge_parent,
              .edge_child = BranchTopology::edge_child,
          },
  };
  EXPECT_EQ(validate_input(tree_input), InputValidationStatus::SUCCESS);

  const InvalidDagTopology dag_topology;
  const Input dag_input = {
      .model_callback = [](const ModelCallbackInput &) {},
      .timeout_callback = []() { return false; },
      .dimensions = dimensions,
      .topology =
          {
              .context = &dag_topology,
              .root = InvalidDagTopology::root,
              .edge_parent = InvalidDagTopology::edge_parent,
              .edge_child = InvalidDagTopology::edge_child,
          },
  };
  EXPECT_EQ(validate_input(dag_input),
            InputValidationStatus::INVALID_TOPOLOGY);
}

TEST(CallbackProvider, SolvesVariableDimensionKKTSystem) {
  const std::vector<int> state_dims = {2, 1, 3};
  const std::vector<int> control_dims = {1, 2};
  const std::vector<int> c_dims = {1, 0, 2};
  const std::vector<int> g_dims = {0, 2, 1};

  const Dimensions dimensions = {
      .num_stages = 2,
      .state_dim = 0,
      .control_dim = 0,
      .c_dim = 0,
      .g_dim = 0,
      .theta_dim = 0,
      .state_dims = state_dims.data(),
      .control_dims = control_dims.data(),
      .c_dims = c_dims.data(),
      .g_dims = g_dims.data(),
  };

  Input input = {
      .model_callback = [](const ModelCallbackInput &) {},
      .timeout_callback = []() { return false; },
      .dimensions = dimensions,
  };

  std::vector<unsigned char> workspace_memory(Workspace::num_bytes(dimensions));
  Workspace workspace;
  EXPECT_EQ(workspace.mem_assign(dimensions, workspace_memory.data()),
            static_cast<int>(workspace_memory.size()));
  auto &mco = workspace.model_callback_output;

  for (int node = 0; node < dimensions.num_nodes(); ++node) {
    const int n = dimensions.get_state_dim(node);
    const int c = dimensions.get_c_dim(node);
    const int g = dimensions.get_g_dim(node);

    std::fill_n(mco.df_dx[node], n, 0.0);
    std::fill_n(mco.dyn_res[node], n, 0.0);
    std::fill_n(mco.c[node], c, 0.0);
    std::fill_n(mco.g[node], g, 0.0);
    fill_spd(mco.d2L_dx2[node], n, 2.0 + 0.2 * node);
    fill_sequence(mco.dc_dx[node], c * n, 0.03);
    fill_sequence(mco.dg_dx[node], g * n, -0.02);
    fill_sequence(mco.d2L_dxdtheta[node], n * dimensions.theta_dim, 0.0);
    fill_sequence(mco.dc_dtheta[node], c * dimensions.theta_dim, 0.0);
    fill_sequence(mco.dg_dtheta[node], g * dimensions.theta_dim, 0.0);
  }

  for (int edge = 0; edge < dimensions.num_edges(); ++edge) {
    const int n_parent = dimensions.get_state_dim(edge);
    const int n_child = dimensions.get_state_dim(edge + 1);
    const int m = dimensions.get_control_dim(edge);
    const int c = dimensions.get_c_dim(edge);
    const int g = dimensions.get_g_dim(edge);

    std::fill_n(mco.df_du[edge], m, 0.0);
    fill_sequence(mco.ddyn_dx[edge], n_child * n_parent, 0.04);
    fill_sequence(mco.ddyn_du[edge], n_child * m, -0.05);
    fill_sequence(mco.ddyn_dtheta[edge], n_child * dimensions.theta_dim, 0.0);
    fill_sequence(mco.dc_du[edge], c * m, 0.06);
    fill_sequence(mco.dg_du[edge], g * m, -0.04);
    fill_sequence(mco.d2L_dxdu[edge], n_parent * m, 0.02);
    fill_spd(mco.d2L_du2[edge], m, 2.5 + 0.1 * edge);
    fill_sequence(mco.d2L_dudtheta[edge], m * dimensions.theta_dim, 0.0);
  }

  auto callback_provider = CallbackProvider(input, workspace);

  const int x_dim = dimensions.get_x_dim();
  const int y_dim = dimensions.get_y_dim();
  const int z_dim = dimensions.get_z_dim();
  const int kkt_dim = x_dim + y_dim + z_dim;

  std::vector<double> w(z_dim, 1.3);
  std::vector<double> r2(y_dim, 0.9);
  std::vector<double> r3(z_dim, 0.4);
  ASSERT_TRUE(callback_provider.factor(w.data(), 0.2, r2.data(), r3.data()));

  std::vector<double> rhs(kkt_dim);
  fill_sequence(rhs.data(), kkt_dim, 0.01);

  std::vector<double> solution(kkt_dim, 0.0);
  callback_provider.solve(rhs.data(), solution.data());

  std::vector<double> product_x(x_dim, 0.0);
  std::vector<double> product_y(y_dim, 0.0);
  std::vector<double> product_z(z_dim, 0.0);
  callback_provider.add_Kx_to_y(
      w.data(), 0.2, r2.data(), r3.data(), solution.data(),
      solution.data() + x_dim, solution.data() + x_dim + y_dim,
      product_x.data(), product_y.data(), product_z.data());

  double squared_error = 0.0;
  for (int i = 0; i < x_dim; ++i) {
    const double residual = product_x[i] - rhs[i];
    squared_error += residual * residual;
  }
  for (int i = 0; i < y_dim; ++i) {
    const double residual = product_y[i] - rhs[x_dim + i];
    squared_error += residual * residual;
  }
  for (int i = 0; i < z_dim; ++i) {
    const double residual = product_z[i] - rhs[x_dim + y_dim + i];
    squared_error += residual * residual;
  }

  EXPECT_LT(std::sqrt(squared_error), 1e-9);
}

TEST(CallbackProvider, SolvesVariableDimensionBranchedKKTSystem) {
  const std::array<int, 3> state_dims = {2, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> c_dims = {0, 0, 0};
  const std::array<int, 3> g_dims = {0, 0, 0};
  const BranchTopology topology;

  const Dimensions dimensions = {
      .num_stages = 2,
      .state_dim = 0,
      .control_dim = 0,
      .c_dim = 0,
      .g_dim = 0,
      .theta_dim = 0,
      .state_dims = state_dims.data(),
      .control_dims = control_dims.data(),
      .c_dims = c_dims.data(),
      .g_dims = g_dims.data(),
  };

  Input input = {
      .model_callback = [](const ModelCallbackInput &) {},
      .timeout_callback = []() { return false; },
      .dimensions = dimensions,
      .topology =
          {
              .context = &topology,
              .root = BranchTopology::root,
              .edge_parent = BranchTopology::edge_parent,
              .edge_child = BranchTopology::edge_child,
          },
  };

  std::vector<unsigned char> workspace_memory(Workspace::num_bytes(dimensions));
  Workspace workspace;
  EXPECT_EQ(workspace.mem_assign(dimensions, workspace_memory.data()),
            static_cast<int>(workspace_memory.size()));
  auto &mco = workspace.model_callback_output;

  for (int node = 0; node < dimensions.num_nodes(); ++node) {
    const int n = dimensions.get_state_dim(node);
    std::fill_n(mco.df_dx[node], n, 0.0);
    std::fill_n(mco.dyn_res[node], n, 0.0);
    fill_spd(mco.d2L_dx2[node], n, 2.0 + 0.2 * node);
    fill_sequence(mco.d2L_dxdtheta[node], n * dimensions.theta_dim, 0.0);
  }

  for (int edge = 0; edge < dimensions.num_edges(); ++edge) {
    const int parent = topology.parent[edge];
    const int child = topology.child[edge];
    const int n_parent = dimensions.get_state_dim(parent);
    const int n_child = dimensions.get_state_dim(child);
    const int m = dimensions.get_control_dim(edge);
    std::fill_n(mco.df_du[edge], m, 0.0);
    fill_sequence(mco.ddyn_dx[edge], n_child * n_parent, 0.04 + 0.01 * edge);
    fill_sequence(mco.ddyn_du[edge], n_child * m, -0.05 - 0.01 * edge);
    fill_sequence(mco.ddyn_dtheta[edge], n_child * dimensions.theta_dim, 0.0);
    fill_sequence(mco.d2L_dxdu[edge], n_parent * m, 0.02 + 0.01 * edge);
    fill_spd(mco.d2L_du2[edge], m, 2.5 + 0.1 * edge);
    fill_sequence(mco.d2L_dudtheta[edge], m * dimensions.theta_dim, 0.0);
  }

  auto callback_provider = CallbackProvider(input, workspace);

  const int x_dim = dimensions.get_x_dim();
  const int y_dim = dimensions.get_y_dim();
  const int z_dim = dimensions.get_z_dim();
  const int kkt_dim = x_dim + y_dim + z_dim;

  std::vector<double> w(z_dim, 1.3);
  std::vector<double> r2(y_dim, 0.9);
  std::vector<double> r3(z_dim, 0.4);
  ASSERT_TRUE(callback_provider.factor(w.data(), 0.2, r2.data(), r3.data()));

  std::vector<double> rhs(kkt_dim);
  fill_sequence(rhs.data(), kkt_dim, 0.01);

  std::vector<double> solution(kkt_dim, 0.0);
  callback_provider.solve(rhs.data(), solution.data());

  std::vector<double> product_x(x_dim, 0.0);
  std::vector<double> product_y(y_dim, 0.0);
  std::vector<double> product_z(z_dim, 0.0);
  callback_provider.add_Kx_to_y(
      w.data(), 0.2, r2.data(), r3.data(), solution.data(),
      solution.data() + x_dim, solution.data() + x_dim + y_dim,
      product_x.data(), product_y.data(), product_z.data());

  double squared_error = 0.0;
  for (int i = 0; i < x_dim; ++i) {
    const double residual = product_x[i] - rhs[i];
    squared_error += residual * residual;
  }
  for (int i = 0; i < y_dim; ++i) {
    const double residual = product_y[i] - rhs[x_dim + i];
    squared_error += residual * residual;
  }
  for (int i = 0; i < z_dim; ++i) {
    const double residual = product_z[i] - rhs[x_dim + y_dim + i];
    squared_error += residual * residual;
  }

  EXPECT_LT(std::sqrt(squared_error), 1e-9);
}

TEST(CallbackProvider, SolvesBranchedKKTSystemWithSchurVariables) {
  const std::array<int, 3> state_dims = {2, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> c_dims = {0, 0, 0};
  const std::array<int, 3> g_dims = {0, 0, 0};
  const BranchTopology topology;

  const Dimensions dimensions = {
      .num_stages = 2,
      .state_dim = 0,
      .control_dim = 0,
      .c_dim = 0,
      .g_dim = 0,
      .theta_dim = 2,
      .state_dims = state_dims.data(),
      .control_dims = control_dims.data(),
      .c_dims = c_dims.data(),
      .g_dims = g_dims.data(),
  };

  Input input = {
      .model_callback = [](const ModelCallbackInput &) {},
      .timeout_callback = []() { return false; },
      .dimensions = dimensions,
      .topology =
          {
              .context = &topology,
              .root = BranchTopology::root,
              .edge_parent = BranchTopology::edge_parent,
              .edge_child = BranchTopology::edge_child,
          },
  };

  std::vector<unsigned char> workspace_memory(Workspace::num_bytes(dimensions));
  Workspace workspace;
  EXPECT_EQ(workspace.mem_assign(dimensions, workspace_memory.data()),
            static_cast<int>(workspace_memory.size()));
  auto &mco = workspace.model_callback_output;
  const int p = dimensions.theta_dim;

  std::fill_n(mco.df_dtheta, p, 0.0);
  fill_spd(mco.d2L_dtheta2, p, 20.0);

  for (int node = 0; node < dimensions.num_nodes(); ++node) {
    const int n = dimensions.get_state_dim(node);
    const int c = dimensions.get_c_dim(node);
    const int g = dimensions.get_g_dim(node);

    std::fill_n(mco.df_dx[node], n, 0.0);
    std::fill_n(mco.dyn_res[node], n, 0.0);
    std::fill_n(mco.c[node], c, 0.0);
    std::fill_n(mco.g[node], g, 0.0);
    fill_spd(mco.d2L_dx2[node], n, 3.0 + 0.2 * node);
    fill_sequence(mco.dc_dx[node], c * n, 0.0);
    fill_sequence(mco.dg_dx[node], g * n, 0.0);
    fill_sequence(mco.dc_dtheta[node], c * p, 0.0);
    fill_sequence(mco.dg_dtheta[node], g * p, 0.0);
    fill_sequence(mco.d2L_dxdtheta[node], n * p, 0.001 * (node + 1));
  }

  for (int edge = 0; edge < dimensions.num_edges(); ++edge) {
    const int parent = topology.parent[edge];
    const int child = topology.child[edge];
    const int n_parent = dimensions.get_state_dim(parent);
    const int n_child = dimensions.get_state_dim(child);
    const int m = dimensions.get_control_dim(edge);
    const int c = dimensions.get_c_dim(parent);
    const int g = dimensions.get_g_dim(parent);

    std::fill_n(mco.df_du[edge], m, 0.0);
    fill_sequence(mco.ddyn_dx[edge], n_child * n_parent, 0.03 + 0.01 * edge);
    fill_sequence(mco.ddyn_du[edge], n_child * m, -0.04 - 0.01 * edge);
    fill_sequence(mco.ddyn_dtheta[edge], n_child * p,
                  0.0015 * (edge + 1));
    fill_sequence(mco.dc_du[edge], c * m, 0.0);
    fill_sequence(mco.dg_du[edge], g * m, 0.0);
    fill_sequence(mco.d2L_dxdu[edge], n_parent * m, 0.015 + 0.01 * edge);
    fill_spd(mco.d2L_du2[edge], m, 3.5 + 0.1 * edge);
    fill_sequence(mco.d2L_dudtheta[edge], m * p, -0.001 * (edge + 1));
  }

  auto callback_provider = CallbackProvider(input, workspace);
  const int root_calls = topology.root_calls;
  const int edge_parent_calls = topology.edge_parent_calls;
  const int edge_child_calls = topology.edge_child_calls;

  const int x_dim = dimensions.get_x_dim();
  const int y_dim = dimensions.get_y_dim();
  const int z_dim = dimensions.get_z_dim();
  const int kkt_dim = x_dim + y_dim + z_dim;

  std::vector<double> w(z_dim, 1.3);
  std::vector<double> r2(y_dim, 0.9);
  std::vector<double> r3(z_dim, 0.4);
  ASSERT_TRUE(callback_provider.factor(w.data(), 0.2, r2.data(), r3.data()));

  std::vector<double> rhs(kkt_dim);
  fill_sequence(rhs.data(), kkt_dim, 0.01);

  std::vector<double> solution(kkt_dim, 0.0);
  callback_provider.solve(rhs.data(), solution.data());

  std::vector<double> product_x(x_dim, 0.0);
  std::vector<double> product_y(y_dim, 0.0);
  std::vector<double> product_z(z_dim, 0.0);
  callback_provider.add_Kx_to_y(
      w.data(), 0.2, r2.data(), r3.data(), solution.data(),
      solution.data() + x_dim, solution.data() + x_dim + y_dim,
      product_x.data(), product_y.data(), product_z.data());

  EXPECT_EQ(topology.root_calls, root_calls);
  EXPECT_EQ(topology.edge_parent_calls, edge_parent_calls);
  EXPECT_EQ(topology.edge_child_calls, edge_child_calls);

  double squared_error = 0.0;
  for (int i = 0; i < x_dim; ++i) {
    const double residual = product_x[i] - rhs[i];
    squared_error += residual * residual;
  }
  for (int i = 0; i < y_dim; ++i) {
    const double residual = product_y[i] - rhs[x_dim + i];
    squared_error += residual * residual;
  }
  for (int i = 0; i < z_dim; ++i) {
    const double residual = product_z[i] - rhs[x_dim + y_dim + i];
    squared_error += residual * residual;
  }

  EXPECT_LT(std::sqrt(squared_error), 1e-9);
}

} // namespace
} // namespace sip::optimal_control

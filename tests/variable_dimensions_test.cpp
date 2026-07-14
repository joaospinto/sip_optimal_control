#include "sip_optimal_control/helpers.hpp"

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace sip::optimal_control {
namespace {

constexpr auto filter_settings() -> ::sip::Settings {
  auto settings = ::sip::Settings{};
  settings.max_iterations = 4;
  settings.line_search.use_filter_line_search = true;
  return settings;
}

constexpr auto kSettings = filter_settings();

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
};

struct ChainTopology {
  std::array<int, 2> parent = {0, 1};
  std::array<int, 2> child = {1, 2};
};

struct InvalidDagTopology {
  std::array<int, 2> parent = {0, 1};
  std::array<int, 2> child = {2, 2};
};

TEST(InputValidation, AcceptsChainAndTreeRejectsNonTreeDag) {
  const std::array<int, 3> state_dims = {2, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> c_dims = {0, 1, 0};
  const std::array<int, 3> g_dims = {1, 0, 2};

  const Dimensions dimensions{2, state_dims.data(), control_dims.data(),
                              c_dims.data(), g_dims.data()};
  ChainTopology chain_topology;

  const Topology chain{2, 0, chain_topology.parent.data(),
                       chain_topology.child.data()};
  EXPECT_EQ(validate_input(dimensions, chain), InputValidationStatus::SUCCESS);

  const BranchTopology tree_topology;
  const Topology tree{2, 0, tree_topology.parent.data(),
                      tree_topology.child.data()};
  EXPECT_EQ(validate_input(dimensions, tree), InputValidationStatus::SUCCESS);

  const InvalidDagTopology dag_topology;
  const Topology dag{2, 0, dag_topology.parent.data(),
                     dag_topology.child.data()};
  EXPECT_EQ(validate_input(dimensions, dag),
            InputValidationStatus::INVALID_TOPOLOGY);

  const std::array<int, 3> zero_root_state_dims = {0, 1, 3};
  const Dimensions zero_root_dimensions{
      2, zero_root_state_dims.data(), control_dims.data(), c_dims.data(),
      g_dims.data()};
  EXPECT_EQ(validate_input(zero_root_dimensions, tree),
            InputValidationStatus::SUCCESS);

  const std::array<int, 3> negative_state_dims = {-1, 1, 3};
  const Dimensions negative_dimensions{
      2, negative_state_dims.data(), control_dims.data(), c_dims.data(),
      g_dims.data()};
  EXPECT_EQ(validate_input(negative_dimensions, tree),
            InputValidationStatus::INVALID_DIMENSIONS);
}

TEST(CallbackProvider, SolvesVariableDimensionKKTSystem) {
  const std::vector<int> state_dims = {2, 1, 3};
  const std::vector<int> control_dims = {1, 2};
  const std::vector<int> c_dims = {1, 0, 2};
  const std::vector<int> g_dims = {0, 2, 1};

  ChainTopology chain_topology;
  Input input = {
      .dimensions = {0, state_dims.data(), control_dims.data(), c_dims.data(),
                     g_dims.data()},
      .topology = {2, 0, chain_topology.parent.data(),
                   chain_topology.child.data()},
      .model_callback = [](const ModelCallbackInput &) {},
      .timeout_callback = []() { return false; },
  };
  Workspace workspace;
  std::vector<unsigned char> workspace_memory(
      Workspace::num_bytes(input.dimensions, input.topology, kSettings));
  EXPECT_EQ(workspace.mem_assign(input.dimensions, input.topology, kSettings,
                                 workspace_memory.data()),
            static_cast<int>(workspace_memory.size()));
  auto &mco = workspace.model_callback_output;

  for (int node = 0; node < 3; ++node) {
    const int n = input.dimensions.get_state_dim(node);
    const int c = input.dimensions.get_c_dim(node);
    const int g = input.dimensions.get_g_dim(node);

    std::fill_n(mco.df_dx[node], n, 0.0);
    std::fill_n(mco.dyn_res[node], n, 0.0);
    std::fill_n(mco.c[node], c, 0.0);
    std::fill_n(mco.g[node], g, 0.0);
    fill_spd(mco.d2L_dx2[node], n, 2.0 + 0.2 * node);
    fill_sequence(mco.dc_dx[node], c * n, 0.03);
    fill_sequence(mco.dg_dx[node], g * n, -0.02);
    fill_sequence(mco.d2L_dxdtheta[node], n * input.dimensions.theta_dim, 0.0);
    fill_sequence(mco.dc_dtheta[node], c * input.dimensions.theta_dim, 0.0);
    fill_sequence(mco.dg_dtheta[node], g * input.dimensions.theta_dim, 0.0);
  }

  for (int edge = 0; edge < 2; ++edge) {
    const int n_parent = input.dimensions.get_state_dim(edge);
    const int n_child = input.dimensions.get_state_dim(edge + 1);
    const int m = input.dimensions.get_control_dim(edge);
    const int c = input.dimensions.get_c_dim(edge);
    const int g = input.dimensions.get_g_dim(edge);

    std::fill_n(mco.df_du[edge], m, 0.0);
    fill_sequence(mco.ddyn_dx[edge], n_child * n_parent, 0.04);
    fill_sequence(mco.ddyn_du[edge], n_child * m, -0.05);
    fill_sequence(mco.ddyn_dtheta[edge], n_child * input.dimensions.theta_dim,
                  0.0);
    fill_sequence(mco.dc_du[edge], c * m, 0.06);
    fill_sequence(mco.dg_du[edge], g * m, -0.04);
    fill_sequence(mco.d2L_dxdu[edge], n_parent * m, 0.02);
    fill_spd(mco.d2L_du2[edge], m, 2.5 + 0.1 * edge);
    fill_sequence(mco.d2L_dudtheta[edge], m * input.dimensions.theta_dim, 0.0);
  }

  auto callback_provider = CallbackProvider(input, workspace);

  const int x_dim = input.dimensions.get_x_dim(input.topology.num_edges);
  const int y_dim = input.dimensions.get_y_dim(input.topology.num_nodes());
  const int z_dim = input.dimensions.get_z_dim(input.topology.num_nodes());
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

TEST(CallbackProvider, SolvesBranchedKKTSystemWithZeroDimensionalRoot) {
  const std::array<int, 3> state_dims = {0, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> c_dims = {0, 0, 0};
  const std::array<int, 3> g_dims = {0, 0, 0};
  const BranchTopology branch_topology;
  Input input = {
      .dimensions = {0, state_dims.data(), control_dims.data(), c_dims.data(),
                     g_dims.data()},
      .topology = {2, 0, branch_topology.parent.data(),
                   branch_topology.child.data()},
      .model_callback = [](const ModelCallbackInput &) {},
      .timeout_callback = []() { return false; },
  };
  Workspace workspace;
  std::vector<unsigned char> workspace_memory(
      Workspace::num_bytes(input.dimensions, input.topology, kSettings));
  EXPECT_EQ(workspace.mem_assign(input.dimensions, input.topology, kSettings,
                                 workspace_memory.data()),
            static_cast<int>(workspace_memory.size()));
  auto &mco = workspace.model_callback_output;

  for (int node = 0; node < 3; ++node) {
    const int n = input.dimensions.get_state_dim(node);
    std::fill_n(mco.df_dx[node], n, 0.0);
    std::fill_n(mco.dyn_res[node], n, 0.0);
    fill_spd(mco.d2L_dx2[node], n, 2.0 + 0.2 * node);
    fill_sequence(mco.d2L_dxdtheta[node], n * input.dimensions.theta_dim, 0.0);
  }

  for (int edge = 0; edge < 2; ++edge) {
    const int parent = branch_topology.parent[edge];
    const int child = branch_topology.child[edge];
    const int n_parent = input.dimensions.get_state_dim(parent);
    const int n_child = input.dimensions.get_state_dim(child);
    const int m = input.dimensions.get_control_dim(edge);
    std::fill_n(mco.df_du[edge], m, 0.0);
    fill_sequence(mco.ddyn_dx[edge], n_child * n_parent, 0.04 + 0.01 * edge);
    fill_sequence(mco.ddyn_du[edge], n_child * m, -0.05 - 0.01 * edge);
    fill_sequence(mco.ddyn_dtheta[edge], n_child * input.dimensions.theta_dim,
                  0.0);
    fill_sequence(mco.d2L_dxdu[edge], n_parent * m, 0.02 + 0.01 * edge);
    fill_spd(mco.d2L_du2[edge], m, 2.5 + 0.1 * edge);
    fill_sequence(mco.d2L_dudtheta[edge], m * input.dimensions.theta_dim, 0.0);
  }

  auto callback_provider = CallbackProvider(input, workspace);

  const int x_dim = input.dimensions.get_x_dim(input.topology.num_edges);
  const int y_dim = input.dimensions.get_y_dim(input.topology.num_nodes());
  const int z_dim = input.dimensions.get_z_dim(input.topology.num_nodes());
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
  const BranchTopology branch_topology;
  Input input = {
      .dimensions = {2, state_dims.data(), control_dims.data(), c_dims.data(),
                     g_dims.data()},
      .topology = {2, 0, branch_topology.parent.data(),
                   branch_topology.child.data()},
      .model_callback = [](const ModelCallbackInput &) {},
      .timeout_callback = []() { return false; },
  };
  Workspace workspace;
  std::vector<unsigned char> workspace_memory(
      Workspace::num_bytes(input.dimensions, input.topology, kSettings));
  EXPECT_EQ(workspace.mem_assign(input.dimensions, input.topology, kSettings,
                                 workspace_memory.data()),
            static_cast<int>(workspace_memory.size()));
  auto &mco = workspace.model_callback_output;
  const int p = input.dimensions.theta_dim;

  std::fill_n(mco.df_dtheta, p, 0.0);
  fill_spd(mco.d2L_dtheta2, p, 20.0);

  for (int node = 0; node < 3; ++node) {
    const int n = input.dimensions.get_state_dim(node);
    const int c = input.dimensions.get_c_dim(node);
    const int g = input.dimensions.get_g_dim(node);

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

  for (int edge = 0; edge < 2; ++edge) {
    const int parent = branch_topology.parent[edge];
    const int child = branch_topology.child[edge];
    const int n_parent = input.dimensions.get_state_dim(parent);
    const int n_child = input.dimensions.get_state_dim(child);
    const int m = input.dimensions.get_control_dim(edge);
    const int c = input.dimensions.get_c_dim(parent);
    const int g = input.dimensions.get_g_dim(parent);

    std::fill_n(mco.df_du[edge], m, 0.0);
    fill_sequence(mco.ddyn_dx[edge], n_child * n_parent, 0.03 + 0.01 * edge);
    fill_sequence(mco.ddyn_du[edge], n_child * m, -0.04 - 0.01 * edge);
    fill_sequence(mco.ddyn_dtheta[edge], n_child * p, 0.0015 * (edge + 1));
    fill_sequence(mco.dc_du[edge], c * m, 0.0);
    fill_sequence(mco.dg_du[edge], g * m, 0.0);
    fill_sequence(mco.d2L_dxdu[edge], n_parent * m, 0.015 + 0.01 * edge);
    fill_spd(mco.d2L_du2[edge], m, 3.5 + 0.1 * edge);
    fill_sequence(mco.d2L_dudtheta[edge], m * p, -0.001 * (edge + 1));
  }

  auto callback_provider = CallbackProvider(input, workspace);

  const int x_dim = input.dimensions.get_x_dim(input.topology.num_edges);
  const int y_dim = input.dimensions.get_y_dim(input.topology.num_nodes());
  const int z_dim = input.dimensions.get_z_dim(input.topology.num_nodes());
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

} // namespace
} // namespace sip::optimal_control

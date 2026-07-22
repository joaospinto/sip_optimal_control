#include "sip_optimal_control/helpers.hpp"
#include "sip_optimal_control/sip_optimal_control.hpp"

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>
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

template <typename T, typename = void>
struct has_control_member : std::false_type {};

template <typename T>
struct has_control_member<T, std::void_t<decltype(std::declval<T>().control)>>
    : std::true_type {};

template <typename T, typename = void>
struct has_control_gradient_member : std::false_type {};

template <typename T>
struct has_control_gradient_member<
    T, std::void_t<decltype(std::declval<T>().df_du)>> : std::true_type {};

static_assert(!has_control_member<NodeModelCallbackInput>::value);
static_assert(has_control_member<EdgeModelCallbackInput>::value);
static_assert(!has_control_gradient_member<NodeModelCallbackOutput>::value);
static_assert(has_control_gradient_member<EdgeModelCallbackOutput>::value);

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

auto empty_model_callback() -> Input::ModelCallback {
  return [](const ModelCallbackInput &, ModelCallbackOutput &) {};
}

void initialize_model(const Input &input, Workspace &workspace,
                      const double theta_diagonal = 0.0) {
  const int p = input.dimensions.theta_dim;
  auto &mco = workspace.model_callback_output;

  for (int node = 0; node < input.topology.num_nodes(); ++node) {
    const int n = input.dimensions.get_state_dim(node);
    const int c = input.dimensions.get_node_c_dim(node);
    const int g = input.dimensions.get_node_g_dim(node);
    auto &output = mco.nodes[node];
    output.f = 0.0;
    std::fill_n(output.df_dx, n, 0.0);
    std::fill_n(output.df_dtheta, p, 0.0);
    std::fill_n(output.c, c, 0.0);
    std::fill_n(output.g, g, 0.0);
    fill_sequence(output.dc_dx, c * n, 0.013 * (node + 1));
    fill_sequence(output.dc_dtheta, c * p, 0.001 * (node + 1));
    fill_sequence(output.dg_dx, g * n, -0.011 * (node + 1));
    fill_sequence(output.dg_dtheta, g * p, -0.0007 * (node + 1));
    fill_spd(output.d2L_dx2, n, 2.5 + 0.2 * node);
    fill_sequence(output.d2L_dxdtheta, n * p, 0.0005 * (node + 1));
    fill_spd(output.d2L_dtheta2, p, theta_diagonal);
  }

  for (int edge = 0; edge < input.topology.num_edges; ++edge) {
    const int parent = input.topology.edge_parents[edge];
    const int child = input.topology.edge_children[edge];
    const int n_parent = input.dimensions.get_state_dim(parent);
    const int n_child = input.dimensions.get_state_dim(child);
    const int m = input.dimensions.get_control_dim(edge);
    const int c = input.dimensions.get_edge_c_dim(edge);
    const int g = input.dimensions.get_edge_g_dim(edge);
    auto &output = mco.edges[edge];
    output.f = 0.0;
    std::fill_n(output.df_dx, n_parent, 0.0);
    std::fill_n(output.df_du, m, 0.0);
    std::fill_n(output.df_dtheta, p, 0.0);
    std::fill_n(output.dyn_res, n_child, 0.0);
    std::fill_n(output.c, c, 0.0);
    std::fill_n(output.g, g, 0.0);
    fill_sequence(output.ddyn_dx, n_child * n_parent, 0.025 + 0.004 * edge);
    fill_sequence(output.ddyn_du, n_child * m, -0.031 - 0.003 * edge);
    fill_sequence(output.ddyn_dtheta, n_child * p, 0.0009 * (edge + 1));
    fill_sequence(output.dc_dx, c * n_parent, 0.017 * (edge + 1));
    fill_sequence(output.dc_du, c * m, 0.019 * (edge + 1));
    fill_sequence(output.dc_dtheta, c * p, 0.0008 * (edge + 1));
    fill_sequence(output.dg_dx, g * n_parent, -0.014 * (edge + 1));
    fill_sequence(output.dg_du, g * m, 0.016 * (edge + 1));
    fill_sequence(output.dg_dtheta, g * p, -0.0006 * (edge + 1));
    fill_spd(output.d2L_dx2, n_parent, 0.3 + 0.05 * edge);
    fill_sequence(output.d2L_dxdu, n_parent * m, 0.009 * (edge + 1));
    fill_spd(output.d2L_du2, m, 3.0 + 0.2 * edge);
    fill_sequence(output.d2L_dxdtheta, n_parent * p, 0.0004 * (edge + 1));
    fill_sequence(output.d2L_dudtheta, m * p, -0.0003 * (edge + 1));
    fill_spd(output.d2L_dtheta2, p, theta_diagonal);
  }
}

void expect_kkt_solve(const Input &input, Workspace &workspace,
                      const double tolerance = 1e-9) {
  auto callback_provider = CallbackProvider(input, workspace);
  const int x_dim = input.dimensions.get_x_dim(input.topology.num_edges);
  const int y_dim = input.dimensions.get_y_dim(input.topology.num_edges);
  const int z_dim = input.dimensions.get_z_dim(input.topology.num_edges);
  const int kkt_dim = x_dim + y_dim + z_dim;

  std::vector<double> w(z_dim, 1.3);
  std::vector<double> r2(y_dim, 0.9);
  std::vector<double> r3(z_dim, 0.4);
  std::vector<double> r1(x_dim);
  fill_sequence(r1.data(), x_dim, 0.03);
  for (double &value : r1) {
    value += 0.2;
  }
  ASSERT_TRUE(
      callback_provider.factor(w.data(), r1.data(), r2.data(), r3.data()));

  std::vector<double> rhs(kkt_dim);
  fill_sequence(rhs.data(), kkt_dim, 0.01);
  std::vector<double> solution(kkt_dim, 0.0);
  callback_provider.solve(rhs.data(), solution.data());

  std::vector<double> product_x(x_dim, 0.0);
  std::vector<double> product_y(y_dim, 0.0);
  std::vector<double> product_z(z_dim, 0.0);
  callback_provider.add_Kx_to_y(
      w.data(), r1.data(), r2.data(), r3.data(), solution.data(),
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
  EXPECT_LT(std::sqrt(squared_error), tolerance);
}

TEST(InputValidation, AcceptsSeparateNodeAndEdgeDimensions) {
  const std::array<int, 3> state_dims = {2, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> node_c_dims = {0, 1, 0};
  const std::array<int, 3> node_g_dims = {1, 0, 2};
  const std::array<int, 2> edge_c_dims = {2, 1};
  const std::array<int, 2> edge_g_dims = {1, 3};
  const Dimensions dimensions{2,
                              state_dims.data(),
                              control_dims.data(),
                              node_c_dims.data(),
                              node_g_dims.data(),
                              edge_c_dims.data(),
                              edge_g_dims.data()};

  const ChainTopology chain_topology;
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

  const std::array<int, 2> negative_edge_c_dims = {-1, 1};
  const Dimensions invalid_dimensions{2,
                                      state_dims.data(),
                                      control_dims.data(),
                                      node_c_dims.data(),
                                      node_g_dims.data(),
                                      negative_edge_c_dims.data(),
                                      edge_g_dims.data()};
  EXPECT_EQ(validate_input(invalid_dimensions, tree),
            InputValidationStatus::INVALID_DIMENSIONS);
}

TEST(Workspace, UniformStaticAndDynamicMemorySizesMatch) {
  constexpr int num_edges = 2;
  constexpr int state_dim = 2;
  constexpr int control_dim = 1;
  constexpr int node_c_dim = 1;
  constexpr int node_g_dim = 2;
  constexpr int edge_c_dim = 3;
  constexpr int edge_g_dim = 1;
  constexpr int theta_dim = 2;
  constexpr int num_bound_sides = 0;

  const std::array<int, 3> state_dims = {state_dim, state_dim, state_dim};
  const std::array<int, 2> control_dims = {control_dim, control_dim};
  const std::array<int, 3> node_c_dims = {node_c_dim, node_c_dim, node_c_dim};
  const std::array<int, 3> node_g_dims = {node_g_dim, node_g_dim, node_g_dim};
  const std::array<int, 2> edge_c_dims = {edge_c_dim, edge_c_dim};
  const std::array<int, 2> edge_g_dims = {edge_g_dim, edge_g_dim};
  const Dimensions dimensions{theta_dim,           state_dims.data(),
                              control_dims.data(), node_c_dims.data(),
                              node_g_dims.data(),  edge_c_dims.data(),
                              edge_g_dims.data()};
  const BranchTopology tree;
  const Topology topology{num_edges, 0, tree.parent.data(), tree.child.data()};

  EXPECT_EQ(ModelCallbackOutput::num_bytes(state_dim, control_dim, num_edges,
                                           node_c_dim, node_g_dim, edge_c_dim,
                                           edge_g_dim, theta_dim),
            ModelCallbackOutput::num_bytes(dimensions, topology));
  EXPECT_EQ(Workspace::RegularizedLQRData::num_bytes(
                state_dim, control_dim, num_edges, node_c_dim, node_g_dim,
                edge_c_dim, edge_g_dim, theta_dim),
            Workspace::RegularizedLQRData::num_bytes(dimensions, num_edges));
  EXPECT_EQ(
      Workspace::num_bytes(state_dim, control_dim, num_edges, node_c_dim,
                           node_g_dim, edge_c_dim, edge_g_dim, theta_dim,
                           num_bound_sides, kSettings),
      Workspace::num_bytes(dimensions, topology, num_bound_sides, kSettings));
}

TEST(CallbackProvider, SolvesChainWithNodeAndEdgeConstraints) {
  const std::array<int, 3> state_dims = {2, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> node_c_dims = {1, 0, 2};
  const std::array<int, 3> node_g_dims = {0, 2, 1};
  const std::array<int, 2> edge_c_dims = {1, 2};
  const std::array<int, 2> edge_g_dims = {2, 1};
  const ChainTopology topology;
  Input input{
      .dimensions = {0, state_dims.data(), control_dims.data(),
                     node_c_dims.data(), node_g_dims.data(), edge_c_dims.data(),
                     edge_g_dims.data()},
      .topology = {2, 0, topology.parent.data(), topology.child.data()},
      .model_callback = empty_model_callback(),
      .timeout_callback = []() { return false; },
  };
  Workspace workspace;
  std::vector<unsigned char> memory(Workspace::num_bytes(
      input.dimensions, input.topology, input.num_bound_sides(), kSettings));
  EXPECT_EQ(workspace.mem_assign(input.dimensions, input.topology,
                                 input.num_bound_sides(), kSettings,
                                 memory.data()),
            static_cast<int>(memory.size()));
  initialize_model(input, workspace);
  expect_kkt_solve(input, workspace);
}

TEST(CallbackProvider, SolvesIndependentConstraintsOnSiblingEdges) {
  const std::array<int, 3> state_dims = {2, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> node_c_dims = {1, 0, 1};
  const std::array<int, 3> node_g_dims = {1, 1, 0};
  const std::array<int, 2> edge_c_dims = {2, 1};
  const std::array<int, 2> edge_g_dims = {1, 2};
  const BranchTopology topology;
  Input input{
      .dimensions = {0, state_dims.data(), control_dims.data(),
                     node_c_dims.data(), node_g_dims.data(), edge_c_dims.data(),
                     edge_g_dims.data()},
      .topology = {2, 0, topology.parent.data(), topology.child.data()},
      .model_callback = empty_model_callback(),
      .timeout_callback = []() { return false; },
  };
  Workspace workspace;
  workspace.reserve(input.dimensions, input.topology, input.num_bound_sides(),
                    kSettings);
  initialize_model(input, workspace);
  expect_kkt_solve(input, workspace);
  workspace.free(input.topology);
}

TEST(CallbackProvider, SolvesBranchedSystemWithZeroDimensionalRoot) {
  const std::array<int, 3> state_dims = {0, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> zero_node_dims = {0, 0, 0};
  const std::array<int, 2> zero_edge_dims = {0, 0};
  const BranchTopology topology;
  Input input{
      .dimensions = {0, state_dims.data(), control_dims.data(),
                     zero_node_dims.data(), zero_node_dims.data(),
                     zero_edge_dims.data(), zero_edge_dims.data()},
      .topology = {2, 0, topology.parent.data(), topology.child.data()},
      .model_callback = empty_model_callback(),
      .timeout_callback = []() { return false; },
  };
  Workspace workspace;
  workspace.reserve(input.dimensions, input.topology, input.num_bound_sides(),
                    kSettings);
  initialize_model(input, workspace);
  expect_kkt_solve(input, workspace);
  workspace.free(input.topology);
}

TEST(CallbackProvider, SolvesBranchedSystemWithSchurVariables) {
  const std::array<int, 3> state_dims = {2, 1, 3};
  const std::array<int, 2> control_dims = {1, 2};
  const std::array<int, 3> node_c_dims = {1, 0, 1};
  const std::array<int, 3> node_g_dims = {0, 1, 1};
  const std::array<int, 2> edge_c_dims = {1, 2};
  const std::array<int, 2> edge_g_dims = {2, 1};
  const BranchTopology topology;
  Input input{
      .dimensions = {2, state_dims.data(), control_dims.data(),
                     node_c_dims.data(), node_g_dims.data(), edge_c_dims.data(),
                     edge_g_dims.data()},
      .topology = {2, 0, topology.parent.data(), topology.child.data()},
      .model_callback = empty_model_callback(),
      .timeout_callback = []() { return false; },
  };
  Workspace workspace;
  std::vector<unsigned char> memory(Workspace::num_bytes(
      input.dimensions, input.topology, input.num_bound_sides(), kSettings));
  EXPECT_EQ(workspace.mem_assign(input.dimensions, input.topology,
                                 input.num_bound_sides(), kSettings,
                                 memory.data()),
            static_cast<int>(memory.size()));
  initialize_model(input, workspace, 6.0);
  expect_kkt_solve(input, workspace, 1e-8);
}

TEST(Solve, DispatchesStructuredCallback) {
  const std::array<int, 2> state_dims = {1, 1};
  const std::array<int, 1> control_dims = {1};
  const std::array<int, 2> zero_node_dims = {0, 0};
  const std::array<int, 1> zero_edge_dims = {0};
  const std::array<int, 1> parents = {0};
  const std::array<int, 1> children = {1};
  const std::array<double, 1> initial_state = {1.0};
  const std::array<double, 3> primal_scaling = {1.0, 1.0, 1.0};
  const std::array<double, 2> equality_scaling = {1.0, 1.0};
  int model_callback_count = 0;

  Input input{
      .dimensions = {0, state_dims.data(), control_dims.data(),
                     zero_node_dims.data(), zero_node_dims.data(),
                     zero_edge_dims.data(), zero_edge_dims.data()},
      .topology = {1, 0, parents.data(), children.data()},
      .initial_state = initial_state.data(),
      .model_callback =
          [&model_callback_count](const ModelCallbackInput &callback_input,
                                  ModelCallbackOutput &output) {
            ++model_callback_count;
            for (int node = 0; node < 2; ++node) {
              const auto &node_input = callback_input.nodes[node];
              auto &node_output = output.nodes[node];
              EXPECT_EQ(node_input.node, node);
              const double state = node_input.state[0];
              if (node == 0) {
                node_output.f = 0.5 * state * state;
                node_output.df_dx[0] = state;
              } else {
                const double residual = state - 2.0;
                node_output.f = 0.5 * residual * residual;
                node_output.df_dx[0] = residual;
              }
              node_output.d2L_dx2[0] = 1.0;
            }

            const auto &edge_input = callback_input.edges[0];
            auto &edge_output = output.edges[0];
            EXPECT_EQ(edge_input.edge, 0);
            EXPECT_EQ(edge_input.parent, 0);
            EXPECT_EQ(edge_input.child, 1);
            const double control = edge_input.control[0];
            edge_output.f = 0.5 * control * control;
            edge_output.df_dx[0] = 0.0;
            edge_output.df_du[0] = control;
            edge_output.dyn_res[0] = edge_input.parent_state[0] + control -
                                     edge_input.child_state[0];
            edge_output.ddyn_dx[0] = 1.0;
            edge_output.ddyn_du[0] = 1.0;
            edge_output.d2L_dx2[0] = 0.0;
            edge_output.d2L_dxdu[0] = 0.0;
            edge_output.d2L_du2[0] = 1.0;
          },
      .timeout_callback = []() { return false; },
      .residual_scaling =
          {
              .dual = primal_scaling.data(),
              .equality = equality_scaling.data(),
              .variable_bound = primal_scaling.data(),
          },
  };
  auto settings = ::sip::Settings{};
  settings.logging.print_logs = false;
  settings.logging.print_line_search_logs = false;
  settings.logging.print_search_direction_logs = false;
  settings.logging.print_derivative_check_logs = false;

  Workspace workspace;
  workspace.reserve(input.dimensions, input.topology, input.num_bound_sides(),
                    settings);
  std::fill_n(workspace.sip_workspace.vars.x, 3, 0.0);
  std::fill_n(workspace.sip_workspace.vars.y, 2, 0.0);

  const auto output = solve(input, settings, workspace);

  EXPECT_EQ(output.exit_status, ::sip::Status::SOLVED);
  EXPECT_GT(model_callback_count, 0);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[0], 1.0, 1e-8);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[1], 0.5, 1e-8);
  EXPECT_NEAR(workspace.sip_workspace.vars.x[2], 1.5, 1e-8);
  workspace.free(input.topology);
}

} // namespace
} // namespace sip::optimal_control

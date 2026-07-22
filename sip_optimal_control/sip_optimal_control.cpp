#include "sip_optimal_control/sip_optimal_control.hpp"
#include "sip/sip.hpp"
#include "sip_optimal_control/helpers.hpp"

#include <algorithm>

namespace sip::optimal_control {

auto solve(const Input &input, const ::sip::Settings &settings,
           Workspace &workspace) -> ::sip::Output {
  CallbackProvider callback_provider(input, workspace);

  const auto model_callback =
      [&input, &workspace](const sip::ModelCallbackInput &mci) -> void {
    const double *theta = mci.x + workspace.stagewise_x_dim;
    workspace.model_callback_input.theta = theta;
    for (int node = 0; node < input.topology.num_nodes(); ++node) {
      workspace.model_callback_input.nodes[node] = NodeModelCallbackInput{
          .node = node,
          .state = mci.x + workspace.x_state_offsets[node],
          .equality_constraint_multipliers =
              mci.y + workspace.y_node_c_offsets[node],
          .inequality_constraint_multipliers =
              mci.z + workspace.z_node_offsets[node],
      };
    }
    for (int edge = 0; edge < input.topology.num_edges; ++edge) {
      const int parent = input.topology.edge_parents[edge];
      const int child = input.topology.edge_children[edge];
      workspace.model_callback_input.edges[edge] = EdgeModelCallbackInput{
          .edge = edge,
          .parent = parent,
          .child = child,
          .parent_state = mci.x + workspace.x_state_offsets[parent],
          .control = mci.x + workspace.x_control_offsets[edge],
          .child_state = mci.x + workspace.x_state_offsets[child],
          .costate = mci.y + workspace.y_dyn_offsets[child],
          .equality_constraint_multipliers =
              mci.y + workspace.y_edge_c_offsets[edge],
          .inequality_constraint_multipliers =
              mci.z + workspace.z_edge_offsets[edge],
      };
    }
    input.model_callback(workspace.model_callback_input,
                         workspace.model_callback_output);

    workspace.f = 0.0;
    for (int node = 0; node < input.topology.num_nodes(); ++node) {
      workspace.f += workspace.model_callback_output.nodes[node].f;
    }
    for (int edge = 0; edge < input.topology.num_edges; ++edge) {
      workspace.f += workspace.model_callback_output.edges[edge].f;
    }

    if (mci.new_x) {
      {
        std::fill_n(workspace.gradient_f, workspace.x_dim, 0.0);
        for (int node = 0; node < input.topology.num_nodes(); ++node) {
          const auto &output = workspace.model_callback_output.nodes[node];
          const int n = input.dimensions.get_state_dim(node);
          for (int row = 0; row < n; ++row) {
            workspace.gradient_f[workspace.x_state_offsets[node] + row] +=
                output.df_dx[row];
          }
          for (int row = 0; row < input.dimensions.theta_dim; ++row) {
            workspace.gradient_f[workspace.stagewise_x_dim + row] +=
                output.df_dtheta[row];
          }
        }
        for (int edge = 0; edge < input.topology.num_edges; ++edge) {
          const int parent = input.topology.edge_parents[edge];
          const auto &output = workspace.model_callback_output.edges[edge];
          const int n = input.dimensions.get_state_dim(parent);
          const int m = input.dimensions.get_control_dim(edge);
          for (int row = 0; row < n; ++row) {
            workspace.gradient_f[workspace.x_state_offsets[parent] + row] +=
                output.df_dx[row];
          }
          for (int row = 0; row < m; ++row) {
            workspace.gradient_f[workspace.x_control_offsets[edge] + row] +=
                output.df_du[row];
          }
          for (int row = 0; row < input.dimensions.theta_dim; ++row) {
            workspace.gradient_f[workspace.stagewise_x_dim + row] +=
                output.df_dtheta[row];
          }
        }
      }

      {
        const int root = input.topology.root;
        const int n_root = input.dimensions.get_state_dim(root);
        for (int row = 0; row < n_root; ++row) {
          workspace.c[workspace.y_dyn_offsets[root] + row] =
              input.initial_state[row] -
              mci.x[workspace.x_state_offsets[root] + row];
        }
        for (int node = 0; node < input.topology.num_nodes(); ++node) {
          std::copy_n(workspace.model_callback_output.nodes[node].c,
                      input.dimensions.get_node_c_dim(node),
                      workspace.c + workspace.y_node_c_offsets[node]);
        }
        for (int edge = 0; edge < input.topology.num_edges; ++edge) {
          const int child = input.topology.edge_children[edge];
          std::copy_n(workspace.model_callback_output.edges[edge].dyn_res,
                      input.dimensions.get_state_dim(child),
                      workspace.c + workspace.y_dyn_offsets[child]);
          std::copy_n(workspace.model_callback_output.edges[edge].c,
                      input.dimensions.get_edge_c_dim(edge),
                      workspace.c + workspace.y_edge_c_offsets[edge]);
        }
      }

      {
        for (int node = 0; node < input.topology.num_nodes(); ++node) {
          std::copy_n(workspace.model_callback_output.nodes[node].g,
                      input.dimensions.get_node_g_dim(node),
                      workspace.g + workspace.z_node_offsets[node]);
        }
        for (int edge = 0; edge < input.topology.num_edges; ++edge) {
          std::copy_n(workspace.model_callback_output.edges[edge].g,
                      input.dimensions.get_edge_g_dim(edge),
                      workspace.g + workspace.z_edge_offsets[edge]);
        }
      }
    }
  };

  const auto factor = [&callback_provider](const double *w, const double *r1,
                                           const double *r2,
                                           const double *r3) -> bool {
    return callback_provider.factor(w, r1, r2, r3);
  };

  const auto solve = [&callback_provider](const double *b, double *v) -> void {
    callback_provider.solve(b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider](const double *w, const double *r1, const double *r2,
                           const double *r3, const double *x_x,
                           const double *x_y, const double *x_z, double *y_x,
                           double *y_y, double *y_z) -> void {
    callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y, y_z);
  };

  const auto add_Hx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    callback_provider.add_Hx_to_y(x, y);
  };

  const auto add_Cx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    callback_provider.add_Cx_to_y(x, y);
  };

  const auto add_CTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    callback_provider.add_CTx_to_y(x, y);
  };

  const auto add_Gx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    callback_provider.add_Gx_to_y(x, y);
  };

  const auto add_GTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    callback_provider.add_GTx_to_y(x, y);
  };

  const auto get_f = [&workspace]() -> double { return workspace.f; };

  const auto get_grad_f = [&workspace]() -> double * {
    return workspace.gradient_f;
  };

  const auto get_c = [&workspace]() -> double * { return workspace.c; };

  const auto get_g = [&workspace]() -> double * { return workspace.g; };

  const auto sip_input = sip::Input{
      .factor = std::cref(factor),
      .solve = std::cref(solve),
      .add_Kx_to_y = std::cref(add_Kx_to_y),
      .add_Hx_to_y = std::cref(add_Hx_to_y),
      .add_Cx_to_y = std::cref(add_Cx_to_y),
      .add_CTx_to_y = std::cref(add_CTx_to_y),
      .add_Gx_to_y = std::cref(add_Gx_to_y),
      .add_GTx_to_y = std::cref(add_GTx_to_y),
      .get_f = std::cref(get_f),
      .get_grad_f = std::cref(get_grad_f),
      .get_c = std::cref(get_c),
      .get_g = std::cref(get_g),
      .model_callback = std::cref(model_callback),
      .timeout_callback = std::cref(input.timeout_callback),
      .lower_bounds = input.lower_bounds,
      .upper_bounds = input.upper_bounds,
      .residual_scaling = input.residual_scaling,
      .dimensions =
          {
              .x_dim = input.dimensions.get_x_dim(input.topology.num_edges),
              .s_dim = input.dimensions.get_z_dim(input.topology.num_edges),
              .y_dim = input.dimensions.get_y_dim(input.topology.num_edges),
          },
  };

  return sip::solve(sip_input, settings, workspace.sip_workspace);
}

} // namespace sip::optimal_control

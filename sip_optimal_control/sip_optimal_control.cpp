#include "sip_optimal_control/sip_optimal_control.hpp"
#include "sip/sip.hpp"
#include "sip_optimal_control/helpers.hpp"

#include <algorithm>

namespace sip::optimal_control {

auto build_sip_input(const Input &input, Workspace &workspace) -> sip::Input {
  CallbackProvider callback_provider(input, workspace);

  const auto model_callback =
      [&input, &workspace](const sip::ModelCallbackInput &mci) -> void {
    {
      double *x = mci.x;
      if (mci.new_x) {
        for (int i = 0; i < input.dimensions.num_stages; ++i) {
          workspace.model_callback_input.states[i] = x;
          x += input.dimensions.state_dim;
          workspace.model_callback_input.controls[i] = x;
          x += input.dimensions.control_dim;
        }
        workspace.model_callback_input.states[input.dimensions.num_stages] = x;
      }
    }

    {
      double *y = mci.y;
      if (mci.new_y) {
        for (int i = 0; i <= input.dimensions.num_stages; ++i) {
          workspace.model_callback_input.costates[i] = y;
          y += input.dimensions.state_dim;
          workspace.model_callback_input.equality_constraint_multipliers[i] = y;
          y += input.dimensions.c_dim;
        }
      }
    }

    {
      double *z = mci.z;
      if (mci.new_z) {
        for (int i = 0; i <= input.dimensions.num_stages; ++i) {
          workspace.model_callback_input.inequality_constraint_multipliers[i] =
              z;
          z += input.dimensions.g_dim;
        }
      }
    }

    input.model_callback(workspace.model_callback_input);

    if (mci.new_x) {
      {
        double *grad_f = workspace.gradient_f;
        for (int i = 0; i < input.dimensions.num_stages; ++i) {
          std::copy_n(workspace.model_callback_output.df_dx[i],
                      input.dimensions.state_dim, grad_f);
          grad_f += input.dimensions.state_dim;
          std::copy_n(workspace.model_callback_output.df_du[i],
                      input.dimensions.control_dim, grad_f);
          grad_f += input.dimensions.control_dim;
        }
        std::copy_n(
            workspace.model_callback_output.df_dx[input.dimensions.num_stages],
            input.dimensions.state_dim, grad_f);
      }

      {
        double *c = workspace.c;
        for (int i = 0; i <= input.dimensions.num_stages; ++i) {
          std::copy_n(workspace.model_callback_output.dyn_res[i],
                      input.dimensions.state_dim, c);
          c += input.dimensions.state_dim;
          std::copy_n(workspace.model_callback_output.c[i],
                      input.dimensions.c_dim, c);
          c += input.dimensions.c_dim;
        }
      }

      {
        double *g = workspace.g;
        for (int i = 0; i <= input.dimensions.num_stages; ++i) {
          std::copy_n(workspace.model_callback_output.g[i],
                      input.dimensions.g_dim, g);
          g += input.dimensions.g_dim;
        }
      }
    }
  };

  const auto factor = [&callback_provider](const double *w, const double r1,
                                           const double r2,
                                           const double r3) -> void {
    return callback_provider.factor(w, r1, r2, r3);
  };

  const auto solve = [&callback_provider](const double *b, double *v) -> void {
    return callback_provider.solve(b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider](const double *w, const double r1, const double r2,
                           const double r3, const double *x_x,
                           const double *x_y, const double *x_z, double *y_x,
                           double *y_y, double *y_z) -> void {
    return callback_provider.add_Kx_to_y(w, r1, r2, r3, x_x, x_y, x_z, y_x, y_y,
                                         y_z);
  };

  const auto add_Hx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    return callback_provider.add_Hx_to_y(x, y);
  };

  const auto add_Cx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    return callback_provider.add_Cx_to_y(x, y);
  };

  const auto add_CTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    return callback_provider.add_CTx_to_y(x, y);
  };

  const auto add_Gx_to_y = [&callback_provider](const double *x,
                                                double *y) -> void {
    return callback_provider.add_Gx_to_y(x, y);
  };

  const auto add_GTx_to_y = [&callback_provider](const double *x,
                                                 double *y) -> void {
    return callback_provider.add_GTx_to_y(x, y);
  };

  const auto get_f = [&workspace]() -> double {
    return workspace.model_callback_output.f;
  };

  const auto get_grad_f = [&workspace]() -> double * {
    return workspace.gradient_f;
  };

  const auto get_c = [&workspace]() -> double * { return workspace.c; };

  const auto get_g = [&workspace]() -> double * { return workspace.g; };

  return sip::Input{
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
      .dimensions =
          {
              .x_dim = input.dimensions.state_dim *
                       (input.dimensions.num_stages + 1),
              .s_dim =
                  input.dimensions.g_dim * (input.dimensions.num_stages + 1),
              .y_dim = (input.dimensions.state_dim + input.dimensions.c_dim) *
                       (input.dimensions.num_stages * 1),
          },
  };
}

auto solve(const Input &input, const ::sip::Settings &settings,
           Workspace &workspace) -> ::sip::Output {
  const auto sip_input = build_sip_input(input, workspace);
  return sip::solve(sip_input, settings, workspace.sip_workspace);
}

} // namespace sip::optimal_control

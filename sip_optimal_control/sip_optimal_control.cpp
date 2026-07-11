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
    {
      double *x = mci.x;
      if (mci.new_x) {
        for (int i = 0; i < input.dimensions.num_stages; ++i) {
          workspace.model_callback_input.states[i] = x;
          x += input.dimensions.get_state_dim(i);
          workspace.model_callback_input.controls[i] = x;
          x += input.dimensions.get_control_dim(i);
        }
        workspace.model_callback_input.states[input.dimensions.num_stages] = x;
        x += input.dimensions.get_state_dim(input.dimensions.num_stages);
        workspace.model_callback_input.theta = x;
      }
    }

    {
      double *y = mci.y;
      if (mci.new_y) {
        for (int i = 0; i <= input.dimensions.num_stages; ++i) {
          workspace.model_callback_input.costates[i] = y;
          y += input.dimensions.get_state_dim(i);
          workspace.model_callback_input.equality_constraint_multipliers[i] = y;
          y += input.dimensions.get_c_dim(i);
        }
      }
    }

    {
      double *z = mci.z;
      if (mci.new_z) {
        for (int i = 0; i <= input.dimensions.num_stages; ++i) {
          workspace.model_callback_input.inequality_constraint_multipliers[i] =
              z;
          z += input.dimensions.get_g_dim(i);
        }
      }
    }

    input.model_callback(workspace.model_callback_input);

    if (mci.new_x) {
      {
        double *grad_f = workspace.gradient_f;
        for (int i = 0; i < input.dimensions.num_stages; ++i) {
          std::copy_n(workspace.model_callback_output.df_dx[i],
                      input.dimensions.get_state_dim(i), grad_f);
          grad_f += input.dimensions.get_state_dim(i);
          std::copy_n(workspace.model_callback_output.df_du[i],
                      input.dimensions.get_control_dim(i), grad_f);
          grad_f += input.dimensions.get_control_dim(i);
        }
        std::copy_n(
            workspace.model_callback_output.df_dx[input.dimensions.num_stages],
            input.dimensions.get_state_dim(input.dimensions.num_stages),
            grad_f);
        grad_f += input.dimensions.get_state_dim(input.dimensions.num_stages);
        std::copy_n(workspace.model_callback_output.df_dtheta,
                    input.dimensions.theta_dim, grad_f);
      }

      {
        double *c = workspace.c;
        for (int i = 0; i <= input.dimensions.num_stages; ++i) {
          std::copy_n(workspace.model_callback_output.dyn_res[i],
                      input.dimensions.get_state_dim(i), c);
          c += input.dimensions.get_state_dim(i);
          std::copy_n(workspace.model_callback_output.c[i],
                      input.dimensions.get_c_dim(i), c);
          c += input.dimensions.get_c_dim(i);
        }
      }

      {
        double *g = workspace.g;
        for (int i = 0; i <= input.dimensions.num_stages; ++i) {
          std::copy_n(workspace.model_callback_output.g[i],
                      input.dimensions.get_g_dim(i), g);
          g += input.dimensions.get_g_dim(i);
        }
      }
    }
  };

  const auto factor = [&callback_provider](const double *w, const double r1,
                                           const double *r2,
                                           const double *r3) -> bool {
    return callback_provider.factor(w, r1, r2, r3);
  };

  const auto solve = [&callback_provider](const double *b, double *v) -> void {
    callback_provider.solve(b, v);
  };

  const auto add_Kx_to_y =
      [&callback_provider](const double *w, const double r1, const double *r2,
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

  const auto get_f = [&workspace]() -> double {
    return workspace.model_callback_output.f;
  };

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
      .dimensions =
          {
              .x_dim = input.dimensions.get_x_dim(),
              .s_dim = input.dimensions.get_z_dim(),
              .y_dim = input.dimensions.get_y_dim(),
          },
  };

  return sip::solve(sip_input, settings, workspace.sip_workspace);
}

} // namespace sip::optimal_control

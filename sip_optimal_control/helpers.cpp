#include "sip_optimal_control/helpers.hpp"

#define EIGEN_NO_MALLOC

#include <Eigen/Dense>

namespace sip::optimal_control {

CallbackProvider::CallbackProvider(const Input &input, Workspace &workspace)
    : input_(input), workspace_(workspace) {}

void CallbackProvider::factor(const double *w, const double r1, const double r2,
                              const double r3) {
  const auto &mco = workspace_.model_callback_output;
  auto &lqr_data = workspace_.regularized_lqr_data;

  int w_offset = 0;

  lqr_data.r2 = r2;
  const double r2_inv = 1.0 / r2;
  lqr_data.r2_inv = r2_inv;

  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto Q_i = Eigen::Map<const Eigen::MatrixXd>(
        mco.d2L_dx2[i], input_.dimensions.state_dim,
        input_.dimensions.state_dim);

    const auto M_i = Eigen::Map<const Eigen::MatrixXd>(
        mco.d2L_dxdu[i], input_.dimensions.state_dim,
        input_.dimensions.control_dim);

    const auto R_i = Eigen::Map<const Eigen::MatrixXd>(
        mco.d2L_du2[i], input_.dimensions.control_dim,
        input_.dimensions.control_dim);

    const auto jac_x_c_i = Eigen::Map<const Eigen::MatrixXd>(
        mco.dc_dx[i], input_.dimensions.c_dim, input_.dimensions.state_dim);

    const auto jac_u_c_i = Eigen::Map<const Eigen::MatrixXd>(
        mco.dc_du[i], input_.dimensions.c_dim, input_.dimensions.control_dim);

    const auto jac_x_g_i = Eigen::Map<const Eigen::MatrixXd>(
        mco.dg_dx[i], input_.dimensions.g_dim, input_.dimensions.state_dim);

    const auto jac_u_g_i = Eigen::Map<const Eigen::MatrixXd>(
        mco.dg_du[i], input_.dimensions.g_dim, input_.dimensions.control_dim);

    auto mod_w_inv_i = Eigen::Map<Eigen::VectorXd>(lqr_data.mod_w_inv[i],
                                                   input_.dimensions.g_dim);

    for (int j = 0; j < input_.dimensions.g_dim; ++j) {
      mod_w_inv_i(j) = 1.0 / (w[w_offset++] + r3);
    }

    auto Q_i_mod = Eigen::Map<Eigen::MatrixXd>(lqr_data.Q_mod[i],
                                               input_.dimensions.state_dim,
                                               input_.dimensions.state_dim);

    auto M_i_mod = Eigen::Map<Eigen::MatrixXd>(lqr_data.M_mod[i],
                                               input_.dimensions.state_dim,
                                               input_.dimensions.control_dim);

    auto R_i_mod = Eigen::Map<Eigen::MatrixXd>(lqr_data.R_mod[i],
                                               input_.dimensions.control_dim,
                                               input_.dimensions.control_dim);

    Q_i_mod.noalias() =
        Q_i + r2_inv * jac_x_c_i.transpose() * jac_x_c_i +
        jac_x_g_i.transpose() * mod_w_inv_i.asDiagonal() * jac_x_g_i;

    M_i_mod.noalias() =
        M_i + r2_inv * jac_x_c_i.transpose() * jac_u_c_i +
        jac_x_g_i.transpose() * mod_w_inv_i.asDiagonal() * jac_u_g_i;

    R_i_mod.noalias() =
        R_i + r2_inv * jac_u_c_i.transpose() * jac_u_c_i +
        jac_u_g_i.transpose() * mod_w_inv_i.asDiagonal() * jac_u_g_i;
  }

  const auto jac_x_c_N = Eigen::Map<const Eigen::MatrixXd>(
      mco.dc_dx[input_.dimensions.num_stages], input_.dimensions.c_dim,
      input_.dimensions.state_dim);

  const auto jac_x_g_N = Eigen::Map<const Eigen::MatrixXd>(
      mco.dg_dx[input_.dimensions.num_stages], input_.dimensions.g_dim,
      input_.dimensions.state_dim);

  const auto Q_N = Eigen::Map<const Eigen::MatrixXd>(
      mco.d2L_dx2[input_.dimensions.num_stages], input_.dimensions.state_dim,
      input_.dimensions.state_dim);

  auto mod_w_inv_N = Eigen::Map<Eigen::VectorXd>(
      lqr_data.mod_w_inv[input_.dimensions.num_stages],
      input_.dimensions.g_dim);

  for (int j = 0; j < input_.dimensions.g_dim; ++j) {
    mod_w_inv_N(j) = 1.0 / (w[w_offset++] + r3);
  }

  auto Q_N_mod = Eigen::Map<Eigen::MatrixXd>(
      lqr_data.Q_mod[input_.dimensions.num_stages], input_.dimensions.state_dim,
      input_.dimensions.state_dim);

  Q_N_mod.noalias() = Q_N;
  for (int i = 0; i < input_.dimensions.state_dim; ++i) {
    Q_N_mod(i, i) += r1;
  }
  Q_N_mod.noalias() += r2_inv * jac_x_c_N.transpose() * jac_x_c_N;
  Q_N_mod.noalias() +=
      jac_x_g_N.transpose() * mod_w_inv_N.asDiagonal() * jac_x_g_N;

  lqr_input_.Q = lqr_data.Q_mod;
  lqr_input_.M = lqr_data.M_mod;
  lqr_input_.R = lqr_data.R_mod;
  lqr_input_.A = mco.ddyn_dx;
  lqr_input_.B = mco.ddyn_du;
  lqr_input_.dimensions = {
      .state_dim = input_.dimensions.state_dim,
      .control_dim = input_.dimensions.control_dim,
      .num_stages = input_.dimensions.num_stages,
  };
  auto lqr_solver = LQR(lqr_input_, workspace_.lqr_workspace);
  lqr_solver.factor(r2);
}

void CallbackProvider::solve(const double *b, double *sol) {
  const int x_dim = input_.dimensions.get_x_dim();
  const int y_dim = input_.dimensions.get_y_dim();

  const double r2 = workspace_.regularized_lqr_data.r2;
  const double r2_inv = workspace_.regularized_lqr_data.r2_inv;

  {
    const double *b_x = b;
    const double *b_y = b_x + x_dim;
    const double *b_z = b_y + y_dim;

    for (int i = 0; i < input_.dimensions.num_stages; ++i) {
      const auto b_x_i =
          Eigen::Map<const Eigen::VectorXd>(b_x, input_.dimensions.state_dim);
      b_x += input_.dimensions.state_dim;

      const auto b_u_i =
          Eigen::Map<const Eigen::VectorXd>(b_x, input_.dimensions.control_dim);
      b_x += input_.dimensions.control_dim;

      const auto b_y_i_prefix =
          Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.state_dim);
      b_y += input_.dimensions.state_dim;

      const auto b_y_i_suffix =
          Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.c_dim);
      b_y += input_.dimensions.c_dim;

      const auto b_z_i =
          Eigen::Map<const Eigen::VectorXd>(b_z, input_.dimensions.g_dim);
      b_z += input_.dimensions.g_dim;

      const auto jac_x_c_i = Eigen::Map<const Eigen::MatrixXd>(
          workspace_.model_callback_output.dc_dx[i], input_.dimensions.c_dim,
          input_.dimensions.state_dim);

      const auto jac_x_g_i = Eigen::Map<const Eigen::MatrixXd>(
          workspace_.model_callback_output.dg_dx[i], input_.dimensions.g_dim,
          input_.dimensions.state_dim);

      const auto jac_u_c_i = Eigen::Map<const Eigen::MatrixXd>(
          workspace_.model_callback_output.dc_du[i], input_.dimensions.c_dim,
          input_.dimensions.control_dim);

      const auto jac_u_g_i = Eigen::Map<const Eigen::MatrixXd>(
          workspace_.model_callback_output.dg_du[i], input_.dimensions.g_dim,
          input_.dimensions.control_dim);

      const auto mod_w_inv_i = Eigen::Map<const Eigen::VectorXd>(
          workspace_.regularized_lqr_data.mod_w_inv[i],
          input_.dimensions.g_dim);

      auto q_i_mod =
          Eigen::Map<Eigen::VectorXd>(workspace_.regularized_lqr_data.q_mod[i],
                                      input_.dimensions.state_dim);

      auto c_i_mod =
          Eigen::Map<Eigen::VectorXd>(workspace_.regularized_lqr_data.c_mod[i],
                                      input_.dimensions.state_dim);

      auto r_i_mod =
          Eigen::Map<Eigen::VectorXd>(workspace_.regularized_lqr_data.r_mod[i],
                                      input_.dimensions.control_dim);

      q_i_mod.noalias() =
          -(b_x_i + r2_inv * jac_x_c_i.transpose() * b_y_i_suffix +
            jac_x_g_i.transpose() * mod_w_inv_i.asDiagonal() * b_z_i);

      r_i_mod.noalias() =
          -(b_u_i + r2_inv * jac_u_c_i.transpose() * b_y_i_suffix +
            jac_u_g_i.transpose() * mod_w_inv_i.asDiagonal() * b_z_i);

      c_i_mod.noalias() = -b_y_i_prefix;
    }

    const auto jac_x_c_N = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dx[input_.dimensions.num_stages],
        input_.dimensions.c_dim, input_.dimensions.state_dim);

    const auto jac_x_g_N = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[input_.dimensions.num_stages],
        input_.dimensions.g_dim, input_.dimensions.state_dim);

    const auto mod_w_inv_N = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.mod_w_inv[input_.dimensions.num_stages],
        input_.dimensions.g_dim);

    const auto b_x_N =
        Eigen::Map<const Eigen::VectorXd>(b_x, input_.dimensions.state_dim);

    const auto b_y_N_prefix =
        Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.state_dim);
    b_y += input_.dimensions.state_dim;

    const auto b_y_N_suffix =
        Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.c_dim);

    const auto b_z_N =
        Eigen::Map<const Eigen::VectorXd>(b_z, input_.dimensions.g_dim);

    auto q_N_mod = Eigen::Map<Eigen::VectorXd>(
        workspace_.regularized_lqr_data.q_mod[input_.dimensions.num_stages],
        input_.dimensions.state_dim);

    auto c_N_mod = Eigen::Map<Eigen::VectorXd>(
        workspace_.regularized_lqr_data.c_mod[input_.dimensions.num_stages],
        input_.dimensions.state_dim);

    q_N_mod.noalias() =
        -(b_x_N + r2_inv * jac_x_c_N.transpose() * b_y_N_suffix +
          jac_x_g_N.transpose() * mod_w_inv_N.asDiagonal() * b_z_N);

    c_N_mod = -b_y_N_prefix;
  }

  auto &lqr_data = workspace_.regularized_lqr_data;
  auto &lqr_output = workspace_.lqr_output;

  {
    double *sol_x = sol;
    double *sol_y = sol_x + input_.dimensions.get_x_dim();

    lqr_input_.q = lqr_data.q_mod;
    lqr_input_.r = lqr_data.r_mod;
    lqr_input_.c = lqr_data.c_mod;

    for (int i = 0; i < input_.dimensions.num_stages; ++i) {
      lqr_output.x[i] = sol_x;
      sol_x += input_.dimensions.state_dim;
      lqr_output.u[i] = sol_x;
      sol_x += input_.dimensions.control_dim;
      lqr_output.y[i] = sol_y;
      sol_y += input_.dimensions.state_dim + input_.dimensions.c_dim;
    }
    lqr_output.x[input_.dimensions.num_stages] = sol_x;
    sol_x += input_.dimensions.state_dim;
    lqr_output.u[input_.dimensions.num_stages] = sol_x;
    lqr_output.y[input_.dimensions.num_stages] = sol_y;
  }

  auto lqr_solver = LQR(lqr_input_, workspace_.lqr_workspace);
  lqr_solver.solve(r2, lqr_output);

  const double *b_y = b + x_dim;
  const double *b_z = b_y + y_dim;

  double *x = sol;
  double *y = x + x_dim;
  double *z = y + y_dim;

  b_y += input_.dimensions.state_dim;
  y += input_.dimensions.state_dim;

  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto jac_x_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dx[i], input_.dimensions.c_dim,
        input_.dimensions.state_dim);

    const auto jac_x_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[i], input_.dimensions.g_dim,
        input_.dimensions.state_dim);

    const auto jac_u_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_du[i], input_.dimensions.c_dim,
        input_.dimensions.control_dim);

    const auto jac_u_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_du[i], input_.dimensions.g_dim,
        input_.dimensions.control_dim);

    const auto mod_w_inv_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.mod_w_inv[i], input_.dimensions.g_dim);

    const auto b_y_i_suffix =
        Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.c_dim);
    b_y += input_.dimensions.state_dim + input_.dimensions.c_dim;

    const auto b_z_i =
        Eigen::Map<const Eigen::VectorXd>(b_z, input_.dimensions.g_dim);

    b_z += input_.dimensions.g_dim;

    const auto x_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);
    x += input_.dimensions.state_dim;

    const auto u_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.control_dim);
    x += input_.dimensions.control_dim;

    auto y_i_suffix = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.c_dim);
    y += input_.dimensions.state_dim + input_.dimensions.c_dim;

    auto z_i = Eigen::Map<Eigen::VectorXd>(z, input_.dimensions.g_dim);
    z += input_.dimensions.g_dim;

    y_i_suffix.noalias() =
        r2_inv * (jac_x_c_i * x_i + jac_u_c_i * u_i - b_y_i_suffix);

    z_i.noalias() =
        mod_w_inv_i.asDiagonal() * (jac_x_g_i * x_i + jac_u_g_i * u_i - b_z_i);
  }

  const auto x_N =
      Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);

  const auto jac_x_c_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dc_dx[input_.dimensions.num_stages],
      input_.dimensions.c_dim, input_.dimensions.state_dim);

  const auto jac_x_g_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dg_dx[input_.dimensions.num_stages],
      input_.dimensions.g_dim, input_.dimensions.state_dim);

  const auto b_y_N_suffix =
      Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.c_dim);

  const auto b_z_N =
      Eigen::Map<const Eigen::VectorXd>(b_z, input_.dimensions.g_dim);

  const auto mod_w_inv_N = Eigen::Map<const Eigen::VectorXd>(
      workspace_.regularized_lqr_data.mod_w_inv[input_.dimensions.num_stages],
      input_.dimensions.g_dim);

  auto y_N_suffix = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.c_dim);
  y_N_suffix.noalias() = r2_inv * (jac_x_c_N * x_N - b_y_N_suffix);

  auto z_N = Eigen::Map<Eigen::VectorXd>(z, input_.dimensions.g_dim);
  z_N.noalias() = mod_w_inv_N.asDiagonal() * (jac_x_g_N * x_N - b_z_N);
}

void CallbackProvider::add_Kx_to_y(const double *w, const double r1,
                                   const double r2, const double r3,
                                   const double *x_x, const double *x_y,
                                   const double *x_z, double *y_x, double *y_y,
                                   double *y_z) {
  add_Hx_to_y(x_x, y_x);
  add_Cx_to_y(x_x, y_y);
  add_CTx_to_y(x_y, y_x);
  add_Gx_to_y(x_x, y_z);
  add_GTx_to_y(x_z, y_x);

  const int x_dim = input_.dimensions.get_x_dim();
  const int y_dim = input_.dimensions.get_y_dim();
  const int z_dim = input_.dimensions.get_z_dim();

  for (int i = 0; i < x_dim; ++i) {
    y_x[i] += r1 * x_x[i];
  }
  for (int i = 0; i < y_dim; ++i) {
    y_y[i] -= r2 * x_y[i];
  }
  for (int i = 0; i < z_dim; ++i) {
    y_z[i] -= (w[i] + r3) * x_z[i];
  }
}

void CallbackProvider::add_Hx_to_y(const double *x, double *y) {
  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto Q_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dx2[i],
        input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto M_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dxdu[i],
        input_.dimensions.state_dim, input_.dimensions.control_dim);

    const auto R_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_du2[i],
        input_.dimensions.control_dim, input_.dimensions.control_dim);

    const auto x_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);
    x += input_.dimensions.state_dim;

    const auto u_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.control_dim);
    x += input_.dimensions.control_dim;

    auto y_i_x = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);
    y += input_.dimensions.state_dim;

    auto y_i_u = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.control_dim);
    y += input_.dimensions.control_dim;

    y_i_x += Q_i * x_i + M_i * u_i;
    y_i_u += M_i.transpose() * x_i + R_i * u_i;
  }

  const auto Q_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.d2L_dx2[input_.dimensions.num_stages],
      input_.dimensions.state_dim, input_.dimensions.state_dim);

  const auto x_N =
      Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);

  auto y_N_x = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);

  y_N_x += Q_N * x_N;
}

void CallbackProvider::add_Cx_to_y(const double *x, double *y) {
  const auto x_0_x =
      Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);

  auto y_0_prefix = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);
  y += input_.dimensions.state_dim;

  y_0_prefix -= x_0_x;

  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto A_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dx[i],
        input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto B_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_du[i],
        input_.dimensions.state_dim, input_.dimensions.control_dim);

    const auto jac_x_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dx[i], input_.dimensions.c_dim,
        input_.dimensions.state_dim);

    const auto jac_u_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_du[i], input_.dimensions.c_dim,
        input_.dimensions.control_dim);

    const auto x_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);
    x += input_.dimensions.state_dim;

    const auto u_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.control_dim);
    x += input_.dimensions.control_dim;

    const auto x_ip1 =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);

    auto y_i_suffix = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.c_dim);
    y += input_.dimensions.c_dim;

    auto y_ip1_prefix =
        Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);
    y += input_.dimensions.state_dim;

    y_i_suffix += jac_x_c_i * x_i + jac_u_c_i * u_i;
    y_ip1_prefix += A_i * x_i + B_i * u_i - x_ip1;
  }

  const auto x_N =
      Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);

  const auto jac_x_c_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dc_dx[input_.dimensions.num_stages],
      input_.dimensions.c_dim, input_.dimensions.state_dim);

  auto y_N_suffix = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.c_dim);

  y_N_suffix += jac_x_c_N * x_N;
}

void CallbackProvider::add_CTx_to_y(const double *x, double *y) {
  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto A_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dx[i],
        input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto B_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_du[i],
        input_.dimensions.state_dim, input_.dimensions.control_dim);

    const auto jac_x_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dx[i], input_.dimensions.c_dim,
        input_.dimensions.state_dim);

    const auto jac_u_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_du[i], input_.dimensions.c_dim,
        input_.dimensions.control_dim);

    const auto x_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);
    x += input_.dimensions.state_dim;

    const auto c_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.c_dim);
    x += input_.dimensions.c_dim;

    const auto x_ip1 =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);

    auto y_i_x = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);
    y += input_.dimensions.state_dim;

    auto y_i_u = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.control_dim);
    y += input_.dimensions.control_dim;

    y_i_x += -x_i + jac_x_c_i.transpose() * c_i + A_i.transpose() * x_ip1;
    y_i_u += jac_u_c_i.transpose() * c_i + B_i.transpose() * x_ip1;
  }

  const auto jac_x_c_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dc_dx[input_.dimensions.num_stages],
      input_.dimensions.c_dim, input_.dimensions.state_dim);

  const auto x_N =
      Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);
  x += input_.dimensions.state_dim;

  const auto c_N =
      Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.c_dim);

  auto y_N_x = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);

  y_N_x += -x_N + jac_x_c_N.transpose() * c_N;
}

void CallbackProvider::add_Gx_to_y(const double *x, double *y) {
  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto jac_x_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[i], input_.dimensions.g_dim,
        input_.dimensions.state_dim);

    const auto jac_u_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_du[i], input_.dimensions.g_dim,
        input_.dimensions.control_dim);

    const auto x_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);
    x += input_.dimensions.state_dim;

    const auto u_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.control_dim);
    x += input_.dimensions.control_dim;

    auto y_i = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.g_dim);
    y += input_.dimensions.g_dim;

    y_i += jac_x_g_i * x_i + jac_u_g_i * u_i;
  }

  const auto jac_x_g_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dg_dx[input_.dimensions.num_stages],
      input_.dimensions.g_dim, input_.dimensions.state_dim);

  const auto x_N =
      Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);

  auto y_N = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.g_dim);

  y_N += jac_x_g_N * x_N;
}

void CallbackProvider::add_GTx_to_y(const double *x, double *y) {
  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto jac_x_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[i], input_.dimensions.g_dim,
        input_.dimensions.state_dim);

    const auto jac_u_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_du[i], input_.dimensions.g_dim,
        input_.dimensions.control_dim);

    const auto x_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.g_dim);
    x += input_.dimensions.g_dim;

    auto y_i_x = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);
    y += input_.dimensions.state_dim;

    auto y_i_u = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.control_dim);
    y += input_.dimensions.control_dim;

    y_i_x += jac_x_g_i.transpose() * x_i;
    y_i_u += jac_u_g_i.transpose() * x_i;
  }

  const auto jac_x_g_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dg_dx[input_.dimensions.num_stages],
      input_.dimensions.g_dim, input_.dimensions.state_dim);

  const auto x_N =
      Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.g_dim);

  auto y_N = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);
  y_N += jac_x_g_N.transpose() * x_N;
}

} // namespace sip::optimal_control

#undef EIGEN_NO_MALLOC

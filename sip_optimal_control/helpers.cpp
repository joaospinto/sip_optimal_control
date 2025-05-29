#include "sip_optimal_control/helpers.hpp"

#define EIGEN_NO_MALLOC

#include <Eigen/Dense>

namespace sip::optimal_control {

CallbackProvider::CallbackProvider(const Input &input, Workspace &workspace)
    : input_(input), workspace_(workspace) {}

void CallbackProvider::factor(const double *w, const double r1, const double r2,
                              const double r3) {
  const auto jac_x_c_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dc_dx[input_.dimensions.num_stages],
      input_.dimensions.c_dim, input_.dimensions.state_dim);

  const auto jac_x_g_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dg_dx[input_.dimensions.num_stages],
      input_.dimensions.g_dim, input_.dimensions.state_dim);

  const auto Q_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.d2L_dx2[input_.dimensions.num_stages],
      input_.dimensions.state_dim, input_.dimensions.state_dim);

  auto mod_w_inv_N = Eigen::Map<Eigen::VectorXd>(
      workspace_.regularized_lqr_data.mod_w_inv[input_.dimensions.num_stages],
      input_.dimensions.g_dim);

  int w_offset = input_.dimensions.g_dim * input_.dimensions.num_stages;

  for (int j = 0; j < input_.dimensions.g_dim; ++j) {
    mod_w_inv_N(j) = 1.0 / (w[w_offset + j] + r3);
    w_offset -= input_.dimensions.g_dim;
  }

  auto V_N = Eigen::Map<Eigen::MatrixXd>(
      workspace_.regularized_lqr_data.V[input_.dimensions.num_stages],
      input_.dimensions.state_dim, input_.dimensions.state_dim);

  workspace_.regularized_lqr_data.r2 = r2;
  const double r2_inv = 1.0 / r2;
  workspace_.regularized_lqr_data.r2_inv = r2_inv;

  // V_N = Q_N + r1 I + (1/r2) C_N^T C_N + G_N^T (W + r3 I)^{-1} G_N
  V_N.noalias() = Q_N;
  V_N.noalias() += r2_inv * jac_x_c_N.transpose() * jac_x_c_N;

  for (int i = 0; i < input_.dimensions.state_dim; ++i) {
    V_N(i, i) += r1;
  }

  V_N.noalias() += jac_x_g_N.transpose() * mod_w_inv_N.asDiagonal() * jac_x_g_N;

  for (int i = input_.dimensions.num_stages - 1; i >= 0; --i) {
    const auto A_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dx[i],
        input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto B_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_du[i],
        input_.dimensions.state_dim, input_.dimensions.control_dim);

    const auto Q_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dx2[i],
        input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto M_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dxdu[i],
        input_.dimensions.state_dim, input_.dimensions.control_dim);

    const auto R_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_du2[i],
        input_.dimensions.control_dim, input_.dimensions.control_dim);

    const auto jac_x_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dx[i], input_.dimensions.c_dim,
        input_.dimensions.state_dim);

    const auto jac_u_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_du[i], input_.dimensions.c_dim,
        input_.dimensions.control_dim);

    const auto jac_x_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[i], input_.dimensions.g_dim,
        input_.dimensions.state_dim);

    const auto jac_u_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_du[i], input_.dimensions.g_dim,
        input_.dimensions.control_dim);

    auto mod_w_inv_i = Eigen::Map<Eigen::VectorXd>(
        workspace_.regularized_lqr_data.mod_w_inv[i], input_.dimensions.g_dim);

    for (int j = 0; j < input_.dimensions.g_dim; ++j) {
      mod_w_inv_i(j) = 1.0 / (w[w_offset + j] + r3);
    }
    w_offset -= input_.dimensions.g_dim;

    auto Q_i_mod = Eigen::Map<Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.Q_mod, input_.dimensions.state_dim,
        input_.dimensions.state_dim);

    auto M_i_mod = Eigen::Map<Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.M_mod, input_.dimensions.state_dim,
        input_.dimensions.control_dim);

    auto R_i_mod = Eigen::Map<Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.R_mod, input_.dimensions.control_dim,
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

    const auto V_ip1 = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.V[i + 1], input_.dimensions.state_dim,
        input_.dimensions.state_dim);

    auto W_i = Eigen::Map<Eigen::MatrixXd>(workspace_.regularized_lqr_data.W[i],
                                           input_.dimensions.state_dim,
                                           input_.dimensions.state_dim);

    auto G_i = Eigen::Map<Eigen::MatrixXd>(workspace_.regularized_lqr_data.G,
                                           input_.dimensions.control_dim,
                                           input_.dimensions.control_dim);

    auto G_i_inv = Eigen::Map<Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.G_inv[i], input_.dimensions.control_dim,
        input_.dimensions.control_dim);

    auto H_i = Eigen::Map<Eigen::MatrixXd>(workspace_.regularized_lqr_data.H,
                                           input_.dimensions.control_dim,
                                           input_.dimensions.state_dim);

    auto K_i = Eigen::Map<Eigen::MatrixXd>(workspace_.regularized_lqr_data.K[i],
                                           input_.dimensions.control_dim,
                                           input_.dimensions.state_dim);

    auto V_i = Eigen::Map<Eigen::MatrixXd>(workspace_.regularized_lqr_data.V[i],
                                           input_.dimensions.state_dim,
                                           input_.dimensions.state_dim);

    // W_i = (I + r2 V_ip1)^{-1} V_ip1 [uses V_i as scratch memory]
    W_i.setIdentity();
    W_i.noalias() += r2 * V_ip1;
    V_i.noalias() = W_i.inverse();
    W_i.noalias() = V_i * V_ip1;

    // G_i = B^T W_i B_i + R_i + r1 I
    G_i.noalias() = B_i.transpose() * W_i * B_i + R_i_mod;
    for (int j = 0; j < input_.dimensions.control_dim; ++j) {
      G_i(j, j) += r1;
    }
    G_i_inv.noalias() = G_i.inverse();

    // H_i = B^T W_i A_i + M_i^T
    H_i.noalias() = B_i.transpose() * W_i * A_i + M_i_mod.transpose();

    // K_i = -G_i^{-1} H_i
    K_i.noalias() = -G_i_inv * H_i;

    // V_i = A_i^T W_i A_i + Q_i + r1 I + K_i^T H_i
    V_i.noalias() =
        A_i.transpose() * W_i * A_i + Q_i_mod + K_i.transpose() * H_i;
    for (int j = 0; j < input_.dimensions.state_dim; ++j) {
      V_i(j, j) += r1;
    }
  }
}

void CallbackProvider::solve(const double *b, double *sol) {
  const double *b_x = b + input_.dimensions.get_x_dim();
  const double *b_y = b_x + input_.dimensions.get_y_dim();
  const double *b_z = b_y + input_.dimensions.get_z_dim();
  b_x -= input_.dimensions.state_dim;
  b_y -= input_.dimensions.control_dim;
  b_z -= input_.dimensions.g_dim;

  auto v_N = Eigen::Map<Eigen::VectorXd>(
      workspace_.regularized_lqr_data.v[input_.dimensions.num_stages],
      input_.dimensions.state_dim);

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
  b_x -= input_.dimensions.control_dim;

  const auto b_y_N_suffix =
      Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.c_dim);
  b_y -= input_.dimensions.state_dim;

  const auto b_z_N =
      Eigen::Map<const Eigen::VectorXd>(b_z, input_.dimensions.g_dim);
  b_z -= input_.dimensions.g_dim;

  const double r2 = workspace_.regularized_lqr_data.r2;
  const double r2_inv = workspace_.regularized_lqr_data.r2_inv;

  // v_N = b_x_N + (1/r2) C_N^T b_y + G_N^T (W + r3 I)^{-1} b_z
  v_N.noalias() = b_x_N + r2_inv * jac_x_c_N.transpose() * b_y_N_suffix +
                  jac_x_g_N.transpose() * mod_w_inv_N.asDiagonal() * b_z_N;

  for (int i = input_.dimensions.num_stages - 1; i >= 0; --i) {
    const auto b_u_i =
        Eigen::Map<const Eigen::VectorXd>(b_x, input_.dimensions.control_dim);
    b_x -= input_.dimensions.state_dim;

    const auto b_x_i =
        Eigen::Map<const Eigen::VectorXd>(b_x, input_.dimensions.state_dim);
    b_x -= input_.dimensions.control_dim;

    const auto b_y_ip1_prefix =
        Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.state_dim);
    b_y -= input_.dimensions.c_dim;

    const auto b_y_i_suffix =
        Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.c_dim);
    b_y -= input_.dimensions.state_dim;

    const auto b_z_i =
        Eigen::Map<const Eigen::VectorXd>(b_z, input_.dimensions.g_dim);
    b_z -= input_.dimensions.g_dim;

    const auto A_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dx[i],
        input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto B_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_du[i],
        input_.dimensions.state_dim, input_.dimensions.control_dim);

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

    const auto v_ip1 = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.v[i + 1], input_.dimensions.state_dim);

    const auto mod_w_inv_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.mod_w_inv[i], input_.dimensions.g_dim);

    auto q_i_mod = Eigen::Map<Eigen::VectorXd>(
        workspace_.regularized_lqr_data.q_mod, input_.dimensions.state_dim);

    auto r_i_mod = Eigen::Map<Eigen::VectorXd>(
        workspace_.regularized_lqr_data.r_mod, input_.dimensions.control_dim);

    auto g_i = Eigen::Map<Eigen::VectorXd>(workspace_.regularized_lqr_data.g,
                                           input_.dimensions.state_dim);

    auto h_i = Eigen::Map<Eigen::VectorXd>(workspace_.regularized_lqr_data.h,
                                           input_.dimensions.control_dim);

    auto k_i = Eigen::Map<Eigen::VectorXd>(workspace_.regularized_lqr_data.k[i],
                                           input_.dimensions.control_dim);

    auto v_i = Eigen::Map<Eigen::VectorXd>(workspace_.regularized_lqr_data.v[i],
                                           input_.dimensions.state_dim);

    const auto W_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.W[i], input_.dimensions.state_dim,
        input_.dimensions.state_dim);

    const auto G_i_inv = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.G_inv[i], input_.dimensions.control_dim,
        input_.dimensions.control_dim);

    const auto K_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.K[i], input_.dimensions.control_dim,
        input_.dimensions.state_dim);

    q_i_mod.noalias() =
        b_x_i + r2_inv * jac_x_c_i.transpose() * b_y_i_suffix +
        jac_x_g_i.transpose() * mod_w_inv_i.asDiagonal() * b_z_i;

    r_i_mod.noalias() =
        b_u_i + r2_inv * jac_u_c_i.transpose() * b_y_i_suffix +
        jac_u_g_i.transpose() * mod_w_inv_i.asDiagonal() * b_z_i;

    // g_i = v_{i+1} + W_i (c_{i+1} - r2 * v_{i+1})
    g_i.noalias() = v_ip1 + W_i * (b_y_ip1_prefix - r2 * v_ip1);

    // h_i = r_i_mod + B_i^T g_i
    h_i.noalias() = r_i_mod + B_i.transpose() * g_i;

    // k_i = -G_i^{-1} h_i
    k_i.noalias() = -G_i_inv * h_i;

    // v_i = q_i_mod + A_i^T g_i + K_i^T h_i
    v_i.noalias() = q_i_mod + A_i.transpose() * g_i + K_i.transpose() * h_i;
  }

  const int x_dim = input_.dimensions.get_x_dim();
  const int y_dim = input_.dimensions.get_y_dim();

  double *x = sol;
  double *y = x + x_dim;
  double *z = y + y_dim;

  // Recover x_0 via (I + r2 V_0) x_0 = c_0 - r2 v_0.
  const auto b_y_0_prefix =
      Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.state_dim);
  b_y += input_.dimensions.state_dim;
  const auto V_0 = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.regularized_lqr_data.V[0], input_.dimensions.state_dim,
      input_.dimensions.state_dim);
  const auto v_0 = Eigen::Map<const Eigen::VectorXd>(
      workspace_.regularized_lqr_data.v[0], input_.dimensions.state_dim);
  auto F_0 = Eigen::Map<Eigen::MatrixXd>(workspace_.regularized_lqr_data.F,
                                         input_.dimensions.state_dim,
                                         input_.dimensions.state_dim);

  auto F_0_inv = Eigen::Map<Eigen::MatrixXd>(
      workspace_.regularized_lqr_data.F_inv, input_.dimensions.state_dim,
      input_.dimensions.state_dim);
  auto f_0 = Eigen::Map<Eigen::VectorXd>(workspace_.regularized_lqr_data.f,
                                         input_.dimensions.state_dim);
  auto x_0 = Eigen::Map<Eigen::VectorXd>(x, input_.dimensions.state_dim);
  F_0.setIdentity();
  F_0.noalias() += workspace_.regularized_lqr_data.r2 * V_0;
  F_0_inv.noalias() = F_0.inverse();
  f_0.noalias() = r2 * v_0 - b_y_0_prefix;
  x_0.noalias() = -F_0_inv * f_0;

  // Recover (the first part of) y_0_prefix via y_0_prefix = V_0 x_0 + v_0.
  auto y_0_prefix = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);
  y += input_.dimensions.state_dim;
  y_0_prefix.noalias() = V_0 * x_0 + v_0;

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

    const auto jac_x_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[i], input_.dimensions.g_dim,
        input_.dimensions.state_dim);

    const auto jac_u_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_du[i], input_.dimensions.c_dim,
        input_.dimensions.control_dim);

    const auto jac_u_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_du[i], input_.dimensions.g_dim,
        input_.dimensions.control_dim);

    const auto K_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.K[i], input_.dimensions.control_dim,
        input_.dimensions.state_dim);
    const auto k_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.k[i], input_.dimensions.control_dim);

    const auto V_ip1 = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.V[i + 1], input_.dimensions.state_dim,
        input_.dimensions.state_dim);
    const auto v_ip1 = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.v[i + 1], input_.dimensions.state_dim);

    const auto mod_w_inv_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.mod_w_inv[i], input_.dimensions.g_dim);

    const auto b_y_i_suffix =
        Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.c_dim);
    b_y += input_.dimensions.c_dim;

    const auto b_y_ip1_prefix =
        Eigen::Map<const Eigen::VectorXd>(b_y, input_.dimensions.state_dim);
    b_y += input_.dimensions.state_dim;

    const auto b_z_i =
        Eigen::Map<const Eigen::VectorXd>(b_z, input_.dimensions.g_dim);

    b_z += input_.dimensions.g_dim;

    auto F_ip1 = Eigen::Map<Eigen::MatrixXd>(workspace_.regularized_lqr_data.F,
                                             input_.dimensions.state_dim,
                                             input_.dimensions.state_dim);

    auto F_ip1_inv = Eigen::Map<Eigen::MatrixXd>(
        workspace_.regularized_lqr_data.F_inv, input_.dimensions.state_dim,
        input_.dimensions.state_dim);
    auto f_ip1 = Eigen::Map<Eigen::VectorXd>(workspace_.regularized_lqr_data.f,
                                             input_.dimensions.state_dim);

    const auto x_i =
        Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);
    x += input_.dimensions.state_dim;
    auto u_i = Eigen::Map<Eigen::VectorXd>(x, input_.dimensions.control_dim);
    x += input_.dimensions.control_dim;
    auto x_ip1 = Eigen::Map<Eigen::VectorXd>(x, input_.dimensions.state_dim);
    auto y_i_suffix = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.c_dim);
    y += input_.dimensions.c_dim;
    auto y_ip1_prefix =
        Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.state_dim);
    y += input_.dimensions.state_dim;
    auto z_i = Eigen::Map<Eigen::VectorXd>(z, input_.dimensions.g_dim);
    z += input_.dimensions.g_dim;

    F_ip1.setIdentity();
    F_ip1.noalias() += workspace_.regularized_lqr_data.r2 * V_ip1;
    F_ip1_inv.noalias() = F_ip1.inverse();

    // f = δ * v[t + 1] - c[t + 1]
    f_ip1.noalias() = r2 * v_ip1 - b_y_ip1_prefix;

    u_i.noalias() = K_i * x_i + k_i;
    x_ip1.noalias() = F_ip1_inv * (A_i * x_i + B_i * u_i - f_ip1);

    // Recover y_i_suffix via y_i_suffix = (1/r2) * (C_i^x x_i + C_i^u u_i +
    // b_y_i_suffix).
    y_i_suffix.noalias() =
        r2_inv * (jac_x_c_i * x_i + jac_u_c_i * u_i + b_y_i_suffix);

    y_ip1_prefix.noalias() = V_ip1 * x_ip1 + v_ip1;

    // Recover z_i via z_i = (W_i + r2 I)^{-1} (G_i^x x_i + G_i^u u_i + b_z_i).
    z_i.noalias() =
        mod_w_inv_i.asDiagonal() * (jac_x_g_i * x_i + jac_u_g_i * u_i + b_z_i);
  }

  const auto x_N =
      Eigen::Map<const Eigen::VectorXd>(x, input_.dimensions.state_dim);

  // Recover y_N_suffix via y_N_suffix = (1/r2) * (C_N^x x_N + b_y_N_suffix).
  auto y_N_suffix = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.c_dim);
  y_N_suffix.noalias() = r2_inv * (jac_x_c_N * x_N + b_y_N_suffix);

  // Recover z_N via z_N = (W_N + r2 I)^{-1} (G_N x_N + b_z_N).
  auto z_N = Eigen::Map<Eigen::VectorXd>(z, input_.dimensions.g_dim);
  z_N.noalias() = mod_w_inv_N.asDiagonal() * (jac_x_g_N * x_N + b_z_N);
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

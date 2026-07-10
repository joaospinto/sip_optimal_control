#include "sip_optimal_control/helpers.hpp"

#define EIGEN_NO_MALLOC

#include <Eigen/Dense>
#include <algorithm>

namespace sip::optimal_control {

CallbackProvider::CallbackProvider(const Input &input, Workspace &workspace)
    : input_(input), workspace_(workspace) {}

void CallbackProvider::form_theta_jacobian() {
  const auto &dim = input_.dimensions;
  const auto &mco = workspace_.model_callback_output;
  auto &lqr_data = workspace_.regularized_lqr_data;

  const int p = dim.theta_dim;
  const int stagewise_kkt_dim = dim.get_stagewise_kkt_dim();
  auto J_theta =
      Eigen::Map<Eigen::MatrixXd>(lqr_data.theta_jacobian, stagewise_kkt_dim, p);
  J_theta.setZero();

  int row = 0;
  for (int i = 0; i < dim.num_stages; ++i) {
    J_theta.block(row, 0, dim.state_dim, p) =
        Eigen::Map<const Eigen::MatrixXd>(mco.d2L_dxdtheta[i], dim.state_dim,
                                          p);
    row += dim.state_dim;
    J_theta.block(row, 0, dim.control_dim, p) =
        Eigen::Map<const Eigen::MatrixXd>(mco.d2L_dudtheta[i], dim.control_dim,
                                          p);
    row += dim.control_dim;
  }
  J_theta.block(row, 0, dim.state_dim, p) = Eigen::Map<const Eigen::MatrixXd>(
      mco.d2L_dxdtheta[dim.num_stages], dim.state_dim, p);
  row += dim.state_dim;

  row += dim.state_dim;
  J_theta.block(row, 0, dim.c_dim, p) =
      Eigen::Map<const Eigen::MatrixXd>(mco.dc_dtheta[0], dim.c_dim, p);
  row += dim.c_dim;
  for (int i = 0; i < dim.num_stages; ++i) {
    J_theta.block(row, 0, dim.state_dim, p) =
        Eigen::Map<const Eigen::MatrixXd>(mco.ddyn_dtheta[i], dim.state_dim, p);
    row += dim.state_dim;
    J_theta.block(row, 0, dim.c_dim, p) = Eigen::Map<const Eigen::MatrixXd>(
        mco.dc_dtheta[i + 1], dim.c_dim, p);
    row += dim.c_dim;
  }

  for (int i = 0; i <= dim.num_stages; ++i) {
    J_theta.block(row, 0, dim.g_dim, p) =
        Eigen::Map<const Eigen::MatrixXd>(mco.dg_dtheta[i], dim.g_dim, p);
    row += dim.g_dim;
  }
}

bool CallbackProvider::factor(const double *w, const double r1,
                              const double *r2, const double *r3) {
  const auto &mco = workspace_.model_callback_output;
  auto &lqr_data = workspace_.regularized_lqr_data;

  int w_offset = 0;
  int y_offset = 0;

  for (int i = 0; i <= input_.dimensions.num_stages; ++i) {
    for (int j = 0; j < input_.dimensions.state_dim; ++j) {
      if (r2[y_offset] <= 0.0) {
        return false;
      }
      lqr_data.dyn_r2[i][j] = r2[y_offset++];
    }
    for (int j = 0; j < input_.dimensions.c_dim; ++j) {
      if (r2[y_offset] <= 0.0) {
        return false;
      }
      lqr_data.c_r2_inv[i][j] = 1.0 / r2[y_offset++];
    }
  }

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
    const auto c_r2_inv_i = Eigen::Map<const Eigen::VectorXd>(
        lqr_data.c_r2_inv[i], input_.dimensions.c_dim);

    for (int j = 0; j < input_.dimensions.g_dim; ++j) {
      const double w_reg = w[w_offset] + r3[w_offset];
      ++w_offset;
      if (w_reg <= 0.0) {
        return false;
      }
      mod_w_inv_i(j) = 1.0 / w_reg;
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

    Q_i_mod.noalias() = Q_i;
    Q_i_mod.diagonal().array() += r1;
    Q_i_mod += jac_x_c_i.transpose() * c_r2_inv_i.asDiagonal() * jac_x_c_i;
    Q_i_mod += jac_x_g_i.transpose() * mod_w_inv_i.asDiagonal() * jac_x_g_i;

    M_i_mod.noalias() =
        M_i + jac_x_c_i.transpose() * c_r2_inv_i.asDiagonal() * jac_u_c_i +
        jac_x_g_i.transpose() * mod_w_inv_i.asDiagonal() * jac_u_g_i;

    R_i_mod.noalias() = R_i;
    R_i_mod.diagonal().array() += r1;
    R_i_mod += jac_u_c_i.transpose() * c_r2_inv_i.asDiagonal() * jac_u_c_i;
    R_i_mod += jac_u_g_i.transpose() * mod_w_inv_i.asDiagonal() * jac_u_g_i;
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
  const auto c_r2_inv_N = Eigen::Map<const Eigen::VectorXd>(
      lqr_data.c_r2_inv[input_.dimensions.num_stages], input_.dimensions.c_dim);

  for (int j = 0; j < input_.dimensions.g_dim; ++j) {
    const double w_reg = w[w_offset] + r3[w_offset];
    ++w_offset;
    if (w_reg <= 0.0) {
      return false;
    }
    mod_w_inv_N(j) = 1.0 / w_reg;
  }

  auto Q_N_mod = Eigen::Map<Eigen::MatrixXd>(
      lqr_data.Q_mod[input_.dimensions.num_stages], input_.dimensions.state_dim,
      input_.dimensions.state_dim);

  Q_N_mod.noalias() = Q_N;
  Q_N_mod.diagonal().array() += r1;
  Q_N_mod += jac_x_c_N.transpose() * c_r2_inv_N.asDiagonal() * jac_x_c_N;
  Q_N_mod += jac_x_g_N.transpose() * mod_w_inv_N.asDiagonal() * jac_x_g_N;

  lqr_input_.Q = lqr_data.Q_mod;
  lqr_input_.M = lqr_data.M_mod;
  lqr_input_.R = lqr_data.R_mod;
  lqr_input_.A = mco.ddyn_dx;
  lqr_input_.B = mco.ddyn_du;
  lqr_input_.delta = lqr_data.dyn_r2;
  lqr_input_.dimensions = {
      .state_dim = input_.dimensions.state_dim,
      .control_dim = input_.dimensions.control_dim,
      .num_stages = input_.dimensions.num_stages,
  };
  auto lqr_solver = LQR(lqr_input_, workspace_.lqr_workspace);
  if (!lqr_solver.factor()) {
    return false;
  }

  if (input_.dimensions.theta_dim == 0) {
    return true;
  }

  form_theta_jacobian();

  auto &theta_data = workspace_.regularized_lqr_data;
  const int p = input_.dimensions.theta_dim;
  const int stagewise_kkt_dim = input_.dimensions.get_stagewise_kkt_dim();

  const auto J_theta = Eigen::Map<const Eigen::MatrixXd>(
      theta_data.theta_jacobian, stagewise_kkt_dim, p);
  auto K_inv_J_theta = Eigen::Map<Eigen::MatrixXd>(
      theta_data.theta_solution, stagewise_kkt_dim, p);

  for (int j = 0; j < p; ++j) {
    solve_stagewise_kkt(J_theta.col(j).data(), K_inv_J_theta.col(j).data());
  }

  const auto H_theta_theta = Eigen::Map<const Eigen::MatrixXd>(
      mco.d2L_dtheta2, input_.dimensions.theta_dim, input_.dimensions.theta_dim);
  auto S_theta =
      Eigen::Map<Eigen::MatrixXd>(theta_data.theta_schur, p, p);
  S_theta.noalias() = H_theta_theta;
  S_theta.diagonal().array() += r1;
  S_theta.noalias() -= J_theta.transpose() * K_inv_J_theta;

  auto S_theta_factor =
      Eigen::Map<Eigen::MatrixXd>(theta_data.theta_schur_factor, p, p);
  S_theta_factor = S_theta;
  Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(S_theta_factor);
  return llt.info() == Eigen::Success;
}

void CallbackProvider::solve_stagewise_kkt(const double *b, double *sol) {
  const int x_dim = input_.dimensions.get_stagewise_x_dim();
  const int y_dim = input_.dimensions.get_y_dim();

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
      const auto c_r2_inv_i = Eigen::Map<const Eigen::VectorXd>(
          workspace_.regularized_lqr_data.c_r2_inv[i], input_.dimensions.c_dim);

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
          -(b_x_i +
            jac_x_c_i.transpose() *
                c_r2_inv_i.cwiseProduct(b_y_i_suffix) +
            jac_x_g_i.transpose() * mod_w_inv_i.cwiseProduct(b_z_i));

      r_i_mod.noalias() =
          -(b_u_i +
            jac_u_c_i.transpose() *
                c_r2_inv_i.cwiseProduct(b_y_i_suffix) +
            jac_u_g_i.transpose() * mod_w_inv_i.cwiseProduct(b_z_i));

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
    const auto c_r2_inv_N = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.c_r2_inv[input_.dimensions.num_stages],
        input_.dimensions.c_dim);

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

    q_N_mod.noalias() = -(
        b_x_N +
        jac_x_c_N.transpose() * c_r2_inv_N.cwiseProduct(b_y_N_suffix) +
        jac_x_g_N.transpose() * mod_w_inv_N.cwiseProduct(b_z_N));

    c_N_mod = -b_y_N_prefix;
  }

  auto &lqr_data = workspace_.regularized_lqr_data;
  auto &lqr_output = workspace_.lqr_output;

  {
    double *sol_x = sol;
    double *sol_y = sol_x + x_dim;

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
    lqr_output.y[input_.dimensions.num_stages] = sol_y;
  }

  auto lqr_solver = LQR(lqr_input_, workspace_.lqr_workspace);
  lqr_solver.solve(lqr_output);

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
    const auto c_r2_inv_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.c_r2_inv[i], input_.dimensions.c_dim);

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

    y_i_suffix.noalias() = c_r2_inv_i.cwiseProduct(
        jac_x_c_i * x_i + jac_u_c_i * u_i - b_y_i_suffix);

    z_i.noalias() = mod_w_inv_i.cwiseProduct(jac_x_g_i * x_i +
                                             jac_u_g_i * u_i - b_z_i);
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
  const auto c_r2_inv_N = Eigen::Map<const Eigen::VectorXd>(
      workspace_.regularized_lqr_data.c_r2_inv[input_.dimensions.num_stages],
      input_.dimensions.c_dim);

  auto y_N_suffix = Eigen::Map<Eigen::VectorXd>(y, input_.dimensions.c_dim);
  y_N_suffix.noalias() =
      c_r2_inv_N.cwiseProduct(jac_x_c_N * x_N - b_y_N_suffix);

  auto z_N = Eigen::Map<Eigen::VectorXd>(z, input_.dimensions.g_dim);
  z_N.noalias() = mod_w_inv_N.cwiseProduct(jac_x_g_N * x_N - b_z_N);
}

void CallbackProvider::solve(const double *b, double *sol) {
  if (input_.dimensions.theta_dim == 0) {
    solve_stagewise_kkt(b, sol);
    return;
  }

  auto &theta_data = workspace_.regularized_lqr_data;
  const int p = input_.dimensions.theta_dim;
  const int stagewise_kkt_dim = input_.dimensions.get_stagewise_kkt_dim();

  const double *b_theta = b + input_.dimensions.get_stagewise_x_dim();
  const double *b_y = b + input_.dimensions.get_x_dim();
  const double *b_z = b_y + input_.dimensions.get_y_dim();

  double *sol_theta = sol + input_.dimensions.get_stagewise_x_dim();
  double *sol_y = sol + input_.dimensions.get_x_dim();
  double *sol_z = sol_y + input_.dimensions.get_y_dim();

  auto stagewise_rhs = Eigen::Map<Eigen::VectorXd>(
      theta_data.theta_stagewise_rhs, stagewise_kkt_dim);
  std::copy_n(b, input_.dimensions.get_stagewise_x_dim(),
              theta_data.theta_stagewise_rhs);
  std::copy_n(b_y, input_.dimensions.get_y_dim(),
              theta_data.theta_stagewise_rhs +
                  input_.dimensions.get_stagewise_x_dim());
  std::copy_n(b_z, input_.dimensions.get_z_dim(),
              theta_data.theta_stagewise_rhs +
                  input_.dimensions.get_stagewise_x_dim() +
                  input_.dimensions.get_y_dim());

  solve_stagewise_kkt(theta_data.theta_stagewise_rhs, sol);

  const auto J_theta = Eigen::Map<const Eigen::MatrixXd>(
      theta_data.theta_jacobian, stagewise_kkt_dim, p);
  const auto K_inv_b =
      Eigen::Map<const Eigen::VectorXd>(sol, stagewise_kkt_dim);
  auto theta_rhs =
      Eigen::Map<Eigen::VectorXd>(theta_data.theta_rhs, p);
  theta_rhs.noalias() = Eigen::Map<const Eigen::VectorXd>(b_theta, p) -
                        J_theta.transpose() * K_inv_b;

  const auto S_theta_factor =
      Eigen::Map<const Eigen::MatrixXd>(theta_data.theta_schur_factor, p, p);
  S_theta_factor.template triangularView<Eigen::Lower>().solveInPlace(
      theta_rhs);
  S_theta_factor.transpose().template triangularView<Eigen::Upper>()
      .solveInPlace(theta_rhs);
  std::copy_n(theta_data.theta_rhs, p, sol_theta);

  stagewise_rhs.noalias() -= J_theta * theta_rhs;

  solve_stagewise_kkt(theta_data.theta_stagewise_rhs, sol);

  const int stagewise_x_dim = input_.dimensions.get_stagewise_x_dim();
  std::copy_backward(sol + stagewise_x_dim + input_.dimensions.get_y_dim(),
                     sol + stagewise_kkt_dim,
                     sol_z + input_.dimensions.get_z_dim());
  std::copy_backward(sol + stagewise_x_dim,
                     sol + stagewise_x_dim + input_.dimensions.get_y_dim(),
                     sol_y + input_.dimensions.get_y_dim());
  std::copy_n(theta_data.theta_rhs, p, sol_theta);
}

void CallbackProvider::add_Kx_to_y(const double *w, const double r1,
                                   const double *r2, const double *r3,
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
    y_y[i] -= r2[i] * x_y[i];
  }
  for (int i = 0; i < z_dim; ++i) {
    y_z[i] -= (w[i] + r3[i]) * x_z[i];
  }
}

void CallbackProvider::add_Hx_to_y(const double *x, double *y) {
  const double *x_begin = x;
  double *y_begin = y;

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

  if (input_.dimensions.theta_dim == 0) {
    return;
  }

  const int p = input_.dimensions.theta_dim;
  const int stagewise_x_dim = input_.dimensions.get_stagewise_x_dim();
  const auto theta =
      Eigen::Map<const Eigen::VectorXd>(x_begin + stagewise_x_dim, p);
  auto y_theta = Eigen::Map<Eigen::VectorXd>(y_begin + stagewise_x_dim, p);

  const double *x_stage = x_begin;
  double *y_stage = y_begin;
  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto H_x_theta_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dxdtheta[i],
        input_.dimensions.state_dim, p);
    const auto H_u_theta_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dudtheta[i],
        input_.dimensions.control_dim, p);

    const auto x_i =
        Eigen::Map<const Eigen::VectorXd>(x_stage, input_.dimensions.state_dim);
    x_stage += input_.dimensions.state_dim;
    const auto u_i = Eigen::Map<const Eigen::VectorXd>(
        x_stage, input_.dimensions.control_dim);
    x_stage += input_.dimensions.control_dim;

    auto y_i_x =
        Eigen::Map<Eigen::VectorXd>(y_stage, input_.dimensions.state_dim);
    y_stage += input_.dimensions.state_dim;
    auto y_i_u =
        Eigen::Map<Eigen::VectorXd>(y_stage, input_.dimensions.control_dim);
    y_stage += input_.dimensions.control_dim;

    y_i_x.noalias() += H_x_theta_i * theta;
    y_i_u.noalias() += H_u_theta_i * theta;
    y_theta.noalias() +=
        H_x_theta_i.transpose() * x_i + H_u_theta_i.transpose() * u_i;
  }

  const auto H_x_theta_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.d2L_dxdtheta
          [input_.dimensions.num_stages],
      input_.dimensions.state_dim, p);
  const auto H_theta_theta = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.d2L_dtheta2, p, p);
  const auto x_N_theta =
      Eigen::Map<const Eigen::VectorXd>(x_stage, input_.dimensions.state_dim);
  auto y_N_theta =
      Eigen::Map<Eigen::VectorXd>(y_stage, input_.dimensions.state_dim);

  y_N_theta.noalias() += H_x_theta_N * theta;
  y_theta.noalias() += H_x_theta_N.transpose() * x_N_theta;
  y_theta.noalias() += H_theta_theta * theta;
}

void CallbackProvider::add_Cx_to_y(const double *x, double *y) {
  const double *x_begin = x;
  double *y_begin = y;

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

  if (input_.dimensions.theta_dim == 0) {
    return;
  }

  const int p = input_.dimensions.theta_dim;
  const auto theta = Eigen::Map<const Eigen::VectorXd>(
      x_begin + input_.dimensions.get_stagewise_x_dim(), p);
  double *y_stage = y_begin + input_.dimensions.state_dim;

  auto y_0_suffix =
      Eigen::Map<Eigen::VectorXd>(y_stage, input_.dimensions.c_dim);
  y_stage += input_.dimensions.c_dim;
  const auto c_theta_0 = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dc_dtheta[0], input_.dimensions.c_dim,
      p);
  y_0_suffix.noalias() += c_theta_0 * theta;

  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    auto y_ip1_prefix =
        Eigen::Map<Eigen::VectorXd>(y_stage, input_.dimensions.state_dim);
    y_stage += input_.dimensions.state_dim;
    auto y_ip1_suffix =
        Eigen::Map<Eigen::VectorXd>(y_stage, input_.dimensions.c_dim);
    y_stage += input_.dimensions.c_dim;

    const auto dyn_theta_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dtheta[i],
        input_.dimensions.state_dim, p);
    const auto c_theta_ip1 = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dtheta[i + 1],
        input_.dimensions.c_dim, p);
    y_ip1_prefix.noalias() += dyn_theta_i * theta;
    y_ip1_suffix.noalias() += c_theta_ip1 * theta;
  }
}

void CallbackProvider::add_CTx_to_y(const double *x, double *y) {
  const double *x_begin = x;
  double *y_begin = y;

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

  if (input_.dimensions.theta_dim == 0) {
    return;
  }

  const int p = input_.dimensions.theta_dim;
  auto y_theta = Eigen::Map<Eigen::VectorXd>(
      y_begin + input_.dimensions.get_stagewise_x_dim(), p);
  const double *x_stage = x_begin + input_.dimensions.state_dim;

  const auto c_0 =
      Eigen::Map<const Eigen::VectorXd>(x_stage, input_.dimensions.c_dim);
  x_stage += input_.dimensions.c_dim;
  const auto c_theta_0 = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dc_dtheta[0], input_.dimensions.c_dim,
      p);
  y_theta.noalias() += c_theta_0.transpose() * c_0;

  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto dyn_ip1 =
        Eigen::Map<const Eigen::VectorXd>(x_stage, input_.dimensions.state_dim);
    x_stage += input_.dimensions.state_dim;
    const auto c_ip1 =
        Eigen::Map<const Eigen::VectorXd>(x_stage, input_.dimensions.c_dim);
    x_stage += input_.dimensions.c_dim;

    const auto dyn_theta_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dtheta[i],
        input_.dimensions.state_dim, p);
    const auto c_theta_ip1 = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dtheta[i + 1],
        input_.dimensions.c_dim, p);
    y_theta.noalias() +=
        dyn_theta_i.transpose() * dyn_ip1 + c_theta_ip1.transpose() * c_ip1;
  }
}

void CallbackProvider::add_Gx_to_y(const double *x, double *y) {
  const double *x_begin = x;
  double *y_begin = y;

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

  if (input_.dimensions.theta_dim == 0) {
    return;
  }

  const int p = input_.dimensions.theta_dim;
  const auto theta = Eigen::Map<const Eigen::VectorXd>(
      x_begin + input_.dimensions.get_stagewise_x_dim(), p);
  double *y_stage = y_begin;
  for (int i = 0; i <= input_.dimensions.num_stages; ++i) {
    auto y_i = Eigen::Map<Eigen::VectorXd>(y_stage, input_.dimensions.g_dim);
    y_stage += input_.dimensions.g_dim;
    const auto g_theta_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dtheta[i], input_.dimensions.g_dim,
        p);
    y_i.noalias() += g_theta_i * theta;
  }
}

void CallbackProvider::add_GTx_to_y(const double *x, double *y) {
  const double *x_begin = x;
  double *y_begin = y;

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

  if (input_.dimensions.theta_dim == 0) {
    return;
  }

  const int p = input_.dimensions.theta_dim;
  auto y_theta = Eigen::Map<Eigen::VectorXd>(
      y_begin + input_.dimensions.get_stagewise_x_dim(), p);
  const double *x_stage = x_begin;
  for (int i = 0; i <= input_.dimensions.num_stages; ++i) {
    const auto x_i =
        Eigen::Map<const Eigen::VectorXd>(x_stage, input_.dimensions.g_dim);
    x_stage += input_.dimensions.g_dim;
    const auto g_theta_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dtheta[i], input_.dimensions.g_dim,
        p);
    y_theta.noalias() += g_theta_i.transpose() * x_i;
  }
}

} // namespace sip::optimal_control

#undef EIGEN_NO_MALLOC

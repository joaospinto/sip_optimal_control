#define EIGEN_NO_MALLOC

#include "sip_optimal_control/helpers.hpp"

#include <Eigen/Dense>

namespace sip::optimal_control {

CallbackProvider::CallbackProvider(const Input &input, Workspace &workspace)
    : input_(input), workspace_(workspace) {}

namespace {

using StridedMatrixMap =
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
               Eigen::Unaligned, Eigen::OuterStride<>>;
using ConstStridedMatrixMap =
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>,
               Eigen::Unaligned, Eigen::OuterStride<>>;

auto matrix_block(double *data, const int offset, const int rows,
                  const int cols, const int stride) -> StridedMatrixMap {
  return StridedMatrixMap(data + offset, rows, cols,
                          Eigen::OuterStride<>(stride));
}

auto matrix_block(const double *data, const int offset, const int rows,
                  const int cols, const int stride) -> ConstStridedMatrixMap {
  return ConstStridedMatrixMap(data + offset, rows, cols,
                               Eigen::OuterStride<>(stride));
}

void set_row_scaled(Eigen::Ref<Eigen::MatrixXd> result,
                    const Eigen::Ref<const Eigen::VectorXd> &weights,
                    const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
  for (int row = 0; row < result.rows(); ++row) {
    result.row(row) = weights(row) * matrix.row(row);
  }
}

void add_weighted_jacobian_products(
    Eigen::Ref<Eigen::MatrixXd> Q, Eigen::Ref<Eigen::MatrixXd> M,
    Eigen::Ref<Eigen::MatrixXd> R,
    const Eigen::Ref<const Eigen::MatrixXd> &J_x,
    const Eigen::Ref<const Eigen::MatrixXd> &J_u,
    const Eigen::Ref<const Eigen::VectorXd> &weights) {
  for (int constraint = 0; constraint < weights.size(); ++constraint) {
    const double weight = weights(constraint);

    for (int col = 0; col < Q.cols(); ++col) {
      const double weighted_j_x_col = weight * J_x(constraint, col);
      for (int row = col; row < Q.rows(); ++row) {
        Q(row, col) += weighted_j_x_col * J_x(constraint, row);
      }
    }

    for (int col = 0; col < M.cols(); ++col) {
      const double weighted_j_u_col = weight * J_u(constraint, col);
      for (int row = 0; row < M.rows(); ++row) {
        M(row, col) += J_x(constraint, row) * weighted_j_u_col;
      }
    }

    for (int col = 0; col < R.cols(); ++col) {
      const double weighted_j_u_col = weight * J_u(constraint, col);
      for (int row = col; row < R.rows(); ++row) {
        R(row, col) += weighted_j_u_col * J_u(constraint, row);
      }
    }
  }
}

void add_weighted_state_jacobian_product(
    Eigen::Ref<Eigen::MatrixXd> Q,
    const Eigen::Ref<const Eigen::MatrixXd> &J_x,
    const Eigen::Ref<const Eigen::VectorXd> &weights) {
  for (int constraint = 0; constraint < weights.size(); ++constraint) {
    const double weight = weights(constraint);
    for (int col = 0; col < Q.cols(); ++col) {
      const double weighted_j_x_col = weight * J_x(constraint, col);
      for (int row = col; row < Q.rows(); ++row) {
        Q(row, col) += weighted_j_x_col * J_x(constraint, row);
      }
    }
  }
}

void mirror_lower_to_upper(Eigen::Ref<Eigen::MatrixXd> matrix) {
  matrix.template triangularView<Eigen::StrictlyUpper>() =
      matrix.transpose().template triangularView<Eigen::StrictlyUpper>();
}

} // namespace

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

    Q_i_mod.template triangularView<Eigen::Lower>() =
        Q_i.template triangularView<Eigen::Lower>();
    Q_i_mod.diagonal().array() += r1;

    M_i_mod.noalias() = M_i;

    R_i_mod.template triangularView<Eigen::Lower>() =
        R_i.template triangularView<Eigen::Lower>();
    R_i_mod.diagonal().array() += r1;

    add_weighted_jacobian_products(Q_i_mod, M_i_mod, R_i_mod, jac_x_c_i,
                                   jac_u_c_i, c_r2_inv_i);
    add_weighted_jacobian_products(Q_i_mod, M_i_mod, R_i_mod, jac_x_g_i,
                                   jac_u_g_i, mod_w_inv_i);
    mirror_lower_to_upper(Q_i_mod);
    mirror_lower_to_upper(R_i_mod);
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

  Q_N_mod.template triangularView<Eigen::Lower>() =
      Q_N.template triangularView<Eigen::Lower>();
  Q_N_mod.diagonal().array() += r1;
  add_weighted_state_jacobian_product(Q_N_mod, jac_x_c_N, c_r2_inv_N);
  add_weighted_state_jacobian_product(Q_N_mod, jac_x_g_N, mod_w_inv_N);
  mirror_lower_to_upper(Q_N_mod);

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

  solve_stagewise_kkt_matrix(J_theta.data(), K_inv_J_theta.data(), p);

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
  solve_stagewise_kkt_matrix(b, sol, 1);
}

void CallbackProvider::solve_stagewise_kkt_matrix(const double *b, double *sol,
                                                  const int num_rhs) {
  const auto &dim = input_.dimensions;
  const int T = dim.num_stages;
  const int n = dim.state_dim;
  const int m = dim.control_dim;
  const int c_dim = dim.c_dim;
  const int g_dim = dim.g_dim;
  const int x_dim = input_.dimensions.get_stagewise_x_dim();
  const int y_dim = input_.dimensions.get_y_dim();
  const int z_dim = input_.dimensions.get_z_dim();
  const int stagewise_kkt_dim = input_.dimensions.get_stagewise_kkt_dim();

  const auto x_offset = [n, m](const int stage) {
    return stage * (n + m);
  };
  const auto u_offset = [n, m](const int stage) {
    return stage * (n + m) + n;
  };
  const auto y_prefix_offset = [x_dim, n, c_dim](const int stage) {
    return x_dim + stage * (n + c_dim);
  };
  const auto y_suffix_offset = [x_dim, n, c_dim](const int stage) {
    return x_dim + stage * (n + c_dim) + n;
  };
  const auto z_offset = [x_dim, y_dim, g_dim](const int stage) {
    return x_dim + y_dim + stage * g_dim;
  };

  const auto terminal_x_rhs =
      matrix_block(b, x_offset(T), n, num_rhs, stagewise_kkt_dim);
  const auto terminal_y_suffix_rhs =
      matrix_block(b, y_suffix_offset(T), c_dim, num_rhs, stagewise_kkt_dim);
  const auto terminal_z_rhs =
      matrix_block(b, z_offset(T), g_dim, num_rhs, stagewise_kkt_dim);
  const auto jac_x_c_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dc_dx[T], c_dim, n);
  const auto jac_x_g_N = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.dg_dx[T], g_dim, n);
  const auto c_r2_inv_N = Eigen::Map<const Eigen::VectorXd>(
      workspace_.regularized_lqr_data.c_r2_inv[T], c_dim);
  const auto mod_w_inv_N = Eigen::Map<const Eigen::VectorXd>(
      workspace_.regularized_lqr_data.mod_w_inv[T], g_dim);

  auto v_N = matrix_block(sol, y_prefix_offset(T), n, num_rhs,
                          stagewise_kkt_dim);
  auto terminal_y_suffix_scratch =
      matrix_block(sol, y_suffix_offset(T), c_dim, num_rhs, stagewise_kkt_dim);
  auto terminal_z_scratch =
      matrix_block(sol, z_offset(T), g_dim, num_rhs, stagewise_kkt_dim);

  v_N.noalias() = -terminal_x_rhs;
  set_row_scaled(terminal_y_suffix_scratch, c_r2_inv_N,
                 terminal_y_suffix_rhs);
  v_N.noalias() -= jac_x_c_N.transpose() * terminal_y_suffix_scratch;
  set_row_scaled(terminal_z_scratch, mod_w_inv_N, terminal_z_rhs);
  v_N.noalias() -= jac_x_g_N.transpose() * terminal_z_scratch;

  for (int i = T - 1; i >= 0; --i) {
    const auto A_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dx[i], n, n);
    const auto B_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_du[i], n, m);
    const auto jac_x_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dx[i], c_dim, n);
    const auto jac_x_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[i], g_dim, n);
    const auto jac_u_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_du[i], c_dim, m);
    const auto jac_u_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_du[i], g_dim, m);
    const auto c_r2_inv_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.c_r2_inv[i], c_dim);
    const auto mod_w_inv_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.mod_w_inv[i], g_dim);
    const auto delta_ip1 = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.dyn_r2[i + 1], n);
    const auto W_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.lqr_workspace.W[i], n, n);
    const auto K_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.lqr_workspace.K[i], m, n);
    const auto G_i_factor = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.lqr_workspace.G_factor[i], m, m);

    const auto b_x_i =
        matrix_block(b, x_offset(i), n, num_rhs, stagewise_kkt_dim);
    const auto b_u_i =
        matrix_block(b, u_offset(i), m, num_rhs, stagewise_kkt_dim);
    const auto b_y_i_suffix =
        matrix_block(b, y_suffix_offset(i), c_dim, num_rhs, stagewise_kkt_dim);
    const auto b_z_i =
        matrix_block(b, z_offset(i), g_dim, num_rhs, stagewise_kkt_dim);
    const auto b_y_ip1_prefix = matrix_block(
        b, y_prefix_offset(i + 1), n, num_rhs, stagewise_kkt_dim);

    auto v_i =
        matrix_block(sol, y_prefix_offset(i), n, num_rhs, stagewise_kkt_dim);
    auto v_ip1 = matrix_block(sol, y_prefix_offset(i + 1), n, num_rhs,
                              stagewise_kkt_dim);
    auto g_i =
        matrix_block(sol, x_offset(i), n, num_rhs, stagewise_kkt_dim);
    auto h_i =
        matrix_block(sol, u_offset(i), m, num_rhs, stagewise_kkt_dim);
    auto weighted_c_i =
        matrix_block(sol, y_suffix_offset(i), c_dim, num_rhs,
                     stagewise_kkt_dim);
    auto weighted_g_i =
        matrix_block(sol, z_offset(i), g_dim, num_rhs, stagewise_kkt_dim);

    set_row_scaled(weighted_c_i, c_r2_inv_i, b_y_i_suffix);
    set_row_scaled(weighted_g_i, mod_w_inv_i, b_z_i);

    h_i.noalias() = -b_u_i;
    h_i.noalias() -= jac_u_c_i.transpose() * weighted_c_i;
    h_i.noalias() -= jac_u_g_i.transpose() * weighted_g_i;

    set_row_scaled(v_i, delta_ip1, v_ip1);
    v_i.noalias() += b_y_ip1_prefix;
    g_i.noalias() = v_ip1 - W_i * v_i;

    h_i.noalias() += B_i.transpose() * g_i;
    v_i.noalias() = -b_x_i;
    v_i.noalias() -= jac_x_c_i.transpose() * weighted_c_i;
    v_i.noalias() -= jac_x_g_i.transpose() * weighted_g_i;
    v_i.noalias() += A_i.transpose() * g_i + K_i.transpose() * h_i;

    G_i_factor.template triangularView<Eigen::Lower>().solveInPlace(h_i);
    G_i_factor.transpose().template triangularView<Eigen::Upper>().solveInPlace(
        h_i);
    h_i *= -1.0;
  }

  auto x_0 = matrix_block(sol, x_offset(0), n, num_rhs, stagewise_kkt_dim);
  auto y_0 = matrix_block(sol, y_prefix_offset(0), n, num_rhs,
                          stagewise_kkt_dim);
  const auto b_y_0_prefix =
      matrix_block(b, y_prefix_offset(0), n, num_rhs, stagewise_kkt_dim);
  const auto delta_0 = Eigen::Map<const Eigen::VectorXd>(
      workspace_.regularized_lqr_data.dyn_r2[0], n);
  const auto F_0_factor = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.lqr_workspace.F_factor[0], n, n);
  const auto sqrt_delta_0 = Eigen::Map<const Eigen::VectorXd>(
      workspace_.lqr_workspace.sqrt_delta[0], n);
  const auto sqrt_delta_0_inv = Eigen::Map<const Eigen::VectorXd>(
      workspace_.lqr_workspace.sqrt_delta_inv[0], n);
  const auto V_0 =
      Eigen::Map<const Eigen::MatrixXd>(workspace_.lqr_workspace.V[0], n, n);

  set_row_scaled(x_0, delta_0, y_0);
  x_0.noalias() += b_y_0_prefix;
  set_row_scaled(x_0, sqrt_delta_0_inv, x_0);
  F_0_factor.template triangularView<Eigen::Lower>().solveInPlace(x_0);
  F_0_factor.transpose().template triangularView<Eigen::Upper>().solveInPlace(
      x_0);
  set_row_scaled(x_0, sqrt_delta_0, x_0);
  x_0 *= -1.0;

  y_0.noalias() += V_0 * x_0;

  for (int i = 0; i < T; ++i) {
    const auto A_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dx[i], n, n);
    const auto B_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_du[i], n, m);
    const auto K_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.lqr_workspace.K[i], m, n);
    const auto V_ip1 =
        Eigen::Map<const Eigen::MatrixXd>(workspace_.lqr_workspace.V[i + 1], n,
                                          n);
    const auto F_ip1_factor = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.lqr_workspace.F_factor[i + 1], n, n);
    const auto sqrt_delta_ip1 = Eigen::Map<const Eigen::VectorXd>(
        workspace_.lqr_workspace.sqrt_delta[i + 1], n);
    const auto sqrt_delta_ip1_inv = Eigen::Map<const Eigen::VectorXd>(
        workspace_.lqr_workspace.sqrt_delta_inv[i + 1], n);
    const auto delta_ip1 = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.dyn_r2[i + 1], n);

    const auto x_i =
        matrix_block(sol, x_offset(i), n, num_rhs, stagewise_kkt_dim);
    auto u_i =
        matrix_block(sol, u_offset(i), m, num_rhs, stagewise_kkt_dim);
    auto x_ip1 =
        matrix_block(sol, x_offset(i + 1), n, num_rhs, stagewise_kkt_dim);
    auto y_ip1 = matrix_block(sol, y_prefix_offset(i + 1), n, num_rhs,
                              stagewise_kkt_dim);
    const auto b_y_ip1_prefix = matrix_block(
        b, y_prefix_offset(i + 1), n, num_rhs, stagewise_kkt_dim);

    u_i.noalias() += K_i * x_i;

    x_ip1.noalias() = -b_y_ip1_prefix;
    x_ip1.noalias() += A_i * x_i + B_i * u_i;
    for (int row = 0; row < n; ++row) {
      x_ip1.row(row).noalias() -= delta_ip1(row) * y_ip1.row(row);
      x_ip1.row(row) *= sqrt_delta_ip1_inv(row);
    }
    F_ip1_factor.template triangularView<Eigen::Lower>().solveInPlace(x_ip1);
    F_ip1_factor.transpose()
        .template triangularView<Eigen::Upper>()
        .solveInPlace(x_ip1);
    set_row_scaled(x_ip1, sqrt_delta_ip1, x_ip1);

    y_ip1.noalias() += V_ip1 * x_ip1;
  }

  for (int i = 0; i < T; ++i) {
    const auto jac_x_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dx[i], c_dim, n);
    const auto jac_x_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[i], g_dim, n);
    const auto jac_u_c_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_du[i], c_dim, m);
    const auto jac_u_g_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_du[i], g_dim, m);
    const auto c_r2_inv_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.c_r2_inv[i], c_dim);
    const auto mod_w_inv_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.regularized_lqr_data.mod_w_inv[i], g_dim);
    const auto x_i =
        matrix_block(sol, x_offset(i), n, num_rhs, stagewise_kkt_dim);
    const auto u_i =
        matrix_block(sol, u_offset(i), m, num_rhs, stagewise_kkt_dim);
    const auto b_y_i_suffix =
        matrix_block(b, y_suffix_offset(i), c_dim, num_rhs, stagewise_kkt_dim);
    const auto b_z_i =
        matrix_block(b, z_offset(i), g_dim, num_rhs, stagewise_kkt_dim);
    auto y_i_suffix =
        matrix_block(sol, y_suffix_offset(i), c_dim, num_rhs,
                     stagewise_kkt_dim);
    auto z_i =
        matrix_block(sol, z_offset(i), g_dim, num_rhs, stagewise_kkt_dim);

    y_i_suffix.noalias() = jac_x_c_i * x_i + jac_u_c_i * u_i - b_y_i_suffix;
    set_row_scaled(y_i_suffix, c_r2_inv_i, y_i_suffix);
    z_i.noalias() = jac_x_g_i * x_i + jac_u_g_i * u_i - b_z_i;
    set_row_scaled(z_i, mod_w_inv_i, z_i);
  }

  const auto x_N = matrix_block(sol, x_offset(T), n, num_rhs,
                                stagewise_kkt_dim);
  const auto b_y_N_suffix =
      matrix_block(b, y_suffix_offset(T), c_dim, num_rhs, stagewise_kkt_dim);
  const auto b_z_N =
      matrix_block(b, z_offset(T), g_dim, num_rhs, stagewise_kkt_dim);
  auto y_N_suffix =
      matrix_block(sol, y_suffix_offset(T), c_dim, num_rhs,
                   stagewise_kkt_dim);
  auto z_N =
      matrix_block(sol, z_offset(T), g_dim, num_rhs, stagewise_kkt_dim);

  y_N_suffix.noalias() = jac_x_c_N * x_N - b_y_N_suffix;
  set_row_scaled(y_N_suffix, c_r2_inv_N, y_N_suffix);
  z_N.noalias() = jac_x_g_N * x_N - b_z_N;
  set_row_scaled(z_N, mod_w_inv_N, z_N);

  (void)z_dim;
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

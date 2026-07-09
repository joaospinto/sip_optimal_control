#include "sip_optimal_control/lqr.hpp"

#define EIGEN_NO_MALLOC

#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace sip::optimal_control {

void LQR::Output::reserve(int num_stages) {
  x = new double *[num_stages + 1];
  u = new double *[num_stages];
  y = new double *[num_stages + 1];
}

void LQR::Output::free() {
  delete[] x;
  delete[] u;
  delete[] y;
}

auto LQR::Output::mem_assign(int num_stages, unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  x = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  u = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  y = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  assert(cum_size == LQR::Output::num_bytes(num_stages));

  return cum_size;
}

void LQR::Workspace::reserve(int state_dim, int control_dim, int num_stages) {
  W = new double *[num_stages];
  K = new double *[num_stages];
  V = new double *[num_stages + 1];
  G_factor = new double *[num_stages];
  F_factor = new double *[num_stages + 1];
  sqrt_delta = new double *[num_stages + 1];
  sqrt_delta_inv = new double *[num_stages + 1];
  k = new double *[num_stages];
  v = new double *[num_stages + 1];

  for (int i = 0; i < num_stages; ++i) {
    W[i] = new double[state_dim * state_dim];
    K[i] = new double[control_dim * state_dim];
    V[i] = new double[state_dim * state_dim];
    G_factor[i] = new double[control_dim * control_dim];
    F_factor[i] = new double[state_dim * state_dim];
    sqrt_delta[i] = new double[state_dim];
    sqrt_delta_inv[i] = new double[state_dim];
    k[i] = new double[control_dim];
    v[i] = new double[state_dim];
  }

  V[num_stages] = new double[state_dim * state_dim];
  v[num_stages] = new double[state_dim];
  F_factor[num_stages] = new double[state_dim * state_dim];
  sqrt_delta[num_stages] = new double[state_dim];
  sqrt_delta_inv[num_stages] = new double[state_dim];

  G = new double[control_dim * control_dim];
  g = new double[state_dim];
  H = new double[control_dim * state_dim];
  h = new double[control_dim];
  F = new double[state_dim * state_dim];
  f = new double[state_dim];
}

void LQR::Workspace::free(int num_stages) {
  delete[] G;
  delete[] g;
  delete[] H;
  delete[] h;
  delete[] F;
  delete[] f;

  for (int i = 0; i < num_stages; ++i) {
    delete[] W[i];
    delete[] K[i];
    delete[] V[i];
    delete[] G_factor[i];
    delete[] F_factor[i];
    delete[] sqrt_delta[i];
    delete[] sqrt_delta_inv[i];
    delete[] k[i];
    delete[] v[i];
  }

  delete[] V[num_stages];
  delete[] v[num_stages];
  delete[] F_factor[num_stages];
  delete[] sqrt_delta[num_stages];
  delete[] sqrt_delta_inv[num_stages];

  delete[] W;
  delete[] K;
  delete[] V;
  delete[] G_factor;
  delete[] F_factor;
  delete[] sqrt_delta;
  delete[] sqrt_delta_inv;
  delete[] k;
  delete[] v;
}

auto LQR::Workspace::mem_assign(int state_dim, int control_dim, int num_stages,
                                unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  W = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  K = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  V = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  G_factor = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  F_factor = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  sqrt_delta = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  sqrt_delta_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  k = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  v = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  for (int i = 0; i < num_stages; ++i) {
    W[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * state_dim * sizeof(double);

    K[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * state_dim * sizeof(double);

    V[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * state_dim * sizeof(double);

    G_factor[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * control_dim * sizeof(double);

    F_factor[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * state_dim * sizeof(double);

    sqrt_delta[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double);

    sqrt_delta_inv[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double);

    k[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * sizeof(double);

    v[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double);
  }

  V[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * state_dim * sizeof(double);

  v[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  F_factor[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * state_dim * sizeof(double);

  sqrt_delta[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  sqrt_delta_inv[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  G = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += control_dim * control_dim * sizeof(double);

  g = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  H = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += control_dim * state_dim * sizeof(double);

  h = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += control_dim * sizeof(double);

  F = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * state_dim * sizeof(double);

  f = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  assert(cum_size ==
         LQR::Workspace::num_bytes(state_dim, control_dim, num_stages));

  return cum_size;
}

LQR::LQR(const LQR::Input &input, LQR::Workspace &workspace)
    : input_(input), workspace_(workspace) {}

namespace {

bool compute_delta_sqrt(const double *delta_data, double *sqrt_delta_data,
                        double *sqrt_delta_inv_data, const int n) {
  for (int i = 0; i < n; ++i) {
    if (delta_data[i] <= 0.0) {
      return false;
    }
    sqrt_delta_data[i] = std::sqrt(delta_data[i]);
    sqrt_delta_inv_data[i] = 1.0 / sqrt_delta_data[i];
  }
  return true;
}

auto factor_F(const double *delta_data, const Eigen::Ref<const Eigen::MatrixXd> &V,
              double *F_factor_data, double *sqrt_delta_data,
              double *sqrt_delta_inv_data, const int n) -> LQR::FactorStatus {
  if (!compute_delta_sqrt(delta_data, sqrt_delta_data, sqrt_delta_inv_data,
                          n)) {
    return LQR::FactorStatus::INVALID_DELTA;
  }

  auto F_factor = Eigen::Map<Eigen::MatrixXd>(F_factor_data, n, n);
  for (int col = 0; col < n; ++col) {
    for (int row = 0; row < n; ++row) {
      F_factor(row, col) =
          sqrt_delta_data[row] * V(row, col) * sqrt_delta_data[col];
    }
    F_factor(col, col) += 1.0;
  }

  Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(F_factor);
  return llt.info() == Eigen::Success
             ? LQR::FactorStatus::SUCCESS
             : LQR::FactorStatus::F_FACTORIZATION_FAILURE;
}

void F_inv_mult_matrix(
    const double *F_factor_data, const Eigen::Ref<const Eigen::MatrixXd> &rhs,
    Eigen::Ref<Eigen::MatrixXd> result, const double *sqrt_delta_data,
    const double *sqrt_delta_inv_data, const int n) {
  const auto F_factor =
      Eigen::Map<const Eigen::MatrixXd>(F_factor_data, n, n);

  for (int col = 0; col < rhs.cols(); ++col) {
    for (int row = 0; row < n; ++row) {
      result(row, col) = sqrt_delta_inv_data[row] * rhs(row, col);
    }
  }

  F_factor.template triangularView<Eigen::Lower>().solveInPlace(result);
  F_factor.transpose()
      .template triangularView<Eigen::Upper>()
      .solveInPlace(result);

  for (int col = 0; col < result.cols(); ++col) {
    for (int row = 0; row < n; ++row) {
      result(row, col) *= sqrt_delta_data[row];
    }
  }
}

void F_inv_mult_vector(
    const double *F_factor_data, const Eigen::Ref<const Eigen::VectorXd> &rhs,
    Eigen::Ref<Eigen::VectorXd> result, const double *sqrt_delta_data,
    const double *sqrt_delta_inv_data, const int n) {
  const auto F_factor =
      Eigen::Map<const Eigen::MatrixXd>(F_factor_data, n, n);

  for (int row = 0; row < n; ++row) {
    result(row) = sqrt_delta_inv_data[row] * rhs(row);
  }

  F_factor.template triangularView<Eigen::Lower>().solveInPlace(result);
  F_factor.transpose()
      .template triangularView<Eigen::Upper>()
      .solveInPlace(result);

  for (int row = 0; row < n; ++row) {
    result(row) *= sqrt_delta_data[row];
  }
}

} // namespace

auto LQR::factor_with_status() -> FactorStatus {
  const auto Q_N = Eigen::Map<const Eigen::MatrixXd>(
      input_.Q[input_.dimensions.num_stages], input_.dimensions.state_dim,
      input_.dimensions.state_dim);

  auto V_N = Eigen::Map<Eigen::MatrixXd>(
      workspace_.V[input_.dimensions.num_stages], input_.dimensions.state_dim,
      input_.dimensions.state_dim);

  V_N.noalias() = Q_N;

  {
    const auto factor_status = factor_F(
        input_.delta[input_.dimensions.num_stages], V_N,
        workspace_.F_factor[input_.dimensions.num_stages],
        workspace_.sqrt_delta[input_.dimensions.num_stages],
        workspace_.sqrt_delta_inv[input_.dimensions.num_stages],
        input_.dimensions.state_dim);
    if (factor_status != FactorStatus::SUCCESS) {
      return factor_status;
    }
  }

  for (int i = input_.dimensions.num_stages - 1; i >= 0; --i) {
    const auto A_i = Eigen::Map<const Eigen::MatrixXd>(
        input_.A[i], input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto B_i = Eigen::Map<const Eigen::MatrixXd>(
        input_.B[i], input_.dimensions.state_dim,
        input_.dimensions.control_dim);

    const auto Q_i = Eigen::Map<const Eigen::MatrixXd>(
        input_.Q[i], input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto M_i = Eigen::Map<const Eigen::MatrixXd>(
        input_.M[i], input_.dimensions.state_dim,
        input_.dimensions.control_dim);

    const auto R_i = Eigen::Map<const Eigen::MatrixXd>(
        input_.R[i], input_.dimensions.control_dim,
        input_.dimensions.control_dim);

    const auto V_ip1 = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.V[i + 1], input_.dimensions.state_dim,
        input_.dimensions.state_dim);

    auto W_i = Eigen::Map<Eigen::MatrixXd>(workspace_.W[i],
                                           input_.dimensions.state_dim,
                                           input_.dimensions.state_dim);

    auto G_i_factor = Eigen::Map<Eigen::MatrixXd>(
        workspace_.G_factor[i], input_.dimensions.control_dim,
        input_.dimensions.control_dim);

    auto H_i =
        Eigen::Map<Eigen::MatrixXd>(workspace_.H, input_.dimensions.control_dim,
                                    input_.dimensions.state_dim);

    auto K_i = Eigen::Map<Eigen::MatrixXd>(workspace_.K[i],
                                           input_.dimensions.control_dim,
                                           input_.dimensions.state_dim);

    auto V_i = Eigen::Map<Eigen::MatrixXd>(workspace_.V[i],
                                           input_.dimensions.state_dim,
                                           input_.dimensions.state_dim);

    auto F_i = Eigen::Map<Eigen::MatrixXd>(
        workspace_.F, input_.dimensions.state_dim, input_.dimensions.state_dim);

    F_inv_mult_matrix(workspace_.F_factor[i + 1], V_ip1, W_i,
                      workspace_.sqrt_delta[i + 1],
                      workspace_.sqrt_delta_inv[i + 1],
                      input_.dimensions.state_dim);

    // NOTE: We use H_i as scratch memory for computing G_i.
    H_i.noalias() = B_i.transpose() * W_i;
    G_i_factor.noalias() = R_i + H_i * B_i;

    {
      Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(G_i_factor);
      if (llt.info() != Eigen::Success) {
        return FactorStatus::G_FACTORIZATION_FAILURE;
      }
    }

    // NOTE: We use F_i as scratch memory for computing H_i and V_i.
    F_i.noalias() = W_i * A_i;

    // NOTE: We use F_i as scratch memory for computing H_i.
    H_i.noalias() = M_i.transpose() + B_i.transpose() * F_i;

    K_i.noalias() = H_i;
    G_i_factor.template triangularView<Eigen::Lower>().solveInPlace(K_i);
    G_i_factor.transpose()
        .template triangularView<Eigen::Upper>()
        .solveInPlace(K_i);
    K_i *= -1.0;

    // NOTE: We use F_i as scratch memory for computing V_i.
    V_i.noalias() = A_i.transpose() * F_i;
    F_i.noalias() = Q_i + K_i.transpose() * H_i;
    V_i += F_i;

    const auto factor_status =
        factor_F(input_.delta[i], V_i, workspace_.F_factor[i],
                 workspace_.sqrt_delta[i], workspace_.sqrt_delta_inv[i],
                 input_.dimensions.state_dim);
    if (factor_status != FactorStatus::SUCCESS) {
      return factor_status;
    }
  }

  return FactorStatus::SUCCESS;
}

bool LQR::factor() {
  return factor_with_status() == FactorStatus::SUCCESS;
}

void LQR::solve(Output &output) {
  const auto q_N = Eigen::Map<const Eigen::VectorXd>(
      input_.q[input_.dimensions.num_stages], input_.dimensions.state_dim);

  auto v_N = Eigen::Map<Eigen::VectorXd>(
      workspace_.v[input_.dimensions.num_stages], input_.dimensions.state_dim);

  v_N.noalias() = q_N;

  for (int i = input_.dimensions.num_stages - 1; i >= 0; --i) {
    const auto A_i = Eigen::Map<const Eigen::MatrixXd>(
        input_.A[i], input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto B_i = Eigen::Map<const Eigen::MatrixXd>(
        input_.B[i], input_.dimensions.state_dim,
        input_.dimensions.control_dim);

    const auto q_i = Eigen::Map<const Eigen::VectorXd>(
        input_.q[i], input_.dimensions.state_dim);

    const auto r_i = Eigen::Map<const Eigen::VectorXd>(
        input_.r[i], input_.dimensions.control_dim);

    const auto c_ip1 = Eigen::Map<const Eigen::VectorXd>(
        input_.c[i + 1], input_.dimensions.state_dim);

    const auto v_ip1 = Eigen::Map<const Eigen::VectorXd>(
        workspace_.v[i + 1], input_.dimensions.state_dim);

    const auto W_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.W[i], input_.dimensions.state_dim,
        input_.dimensions.state_dim);

    const auto G_i_factor = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.G_factor[i], input_.dimensions.control_dim,
        input_.dimensions.control_dim);

    const auto K_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.K[i], input_.dimensions.control_dim,
        input_.dimensions.state_dim);

    auto g_i =
        Eigen::Map<Eigen::VectorXd>(workspace_.g, input_.dimensions.state_dim);

    auto h_i = Eigen::Map<Eigen::VectorXd>(workspace_.h,
                                           input_.dimensions.control_dim);

    auto k_i = Eigen::Map<Eigen::VectorXd>(workspace_.k[i],
                                           input_.dimensions.control_dim);

    auto v_i = Eigen::Map<Eigen::VectorXd>(workspace_.v[i],
                                           input_.dimensions.state_dim);

    auto f_i =
        Eigen::Map<Eigen::VectorXd>(workspace_.f, input_.dimensions.state_dim);

    const auto delta_ip1 = Eigen::Map<const Eigen::VectorXd>(
        input_.delta[i + 1], input_.dimensions.state_dim);

    f_i.noalias() = delta_ip1.cwiseProduct(v_ip1) - c_ip1;
    g_i.noalias() = v_ip1 - W_i * f_i;

    h_i.noalias() = r_i + B_i.transpose() * g_i;
    k_i.noalias() = h_i;
    G_i_factor.template triangularView<Eigen::Lower>().solveInPlace(k_i);
    G_i_factor.transpose()
        .template triangularView<Eigen::Upper>()
        .solveInPlace(k_i);
    k_i *= -1.0;

    v_i.noalias() = q_i + A_i.transpose() * g_i + K_i.transpose() * h_i;
  }

  const auto c_0 = Eigen::Map<const Eigen::VectorXd>(
      input_.c[0], input_.dimensions.state_dim);

  const auto V_0 = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.V[0], input_.dimensions.state_dim,
      input_.dimensions.state_dim);

  const auto v_0 = Eigen::Map<const Eigen::VectorXd>(
      workspace_.v[0], input_.dimensions.state_dim);

  auto f_0 =
      Eigen::Map<Eigen::VectorXd>(workspace_.f, input_.dimensions.state_dim);

  auto x_0 =
      Eigen::Map<Eigen::VectorXd>(output.x[0], input_.dimensions.state_dim);
  const auto delta_0 = Eigen::Map<const Eigen::VectorXd>(
      input_.delta[0], input_.dimensions.state_dim);

  f_0.noalias() = delta_0.cwiseProduct(v_0) - c_0;
  F_inv_mult_vector(workspace_.F_factor[0], f_0, x_0, workspace_.sqrt_delta[0],
                    workspace_.sqrt_delta_inv[0],
                    input_.dimensions.state_dim);
  x_0 *= -1.0;

  auto y_0 =
      Eigen::Map<Eigen::VectorXd>(output.y[0], input_.dimensions.state_dim);
  y_0.noalias() = v_0 + V_0 * x_0;

  for (int i = 0; i < input_.dimensions.num_stages; ++i) {
    const auto A_i = Eigen::Map<const Eigen::MatrixXd>(
        input_.A[i], input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto B_i = Eigen::Map<const Eigen::MatrixXd>(
        input_.B[i], input_.dimensions.state_dim,
        input_.dimensions.control_dim);

    const auto K_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.K[i], input_.dimensions.control_dim,
        input_.dimensions.state_dim);
    const auto k_i = Eigen::Map<const Eigen::VectorXd>(
        workspace_.k[i], input_.dimensions.control_dim);

    const auto V_ip1 = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.V[i + 1], input_.dimensions.state_dim,
        input_.dimensions.state_dim);
    const auto v_ip1 = Eigen::Map<const Eigen::VectorXd>(
        workspace_.v[i + 1], input_.dimensions.state_dim);

    const auto c_ip1 = Eigen::Map<const Eigen::VectorXd>(
        input_.c[i + 1], input_.dimensions.state_dim);

    const auto x_i = Eigen::Map<const Eigen::VectorXd>(
        output.x[i], input_.dimensions.state_dim);
    auto u_i =
        Eigen::Map<Eigen::VectorXd>(output.u[i], input_.dimensions.control_dim);
    auto x_ip1 = Eigen::Map<Eigen::VectorXd>(output.x[i + 1],
                                             input_.dimensions.state_dim);
    auto y_ip1_prefix = Eigen::Map<Eigen::VectorXd>(
        output.y[i + 1], input_.dimensions.state_dim);

    auto f_ip1 =
        Eigen::Map<Eigen::VectorXd>(workspace_.f, input_.dimensions.state_dim);

    u_i.noalias() = k_i + K_i * x_i;

    // NOTE: We use f_ip1 as scratch memory for computing x_ip1.
    const auto delta_ip1 = Eigen::Map<const Eigen::VectorXd>(
        input_.delta[i + 1], input_.dimensions.state_dim);

    f_ip1.noalias() =
        c_ip1 - delta_ip1.cwiseProduct(v_ip1) + A_i * x_i + B_i * u_i;
    F_inv_mult_vector(workspace_.F_factor[i + 1], f_ip1, x_ip1,
                      workspace_.sqrt_delta[i + 1],
                      workspace_.sqrt_delta_inv[i + 1],
                      input_.dimensions.state_dim);

    y_ip1_prefix.noalias() = v_ip1 + V_ip1 * x_ip1;
  }
}

} // namespace sip::optimal_control

#undef EIGEN_NO_MALLOC

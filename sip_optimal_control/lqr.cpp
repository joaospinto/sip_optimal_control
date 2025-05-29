#include "sip_optimal_control/lqr.hpp"

#define EIGEN_NO_MALLOC

#include <Eigen/Dense>
#include <cassert>

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
  G_inv = new double *[num_stages];
  k = new double *[num_stages];
  v = new double *[num_stages + 1];

  for (int i = 0; i < num_stages; ++i) {
    W[i] = new double[state_dim * state_dim];
    K[i] = new double[control_dim * state_dim];
    V[i] = new double[state_dim * state_dim];
    G_inv[i] = new double[control_dim * control_dim];
    k[i] = new double[control_dim];
    v[i] = new double[state_dim];
  }

  V[num_stages] = new double[state_dim * state_dim];
  v[num_stages] = new double[state_dim];

  G = new double[control_dim * control_dim];
  g = new double[control_dim];
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
    delete[] G_inv[i];
    delete[] k[i];
    delete[] v[i];
  }

  delete[] V[num_stages];
  delete[] v[num_stages];

  delete[] W;
  delete[] K;
  delete[] V;
  delete[] G_inv;
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

  G_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

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

    G_inv[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * control_dim * sizeof(double);

    k[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * sizeof(double);

    v[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double);
  }

  V[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * state_dim * sizeof(double);

  v[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  G = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += control_dim * control_dim * sizeof(double);

  g = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += control_dim * sizeof(double);

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

void LQR::factor(const double δ) {
  const auto Q_N = Eigen::Map<const Eigen::MatrixXd>(
      input_.Q[input_.dimensions.num_stages], input_.dimensions.state_dim,
      input_.dimensions.state_dim);

  auto V_N = Eigen::Map<Eigen::MatrixXd>(
      workspace_.V[input_.dimensions.num_stages], input_.dimensions.state_dim,
      input_.dimensions.state_dim);

  V_N.noalias() = Q_N;

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

    auto G_i =
        Eigen::Map<Eigen::MatrixXd>(workspace_.G, input_.dimensions.control_dim,
                                    input_.dimensions.control_dim);

    auto G_i_inv = Eigen::Map<Eigen::MatrixXd>(workspace_.G_inv[i],
                                               input_.dimensions.control_dim,
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

    auto scratch_F = Eigen::Map<Eigen::MatrixXd>(
        workspace_.F, input_.dimensions.state_dim, input_.dimensions.state_dim);

    // W_i = (I + δ V_ip1)^{-1} V_ip1
    // NOTE: We use V_i as scratch memory for computing W_i.
    V_i.setIdentity();
    V_i += δ * V_ip1;
    {
      Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(V_i);
      W_i.noalias() = llt.solve(V_ip1);
    }

    // NOTE: We use H_i as scratch memory for computing G_i.
    H_i.noalias() = B_i.transpose() * W_i;
    G_i.noalias() = H_i * B_i;
    G_i += R_i;

    G_i_inv.setIdentity();
    {
      Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(G_i);
      llt.solveInPlace(G_i_inv);
    }

    // NOTE: We use K_i as scratch memory for computing H_i.
    K_i.noalias() = B_i.transpose() * W_i;
    H_i.noalias() = K_i * A_i;
    H_i += M_i.transpose();

    K_i.noalias() = -G_i_inv * H_i;

    // NOTE: We use scratch_F as scratch memory for computing V_i.
    scratch_F.noalias() = A_i.transpose() * W_i;
    V_i.noalias() = scratch_F * A_i;
    scratch_F.noalias() = K_i.transpose() * H_i;
    V_i += scratch_F;
    V_i += Q_i;
  }
}

void LQR::solve(const double δ, Output &output) {
  const auto q_N = Eigen::Map<const Eigen::VectorXd>(
      input_.q[input_.dimensions.num_stages], input_.dimensions.state_dim);

  auto v_N = Eigen::Map<Eigen::VectorXd>(
      workspace_.v[input_.dimensions.num_stages], input_.dimensions.state_dim);

  // v_N = q_N
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

    auto g_i =
        Eigen::Map<Eigen::VectorXd>(workspace_.g, input_.dimensions.state_dim);

    auto h_i = Eigen::Map<Eigen::VectorXd>(workspace_.h,
                                           input_.dimensions.control_dim);

    auto k_i = Eigen::Map<Eigen::VectorXd>(workspace_.k[i],
                                           input_.dimensions.control_dim);

    auto v_i = Eigen::Map<Eigen::VectorXd>(workspace_.v[i],
                                           input_.dimensions.state_dim);

    const auto W_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.W[i], input_.dimensions.state_dim,
        input_.dimensions.state_dim);

    const auto G_i_inv = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.G_inv[i], input_.dimensions.control_dim,
        input_.dimensions.control_dim);

    const auto K_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.K[i], input_.dimensions.control_dim,
        input_.dimensions.state_dim);

    auto scratch_f =
        Eigen::Map<Eigen::VectorXd>(workspace_.f, input_.dimensions.state_dim);

    // NOTE: We use scratch_f as scratch memory for computing g_i.
    scratch_f.noalias() = c_ip1 - δ * v_ip1;
    g_i.noalias() = W_i * scratch_f;
    g_i += v_ip1;

    h_i.noalias() = r_i + B_i.transpose() * g_i;
    k_i.noalias() = -G_i_inv * h_i;

    // NOTE: We use scratch_f as scratch memory for computing v_i.
    v_i.noalias() = A_i.transpose() * g_i;
    scratch_f.noalias() = K_i.transpose() * h_i;
    v_i += scratch_f;
    v_i += q_i;
  }

  const auto c_0 = Eigen::Map<const Eigen::VectorXd>(
      input_.c[0], input_.dimensions.state_dim);
  const auto V_0 = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.V[0], input_.dimensions.state_dim,
      input_.dimensions.state_dim);
  const auto v_0 = Eigen::Map<const Eigen::VectorXd>(
      workspace_.v[0], input_.dimensions.state_dim);
  auto F_0 = Eigen::Map<Eigen::MatrixXd>(
      workspace_.F, input_.dimensions.state_dim, input_.dimensions.state_dim);

  auto x_0 =
      Eigen::Map<Eigen::VectorXd>(output.x[0], input_.dimensions.state_dim);
  F_0.setIdentity();
  F_0 += δ * V_0;
  x_0.noalias() = c_0 - δ * v_0;
  {
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(F_0);
    llt.solveInPlace(x_0);
  }

  auto y_0 =
      Eigen::Map<Eigen::VectorXd>(output.y[0], input_.dimensions.state_dim);
  y_0.noalias() = V_0 * x_0;
  y_0 += v_0;

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

    auto F_ip1 = Eigen::Map<Eigen::MatrixXd>(
        workspace_.F, input_.dimensions.state_dim, input_.dimensions.state_dim);

    const auto x_i = Eigen::Map<const Eigen::VectorXd>(
        output.x[i], input_.dimensions.state_dim);
    auto u_i =
        Eigen::Map<Eigen::VectorXd>(output.u[i], input_.dimensions.control_dim);
    auto x_ip1 = Eigen::Map<Eigen::VectorXd>(output.x[i + 1],
                                             input_.dimensions.state_dim);
    auto y_ip1_prefix = Eigen::Map<Eigen::VectorXd>(
        output.y[i + 1], input_.dimensions.state_dim);

    auto f_i =
        Eigen::Map<Eigen::VectorXd>(workspace_.f, input_.dimensions.state_dim);

    u_i.noalias() = K_i * x_i;
    u_i += k_i;

    F_ip1.setIdentity();
    F_ip1 += δ * V_ip1;
    x_ip1.noalias() = A_i * x_i;
    // NOTE: We use f_i as scratch memory for computing x_ip1.
    f_i.noalias() = B_i * u_i;
    x_ip1 += f_i;
    f_i.noalias() = δ * v_ip1 - c_ip1;
    x_ip1 -= f_i;
    {
      Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(F_ip1);
      llt.solveInPlace(x_ip1);
    }

    y_ip1_prefix.noalias() = V_ip1 * x_ip1;
    y_ip1_prefix += v_ip1;
  }
}

} // namespace sip::optimal_control

#undef EIGEN_NO_MALLOC

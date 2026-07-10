#include "sip_optimal_control/lqr.hpp"

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace sip::optimal_control {
namespace {

struct LQRProblem {
  int state_dim = 2;
  int control_dim = 1;
  int num_stages = 2;

  std::vector<Eigen::MatrixXd> Q;
  std::vector<Eigen::MatrixXd> M;
  std::vector<Eigen::MatrixXd> R;
  std::vector<Eigen::MatrixXd> A;
  std::vector<Eigen::MatrixXd> B;
  std::vector<Eigen::VectorXd> q;
  std::vector<Eigen::VectorXd> r;
  std::vector<Eigen::VectorXd> c;
  std::vector<Eigen::VectorXd> delta;

  std::vector<double *> Q_ptr;
  std::vector<double *> M_ptr;
  std::vector<double *> R_ptr;
  std::vector<double *> A_ptr;
  std::vector<double *> B_ptr;
  std::vector<double *> q_ptr;
  std::vector<double *> r_ptr;
  std::vector<double *> c_ptr;
  std::vector<double *> delta_ptr;

  LQRProblem(int n, int m, int T)
      : state_dim(n), control_dim(m), num_stages(T), Q(T + 1), M(T), R(T), A(T),
        B(T), q(T + 1), r(T), c(T + 1), delta(T + 1), Q_ptr(T + 1), M_ptr(T),
        R_ptr(T), A_ptr(T), B_ptr(T), q_ptr(T + 1), r_ptr(T), c_ptr(T + 1),
        delta_ptr(T + 1) {
    for (int i = 0; i <= T; ++i) {
      Q[i] = Eigen::MatrixXd::Identity(n, n);
      q[i] = Eigen::VectorXd::Zero(n);
      c[i] = Eigen::VectorXd::Zero(n);
      delta[i] = Eigen::VectorXd::Ones(n);
    }

    for (int i = 0; i < T; ++i) {
      M[i] = Eigen::MatrixXd::Zero(n, m);
      R[i] = Eigen::MatrixXd::Identity(m, m);
      A[i] = Eigen::MatrixXd::Identity(n, n);
      B[i] = Eigen::MatrixXd::Ones(n, m);
      r[i] = Eigen::VectorXd::Zero(m);
    }

    refresh_pointers();
  }

  void refresh_pointers() {
    for (int i = 0; i <= num_stages; ++i) {
      Q_ptr[i] = Q[i].data();
      q_ptr[i] = q[i].data();
      c_ptr[i] = c[i].data();
      delta_ptr[i] = delta[i].data();
    }
    for (int i = 0; i < num_stages; ++i) {
      M_ptr[i] = M[i].data();
      R_ptr[i] = R[i].data();
      A_ptr[i] = A[i].data();
      B_ptr[i] = B[i].data();
      r_ptr[i] = r[i].data();
    }
  }

  auto input() -> LQR::Input {
    refresh_pointers();
    return LQR::Input{
        .Q = Q_ptr.data(),
        .M = M_ptr.data(),
        .R = R_ptr.data(),
        .q = q_ptr.data(),
        .r = r_ptr.data(),
        .A = A_ptr.data(),
        .B = B_ptr.data(),
        .c = c_ptr.data(),
        .delta = delta_ptr.data(),
        .dimensions =
            {
                .state_dim = state_dim,
                .control_dim = control_dim,
                .num_stages = num_stages,
            },
    };
  }
};

struct LQROutputStorage {
  std::vector<Eigen::VectorXd> x;
  std::vector<Eigen::VectorXd> u;
  std::vector<Eigen::VectorXd> y;
  std::vector<double *> x_ptr;
  std::vector<double *> u_ptr;
  std::vector<double *> y_ptr;

  LQROutputStorage(int n, int m, int T)
      : x(T + 1), u(T), y(T + 1), x_ptr(T + 1), u_ptr(T), y_ptr(T + 1) {
    for (int i = 0; i <= T; ++i) {
      x[i] = Eigen::VectorXd::Zero(n);
      y[i] = Eigen::VectorXd::Zero(n);
      x_ptr[i] = x[i].data();
      y_ptr[i] = y[i].data();
    }
    for (int i = 0; i < T; ++i) {
      u[i] = Eigen::VectorXd::Zero(m);
      u_ptr[i] = u[i].data();
    }
  }

  auto output() -> LQR::Output {
    return LQR::Output{
        .x = x_ptr.data(),
        .u = u_ptr.data(),
        .y = y_ptr.data(),
    };
  }
};

auto factor_status(LQRProblem &problem) -> LQR::FactorStatus {
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_stages);
  auto lqr = LQR(input, workspace);
  const auto status = lqr.factor_with_status();
  workspace.free(problem.num_stages);
  return status;
}

auto compute_residual_norm(const LQRProblem &problem,
                           const LQROutputStorage &output) -> double {
  const int T = problem.num_stages;
  double squared_norm = 0.0;

  for (int i = 0; i < T; ++i) {
    const Eigen::VectorXd stationarity_x =
        problem.Q[i].selfadjointView<Eigen::Lower>() * output.x[i] +
        problem.M[i] * output.u[i] - output.y[i] +
        problem.A[i].transpose() * output.y[i + 1] + problem.q[i];
    const Eigen::VectorXd stationarity_u =
        problem.M[i].transpose() * output.x[i] +
        problem.R[i].selfadjointView<Eigen::Lower>() * output.u[i] +
        problem.B[i].transpose() * output.y[i + 1] + problem.r[i];
    const Eigen::VectorXd dynamics =
        problem.A[i] * output.x[i] + problem.B[i] * output.u[i] -
        output.x[i + 1] + problem.c[i + 1] -
        problem.delta[i + 1].cwiseProduct(output.y[i + 1]);

    squared_norm += stationarity_x.squaredNorm();
    squared_norm += stationarity_u.squaredNorm();
    squared_norm += dynamics.squaredNorm();
  }

  const Eigen::VectorXd terminal_stationarity =
      problem.Q[T].selfadjointView<Eigen::Lower>() * output.x[T] -
      output.y[T] + problem.q[T];
  const Eigen::VectorXd initial_dynamics =
      -output.x[0] - problem.delta[0].cwiseProduct(output.y[0]) + problem.c[0];

  squared_norm += terminal_stationarity.squaredNorm();
  squared_norm += initial_dynamics.squaredNorm();

  return std::sqrt(squared_norm);
}

TEST(LQRFactor, ReportsSuccess) {
  auto problem = LQRProblem(/*n=*/2, /*m=*/1, /*T=*/2);

  EXPECT_EQ(factor_status(problem), LQR::FactorStatus::SUCCESS);
}

TEST(LQRFactor, BoolFactorWrapsStatusApi) {
  auto problem = LQRProblem(/*n=*/2, /*m=*/1, /*T=*/2);
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_stages);
  auto lqr = LQR(input, workspace);

  EXPECT_TRUE(lqr.factor());

  workspace.free(problem.num_stages);
}

TEST(LQRFactor, ReportsInvalidDelta) {
  auto problem = LQRProblem(/*n=*/2, /*m=*/1, /*T=*/2);
  problem.delta[problem.num_stages](0) = 0.0;

  EXPECT_EQ(factor_status(problem), LQR::FactorStatus::INVALID_DELTA);
}

TEST(LQRFactor, ReportsFFactorizationFailure) {
  auto problem = LQRProblem(/*n=*/1, /*m=*/1, /*T=*/1);
  problem.Q[problem.num_stages](0, 0) = -2.0;
  problem.delta[problem.num_stages](0) = 1.0;

  EXPECT_EQ(factor_status(problem), LQR::FactorStatus::F_FACTORIZATION_FAILURE);
}

TEST(LQRFactor, ReportsGFactorizationFailure) {
  auto problem = LQRProblem(/*n=*/1, /*m=*/1, /*T=*/1);
  problem.Q[problem.num_stages](0, 0) = 0.0;
  problem.R[0](0, 0) = -1.0;

  EXPECT_EQ(factor_status(problem), LQR::FactorStatus::G_FACTORIZATION_FAILURE);
}

TEST(LQRSolve, SolvesNonuniformDiagonalDeltaProblem) {
  auto problem = LQRProblem(/*n=*/3, /*m=*/2, /*T=*/3);

  for (int i = 0; i < problem.num_stages; ++i) {
    problem.A[i] << 1.0 + 0.02 * i, 0.03, -0.01, -0.02, 0.95 + 0.01 * i, 0.04,
        0.01, -0.03, 1.02 - 0.01 * i;
    problem.B[i] << 0.2, -0.1, 0.05, 0.15, -0.1, 0.08;
    problem.Q[i].diagonal() << 1.0 + 0.1 * i, 1.4 + 0.05 * i, 1.8 + 0.03 * i;
    problem.R[i].diagonal() << 1.2 + 0.1 * i, 1.6 + 0.07 * i;
    problem.q[i] << 0.2 + 0.01 * i, -0.1 + 0.02 * i, 0.05 - 0.03 * i;
    problem.r[i] << -0.2 + 0.03 * i, 0.1 - 0.01 * i;
    problem.c[i] << 0.03 + 0.01 * i, -0.04 + 0.02 * i, 0.02 - 0.01 * i;
    problem.delta[i] << 0.03 + 0.01 * i, 0.11 + 0.02 * i, 0.19 + 0.03 * i;
  }

  problem.Q[problem.num_stages].diagonal() << 1.3, 1.7, 2.1;
  problem.q[problem.num_stages] << 0.06, -0.08, 0.12;
  problem.c[problem.num_stages] << -0.02, 0.05, -0.01;
  problem.delta[problem.num_stages] << 0.07, 0.17, 0.29;

  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_stages);
  auto lqr = LQR(input, workspace);
  ASSERT_EQ(lqr.factor_with_status(), LQR::FactorStatus::SUCCESS);

  auto output_storage = LQROutputStorage(problem.state_dim, problem.control_dim,
                                         problem.num_stages);
  auto output = output_storage.output();
  lqr.solve(output);

  EXPECT_LT(compute_residual_norm(problem, output_storage), 1e-12);

  workspace.free(problem.num_stages);
}

} // namespace
} // namespace sip::optimal_control

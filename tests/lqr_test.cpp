#include "sip_optimal_control/lqr.hpp"

#include <Eigen/Dense>

#include <gtest/gtest.h>

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
      : state_dim(n), control_dim(m), num_stages(T), Q(T + 1), M(T), R(T),
        A(T), B(T), q(T + 1), r(T), c(T + 1), delta(T + 1), Q_ptr(T + 1),
        M_ptr(T), R_ptr(T), A_ptr(T), B_ptr(T), q_ptr(T + 1), r_ptr(T),
        c_ptr(T + 1), delta_ptr(T + 1) {
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

auto factor_status(LQRProblem &problem) -> LQR::FactorStatus {
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_stages);
  auto lqr = LQR(input, workspace);
  const auto status = lqr.factor_with_status();
  workspace.free(problem.num_stages);
  return status;
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

  EXPECT_EQ(factor_status(problem),
            LQR::FactorStatus::F_FACTORIZATION_FAILURE);
}

TEST(LQRFactor, ReportsGFactorizationFailure) {
  auto problem = LQRProblem(/*n=*/1, /*m=*/1, /*T=*/1);
  problem.Q[problem.num_stages](0, 0) = 0.0;
  problem.R[0](0, 0) = -1.0;

  EXPECT_EQ(factor_status(problem),
            LQR::FactorStatus::G_FACTORIZATION_FAILURE);
}

} // namespace
} // namespace sip::optimal_control

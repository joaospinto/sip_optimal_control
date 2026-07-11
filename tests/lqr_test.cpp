#include "sip_optimal_control/lqr.hpp"

#include <Eigen/Dense>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

namespace sip::optimal_control {
namespace {

struct LQRProblem {
  int state_dim = 2;
  int control_dim = 1;
  int num_edges = 2;
  std::vector<int> state_dims;
  std::vector<int> control_dims;
  std::vector<int> edge_parents;
  std::vector<int> edge_children;
  Dimensions input_dimensions;
  Topology input_topology;

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
      : state_dim(n), control_dim(m), num_edges(T), state_dims(T + 1, n),
        control_dims(T, m), edge_parents(T), edge_children(T), Q(T + 1), M(T), R(T), A(T),
        B(T), q(T + 1), r(T), c(T + 1), delta(T + 1), Q_ptr(T + 1), M_ptr(T),
        R_ptr(T), A_ptr(T), B_ptr(T), q_ptr(T + 1), r_ptr(T), c_ptr(T + 1),
        delta_ptr(T + 1) {
    for (int edge = 0; edge < T; ++edge) {
      edge_parents[edge] = edge;
      edge_children[edge] = edge + 1;
    }
    input_dimensions = Dimensions{0, state_dims.data(), control_dims.data(),
                                  nullptr, nullptr};
    input_topology = Topology{num_edges, 0, edge_parents.data(),
                              edge_children.data()};
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
    for (int i = 0; i <= num_edges; ++i) {
      Q_ptr[i] = Q[i].data();
      q_ptr[i] = q[i].data();
      c_ptr[i] = c[i].data();
      delta_ptr[i] = delta[i].data();
    }
    for (int i = 0; i < num_edges; ++i) {
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
        .dimensions = input_dimensions,
        .topology = input_topology,
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
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_edges);
  auto lqr = LQR(input, workspace);
  const auto status = lqr.factor_with_status();
  workspace.free(problem.num_edges);
  return status;
}

auto compute_residual_norm(const LQRProblem &problem,
                           const LQROutputStorage &output) -> double {
  const int T = problem.num_edges;
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
      problem.Q[T].selfadjointView<Eigen::Lower>() * output.x[T] - output.y[T] +
      problem.q[T];
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
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_edges);
  auto lqr = LQR(input, workspace);

  EXPECT_TRUE(lqr.factor());

  workspace.free(problem.num_edges);
}

TEST(LQRFactor, ReportsInvalidDelta) {
  auto problem = LQRProblem(/*n=*/2, /*m=*/1, /*T=*/2);
  problem.delta[problem.num_edges](0) = 0.0;

  EXPECT_EQ(factor_status(problem), LQR::FactorStatus::INVALID_DELTA);
}

TEST(LQRFactor, ReportsFFactorizationFailure) {
  auto problem = LQRProblem(/*n=*/1, /*m=*/1, /*T=*/1);
  problem.Q[problem.num_edges](0, 0) = -2.0;
  problem.delta[problem.num_edges](0) = 1.0;

  EXPECT_EQ(factor_status(problem), LQR::FactorStatus::F_FACTORIZATION_FAILURE);
}

TEST(LQRFactor, ReportsGFactorizationFailure) {
  auto problem = LQRProblem(/*n=*/1, /*m=*/1, /*T=*/1);
  problem.Q[problem.num_edges](0, 0) = 0.0;
  problem.R[0](0, 0) = -1.0;

  EXPECT_EQ(factor_status(problem), LQR::FactorStatus::G_FACTORIZATION_FAILURE);
}

TEST(LQRSolve, SolvesNonuniformDiagonalDeltaProblem) {
  auto problem = LQRProblem(/*n=*/3, /*m=*/2, /*T=*/3);

  for (int i = 0; i < problem.num_edges; ++i) {
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

  problem.Q[problem.num_edges].diagonal() << 1.3, 1.7, 2.1;
  problem.q[problem.num_edges] << 0.06, -0.08, 0.12;
  problem.c[problem.num_edges] << -0.02, 0.05, -0.01;
  problem.delta[problem.num_edges] << 0.07, 0.17, 0.29;

  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_edges);
  auto lqr = LQR(input, workspace);
  ASSERT_EQ(lqr.factor_with_status(), LQR::FactorStatus::SUCCESS);

  auto output_storage = LQROutputStorage(problem.state_dim, problem.control_dim,
                                         problem.num_edges);
  auto output = output_storage.output();
  lqr.solve(output);

  EXPECT_LT(compute_residual_norm(problem, output_storage), 1e-12);

  workspace.free(problem.num_edges);
}

struct BranchTopology {
  int parent[2] = {0, 0};
  int child[2] = {1, 2};
};

struct BranchLQRProblem {
  int state_dim = 2;
  int control_dim = 1;
  int num_edges = 2;
  BranchTopology topology;
  std::array<int, 3> state_dims = {2, 2, 2};
  std::array<int, 2> control_dims = {1, 1};
  Dimensions input_dimensions;
  Topology input_topology;

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

  BranchLQRProblem()
      : Q(3), M(2), R(2), A(2), B(2), q(3), r(2), c(3), delta(3), Q_ptr(3),
        M_ptr(2), R_ptr(2), A_ptr(2), B_ptr(2), q_ptr(3), r_ptr(2), c_ptr(3),
        delta_ptr(3) {
    Q[0] = (Eigen::Matrix2d() << 2.0, 0.1, 0.1, 1.5).finished();
    Q[1] = (Eigen::Matrix2d() << 1.3, 0.2, 0.2, 1.7).finished();
    Q[2] = (Eigen::Matrix2d() << 1.8, -0.1, -0.1, 1.4).finished();

    M[0] = (Eigen::Matrix<double, 2, 1>() << 0.2, -0.1).finished();
    M[1] = (Eigen::Matrix<double, 2, 1>() << -0.15, 0.05).finished();
    R[0] = Eigen::MatrixXd::Constant(1, 1, 1.6);
    R[1] = Eigen::MatrixXd::Constant(1, 1, 1.9);

    A[0] = (Eigen::Matrix2d() << 1.0, 0.2, 0.0, 0.9).finished();
    A[1] = (Eigen::Matrix2d() << 0.8, -0.1, 0.3, 1.1).finished();
    B[0] = (Eigen::Matrix<double, 2, 1>() << 0.4, 0.2).finished();
    B[1] = (Eigen::Matrix<double, 2, 1>() << -0.1, 0.5).finished();

    q[0] = (Eigen::Vector2d() << 0.3, -0.2).finished();
    q[1] = (Eigen::Vector2d() << -0.1, 0.4).finished();
    q[2] = (Eigen::Vector2d() << 0.2, 0.1).finished();
    r[0] = Eigen::VectorXd::Constant(1, -0.3);
    r[1] = Eigen::VectorXd::Constant(1, 0.25);

    c[0] = (Eigen::Vector2d() << 0.1, -0.2).finished();
    c[1] = (Eigen::Vector2d() << -0.05, 0.1).finished();
    c[2] = (Eigen::Vector2d() << 0.2, 0.15).finished();
    delta[0] = (Eigen::Vector2d() << 0.7, 0.9).finished();
    delta[1] = (Eigen::Vector2d() << 0.8, 1.1).finished();
    delta[2] = (Eigen::Vector2d() << 1.0, 0.6).finished();

    refresh_pointers();
    input_dimensions = Dimensions{0, state_dims.data(), control_dims.data(),
                                  nullptr, nullptr};
    input_topology = Topology{num_edges, 0, topology.parent, topology.child};
  }

  void refresh_pointers() {
    for (int node = 0; node < 3; ++node) {
      Q_ptr[node] = Q[node].data();
      q_ptr[node] = q[node].data();
      c_ptr[node] = c[node].data();
      delta_ptr[node] = delta[node].data();
    }
    for (int edge = 0; edge < num_edges; ++edge) {
      M_ptr[edge] = M[edge].data();
      R_ptr[edge] = R[edge].data();
      A_ptr[edge] = A[edge].data();
      B_ptr[edge] = B[edge].data();
      r_ptr[edge] = r[edge].data();
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
        .dimensions = input_dimensions,
        .topology = input_topology,
    };
  }
};

auto compute_branch_residual_norm(const BranchLQRProblem &problem,
                                  const LQROutputStorage &output) -> double {
  double squared_norm = 0.0;

  for (int node = 0; node < 3; ++node) {
    Eigen::VectorXd stationarity_x =
        problem.Q[node].selfadjointView<Eigen::Lower>() * output.x[node] -
        output.y[node] + problem.q[node];
    for (int edge = 0; edge < problem.num_edges; ++edge) {
      if (problem.topology.parent[edge] == node) {
        const int child = problem.topology.child[edge];
        stationarity_x += problem.M[edge] * output.u[edge] +
                          problem.A[edge].transpose() * output.y[child];
      }
    }
    squared_norm += stationarity_x.squaredNorm();
  }

  for (int edge = 0; edge < problem.num_edges; ++edge) {
    const int parent = problem.topology.parent[edge];
    const int child = problem.topology.child[edge];
    const Eigen::VectorXd stationarity_u =
        problem.M[edge].transpose() * output.x[parent] +
        problem.R[edge].selfadjointView<Eigen::Lower>() * output.u[edge] +
        problem.B[edge].transpose() * output.y[child] + problem.r[edge];
    const Eigen::VectorXd dynamics =
        problem.A[edge] * output.x[parent] + problem.B[edge] * output.u[edge] -
        output.x[child] + problem.c[child] -
        problem.delta[child].cwiseProduct(output.y[child]);
    squared_norm += stationarity_u.squaredNorm();
    squared_norm += dynamics.squaredNorm();
  }

  const Eigen::VectorXd root_dynamics =
      -output.x[0] - problem.delta[0].cwiseProduct(output.y[0]) + problem.c[0];
  squared_norm += root_dynamics.squaredNorm();

  return std::sqrt(squared_norm);
}

TEST(LQRSolve, SolvesBranchingTreeProblem) {
  auto problem = BranchLQRProblem();
  auto input = problem.input();

  LQR::Workspace workspace;
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_edges);
  auto lqr = LQR(input, workspace);

  ASSERT_EQ(lqr.factor_with_status(), LQR::FactorStatus::SUCCESS);

  auto output_storage = LQROutputStorage(problem.state_dim, problem.control_dim,
                                         problem.num_edges);
  auto output = output_storage.output();
  lqr.solve(output);

  EXPECT_LT(compute_branch_residual_norm(problem, output_storage), 1e-12);

  workspace.free(problem.num_edges);
}

TEST(LQRTopology, ReusesCompiledTopologyAcrossFactorAndSolveCalls) {
  auto problem = BranchLQRProblem();
  auto input = problem.input();

  LQR::Workspace workspace;
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_edges);
  auto lqr = LQR(input, workspace);

  ASSERT_EQ(lqr.factor_with_status(), LQR::FactorStatus::SUCCESS);

  ASSERT_EQ(lqr.factor_with_status(), LQR::FactorStatus::SUCCESS);

  auto output_storage = LQROutputStorage(problem.state_dim, problem.control_dim,
                                         problem.num_edges);
  auto output = output_storage.output();
  lqr.solve(output);
  lqr.solve(output);


  workspace.free(problem.num_edges);
}

TEST(LQRFactor, RejectsInvalidTreeTopology) {
  auto problem = BranchLQRProblem();
  problem.topology.child[1] = 1;
  auto input = problem.input();

  LQR::Workspace workspace;
  workspace.reserve(problem.state_dim, problem.control_dim, problem.num_edges);
  auto lqr = LQR(input, workspace);

  EXPECT_EQ(lqr.factor_with_status(), LQR::FactorStatus::INVALID_TOPOLOGY);

  workspace.free(problem.num_edges);
}

struct VariableDimensionBranchProblem {
  int num_edges = 2;
  std::vector<int> state_dims = {2, 1, 3};
  std::vector<int> control_dims = {2, 1};
  BranchTopology topology;
  Dimensions input_dimensions;
  Topology input_topology;

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

  VariableDimensionBranchProblem()
      : Q(3), M(2), R(2), A(2), B(2), q(3), r(2), c(3), delta(3), Q_ptr(3),
        M_ptr(2), R_ptr(2), A_ptr(2), B_ptr(2), q_ptr(3), r_ptr(2), c_ptr(3),
        delta_ptr(3) {
    Q[0] = (Eigen::Matrix2d() << 2.0, 0.1, 0.1, 1.7).finished();
    Q[1] = Eigen::MatrixXd::Constant(1, 1, 1.3);
    Q[2] =
        (Eigen::Matrix3d() << 1.8, 0.1, -0.2, 0.1, 1.6, 0.05, -0.2, 0.05, 2.1)
            .finished();

    M[0] = (Eigen::Matrix2d() << 0.1, -0.2, 0.05, 0.15).finished();
    M[1] = (Eigen::Matrix<double, 2, 1>() << -0.1, 0.2).finished();
    R[0] = (Eigen::Matrix2d() << 1.8, 0.1, 0.1, 1.5).finished();
    R[1] = Eigen::MatrixXd::Constant(1, 1, 1.4);

    A[0] = (Eigen::Matrix<double, 1, 2>() << 0.8, -0.3).finished();
    A[1] = (Eigen::Matrix<double, 3, 2>() << 1.0, 0.2, -0.1, 0.7, 0.3, -0.4)
               .finished();
    B[0] = (Eigen::Matrix<double, 1, 2>() << 0.4, -0.2).finished();
    B[1] = (Eigen::Matrix<double, 3, 1>() << 0.2, -0.1, 0.5).finished();

    q[0] = (Eigen::Vector2d() << 0.2, -0.15).finished();
    q[1] = Eigen::VectorXd::Constant(1, -0.05);
    q[2] = (Eigen::Vector3d() << 0.1, -0.2, 0.05).finished();
    r[0] = (Eigen::Vector2d() << -0.1, 0.25).finished();
    r[1] = Eigen::VectorXd::Constant(1, -0.2);

    c[0] = (Eigen::Vector2d() << 0.05, -0.1).finished();
    c[1] = Eigen::VectorXd::Constant(1, 0.12);
    c[2] = (Eigen::Vector3d() << -0.02, 0.04, -0.08).finished();
    delta[0] = (Eigen::Vector2d() << 0.8, 1.1).finished();
    delta[1] = Eigen::VectorXd::Constant(1, 0.9);
    delta[2] = (Eigen::Vector3d() << 0.7, 1.0, 1.2).finished();

    refresh_pointers();
    input_dimensions = Dimensions{0, state_dims.data(), control_dims.data(),
                                  nullptr, nullptr};
    input_topology = Topology{num_edges, 0, topology.parent, topology.child};
  }

  void refresh_pointers() {
    for (int node = 0; node < 3; ++node) {
      Q_ptr[node] = Q[node].data();
      q_ptr[node] = q[node].data();
      c_ptr[node] = c[node].data();
      delta_ptr[node] = delta[node].data();
    }
    for (int edge = 0; edge < num_edges; ++edge) {
      M_ptr[edge] = M[edge].data();
      R_ptr[edge] = R[edge].data();
      A_ptr[edge] = A[edge].data();
      B_ptr[edge] = B[edge].data();
      r_ptr[edge] = r[edge].data();
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
        .dimensions = input_dimensions,
        .topology = input_topology,
    };
  }
};

struct VariableDimensionOutputStorage {
  std::vector<Eigen::VectorXd> x;
  std::vector<Eigen::VectorXd> u;
  std::vector<Eigen::VectorXd> y;
  std::vector<double *> x_ptr;
  std::vector<double *> u_ptr;
  std::vector<double *> y_ptr;

  explicit VariableDimensionOutputStorage(
      const VariableDimensionBranchProblem &problem)
      : x(3), u(2), y(3), x_ptr(3), u_ptr(2), y_ptr(3) {
    for (int node = 0; node < 3; ++node) {
      x[node] = Eigen::VectorXd::Zero(problem.state_dims[node]);
      y[node] = Eigen::VectorXd::Zero(problem.state_dims[node]);
      x_ptr[node] = x[node].data();
      y_ptr[node] = y[node].data();
    }
    for (int edge = 0; edge < problem.num_edges; ++edge) {
      u[edge] = Eigen::VectorXd::Zero(problem.control_dims[edge]);
      u_ptr[edge] = u[edge].data();
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

auto compute_variable_branch_residual_norm(
    const VariableDimensionBranchProblem &problem,
    const VariableDimensionOutputStorage &output) -> double {
  double squared_norm = 0.0;

  for (int node = 0; node < 3; ++node) {
    Eigen::VectorXd stationarity_x =
        problem.Q[node].selfadjointView<Eigen::Lower>() * output.x[node] -
        output.y[node] + problem.q[node];
    for (int edge = 0; edge < problem.num_edges; ++edge) {
      if (problem.topology.parent[edge] == node) {
        const int child = problem.topology.child[edge];
        stationarity_x += problem.M[edge] * output.u[edge] +
                          problem.A[edge].transpose() * output.y[child];
      }
    }
    squared_norm += stationarity_x.squaredNorm();
  }

  for (int edge = 0; edge < problem.num_edges; ++edge) {
    const int parent = problem.topology.parent[edge];
    const int child = problem.topology.child[edge];
    const Eigen::VectorXd stationarity_u =
        problem.M[edge].transpose() * output.x[parent] +
        problem.R[edge].selfadjointView<Eigen::Lower>() * output.u[edge] +
        problem.B[edge].transpose() * output.y[child] + problem.r[edge];
    const Eigen::VectorXd dynamics =
        problem.A[edge] * output.x[parent] + problem.B[edge] * output.u[edge] -
        output.x[child] + problem.c[child] -
        problem.delta[child].cwiseProduct(output.y[child]);
    squared_norm += stationarity_u.squaredNorm();
    squared_norm += dynamics.squaredNorm();
  }

  const Eigen::VectorXd root_dynamics =
      -output.x[0] - problem.delta[0].cwiseProduct(output.y[0]) + problem.c[0];
  squared_norm += root_dynamics.squaredNorm();

  return std::sqrt(squared_norm);
}

TEST(LQRSolve, SolvesVariableDimensionBranchingTreeProblem) {
  auto problem = VariableDimensionBranchProblem();
  auto input = problem.input();

  LQR::Workspace workspace;
  workspace.reserve(input.dimensions, input.topology);
  auto lqr = LQR(input, workspace);

  ASSERT_EQ(lqr.factor_with_status(), LQR::FactorStatus::SUCCESS);

  auto output_storage = VariableDimensionOutputStorage(problem);
  auto output = output_storage.output();
  lqr.solve(output);

  EXPECT_LT(compute_variable_branch_residual_norm(problem, output_storage),
            1e-12);

  workspace.free(problem.num_edges);
}

struct FiveNodeTopology {
  std::array<int, 4> parent = {0, 0, 1, 1};
  std::array<int, 4> child = {1, 2, 3, 4};

};

struct FiveNodeVariableTreeProblem {
  static constexpr int num_nodes = 5;
  static constexpr int num_edges = 4;
  std::array<int, num_nodes> state_dims = {3, 1, 2, 4, 2};
  std::array<int, num_edges> control_dims = {2, 1, 3, 1};
  FiveNodeTopology topology;
  Dimensions input_dimensions;
  Topology input_topology;

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

  FiveNodeVariableTreeProblem()
      : Q(num_nodes), M(num_edges), R(num_edges), A(num_edges), B(num_edges),
        q(num_nodes), r(num_edges), c(num_nodes), delta(num_nodes),
        Q_ptr(num_nodes), M_ptr(num_edges), R_ptr(num_edges), A_ptr(num_edges),
        B_ptr(num_edges), q_ptr(num_nodes), r_ptr(num_edges), c_ptr(num_nodes),
        delta_ptr(num_nodes) {
    for (int node = 0; node < num_nodes; ++node) {
      const int n = state_dims[node];
      Q[node] = Eigen::MatrixXd::Identity(n, n) * (1.5 + 0.2 * node);
      for (int col = 0; col < n; ++col) {
        for (int row = col + 1; row < n; ++row) {
          Q[node](row, col) = 0.02 * static_cast<double>(row + col + node + 1);
          Q[node](col, row) = Q[node](row, col);
        }
      }
      q[node] = Eigen::VectorXd::LinSpaced(n, -0.15 + 0.03 * node,
                                           0.12 + 0.02 * node);
      c[node] = Eigen::VectorXd::LinSpaced(n, 0.05 * node, 0.04 + 0.03 * node);
      delta[node] =
          Eigen::VectorXd::LinSpaced(n, 0.7 + 0.05 * node, 1.0 + 0.04 * node);
    }

    for (int edge = 0; edge < num_edges; ++edge) {
      const int parent = topology.parent[edge];
      const int child = topology.child[edge];
      const int n_parent = state_dims[parent];
      const int n_child = state_dims[child];
      const int m = control_dims[edge];

      M[edge] = Eigen::MatrixXd(n_parent, m);
      A[edge] = Eigen::MatrixXd(n_child, n_parent);
      B[edge] = Eigen::MatrixXd(n_child, m);
      for (int col = 0; col < m; ++col) {
        for (int row = 0; row < n_parent; ++row) {
          M[edge](row, col) =
              0.015 * static_cast<double>((edge + 1) * (row + 1) - col);
        }
      }
      for (int col = 0; col < n_parent; ++col) {
        for (int row = 0; row < n_child; ++row) {
          A[edge](row, col) = 0.08 * static_cast<double>(row + 1) /
                              static_cast<double>(edge + col + 2);
        }
      }
      for (int col = 0; col < m; ++col) {
        for (int row = 0; row < n_child; ++row) {
          B[edge](row, col) = -0.06 * static_cast<double>(col + 1) /
                              static_cast<double>(edge + row + 2);
        }
      }

      R[edge] = Eigen::MatrixXd::Identity(m, m) * (1.8 + 0.1 * edge);
      for (int col = 0; col < m; ++col) {
        for (int row = col + 1; row < m; ++row) {
          R[edge](row, col) = 0.03 * static_cast<double>(row + col + 1);
          R[edge](col, row) = R[edge](row, col);
        }
      }
      r[edge] =
          Eigen::VectorXd::LinSpaced(m, -0.2 + 0.04 * edge, 0.1 + 0.03 * edge);
    }

    refresh_pointers();
    input_dimensions = Dimensions{0, state_dims.data(), control_dims.data(),
                                  nullptr, nullptr};
    input_topology = Topology{num_edges, 0, topology.parent.data(),
                              topology.child.data()};
  }

  void refresh_pointers() {
    for (int node = 0; node < num_nodes; ++node) {
      Q_ptr[node] = Q[node].data();
      q_ptr[node] = q[node].data();
      c_ptr[node] = c[node].data();
      delta_ptr[node] = delta[node].data();
    }
    for (int edge = 0; edge < num_edges; ++edge) {
      M_ptr[edge] = M[edge].data();
      R_ptr[edge] = R[edge].data();
      A_ptr[edge] = A[edge].data();
      B_ptr[edge] = B[edge].data();
      r_ptr[edge] = r[edge].data();
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
        .dimensions = input_dimensions,
        .topology = input_topology,
    };
  }
};

struct FiveNodeVariableOutputStorage {
  std::vector<Eigen::VectorXd> x;
  std::vector<Eigen::VectorXd> u;
  std::vector<Eigen::VectorXd> y;
  std::vector<double *> x_ptr;
  std::vector<double *> u_ptr;
  std::vector<double *> y_ptr;

  explicit FiveNodeVariableOutputStorage(
      const FiveNodeVariableTreeProblem &problem)
      : x(FiveNodeVariableTreeProblem::num_nodes),
        u(FiveNodeVariableTreeProblem::num_edges),
        y(FiveNodeVariableTreeProblem::num_nodes),
        x_ptr(FiveNodeVariableTreeProblem::num_nodes),
        u_ptr(FiveNodeVariableTreeProblem::num_edges),
        y_ptr(FiveNodeVariableTreeProblem::num_nodes) {
    for (int node = 0; node < FiveNodeVariableTreeProblem::num_nodes; ++node) {
      x[node] = Eigen::VectorXd::Zero(problem.state_dims[node]);
      y[node] = Eigen::VectorXd::Zero(problem.state_dims[node]);
      x_ptr[node] = x[node].data();
      y_ptr[node] = y[node].data();
    }
    for (int edge = 0; edge < FiveNodeVariableTreeProblem::num_edges; ++edge) {
      u[edge] = Eigen::VectorXd::Zero(problem.control_dims[edge]);
      u_ptr[edge] = u[edge].data();
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

struct DenseLayout {
  std::array<int, FiveNodeVariableTreeProblem::num_nodes> x_offset = {};
  std::array<int, FiveNodeVariableTreeProblem::num_edges> u_offset = {};
  std::array<int, FiveNodeVariableTreeProblem::num_nodes> y_offset = {};
  int total_dim = 0;
};

auto dense_layout(const FiveNodeVariableTreeProblem &problem) -> DenseLayout {
  auto layout = DenseLayout();
  for (int node = 0; node < FiveNodeVariableTreeProblem::num_nodes; ++node) {
    layout.x_offset[node] = layout.total_dim;
    layout.total_dim += problem.state_dims[node];
  }
  for (int edge = 0; edge < FiveNodeVariableTreeProblem::num_edges; ++edge) {
    layout.u_offset[edge] = layout.total_dim;
    layout.total_dim += problem.control_dims[edge];
  }
  for (int node = 0; node < FiveNodeVariableTreeProblem::num_nodes; ++node) {
    layout.y_offset[node] = layout.total_dim;
    layout.total_dim += problem.state_dims[node];
  }
  return layout;
}

auto solve_dense_kkt(const FiveNodeVariableTreeProblem &problem)
    -> Eigen::VectorXd {
  const auto layout = dense_layout(problem);
  Eigen::MatrixXd matrix =
      Eigen::MatrixXd::Zero(layout.total_dim, layout.total_dim);
  Eigen::VectorXd rhs = Eigen::VectorXd::Zero(layout.total_dim);
  int row_offset = 0;

  for (int node = 0; node < FiveNodeVariableTreeProblem::num_nodes; ++node) {
    const int n = problem.state_dims[node];
    matrix.block(row_offset, layout.x_offset[node], n, n) += problem.Q[node];
    matrix.block(row_offset, layout.y_offset[node], n, n).diagonal().array() -=
        1.0;
    for (int edge = 0; edge < FiveNodeVariableTreeProblem::num_edges; ++edge) {
      if (problem.topology.parent[edge] == node) {
        const int child = problem.topology.child[edge];
        const int m = problem.control_dims[edge];
        matrix.block(row_offset, layout.u_offset[edge], n, m) +=
            problem.M[edge];
        matrix.block(row_offset, layout.y_offset[child], n,
                     problem.state_dims[child]) += problem.A[edge].transpose();
      }
    }
    rhs.segment(row_offset, n) = -problem.q[node];
    row_offset += n;
  }

  for (int edge = 0; edge < FiveNodeVariableTreeProblem::num_edges; ++edge) {
    const int parent = problem.topology.parent[edge];
    const int child = problem.topology.child[edge];
    const int m = problem.control_dims[edge];
    matrix.block(row_offset, layout.x_offset[parent], m,
                 problem.state_dims[parent]) += problem.M[edge].transpose();
    matrix.block(row_offset, layout.u_offset[edge], m, m) += problem.R[edge];
    matrix.block(row_offset, layout.y_offset[child], m,
                 problem.state_dims[child]) += problem.B[edge].transpose();
    rhs.segment(row_offset, m) = -problem.r[edge];
    row_offset += m;
  }

  {
    const int root = 0;
    const int n = problem.state_dims[root];
    matrix.block(row_offset, layout.x_offset[root], n, n).diagonal().array() -=
        1.0;
    matrix.block(row_offset, layout.y_offset[root], n, n).diagonal() -=
        problem.delta[root];
    rhs.segment(row_offset, n) = -problem.c[root];
    row_offset += n;
  }

  for (int edge = 0; edge < FiveNodeVariableTreeProblem::num_edges; ++edge) {
    const int parent = problem.topology.parent[edge];
    const int child = problem.topology.child[edge];
    const int n_child = problem.state_dims[child];
    matrix.block(row_offset, layout.x_offset[parent], n_child,
                 problem.state_dims[parent]) += problem.A[edge];
    matrix.block(row_offset, layout.u_offset[edge], n_child,
                 problem.control_dims[edge]) += problem.B[edge];
    matrix.block(row_offset, layout.x_offset[child], n_child, n_child)
        .diagonal()
        .array() -= 1.0;
    matrix.block(row_offset, layout.y_offset[child], n_child, n_child)
        .diagonal() -= problem.delta[child];
    rhs.segment(row_offset, n_child) = -problem.c[child];
    row_offset += n_child;
  }

  EXPECT_EQ(row_offset, layout.total_dim);
  return matrix.colPivHouseholderQr().solve(rhs);
}

TEST(LQRTopology, CompilesMultiChildPreorderAndPostorder) {
  auto problem = FiveNodeVariableTreeProblem();
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(input.dimensions, input.topology);
  auto lqr = LQR(input, workspace);

  ASSERT_EQ(lqr.factor_with_status(), LQR::FactorStatus::SUCCESS);

  EXPECT_EQ(
      std::vector<int>(workspace.child_offsets, workspace.child_offsets + 6),
      (std::vector<int>{0, 2, 4, 4, 4, 4}));
  EXPECT_EQ(std::vector<int>(workspace.child_edges, workspace.child_edges + 4),
            (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(
      std::vector<int>(workspace.preorder_nodes, workspace.preorder_nodes + 5),
      (std::vector<int>{0, 1, 3, 4, 2}));
  EXPECT_EQ(std::vector<int>(workspace.postorder_nodes,
                             workspace.postorder_nodes + 5),
            (std::vector<int>{2, 4, 3, 1, 0}));

  workspace.free(problem.num_edges);
}

TEST(LQRTopology, RejectsDisconnectedTree) {
  auto problem = FiveNodeVariableTreeProblem();
  problem.topology.parent[3] = 4;
  problem.topology.child[3] = 3;
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(input.dimensions, input.topology);
  auto lqr = LQR(input, workspace);

  EXPECT_EQ(lqr.factor_with_status(), LQR::FactorStatus::INVALID_TOPOLOGY);

  workspace.free(problem.num_edges);
}

TEST(LQRTopology, RejectsCycle) {
  auto problem = FiveNodeVariableTreeProblem();
  problem.topology.parent[0] = 4;
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(input.dimensions, input.topology);
  auto lqr = LQR(input, workspace);

  EXPECT_EQ(lqr.factor_with_status(), LQR::FactorStatus::INVALID_TOPOLOGY);

  workspace.free(problem.num_edges);
}

TEST(LQRSolve, MatchesDenseKKTOnVariableDimensionTreeProblem) {
  auto problem = FiveNodeVariableTreeProblem();
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(input.dimensions, input.topology);
  auto lqr = LQR(input, workspace);

  ASSERT_EQ(lqr.factor_with_status(), LQR::FactorStatus::SUCCESS);

  auto output_storage = FiveNodeVariableOutputStorage(problem);
  auto output = output_storage.output();
  lqr.solve(output);

  const auto dense_solution = solve_dense_kkt(problem);
  const auto layout = dense_layout(problem);
  for (int node = 0; node < FiveNodeVariableTreeProblem::num_nodes; ++node) {
    EXPECT_TRUE(output_storage.x[node].isApprox(
        dense_solution.segment(layout.x_offset[node], problem.state_dims[node]),
        1e-10));
    EXPECT_TRUE(output_storage.y[node].isApprox(
        dense_solution.segment(layout.y_offset[node], problem.state_dims[node]),
        1e-10));
  }
  for (int edge = 0; edge < FiveNodeVariableTreeProblem::num_edges; ++edge) {
    EXPECT_TRUE(output_storage.u[edge].isApprox(
        dense_solution.segment(layout.u_offset[edge],
                               problem.control_dims[edge]),
        1e-10));
  }

  workspace.free(problem.num_edges);
}

} // namespace
} // namespace sip::optimal_control

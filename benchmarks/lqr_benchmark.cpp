#include "sip_optimal_control/lqr.hpp"

#include <Eigen/Dense>

#include <benchmark/benchmark.h>

#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace sip::optimal_control {
namespace {

struct LQRProblem {
  int state_dim;
  int control_dim;
  int num_stages;

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
    auto rng = std::mt19937(0);
    auto normal = std::normal_distribution<double>(0.0, 1.0);
    auto uniform = std::uniform_real_distribution<double>(0.0, 1.0);

    const double psd_floor = 1e-3;
    const double r_floor = 1e-2;

    for (int i = 0; i < T; ++i) {
      A[i] = 0.05 * random_matrix(rng, normal, n, n);
      A[i].diagonal().array() += 1.0;
      B[i] = 0.1 * random_matrix(rng, normal, n, m);
      M[i] = Eigen::MatrixXd::Zero(n, m);

      const auto R_root = random_matrix(rng, normal, m, m);
      R[i].noalias() = R_root.transpose() * R_root;
      R[i].diagonal().array() += r_floor + 1.0;

      const auto S_root = random_matrix(rng, normal, n, n);
      auto S = Eigen::MatrixXd(n, n);
      S.noalias() = S_root.transpose() * S_root;
      S.diagonal().array() += psd_floor;

      Q[i] = S;

      q[i] = random_vector(rng, normal, n);
      r[i] = random_vector(rng, normal, m);
      c[i] = random_vector(rng, normal, n);
      delta[i] = positive_delta(rng, uniform, n);
    }

    const auto Q_terminal_root = random_matrix(rng, normal, n, n);
    Q[T].noalias() = Q_terminal_root.transpose() * Q_terminal_root;
    Q[T].diagonal().array() += psd_floor;
    q[T] = random_vector(rng, normal, n);
    c[T] = random_vector(rng, normal, n);
    delta[T] = positive_delta(rng, uniform, n);

    refresh_pointers();
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

private:
  static auto random_matrix(std::mt19937 &rng,
                            std::normal_distribution<double> &normal, int rows,
                            int cols) -> Eigen::MatrixXd {
    auto matrix = Eigen::MatrixXd(rows, cols);
    for (int col = 0; col < cols; ++col) {
      for (int row = 0; row < rows; ++row) {
        matrix(row, col) = normal(rng);
      }
    }
    return matrix;
  }

  static auto random_vector(std::mt19937 &rng,
                            std::normal_distribution<double> &normal, int rows)
      -> Eigen::VectorXd {
    auto vector = Eigen::VectorXd(rows);
    for (int row = 0; row < rows; ++row) {
      vector(row) = normal(rng);
    }
    return vector;
  }

  static auto positive_delta(std::mt19937 &rng,
                             std::uniform_real_distribution<double> &uniform,
                             int rows) -> Eigen::VectorXd {
    auto vector = Eigen::VectorXd(rows);
    for (int row = 0; row < rows; ++row) {
      vector(row) = 1e-3 + 1e-1 * uniform(rng);
    }
    return vector;
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

auto compute_residual_norm(const LQRProblem &problem,
                           const LQROutputStorage &output) -> double {
  const int T = problem.num_stages;
  double squared_norm = 0.0;

  for (int i = 0; i < T; ++i) {
    const Eigen::VectorXd stationarity_x =
        problem.Q[i] * output.x[i] + problem.M[i] * output.u[i] - output.y[i] +
        problem.A[i].transpose() * output.y[i + 1] + problem.q[i];
    const Eigen::VectorXd stationarity_u =
        problem.M[i].transpose() * output.x[i] + problem.R[i] * output.u[i] +
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
      problem.Q[T] * output.x[T] - output.y[T] + problem.q[T];
  const Eigen::VectorXd initial_dynamics =
      -output.x[0] - problem.delta[0].cwiseProduct(output.y[0]) + problem.c[0];

  squared_norm += terminal_stationarity.squaredNorm();
  squared_norm += initial_dynamics.squaredNorm();

  return std::sqrt(squared_norm);
}

auto factor_status_name(const LQR::FactorStatus status) -> const char * {
  switch (status) {
  case LQR::FactorStatus::SUCCESS:
    return "SUCCESS";
  case LQR::FactorStatus::INVALID_DELTA:
    return "INVALID_DELTA";
  case LQR::FactorStatus::F_FACTORIZATION_FAILURE:
    return "F_FACTORIZATION_FAILURE";
  case LQR::FactorStatus::G_FACTORIZATION_FAILURE:
    return "G_FACTORIZATION_FAILURE";
  }
  return "UNKNOWN";
}

void set_residual_counter(benchmark::State &state, LQRProblem &problem,
                          LQR::Workspace &workspace) {
  auto input = problem.input();
  auto lqr = LQR(input, workspace);
  const auto status = lqr.factor_with_status();
  if (status != LQR::FactorStatus::SUCCESS) {
    state.SkipWithError(
        (std::string("LQR factorization failed: ") + factor_status_name(status))
            .c_str());
    return;
  }

  auto output_storage = LQROutputStorage(problem.state_dim, problem.control_dim,
                                         problem.num_stages);
  auto output = output_storage.output();
  lqr.solve(output);

  state.counters["residual_norm"] =
      compute_residual_norm(problem, output_storage);
}

void Args(benchmark::Benchmark *benchmark) {
  for (const int num_stages : {16, 32, 64, 128}) {
    for (const int state_dim : {4, 6, 8, 16}) {
      for (const int control_dim : {1, 2, 3, 4}) {
        benchmark->Args({num_stages, state_dim, control_dim});
      }
    }
  }
}

void BM_LQRFactor(benchmark::State &state) {
  const int T = static_cast<int>(state.range(0));
  const int n = static_cast<int>(state.range(1));
  const int m = static_cast<int>(state.range(2));

  auto problem = LQRProblem(n, m, T);
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(n, m, T);
  auto lqr = LQR(input, workspace);

  for (auto _ : state) {
    const auto status = lqr.factor_with_status();
    benchmark::DoNotOptimize(static_cast<int>(status));
    if (status != LQR::FactorStatus::SUCCESS) {
      state.SkipWithError((std::string("LQR factorization failed: ") +
                           factor_status_name(status))
                              .c_str());
      break;
    }
  }

  set_residual_counter(state, problem, workspace);
  workspace.free(T);
}

void BM_LQRSolve(benchmark::State &state) {
  const int T = static_cast<int>(state.range(0));
  const int n = static_cast<int>(state.range(1));
  const int m = static_cast<int>(state.range(2));

  auto problem = LQRProblem(n, m, T);
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(n, m, T);
  auto lqr = LQR(input, workspace);
  if (lqr.factor_with_status() != LQR::FactorStatus::SUCCESS) {
    state.SkipWithError("LQR factorization failed");
    workspace.free(T);
    return;
  }

  auto output_storage = LQROutputStorage(n, m, T);
  auto output = output_storage.output();

  for (auto _ : state) {
    lqr.solve(output);
    benchmark::DoNotOptimize(output.x[0][0]);
  }

  state.counters["residual_norm"] =
      compute_residual_norm(problem, output_storage);
  workspace.free(T);
}

void BM_LQRFactorSolve(benchmark::State &state) {
  const int T = static_cast<int>(state.range(0));
  const int n = static_cast<int>(state.range(1));
  const int m = static_cast<int>(state.range(2));

  auto problem = LQRProblem(n, m, T);
  auto input = problem.input();
  LQR::Workspace workspace;
  workspace.reserve(n, m, T);
  auto lqr = LQR(input, workspace);
  auto output_storage = LQROutputStorage(n, m, T);
  auto output = output_storage.output();

  for (auto _ : state) {
    const auto status = lqr.factor_with_status();
    if (status != LQR::FactorStatus::SUCCESS) {
      state.SkipWithError((std::string("LQR factorization failed: ") +
                           factor_status_name(status))
                              .c_str());
      break;
    }
    lqr.solve(output);
    benchmark::DoNotOptimize(output.x[0][0]);
  }

  state.counters["residual_norm"] =
      compute_residual_norm(problem, output_storage);
  workspace.free(T);
}

BENCHMARK(BM_LQRFactor)->Apply(Args);
BENCHMARK(BM_LQRSolve)->Apply(Args);
BENCHMARK(BM_LQRFactorSolve)->Apply(Args);

} // namespace
} // namespace sip::optimal_control

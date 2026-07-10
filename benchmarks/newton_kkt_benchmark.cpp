#include "sip_optimal_control/helpers.hpp"

#include <Eigen/Dense>

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace sip::optimal_control {
namespace {

struct NewtonKKTProblem {
  int state_dim;
  int control_dim;
  int num_stages;
  int c_dim;
  int g_dim;
  int x_dim;
  int y_dim;
  int z_dim;
  int kkt_dim;
  double r1;

  Workspace workspace;
  Input input;
  std::unique_ptr<CallbackProvider> callback_provider;

  std::vector<double> w;
  std::vector<double> r2;
  std::vector<double> r3;
  std::vector<double> rhs;
  std::vector<double> solution;
  std::vector<double> kkt_times_solution;

  NewtonKKTProblem(int n, int m, int T)
      : state_dim(n), control_dim(m), num_stages(T), c_dim(std::max(1, n / 2)),
        g_dim(std::max(1, 2 * m)),
        x_dim(T * (n + m) + n), y_dim((c_dim + n) * (T + 1)),
        z_dim(g_dim * (T + 1)), kkt_dim(x_dim + y_dim + z_dim), r1(1e-8),
        input{
            .model_callback = [](const ModelCallbackInput &) {},
            .timeout_callback = []() -> bool { return false; },
            .dimensions =
                {
                    .num_stages = T,
                    .state_dim = n,
                    .control_dim = m,
                    .c_dim = c_dim,
                    .g_dim = g_dim,
                },
        },
        w(z_dim), r2(y_dim), r3(z_dim), rhs(kkt_dim), solution(kkt_dim),
        kkt_times_solution(kkt_dim) {
    workspace.reserve(state_dim, control_dim, num_stages, c_dim, g_dim);

    auto rng = std::mt19937(0);
    auto normal = std::normal_distribution<double>(0.0, 1.0);
    auto unit = std::uniform_real_distribution<double>(0.0, 1.0);

    fill_model_callback_output(rng, normal);
    fill_regularization_vectors(rng, unit);
    fill_vector(rhs, rng, normal);

    callback_provider = std::make_unique<CallbackProvider>(input, workspace);
  }

  ~NewtonKKTProblem() { workspace.free(num_stages); }

  NewtonKKTProblem(const NewtonKKTProblem &) = delete;
  auto operator=(const NewtonKKTProblem &) -> NewtonKKTProblem & = delete;

  bool factor() {
    return callback_provider->factor(w.data(), r1, r2.data(), r3.data());
  }

  void solve() { callback_provider->solve(rhs.data(), solution.data()); }

  auto residual_norm() -> double {
    std::fill(kkt_times_solution.begin(), kkt_times_solution.end(), 0.0);
    callback_provider->add_Kx_to_y(
        w.data(), r1, r2.data(), r3.data(), solution.data(),
        solution.data() + x_dim, solution.data() + x_dim + y_dim,
        kkt_times_solution.data(), kkt_times_solution.data() + x_dim,
        kkt_times_solution.data() + x_dim + y_dim);

    double squared_norm = 0.0;
    for (int i = 0; i < kkt_dim; ++i) {
      const double residual_i = rhs[i] - kkt_times_solution[i];
      squared_norm += residual_i * residual_i;
    }
    return std::sqrt(squared_norm);
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

  static void fill_vector(std::vector<double> &vector, std::mt19937 &rng,
                          std::normal_distribution<double> &normal) {
    for (double &value : vector) {
      value = normal(rng);
    }
  }

  static void fill_spd_matrix(double *data, std::mt19937 &rng,
                              std::normal_distribution<double> &normal,
                              const int dim, const double diagonal_shift) {
    const auto root = random_matrix(rng, normal, dim, dim);
    auto matrix = Eigen::Map<Eigen::MatrixXd>(data, dim, dim);
    matrix.noalias() = root.transpose() * root;
    matrix.diagonal().array() += diagonal_shift;
  }

  static void fill_matrix(double *data, std::mt19937 &rng,
                          std::normal_distribution<double> &normal,
                          const int rows, const int cols,
                          const double scale) {
    auto matrix = Eigen::Map<Eigen::MatrixXd>(data, rows, cols);
    matrix.noalias() = scale * random_matrix(rng, normal, rows, cols);
  }

  static void fill_zero(double *data, const int size) {
    std::fill_n(data, size, 0.0);
  }

  static auto positive_log_uniform(std::mt19937 &rng,
                                   std::uniform_real_distribution<double> &unit,
                                   const double min_value,
                                   const double max_value) -> double {
    const double log_min = std::log(min_value);
    const double log_max = std::log(max_value);
    return std::exp(log_min + (log_max - log_min) * unit(rng));
  }

  void fill_model_callback_output(std::mt19937 &rng,
                                  std::normal_distribution<double> &normal) {
    auto &mco = workspace.model_callback_output;
    mco.f = 0.0;

    for (int i = 0; i < num_stages; ++i) {
      fill_zero(mco.df_dx[i], state_dim);
      fill_zero(mco.df_du[i], control_dim);
      fill_zero(mco.dyn_res[i], state_dim);
      fill_matrix(mco.ddyn_dx[i], rng, normal, state_dim, state_dim, 0.05);
      Eigen::Map<Eigen::MatrixXd>(mco.ddyn_dx[i], state_dim, state_dim)
          .diagonal()
          .array() += 1.0;
      fill_matrix(mco.ddyn_du[i], rng, normal, state_dim, control_dim, 0.1);
      fill_zero(mco.c[i], c_dim);
      fill_matrix(mco.dc_dx[i], rng, normal, c_dim, state_dim, 0.1);
      fill_matrix(mco.dc_du[i], rng, normal, c_dim, control_dim, 0.1);
      fill_zero(mco.g[i], g_dim);
      fill_matrix(mco.dg_dx[i], rng, normal, g_dim, state_dim, 0.1);
      fill_matrix(mco.dg_du[i], rng, normal, g_dim, control_dim, 0.1);
      fill_spd_matrix(mco.d2L_dx2[i], rng, normal, state_dim, 1e-3);
      fill_matrix(mco.d2L_dxdu[i], rng, normal, state_dim, control_dim, 0.01);
      fill_spd_matrix(mco.d2L_du2[i], rng, normal, control_dim, 1.0);
    }

    fill_zero(mco.df_dx[num_stages], state_dim);
    fill_zero(mco.dyn_res[num_stages], state_dim);
    fill_zero(mco.c[num_stages], c_dim);
    fill_matrix(mco.dc_dx[num_stages], rng, normal, c_dim, state_dim, 0.1);
    fill_zero(mco.g[num_stages], g_dim);
    fill_matrix(mco.dg_dx[num_stages], rng, normal, g_dim, state_dim, 0.1);
    fill_spd_matrix(mco.d2L_dx2[num_stages], rng, normal, state_dim, 1e-3);
  }

  void fill_regularization_vectors(
      std::mt19937 &rng, std::uniform_real_distribution<double> &unit) {
    for (double &value : r2) {
      value = positive_log_uniform(rng, unit, 1e-3, 1e9);
    }
    for (double &value : w) {
      value = positive_log_uniform(rng, unit, 1e-2, 1e3);
    }
    for (double &value : r3) {
      value = positive_log_uniform(rng, unit, 1e-3, 1e1);
    }
  }
};

void Args(benchmark::Benchmark *benchmark) {
  for (const int num_stages : {16, 32, 64, 128}) {
    for (const int state_dim : {4, 6, 8, 16}) {
      for (const int control_dim : {1, 2, 3, 4}) {
        benchmark->Args({num_stages, state_dim, control_dim});
      }
    }
  }
}

void BM_NewtonKKTFactor(benchmark::State &state) {
  const int T = static_cast<int>(state.range(0));
  const int n = static_cast<int>(state.range(1));
  const int m = static_cast<int>(state.range(2));

  auto problem = NewtonKKTProblem(n, m, T);

  for (auto _ : state) {
    const bool success = problem.factor();
    benchmark::DoNotOptimize(static_cast<int>(success));
    if (!success) {
      state.SkipWithError("Newton-KKT factorization failed");
      break;
    }
  }

  if (!problem.factor()) {
    state.SkipWithError("Newton-KKT factorization failed");
    return;
  }
  problem.solve();
  state.counters["residual_norm"] = problem.residual_norm();
}

void BM_NewtonKKTSolve(benchmark::State &state) {
  const int T = static_cast<int>(state.range(0));
  const int n = static_cast<int>(state.range(1));
  const int m = static_cast<int>(state.range(2));

  auto problem = NewtonKKTProblem(n, m, T);
  if (!problem.factor()) {
    state.SkipWithError("Newton-KKT factorization failed");
    return;
  }

  for (auto _ : state) {
    problem.solve();
    benchmark::DoNotOptimize(problem.solution.data());
    benchmark::ClobberMemory();
  }

  state.counters["residual_norm"] = problem.residual_norm();
}

void BM_NewtonKKTFactorSolve(benchmark::State &state) {
  const int T = static_cast<int>(state.range(0));
  const int n = static_cast<int>(state.range(1));
  const int m = static_cast<int>(state.range(2));

  auto problem = NewtonKKTProblem(n, m, T);

  for (auto _ : state) {
    if (!problem.factor()) {
      state.SkipWithError("Newton-KKT factorization failed");
      break;
    }
    problem.solve();
    benchmark::DoNotOptimize(problem.solution.data());
    benchmark::ClobberMemory();
  }

  state.counters["residual_norm"] = problem.residual_norm();
}

void BM_NewtonKKTResidual(benchmark::State &state) {
  const int T = static_cast<int>(state.range(0));
  const int n = static_cast<int>(state.range(1));
  const int m = static_cast<int>(state.range(2));

  auto problem = NewtonKKTProblem(n, m, T);
  if (!problem.factor()) {
    state.SkipWithError("Newton-KKT factorization failed");
    return;
  }
  problem.solve();

  double residual_norm = 0.0;
  for (auto _ : state) {
    residual_norm = problem.residual_norm();
    benchmark::DoNotOptimize(residual_norm);
  }

  state.counters["residual_norm"] = residual_norm;
}

BENCHMARK(BM_NewtonKKTFactor)->Apply(Args);
BENCHMARK(BM_NewtonKKTSolve)->Apply(Args);
BENCHMARK(BM_NewtonKKTFactorSolve)->Apply(Args);
BENCHMARK(BM_NewtonKKTResidual)->Apply(Args);

} // namespace
} // namespace sip::optimal_control

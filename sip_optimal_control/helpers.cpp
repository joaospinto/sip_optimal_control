#define EIGEN_NO_MALLOC

#include "sip_optimal_control/helpers.hpp"

#include <Eigen/Dense>

#include <cassert>

namespace sip::optimal_control {

CallbackProvider::CallbackProvider(const Input &input, Workspace &workspace)
    : input_(input), workspace_(workspace),
      lqr_input_{.Q = nullptr,
                 .M = nullptr,
                 .R = nullptr,
                 .q = nullptr,
                 .r = nullptr,
                 .A = nullptr,
                 .B = nullptr,
                 .c = nullptr,
                 .delta = nullptr,
                 .dimensions = input.dimensions,
                 .topology = input.topology},
      lqr_solver_(lqr_input_, workspace.lqr_workspace),
      input_is_valid_(validate_input(input.dimensions, input.topology) ==
                      InputValidationStatus::SUCCESS) {}

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

auto state_dim(const Input &input, const int node) -> int {
  return input.dimensions.get_state_dim(node);
}

auto control_dim(const Input &input, const int edge) -> int {
  return input.dimensions.get_control_dim(edge);
}

auto c_dim(const Input &input, const int node) -> int {
  return input.dimensions.get_c_dim(node);
}

auto g_dim(const Input &input, const int node) -> int {
  return input.dimensions.get_g_dim(node);
}

void set_row_scaled(Eigen::Ref<Eigen::MatrixXd> result,
                    const Eigen::Ref<const Eigen::VectorXd> &weights,
                    const Eigen::Ref<const Eigen::MatrixXd> &matrix) {
  result.array() = matrix.array().colwise() * weights.array();
}

void add_weighted_control_jacobian_products(
    Eigen::Ref<Eigen::MatrixXd> M, Eigen::Ref<Eigen::MatrixXd> R,
    const Eigen::Ref<const Eigen::MatrixXd> &J_x,
    const Eigen::Ref<const Eigen::MatrixXd> &J_u,
    const Eigen::Ref<const Eigen::VectorXd> &weights) {
  for (int constraint = 0; constraint < weights.size(); ++constraint) {
    const double weight = weights(constraint);

    for (int col = 0; col < M.cols(); ++col) {
      const double weighted_j_u_col = weight * J_u(constraint, col);
      if (weighted_j_u_col == 0.0) {
        continue;
      }
      for (int row = 0; row < M.rows(); ++row) {
        const double j_x = J_x(constraint, row);
        if (j_x == 0.0) {
          continue;
        }
        M(row, col) += j_x * weighted_j_u_col;
      }
    }

    for (int col = 0; col < R.cols(); ++col) {
      const double weighted_j_u_col = weight * J_u(constraint, col);
      if (weighted_j_u_col == 0.0) {
        continue;
      }
      for (int row = col; row < R.rows(); ++row) {
        const double j_u = J_u(constraint, row);
        if (j_u == 0.0) {
          continue;
        }
        R(row, col) += weighted_j_u_col * j_u;
      }
    }
  }
}

void add_weighted_state_jacobian_product(
    Eigen::Ref<Eigen::MatrixXd> Q, const Eigen::Ref<const Eigen::MatrixXd> &J_x,
    const Eigen::Ref<const Eigen::VectorXd> &weights) {
  for (int constraint = 0; constraint < weights.size(); ++constraint) {
    const double weight = weights(constraint);
    for (int col = 0; col < Q.cols(); ++col) {
      const double weighted_j_x_col = weight * J_x(constraint, col);
      if (weighted_j_x_col == 0.0) {
        continue;
      }
      for (int row = col; row < Q.rows(); ++row) {
        const double j_x = J_x(constraint, row);
        if (j_x == 0.0) {
          continue;
        }
        Q(row, col) += weighted_j_x_col * j_x;
      }
    }
  }
}

void subtract_weighted_jacobian_rhs(
    Eigen::Ref<Eigen::VectorXd> result,
    const Eigen::Ref<const Eigen::MatrixXd> &jacobian,
    const Eigen::Ref<const Eigen::VectorXd> &weights,
    const Eigen::Ref<const Eigen::VectorXd> &rhs) {
  for (int constraint = 0; constraint < weights.size(); ++constraint) {
    const double weighted_rhs = weights(constraint) * rhs(constraint);
    for (int col = 0; col < jacobian.cols(); ++col) {
      const double jacobian_entry = jacobian(constraint, col);
      if (jacobian_entry == 0.0) {
        continue;
      }
      result(col) -= jacobian_entry * weighted_rhs;
    }
  }
}

void mirror_lower_to_upper(Eigen::Ref<Eigen::MatrixXd> matrix) {
  matrix.template triangularView<Eigen::StrictlyUpper>() =
      matrix.transpose().template triangularView<Eigen::StrictlyUpper>();
}

auto x_state_offset(const Workspace &workspace, const int node) -> int {
  return workspace.x_state_offsets[node];
}

auto x_control_offset(const Workspace &workspace, const int edge) -> int {
  return workspace.x_control_offsets[edge];
}

auto y_dyn_offset(const Workspace &workspace, const int node) -> int {
  return workspace.y_dyn_offsets[node];
}

auto y_c_offset(const Workspace &workspace, const int node) -> int {
  return workspace.y_c_offsets[node];
}

auto z_offset(const Workspace &workspace, const int node) -> int {
  return workspace.z_offsets[node];
}

} // namespace

void CallbackProvider::form_theta_jacobian() {
  const auto &dim = input_.dimensions;
  const auto &mco = workspace_.model_callback_output;
  auto &lqr_data = workspace_.regularized_lqr_data;

  const int p = dim.theta_dim;
  const int stagewise_kkt_dim = workspace_.stagewise_kkt_dim;
  auto J_theta = Eigen::Map<Eigen::MatrixXd>(lqr_data.theta_jacobian,
                                             stagewise_kkt_dim, p);
  J_theta.setZero();

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n = state_dim(input_, node);
    J_theta.block(x_state_offset(workspace_, node), 0, n, p) =
        Eigen::Map<const Eigen::MatrixXd>(mco.d2L_dxdtheta[node], n, p);
  }
  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int m = control_dim(input_, edge);
    J_theta.block(x_control_offset(workspace_, edge), 0, m, p) =
        Eigen::Map<const Eigen::MatrixXd>(mco.d2L_dudtheta[edge], m, p);
  }

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int c = c_dim(input_, node);
    J_theta.block(workspace_.stagewise_x_dim + y_c_offset(workspace_, node), 0,
                  c, p) =
        Eigen::Map<const Eigen::MatrixXd>(mco.dc_dtheta[node], c, p);
  }
  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int child = workspace_.lqr_workspace.edge_children[edge];
    const int n_child = state_dim(input_, child);
    J_theta.block(workspace_.stagewise_x_dim + y_dyn_offset(workspace_, child),
                  0, n_child, p) =
        Eigen::Map<const Eigen::MatrixXd>(mco.ddyn_dtheta[edge], n_child, p);
  }

  const int z_row = workspace_.stagewise_x_dim + workspace_.y_dim;
  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int g = g_dim(input_, node);
    J_theta.block(z_row + z_offset(workspace_, node), 0, g, p) =
        Eigen::Map<const Eigen::MatrixXd>(mco.dg_dtheta[node], g, p);
  }
}

bool CallbackProvider::factor(const double *w, const double r1,
                              const double *r2, const double *r3) {
  if (!input_is_valid_) {
    return false;
  }

  const auto &mco = workspace_.model_callback_output;
  auto &lqr_data = workspace_.regularized_lqr_data;

  int w_offset = 0;
  int y_offset = 0;

  for (int i = 0; i < input_.topology.num_nodes(); ++i) {
    const int n_i = input_.dimensions.get_state_dim(i);
    const int c_i = input_.dimensions.get_c_dim(i);
    for (int j = 0; j < n_i; ++j) {
      if (r2[y_offset] <= 0.0) {
        return false;
      }
      lqr_data.dyn_r2[i][j] = r2[y_offset++];
    }
    for (int j = 0; j < c_i; ++j) {
      if (r2[y_offset] <= 0.0) {
        return false;
      }
      lqr_data.c_r2_inv[i][j] = 1.0 / r2[y_offset++];
    }
  }

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n_i = input_.dimensions.get_state_dim(node);
    const int c_i = input_.dimensions.get_c_dim(node);
    const int g_i = input_.dimensions.get_g_dim(node);

    const auto Q_i =
        Eigen::Map<const Eigen::MatrixXd>(mco.d2L_dx2[node], n_i, n_i);

    const auto jac_x_c_i =
        Eigen::Map<const Eigen::MatrixXd>(mco.dc_dx[node], c_i, n_i);

    const auto jac_x_g_i =
        Eigen::Map<const Eigen::MatrixXd>(mco.dg_dx[node], g_i, n_i);

    auto mod_w_inv_i =
        Eigen::Map<Eigen::VectorXd>(lqr_data.mod_w_inv[node], g_i);
    const auto c_r2_inv_i =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.c_r2_inv[node], c_i);

    for (int j = 0; j < g_i; ++j) {
      const double w_reg = w[w_offset] + r3[w_offset];
      ++w_offset;
      if (w_reg <= 0.0) {
        return false;
      }
      mod_w_inv_i(j) = 1.0 / w_reg;
    }

    auto Q_i_mod = Eigen::Map<Eigen::MatrixXd>(lqr_data.Q_mod[node], n_i, n_i);

    Q_i_mod.template triangularView<Eigen::Lower>() =
        Q_i.template triangularView<Eigen::Lower>();
    Q_i_mod.diagonal().array() += r1;

    add_weighted_state_jacobian_product(Q_i_mod, jac_x_c_i, c_r2_inv_i);
    add_weighted_state_jacobian_product(Q_i_mod, jac_x_g_i, mod_w_inv_i);
    mirror_lower_to_upper(Q_i_mod);
  }

  for (int i = 0; i < input_.topology.num_edges; ++i) {
    const int parent = workspace_.lqr_workspace.edge_parents[i];
    const int n_i = input_.dimensions.get_state_dim(parent);
    const int m_i = input_.dimensions.get_control_dim(i);
    const int c_i = input_.dimensions.get_c_dim(parent);
    const int g_i = input_.dimensions.get_g_dim(parent);

    const auto M_i =
        Eigen::Map<const Eigen::MatrixXd>(mco.d2L_dxdu[i], n_i, m_i);

    const auto R_i =
        Eigen::Map<const Eigen::MatrixXd>(mco.d2L_du2[i], m_i, m_i);

    const auto jac_x_c_i =
        Eigen::Map<const Eigen::MatrixXd>(mco.dc_dx[parent], c_i, n_i);

    const auto jac_u_c_i =
        Eigen::Map<const Eigen::MatrixXd>(mco.dc_du[i], c_i, m_i);

    const auto jac_x_g_i =
        Eigen::Map<const Eigen::MatrixXd>(mco.dg_dx[parent], g_i, n_i);

    const auto jac_u_g_i =
        Eigen::Map<const Eigen::MatrixXd>(mco.dg_du[i], g_i, m_i);

    const auto mod_w_inv_i =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.mod_w_inv[parent], g_i);
    const auto c_r2_inv_i =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.c_r2_inv[parent], c_i);

    auto M_i_mod = Eigen::Map<Eigen::MatrixXd>(lqr_data.M_mod[i], n_i, m_i);

    auto R_i_mod = Eigen::Map<Eigen::MatrixXd>(lqr_data.R_mod[i], m_i, m_i);

    M_i_mod.noalias() = M_i;

    R_i_mod.template triangularView<Eigen::Lower>() =
        R_i.template triangularView<Eigen::Lower>();
    R_i_mod.diagonal().array() += r1;

    add_weighted_control_jacobian_products(M_i_mod, R_i_mod, jac_x_c_i,
                                           jac_u_c_i, c_r2_inv_i);
    add_weighted_control_jacobian_products(M_i_mod, R_i_mod, jac_x_g_i,
                                           jac_u_g_i, mod_w_inv_i);
    mirror_lower_to_upper(R_i_mod);
  }

  lqr_input_.Q = lqr_data.Q_mod;
  lqr_input_.M = lqr_data.M_mod;
  lqr_input_.R = lqr_data.R_mod;
  lqr_input_.A = mco.ddyn_dx;
  lqr_input_.B = mco.ddyn_du;
  lqr_input_.delta = lqr_data.dyn_r2;
  if (!lqr_solver_.factor()) {
    return false;
  }

  if (input_.dimensions.theta_dim == 0) {
    return true;
  }

  form_theta_jacobian();

  auto &theta_data = workspace_.regularized_lqr_data;
  const int p = input_.dimensions.theta_dim;
  const int stagewise_kkt_dim = workspace_.stagewise_kkt_dim;

  const auto J_theta = Eigen::Map<const Eigen::MatrixXd>(
      theta_data.theta_jacobian, stagewise_kkt_dim, p);
  auto K_inv_J_theta = Eigen::Map<Eigen::MatrixXd>(theta_data.theta_solution,
                                                   stagewise_kkt_dim, p);

  solve_stagewise_kkt_matrix(J_theta.data(), K_inv_J_theta.data(), p);

  const auto H_theta_theta = Eigen::Map<const Eigen::MatrixXd>(
      mco.d2L_dtheta2, input_.dimensions.theta_dim,
      input_.dimensions.theta_dim);
  auto S_theta = Eigen::Map<Eigen::MatrixXd>(theta_data.theta_schur, p, p);
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
  const int x_dim = workspace_.stagewise_x_dim;
  const int y_dim = workspace_.y_dim;
  const int stagewise_kkt_dim = workspace_.stagewise_kkt_dim;
  assert(num_rhs <= (dim.theta_dim > 0 ? dim.theta_dim : 1));

  if (num_rhs > 1) {
    auto &mco = workspace_.model_callback_output;
    auto &lqr_data = workspace_.regularized_lqr_data;

    for (int node = 0; node < input_.topology.num_nodes(); ++node) {
      const int n = state_dim(input_, node);
      const int c = c_dim(input_, node);
      const int g = g_dim(input_, node);
      const auto b_x = matrix_block(b, x_state_offset(workspace_, node), n,
                                    num_rhs, stagewise_kkt_dim);
      const auto b_y_c = matrix_block(b, x_dim + y_c_offset(workspace_, node),
                                      c, num_rhs, stagewise_kkt_dim);
      const auto b_z =
          matrix_block(b, x_dim + y_dim + z_offset(workspace_, node), g,
                       num_rhs, stagewise_kkt_dim);
      const auto jac_x_c =
          Eigen::Map<const Eigen::MatrixXd>(mco.dc_dx[node], c, n);
      const auto jac_x_g =
          Eigen::Map<const Eigen::MatrixXd>(mco.dg_dx[node], g, n);
      const auto c_r2_inv =
          Eigen::Map<const Eigen::VectorXd>(lqr_data.c_r2_inv[node], c);
      const auto mod_w_inv =
          Eigen::Map<const Eigen::VectorXd>(lqr_data.mod_w_inv[node], g);
      auto v_node = matrix_block(sol, x_dim + y_dyn_offset(workspace_, node), n,
                                 num_rhs, stagewise_kkt_dim);
      auto weighted_c = matrix_block(sol, x_dim + y_c_offset(workspace_, node),
                                     c, num_rhs, stagewise_kkt_dim);
      auto weighted_g =
          matrix_block(sol, x_dim + y_dim + z_offset(workspace_, node), g,
                       num_rhs, stagewise_kkt_dim);

      v_node.noalias() = -b_x;
      set_row_scaled(weighted_c, c_r2_inv, b_y_c);
      v_node.noalias() -= jac_x_c.transpose() * weighted_c;
      set_row_scaled(weighted_g, mod_w_inv, b_z);
      v_node.noalias() -= jac_x_g.transpose() * weighted_g;
    }

    for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
      const int parent = workspace_.lqr_workspace.edge_parents[edge];
      const int m = control_dim(input_, edge);
      const int c = c_dim(input_, parent);
      const int g = g_dim(input_, parent);
      const auto b_u = matrix_block(b, x_control_offset(workspace_, edge), m,
                                    num_rhs, stagewise_kkt_dim);
      const auto jac_u_c =
          Eigen::Map<const Eigen::MatrixXd>(mco.dc_du[edge], c, m);
      const auto jac_u_g =
          Eigen::Map<const Eigen::MatrixXd>(mco.dg_du[edge], g, m);
      const auto weighted_c =
          matrix_block(sol, x_dim + y_c_offset(workspace_, parent), c, num_rhs,
                       stagewise_kkt_dim);
      const auto weighted_g =
          matrix_block(sol, x_dim + y_dim + z_offset(workspace_, parent), g,
                       num_rhs, stagewise_kkt_dim);
      auto h_edge = matrix_block(sol, x_control_offset(workspace_, edge), m,
                                 num_rhs, stagewise_kkt_dim);

      h_edge.noalias() = -b_u;
      h_edge.noalias() -= jac_u_c.transpose() * weighted_c;
      h_edge.noalias() -= jac_u_g.transpose() * weighted_g;
    }

    auto f_scratch =
        matrix_block(lqr_data.stagewise_scratch, 0,
                     dim.max_state_dim(input_.topology.num_nodes()), num_rhs,
                     dim.max_state_dim(input_.topology.num_nodes()));
    auto g_scratch =
        matrix_block(lqr_data.stagewise_scratch,
                     dim.max_state_dim(input_.topology.num_nodes()) * num_rhs,
                     dim.max_state_dim(input_.topology.num_nodes()), num_rhs,
                     dim.max_state_dim(input_.topology.num_nodes()));

    for (int order = 0; order < input_.topology.num_nodes(); ++order) {
      const int node = workspace_.lqr_workspace.postorder_nodes[order];
      const int node_dim = state_dim(input_, node);
      auto v_node = matrix_block(sol, x_dim + y_dyn_offset(workspace_, node),
                                 node_dim, num_rhs, stagewise_kkt_dim);

      for (int child_index = workspace_.lqr_workspace.child_offsets[node];
           child_index < workspace_.lqr_workspace.child_offsets[node + 1];
           ++child_index) {
        const int edge = workspace_.lqr_workspace.child_edges[child_index];
        const int child = workspace_.lqr_workspace.edge_children[edge];
        const int child_dim = state_dim(input_, child);
        const int m = control_dim(input_, edge);

        const auto A_edge = Eigen::Map<const Eigen::MatrixXd>(
            mco.ddyn_dx[edge], child_dim, node_dim);
        const auto B_edge =
            Eigen::Map<const Eigen::MatrixXd>(mco.ddyn_du[edge], child_dim, m);
        const auto delta_child = Eigen::Map<const Eigen::VectorXd>(
            lqr_data.dyn_r2[child], child_dim);
        const auto W_edge = Eigen::Map<const Eigen::MatrixXd>(
            workspace_.lqr_workspace.W[edge], child_dim, child_dim);
        const auto G_edge_factor = Eigen::Map<const Eigen::MatrixXd>(
            workspace_.lqr_workspace.G_factor[edge], m, m);
        const auto K_edge = Eigen::Map<const Eigen::MatrixXd>(
            workspace_.lqr_workspace.K[edge], m, node_dim);
        const auto b_y_dyn_child =
            matrix_block(b, x_dim + y_dyn_offset(workspace_, child), child_dim,
                         num_rhs, stagewise_kkt_dim);
        const auto v_child =
            matrix_block(sol, x_dim + y_dyn_offset(workspace_, child),
                         child_dim, num_rhs, stagewise_kkt_dim);
        auto h_edge = matrix_block(sol, x_control_offset(workspace_, edge), m,
                                   num_rhs, stagewise_kkt_dim);
        auto f_child = f_scratch.block(0, 0, child_dim, num_rhs);
        auto g_child = g_scratch.block(0, 0, child_dim, num_rhs);

        f_child.array() = v_child.array().colwise() * delta_child.array();
        f_child.noalias() += b_y_dyn_child;
        g_child.noalias() = v_child;
        g_child.noalias() -= W_edge * f_child;

        h_edge.noalias() += B_edge.transpose() * g_child;
        v_node.noalias() += A_edge.transpose() * g_child;
        v_node.noalias() += K_edge.transpose() * h_edge;

        G_edge_factor.template triangularView<Eigen::Lower>().solveInPlace(
            h_edge);
        G_edge_factor.transpose()
            .template triangularView<Eigen::Upper>()
            .solveInPlace(h_edge);
        h_edge *= -1.0;
      }
    }

    const int root = workspace_.lqr_workspace.preorder_nodes[0];
    const int root_dim = state_dim(input_, root);
    const auto b_y_dyn_root =
        matrix_block(b, x_dim + y_dyn_offset(workspace_, root), root_dim,
                     num_rhs, stagewise_kkt_dim);
    const auto delta_root =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.dyn_r2[root], root_dim);
    const auto V_root = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.lqr_workspace.V[root], root_dim, root_dim);
    const auto F_root_factor = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.lqr_workspace.F_factor[root], root_dim, root_dim);
    const auto sqrt_delta_root = Eigen::Map<const Eigen::VectorXd>(
        workspace_.lqr_workspace.sqrt_delta[root], root_dim);
    const auto sqrt_delta_root_inv = Eigen::Map<const Eigen::VectorXd>(
        workspace_.lqr_workspace.sqrt_delta_inv[root], root_dim);
    auto x_root = matrix_block(sol, x_state_offset(workspace_, root), root_dim,
                               num_rhs, stagewise_kkt_dim);
    auto y_root = matrix_block(sol, x_dim + y_dyn_offset(workspace_, root),
                               root_dim, num_rhs, stagewise_kkt_dim);

    x_root.array() = y_root.array().colwise() * delta_root.array();
    x_root.noalias() += b_y_dyn_root;
    x_root.array().colwise() *= sqrt_delta_root_inv.array();
    F_root_factor.template triangularView<Eigen::Lower>().solveInPlace(x_root);
    F_root_factor.transpose()
        .template triangularView<Eigen::Upper>()
        .solveInPlace(x_root);
    x_root.array().colwise() *= sqrt_delta_root.array();
    x_root *= -1.0;
    y_root.noalias() += V_root * x_root;

    for (int order = 0; order < input_.topology.num_nodes(); ++order) {
      const int node = workspace_.lqr_workspace.preorder_nodes[order];
      const int node_dim = state_dim(input_, node);
      const auto x_node = matrix_block(sol, x_state_offset(workspace_, node),
                                       node_dim, num_rhs, stagewise_kkt_dim);

      for (int child_index = workspace_.lqr_workspace.child_offsets[node];
           child_index < workspace_.lqr_workspace.child_offsets[node + 1];
           ++child_index) {
        const int edge = workspace_.lqr_workspace.child_edges[child_index];
        const int child = workspace_.lqr_workspace.edge_children[edge];
        const int child_dim = state_dim(input_, child);
        const int m = control_dim(input_, edge);

        const auto A_edge = Eigen::Map<const Eigen::MatrixXd>(
            mco.ddyn_dx[edge], child_dim, node_dim);
        const auto B_edge =
            Eigen::Map<const Eigen::MatrixXd>(mco.ddyn_du[edge], child_dim, m);
        const auto K_edge = Eigen::Map<const Eigen::MatrixXd>(
            workspace_.lqr_workspace.K[edge], m, node_dim);
        const auto V_child = Eigen::Map<const Eigen::MatrixXd>(
            workspace_.lqr_workspace.V[child], child_dim, child_dim);
        const auto F_child_factor = Eigen::Map<const Eigen::MatrixXd>(
            workspace_.lqr_workspace.F_factor[child], child_dim, child_dim);
        const auto sqrt_delta_child = Eigen::Map<const Eigen::VectorXd>(
            workspace_.lqr_workspace.sqrt_delta[child], child_dim);
        const auto sqrt_delta_child_inv = Eigen::Map<const Eigen::VectorXd>(
            workspace_.lqr_workspace.sqrt_delta_inv[child], child_dim);
        const auto delta_child = Eigen::Map<const Eigen::VectorXd>(
            lqr_data.dyn_r2[child], child_dim);
        const auto b_y_dyn_child =
            matrix_block(b, x_dim + y_dyn_offset(workspace_, child), child_dim,
                         num_rhs, stagewise_kkt_dim);
        auto u_edge = matrix_block(sol, x_control_offset(workspace_, edge), m,
                                   num_rhs, stagewise_kkt_dim);
        auto x_child = matrix_block(sol, x_state_offset(workspace_, child),
                                    child_dim, num_rhs, stagewise_kkt_dim);
        auto y_child =
            matrix_block(sol, x_dim + y_dyn_offset(workspace_, child),
                         child_dim, num_rhs, stagewise_kkt_dim);

        u_edge.noalias() += K_edge * x_node;
        x_child.noalias() = -b_y_dyn_child;
        x_child.array() -= y_child.array().colwise() * delta_child.array();
        x_child.noalias() += A_edge * x_node;
        x_child.noalias() += B_edge * u_edge;
        x_child.array().colwise() *= sqrt_delta_child_inv.array();
        F_child_factor.template triangularView<Eigen::Lower>().solveInPlace(
            x_child);
        F_child_factor.transpose()
            .template triangularView<Eigen::Upper>()
            .solveInPlace(x_child);
        x_child.array().colwise() *= sqrt_delta_child.array();

        y_child.noalias() += V_child * x_child;
      }
    }

    for (int node = 0; node < input_.topology.num_nodes(); ++node) {
      const int n = state_dim(input_, node);
      const int c = c_dim(input_, node);
      const int g = g_dim(input_, node);
      const auto x_node = matrix_block(sol, x_state_offset(workspace_, node), n,
                                       num_rhs, stagewise_kkt_dim);
      const auto b_y_c = matrix_block(b, x_dim + y_c_offset(workspace_, node),
                                      c, num_rhs, stagewise_kkt_dim);
      const auto b_z =
          matrix_block(b, x_dim + y_dim + z_offset(workspace_, node), g,
                       num_rhs, stagewise_kkt_dim);
      const auto jac_x_c =
          Eigen::Map<const Eigen::MatrixXd>(mco.dc_dx[node], c, n);
      const auto jac_x_g =
          Eigen::Map<const Eigen::MatrixXd>(mco.dg_dx[node], g, n);
      const auto c_r2_inv =
          Eigen::Map<const Eigen::VectorXd>(lqr_data.c_r2_inv[node], c);
      const auto mod_w_inv =
          Eigen::Map<const Eigen::VectorXd>(lqr_data.mod_w_inv[node], g);
      auto y_c = matrix_block(sol, x_dim + y_c_offset(workspace_, node), c,
                              num_rhs, stagewise_kkt_dim);
      auto z = matrix_block(sol, x_dim + y_dim + z_offset(workspace_, node), g,
                            num_rhs, stagewise_kkt_dim);

      y_c.noalias() = jac_x_c * x_node;
      z.noalias() = jac_x_g * x_node;
      for (int child_index = workspace_.lqr_workspace.child_offsets[node];
           child_index < workspace_.lqr_workspace.child_offsets[node + 1];
           ++child_index) {
        const int edge = workspace_.lqr_workspace.child_edges[child_index];
        const int m = control_dim(input_, edge);
        const auto u_edge =
            matrix_block(sol, x_control_offset(workspace_, edge), m, num_rhs,
                         stagewise_kkt_dim);
        const auto jac_u_c =
            Eigen::Map<const Eigen::MatrixXd>(mco.dc_du[edge], c, m);
        const auto jac_u_g =
            Eigen::Map<const Eigen::MatrixXd>(mco.dg_du[edge], g, m);
        y_c.noalias() += jac_u_c * u_edge;
        z.noalias() += jac_u_g * u_edge;
      }
      y_c.noalias() -= b_y_c;
      set_row_scaled(y_c, c_r2_inv, y_c);
      z.noalias() -= b_z;
      set_row_scaled(z, mod_w_inv, z);
    }
    return;
  }

  auto &mco = workspace_.model_callback_output;
  auto &lqr_data = workspace_.regularized_lqr_data;

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n = state_dim(input_, node);
    const int c = c_dim(input_, node);
    const int g = g_dim(input_, node);
    const auto b_x = Eigen::Map<const Eigen::VectorXd>(
        b + x_state_offset(workspace_, node), n);
    const auto b_y_c = Eigen::Map<const Eigen::VectorXd>(
        b + x_dim + y_c_offset(workspace_, node), c);
    const auto b_z = Eigen::Map<const Eigen::VectorXd>(
        b + x_dim + y_dim + z_offset(workspace_, node), g);
    const auto jac_x_c =
        Eigen::Map<const Eigen::MatrixXd>(mco.dc_dx[node], c, n);
    const auto jac_x_g =
        Eigen::Map<const Eigen::MatrixXd>(mco.dg_dx[node], g, n);
    const auto c_r2_inv =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.c_r2_inv[node], c);
    const auto mod_w_inv =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.mod_w_inv[node], g);
    auto q = Eigen::Map<Eigen::VectorXd>(lqr_data.q_mod[node], n);
    auto c_mod = Eigen::Map<Eigen::VectorXd>(lqr_data.c_mod[node], n);

    q.noalias() = -b_x;
    subtract_weighted_jacobian_rhs(q, jac_x_c, c_r2_inv, b_y_c);
    subtract_weighted_jacobian_rhs(q, jac_x_g, mod_w_inv, b_z);

    c_mod.noalias() = -Eigen::Map<const Eigen::VectorXd>(
        b + x_dim + y_dyn_offset(workspace_, node), n);
  }

  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int parent = workspace_.lqr_workspace.edge_parents[edge];
    const int n_parent = state_dim(input_, parent);
    const int m = control_dim(input_, edge);
    const int c = c_dim(input_, parent);
    const int g = g_dim(input_, parent);
    const auto b_u = Eigen::Map<const Eigen::VectorXd>(
        b + x_control_offset(workspace_, edge), m);
    const auto b_y_c = Eigen::Map<const Eigen::VectorXd>(
        b + x_dim + y_c_offset(workspace_, parent), c);
    const auto b_z = Eigen::Map<const Eigen::VectorXd>(
        b + x_dim + y_dim + z_offset(workspace_, parent), g);
    const auto jac_u_c =
        Eigen::Map<const Eigen::MatrixXd>(mco.dc_du[edge], c, m);
    const auto jac_u_g =
        Eigen::Map<const Eigen::MatrixXd>(mco.dg_du[edge], g, m);
    const auto c_r2_inv =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.c_r2_inv[parent], c);
    const auto mod_w_inv =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.mod_w_inv[parent], g);
    auto r = Eigen::Map<Eigen::VectorXd>(lqr_data.r_mod[edge], m);
    (void)n_parent;

    r.noalias() = -b_u;
    subtract_weighted_jacobian_rhs(r, jac_u_c, c_r2_inv, b_y_c);
    subtract_weighted_jacobian_rhs(r, jac_u_g, mod_w_inv, b_z);
  }

  lqr_input_.q = lqr_data.q_mod;
  lqr_input_.r = lqr_data.r_mod;
  lqr_input_.c = lqr_data.c_mod;
  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    workspace_.lqr_output.x[node] = sol + x_state_offset(workspace_, node);
    workspace_.lqr_output.y[node] =
        sol + x_dim + y_dyn_offset(workspace_, node);
  }
  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    workspace_.lqr_output.u[edge] = sol + x_control_offset(workspace_, edge);
  }

  lqr_solver_.solve(workspace_.lqr_output);

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n = state_dim(input_, node);
    const int c = c_dim(input_, node);
    const int g = g_dim(input_, node);
    const auto x_node = Eigen::Map<const Eigen::VectorXd>(
        sol + x_state_offset(workspace_, node), n);
    const auto b_y_c = Eigen::Map<const Eigen::VectorXd>(
        b + x_dim + y_c_offset(workspace_, node), c);
    const auto b_z = Eigen::Map<const Eigen::VectorXd>(
        b + x_dim + y_dim + z_offset(workspace_, node), g);
    const auto jac_x_c =
        Eigen::Map<const Eigen::MatrixXd>(mco.dc_dx[node], c, n);
    const auto jac_x_g =
        Eigen::Map<const Eigen::MatrixXd>(mco.dg_dx[node], g, n);
    const auto c_r2_inv =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.c_r2_inv[node], c);
    const auto mod_w_inv =
        Eigen::Map<const Eigen::VectorXd>(lqr_data.mod_w_inv[node], g);
    auto y_c = Eigen::Map<Eigen::VectorXd>(
        sol + x_dim + y_c_offset(workspace_, node), c);
    auto z = Eigen::Map<Eigen::VectorXd>(
        sol + x_dim + y_dim + z_offset(workspace_, node), g);

    y_c.noalias() = jac_x_c * x_node;
    z.noalias() = jac_x_g * x_node;
    for (int child_index = workspace_.lqr_workspace.child_offsets[node];
         child_index < workspace_.lqr_workspace.child_offsets[node + 1];
         ++child_index) {
      const int edge = workspace_.lqr_workspace.child_edges[child_index];
      const int m = control_dim(input_, edge);
      const auto u_edge = Eigen::Map<const Eigen::VectorXd>(
          sol + x_control_offset(workspace_, edge), m);
      const auto jac_u_c =
          Eigen::Map<const Eigen::MatrixXd>(mco.dc_du[edge], c, m);
      const auto jac_u_g =
          Eigen::Map<const Eigen::MatrixXd>(mco.dg_du[edge], g, m);
      y_c.noalias() += jac_u_c * u_edge;
      z.noalias() += jac_u_g * u_edge;
    }
    y_c.noalias() -= b_y_c;
    set_row_scaled(y_c, c_r2_inv, y_c);
    z.noalias() -= b_z;
    set_row_scaled(z, mod_w_inv, z);
  }
}

void CallbackProvider::solve(const double *b, double *sol) {
  if (input_.dimensions.theta_dim == 0) {
    solve_stagewise_kkt(b, sol);
    return;
  }

  auto &theta_data = workspace_.regularized_lqr_data;
  const int p = input_.dimensions.theta_dim;
  const int stagewise_kkt_dim = workspace_.stagewise_kkt_dim;

  const double *b_theta = b + workspace_.stagewise_x_dim;
  const double *b_y = b + workspace_.x_dim;
  const double *b_z = b_y + workspace_.y_dim;

  double *sol_theta = sol + workspace_.stagewise_x_dim;
  double *sol_y = sol + workspace_.x_dim;
  double *sol_z = sol_y + workspace_.y_dim;

  std::copy_n(b, workspace_.stagewise_x_dim, theta_data.theta_stagewise_rhs);
  std::copy_n(b_y, workspace_.y_dim,
              theta_data.theta_stagewise_rhs + workspace_.stagewise_x_dim);
  std::copy_n(b_z, workspace_.z_dim,
              theta_data.theta_stagewise_rhs + workspace_.stagewise_x_dim +
                  workspace_.y_dim);

  solve_stagewise_kkt(theta_data.theta_stagewise_rhs, sol);

  const auto J_theta = Eigen::Map<const Eigen::MatrixXd>(
      theta_data.theta_jacobian, stagewise_kkt_dim, p);
  const auto K_inv_b =
      Eigen::Map<const Eigen::VectorXd>(sol, stagewise_kkt_dim);
  auto theta_rhs = Eigen::Map<Eigen::VectorXd>(theta_data.theta_rhs, p);
  theta_rhs.noalias() = Eigen::Map<const Eigen::VectorXd>(b_theta, p);
  theta_rhs.noalias() -= J_theta.transpose() * K_inv_b;

  const auto S_theta_factor =
      Eigen::Map<const Eigen::MatrixXd>(theta_data.theta_schur_factor, p, p);
  S_theta_factor.template triangularView<Eigen::Lower>().solveInPlace(
      theta_rhs);
  S_theta_factor.transpose()
      .template triangularView<Eigen::Upper>()
      .solveInPlace(theta_rhs);

  const auto K_inv_J_theta = Eigen::Map<const Eigen::MatrixXd>(
      theta_data.theta_solution, stagewise_kkt_dim, p);
  auto stagewise_solution = Eigen::Map<Eigen::VectorXd>(sol, stagewise_kkt_dim);
  stagewise_solution.noalias() -= K_inv_J_theta * theta_rhs;

  const int stagewise_x_dim = workspace_.stagewise_x_dim;
  std::copy_backward(sol + stagewise_x_dim + workspace_.y_dim,
                     sol + stagewise_kkt_dim, sol_z + workspace_.z_dim);
  std::copy_backward(sol + stagewise_x_dim,
                     sol + stagewise_x_dim + workspace_.y_dim,
                     sol_y + workspace_.y_dim);
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

  const int x_dim = workspace_.x_dim;
  const int y_dim = workspace_.y_dim;
  const int z_dim = workspace_.z_dim;

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
  const auto &dim = input_.dimensions;
  const double *x_begin = x;
  double *y_begin = y;

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n = state_dim(input_, node);
    const auto Q = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dx2[node], n, n);
    const auto x_node = Eigen::Map<const Eigen::VectorXd>(
        x + x_state_offset(workspace_, node), n);
    auto y_node =
        Eigen::Map<Eigen::VectorXd>(y + x_state_offset(workspace_, node), n);
    y_node.noalias() += Q * x_node;
  }

  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int parent = workspace_.lqr_workspace.edge_parents[edge];
    const int n_parent = state_dim(input_, parent);
    const int m = control_dim(input_, edge);
    const auto M = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dxdu[edge], n_parent, m);
    const auto R = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_du2[edge], m, m);
    const auto x_parent = Eigen::Map<const Eigen::VectorXd>(
        x + x_state_offset(workspace_, parent), n_parent);
    const auto u_edge = Eigen::Map<const Eigen::VectorXd>(
        x + x_control_offset(workspace_, edge), m);
    auto y_parent = Eigen::Map<Eigen::VectorXd>(
        y + x_state_offset(workspace_, parent), n_parent);
    auto y_u =
        Eigen::Map<Eigen::VectorXd>(y + x_control_offset(workspace_, edge), m);
    y_parent.noalias() += M * u_edge;
    y_u.noalias() += M.transpose() * x_parent;
    y_u.noalias() += R * u_edge;
  }

  if (dim.theta_dim == 0) {
    return;
  }

  const int p = dim.theta_dim;
  const int stagewise_x_dim = workspace_.stagewise_x_dim;
  const auto theta =
      Eigen::Map<const Eigen::VectorXd>(x_begin + stagewise_x_dim, p);
  auto y_theta = Eigen::Map<Eigen::VectorXd>(y_begin + stagewise_x_dim, p);

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n = state_dim(input_, node);
    const auto H_x_theta = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dxdtheta[node], n, p);
    const auto x_node = Eigen::Map<const Eigen::VectorXd>(
        x_begin + x_state_offset(workspace_, node), n);
    auto y_node = Eigen::Map<Eigen::VectorXd>(
        y_begin + x_state_offset(workspace_, node), n);
    y_node.noalias() += H_x_theta * theta;
    y_theta.noalias() += H_x_theta.transpose() * x_node;
  }
  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int m = control_dim(input_, edge);
    const auto H_u_theta = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.d2L_dudtheta[edge], m, p);
    const auto u_edge = Eigen::Map<const Eigen::VectorXd>(
        x_begin + x_control_offset(workspace_, edge), m);
    auto y_u = Eigen::Map<Eigen::VectorXd>(
        y_begin + x_control_offset(workspace_, edge), m);
    y_u.noalias() += H_u_theta * theta;
    y_theta.noalias() += H_u_theta.transpose() * u_edge;
  }
  const auto H_theta_theta = Eigen::Map<const Eigen::MatrixXd>(
      workspace_.model_callback_output.d2L_dtheta2, p, p);
  y_theta.noalias() += H_theta_theta * theta;
}

void CallbackProvider::add_Cx_to_y(const double *x, double *y) {
  const auto &dim = input_.dimensions;
  const double *x_begin = x;
  double *y_begin = y;

  const int root = workspace_.lqr_workspace.preorder_nodes[0];
  const int n_root = state_dim(input_, root);
  const auto x_root = Eigen::Map<const Eigen::VectorXd>(
      x + x_state_offset(workspace_, root), n_root);
  auto y_root =
      Eigen::Map<Eigen::VectorXd>(y + y_dyn_offset(workspace_, root), n_root);
  y_root.noalias() -= x_root;

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n = state_dim(input_, node);
    const int c = c_dim(input_, node);
    const auto x_node = Eigen::Map<const Eigen::VectorXd>(
        x + x_state_offset(workspace_, node), n);
    const auto jac_x_c = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dx[node], c, n);
    auto y_c_node =
        Eigen::Map<Eigen::VectorXd>(y + y_c_offset(workspace_, node), c);
    y_c_node.noalias() += jac_x_c * x_node;
  }

  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int parent = workspace_.lqr_workspace.edge_parents[edge];
    const int child = workspace_.lqr_workspace.edge_children[edge];
    const int n_parent = state_dim(input_, parent);
    const int n_child = state_dim(input_, child);
    const int m = control_dim(input_, edge);
    const int c_parent = c_dim(input_, parent);
    const auto A = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dx[edge], n_child, n_parent);
    const auto B = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_du[edge], n_child, m);
    const auto jac_u_c = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_du[edge], c_parent, m);
    const auto x_parent = Eigen::Map<const Eigen::VectorXd>(
        x + x_state_offset(workspace_, parent), n_parent);
    const auto x_child = Eigen::Map<const Eigen::VectorXd>(
        x + x_state_offset(workspace_, child), n_child);
    const auto u_edge = Eigen::Map<const Eigen::VectorXd>(
        x + x_control_offset(workspace_, edge), m);
    auto y_dyn_child = Eigen::Map<Eigen::VectorXd>(
        y + y_dyn_offset(workspace_, child), n_child);
    auto y_c_parent = Eigen::Map<Eigen::VectorXd>(
        y + y_c_offset(workspace_, parent), c_parent);
    y_dyn_child.noalias() += A * x_parent;
    y_dyn_child.noalias() += B * u_edge;
    y_dyn_child.noalias() -= x_child;
    y_c_parent.noalias() += jac_u_c * u_edge;
  }

  if (dim.theta_dim == 0) {
    return;
  }

  const int p = dim.theta_dim;
  const auto theta = Eigen::Map<const Eigen::VectorXd>(
      x_begin + workspace_.stagewise_x_dim, p);
  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int c = c_dim(input_, node);
    auto y_c_node =
        Eigen::Map<Eigen::VectorXd>(y_begin + y_c_offset(workspace_, node), c);
    const auto c_theta = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dtheta[node], c, p);
    y_c_node.noalias() += c_theta * theta;
  }
  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int child = workspace_.lqr_workspace.edge_children[edge];
    const int n_child = state_dim(input_, child);
    auto y_dyn_child = Eigen::Map<Eigen::VectorXd>(
        y_begin + y_dyn_offset(workspace_, child), n_child);
    const auto dyn_theta = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dtheta[edge], n_child, p);
    y_dyn_child.noalias() += dyn_theta * theta;
  }
}

void CallbackProvider::add_CTx_to_y(const double *x, double *y) {
  const auto &dim = input_.dimensions;
  const double *x_begin = x;
  double *y_begin = y;

  const int root = workspace_.lqr_workspace.preorder_nodes[0];
  const int n_root = state_dim(input_, root);
  const auto dyn_root = Eigen::Map<const Eigen::VectorXd>(
      x + y_dyn_offset(workspace_, root), n_root);
  auto y_root =
      Eigen::Map<Eigen::VectorXd>(y + x_state_offset(workspace_, root), n_root);
  y_root.noalias() -= dyn_root;

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n = state_dim(input_, node);
    const int c = c_dim(input_, node);
    const auto c_vec =
        Eigen::Map<const Eigen::VectorXd>(x + y_c_offset(workspace_, node), c);
    const auto jac_x_c = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dx[node], c, n);
    auto y_state =
        Eigen::Map<Eigen::VectorXd>(y + x_state_offset(workspace_, node), n);
    y_state.noalias() += jac_x_c.transpose() * c_vec;
  }

  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int parent = workspace_.lqr_workspace.edge_parents[edge];
    const int child = workspace_.lqr_workspace.edge_children[edge];
    const int n_parent = state_dim(input_, parent);
    const int n_child = state_dim(input_, child);
    const int m = control_dim(input_, edge);
    const int c_parent = c_dim(input_, parent);
    const auto dyn_child = Eigen::Map<const Eigen::VectorXd>(
        x + y_dyn_offset(workspace_, child), n_child);
    const auto c_vec = Eigen::Map<const Eigen::VectorXd>(
        x + y_c_offset(workspace_, parent), c_parent);
    const auto A = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dx[edge], n_child, n_parent);
    const auto B = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_du[edge], n_child, m);
    const auto jac_u_c = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_du[edge], c_parent, m);
    auto y_parent = Eigen::Map<Eigen::VectorXd>(
        y + x_state_offset(workspace_, parent), n_parent);
    auto y_child = Eigen::Map<Eigen::VectorXd>(
        y + x_state_offset(workspace_, child), n_child);
    auto y_u =
        Eigen::Map<Eigen::VectorXd>(y + x_control_offset(workspace_, edge), m);
    y_parent.noalias() += A.transpose() * dyn_child;
    y_child.noalias() -= dyn_child;
    y_u.noalias() += B.transpose() * dyn_child;
    y_u.noalias() += jac_u_c.transpose() * c_vec;
  }

  if (dim.theta_dim == 0) {
    return;
  }

  const int p = dim.theta_dim;
  auto y_theta =
      Eigen::Map<Eigen::VectorXd>(y_begin + workspace_.stagewise_x_dim, p);
  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int c = c_dim(input_, node);
    const auto c_vec = Eigen::Map<const Eigen::VectorXd>(
        x_begin + y_c_offset(workspace_, node), c);
    const auto c_theta = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dc_dtheta[node], c, p);
    y_theta.noalias() += c_theta.transpose() * c_vec;
  }
  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int child = workspace_.lqr_workspace.edge_children[edge];
    const int n_child = state_dim(input_, child);
    const auto dyn_child = Eigen::Map<const Eigen::VectorXd>(
        x_begin + y_dyn_offset(workspace_, child), n_child);
    const auto dyn_theta = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.ddyn_dtheta[edge], n_child, p);
    y_theta.noalias() += dyn_theta.transpose() * dyn_child;
  }
}

void CallbackProvider::add_Gx_to_y(const double *x, double *y) {
  const auto &dim = input_.dimensions;
  const double *x_begin = x;
  double *y_begin = y;

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n = state_dim(input_, node);
    const int g = g_dim(input_, node);
    const auto jac_x_g = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[node], g, n);
    const auto x_node = Eigen::Map<const Eigen::VectorXd>(
        x + x_state_offset(workspace_, node), n);
    auto y_node =
        Eigen::Map<Eigen::VectorXd>(y + z_offset(workspace_, node), g);
    y_node.noalias() += jac_x_g * x_node;
  }
  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int parent = workspace_.lqr_workspace.edge_parents[edge];
    const int m = control_dim(input_, edge);
    const int g = g_dim(input_, parent);
    const auto jac_u_g = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_du[edge], g, m);
    const auto u_edge = Eigen::Map<const Eigen::VectorXd>(
        x + x_control_offset(workspace_, edge), m);
    auto y_parent =
        Eigen::Map<Eigen::VectorXd>(y + z_offset(workspace_, parent), g);
    y_parent.noalias() += jac_u_g * u_edge;
  }

  if (dim.theta_dim == 0) {
    return;
  }

  const int p = dim.theta_dim;
  const auto theta = Eigen::Map<const Eigen::VectorXd>(
      x_begin + workspace_.stagewise_x_dim, p);
  double *y_stage = y_begin;
  for (int i = 0; i < input_.topology.num_nodes(); ++i) {
    const int g_i = g_dim(input_, i);
    auto y_i = Eigen::Map<Eigen::VectorXd>(y_stage, g_i);
    y_stage += g_i;
    const auto g_theta_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dtheta[i], g_i, p);
    y_i.noalias() += g_theta_i * theta;
  }
}

void CallbackProvider::add_GTx_to_y(const double *x, double *y) {
  const auto &dim = input_.dimensions;
  const double *x_begin = x;
  double *y_begin = y;

  for (int node = 0; node < input_.topology.num_nodes(); ++node) {
    const int n = state_dim(input_, node);
    const int g = g_dim(input_, node);
    const auto x_node =
        Eigen::Map<const Eigen::VectorXd>(x + z_offset(workspace_, node), g);
    const auto jac_x_g = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dx[node], g, n);
    auto y_state =
        Eigen::Map<Eigen::VectorXd>(y + x_state_offset(workspace_, node), n);
    y_state.noalias() += jac_x_g.transpose() * x_node;
  }
  for (int edge = 0; edge < input_.topology.num_edges; ++edge) {
    const int parent = workspace_.lqr_workspace.edge_parents[edge];
    const int m = control_dim(input_, edge);
    const int g = g_dim(input_, parent);
    const auto x_parent =
        Eigen::Map<const Eigen::VectorXd>(x + z_offset(workspace_, parent), g);
    const auto jac_u_g = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_du[edge], g, m);
    auto y_u =
        Eigen::Map<Eigen::VectorXd>(y + x_control_offset(workspace_, edge), m);
    y_u.noalias() += jac_u_g.transpose() * x_parent;
  }

  if (dim.theta_dim == 0) {
    return;
  }

  const int p = dim.theta_dim;
  auto y_theta =
      Eigen::Map<Eigen::VectorXd>(y_begin + workspace_.stagewise_x_dim, p);
  const double *x_stage = x_begin;
  for (int i = 0; i < input_.topology.num_nodes(); ++i) {
    const int g_i = g_dim(input_, i);
    const auto x_i = Eigen::Map<const Eigen::VectorXd>(x_stage, g_i);
    x_stage += g_i;
    const auto g_theta_i = Eigen::Map<const Eigen::MatrixXd>(
        workspace_.model_callback_output.dg_dtheta[i], g_i, p);
    y_theta.noalias() += g_theta_i.transpose() * x_i;
  }
}

} // namespace sip::optimal_control

#undef EIGEN_NO_MALLOC

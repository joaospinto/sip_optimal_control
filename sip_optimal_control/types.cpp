#include "sip_optimal_control/types.hpp"

#include <cassert>

namespace sip::optimal_control {

void ModelCallbackInput::reserve(int num_stages) {
  states = new double *[num_stages + 1];
  controls = new double *[num_stages];
  costates = new double *[num_stages + 1];
  equality_constraint_multipliers = new double *[num_stages + 1];
  inequality_constraint_multipliers = new double *[num_stages + 1];
}

void ModelCallbackInput::free() {
  delete[] states;
  delete[] controls;
  delete[] costates;
  delete[] equality_constraint_multipliers;
  delete[] inequality_constraint_multipliers;
}

auto ModelCallbackInput::mem_assign(int num_stages, unsigned char *mem_ptr)
    -> int {
  int cum_size = 0;

  states = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  controls = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  costates = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  equality_constraint_multipliers =
      reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  inequality_constraint_multipliers =
      reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  assert(cum_size == ModelCallbackInput::num_bytes(num_stages));

  return cum_size;
}

void ModelCallbackOutput::reserve(int state_dim, int control_dim,
                                  int num_stages, int c_dim, int g_dim) {
  df_dx = new double *[num_stages + 1];
  df_du = new double *[num_stages];
  dyn_res = new double *[num_stages + 1];
  ddyn_dx = new double *[num_stages];
  ddyn_du = new double *[num_stages];
  c = new double *[num_stages + 1];
  dc_dx = new double *[num_stages + 1];
  dc_du = new double *[num_stages];
  g = new double *[num_stages + 1];
  dg_dx = new double *[num_stages + 1];
  dg_du = new double *[num_stages];
  d2L_dx2 = new double *[num_stages + 1];
  d2L_dxdu = new double *[num_stages];
  d2L_du2 = new double *[num_stages];

  for (int i = 0; i < num_stages; ++i) {
    df_dx[i] = new double[state_dim];
    df_du[i] = new double[control_dim];
    dyn_res[i] = new double[state_dim];
    ddyn_dx[i] = new double[state_dim];
    ddyn_du[i] = new double[control_dim];
    c[i] = new double[c_dim];
    dc_dx[i] = new double[c_dim * state_dim];
    dc_du[i] = new double[c_dim * control_dim];
    g[i] = new double[g_dim];
    dg_dx[i] = new double[g_dim * state_dim];
    dg_du[i] = new double[g_dim * control_dim];
    d2L_dx2[i] = new double[state_dim * state_dim];
    d2L_dxdu[i] = new double[state_dim * control_dim];
    d2L_du2[i] = new double[control_dim * control_dim];
  }

  df_dx[num_stages] = new double[state_dim];
  dyn_res[num_stages] = new double[state_dim];
  c[num_stages] = new double[c_dim];
  dc_dx[num_stages] = new double[c_dim * state_dim];
  g[num_stages] = new double[g_dim];
  dg_dx[num_stages] = new double[g_dim * state_dim];
  d2L_dx2[num_stages] = new double[state_dim * state_dim];
}

void ModelCallbackOutput::free(int num_stages) {
  for (int i = 0; i <= num_stages; ++i) {
    delete[] df_dx[i];
    delete[] df_du[i];
    delete[] dyn_res[i];
    delete[] ddyn_dx[i];
    delete[] ddyn_du[i];
    delete[] c[i];
    delete[] dc_dx[i];
    delete[] dc_du[i];
    delete[] g[i];
    delete[] dg_dx[i];
    delete[] dg_du[i];
    delete[] d2L_dx2[i];
    delete[] d2L_dxdu[i];
    delete[] d2L_du2[i];
  }

  delete[] df_dx[num_stages];
  delete[] dyn_res[num_stages];
  delete[] c[num_stages];
  delete[] dc_dx[num_stages];
  delete[] g[num_stages];
  delete[] dg_dx[num_stages];
  delete[] d2L_dx2[num_stages];

  delete[] df_dx;
  delete[] df_du;
  delete[] dyn_res;
  delete[] ddyn_dx;
  delete[] ddyn_du;
  delete[] c;
  delete[] dc_dx;
  delete[] dc_du;
  delete[] g;
  delete[] dg_dx;
  delete[] dg_du;
  delete[] d2L_dx2;
  delete[] d2L_dxdu;
  delete[] d2L_du2;
}

auto ModelCallbackOutput::mem_assign(int state_dim, int control_dim,
                                     int num_stages, int c_dim, int g_dim,
                                     unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  df_dx = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  df_du = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  dyn_res = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  ddyn_dx = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  ddyn_du = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  c = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  dc_dx = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  dc_du = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  g = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  dg_dx = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  dg_du = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  d2L_dx2 = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  d2L_dxdu = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  d2L_du2 = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  for (int i = 0; i < num_stages; ++i) {
    df_dx[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double *);

    df_du[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * sizeof(double *);

    dyn_res[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double *);

    ddyn_dx[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * state_dim * sizeof(double *);

    ddyn_du[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * control_dim * sizeof(double *);

    c[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += c_dim * sizeof(double *);

    dc_dx[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += c_dim * state_dim * sizeof(double *);

    dc_du[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += c_dim * control_dim * sizeof(double *);

    g[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += g_dim * sizeof(double *);

    dg_dx[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += g_dim * state_dim * sizeof(double *);

    dg_du[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += g_dim * control_dim * sizeof(double *);

    d2L_dx2[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * state_dim * sizeof(double *);

    d2L_dxdu[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * control_dim * sizeof(double *);

    d2L_du2[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * control_dim * sizeof(double *);
  }

  df_dx[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double *);

  dyn_res[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double *);

  c[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += c_dim * sizeof(double *);

  dc_dx[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += c_dim * state_dim * sizeof(double *);

  g[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += g_dim * sizeof(double *);

  dg_dx[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += g_dim * state_dim * sizeof(double *);

  d2L_dx2[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * state_dim * sizeof(double *);

  assert(cum_size == ModelCallbackOutput::num_bytes(state_dim, control_dim,
                                                    num_stages, c_dim, g_dim));

  return cum_size;
}

void Workspace::RegularizedLQRData::reserve(int state_dim, int control_dim,
                                            int num_stages, int g_dim) {

  mod_w_inv = new double *[num_stages + 1];
  Q_mod = new double *[num_stages + 1];
  M_mod = new double *[num_stages];
  R_mod = new double *[num_stages];
  q_mod = new double *[num_stages + 1];
  r_mod = new double *[num_stages];
  c_mod = new double *[num_stages + 1];

  for (int i = 0; i < num_stages; ++i) {
    mod_w_inv[i] = new double[g_dim];
    Q_mod[i] = new double[state_dim * state_dim];
    M_mod[i] = new double[state_dim * control_dim];
    R_mod[i] = new double[control_dim * control_dim];
    q_mod[i] = new double[state_dim];
    r_mod[i] = new double[control_dim];
    c_mod[i] = new double[state_dim];
  }

  mod_w_inv[num_stages] = new double[g_dim];
  Q_mod[num_stages] = new double[state_dim * state_dim];
  q_mod[num_stages] = new double[state_dim];
  c_mod[num_stages] = new double[state_dim];
}

void Workspace::RegularizedLQRData::free(int num_stages) {
  for (int i = 0; i < num_stages; ++i) {
    delete[] mod_w_inv[i];
    delete[] Q_mod[i];
    delete[] M_mod[i];
    delete[] R_mod[i];
    delete[] q_mod[i];
    delete[] r_mod[i];
    delete[] c_mod[i];
  }

  delete[] mod_w_inv[num_stages];
  delete[] Q_mod[num_stages];
  delete[] q_mod[num_stages];
  delete[] c_mod[num_stages];

  delete[] mod_w_inv;
  delete[] Q_mod;
  delete[] M_mod;
  delete[] R_mod;
  delete[] q_mod;
  delete[] r_mod;
  delete[] c_mod;
}

auto Workspace::RegularizedLQRData::mem_assign(int state_dim, int control_dim,
                                               int num_stages, int g_dim,
                                               unsigned char *mem_ptr) -> int {
  int cum_size = 0;

  mod_w_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  Q_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  M_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  R_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  q_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  r_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += num_stages * sizeof(double *);

  c_mod = reinterpret_cast<double **>(mem_ptr + cum_size);
  cum_size += (num_stages + 1) * sizeof(double *);

  for (int i = 0; i < num_stages; ++i) {
    mod_w_inv[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += g_dim * sizeof(double);

    Q_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * state_dim * sizeof(double);

    M_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * control_dim * sizeof(double);

    R_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * control_dim * sizeof(double);

    q_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double);

    r_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += control_dim * sizeof(double);

    c_mod[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += state_dim * sizeof(double);
  }

  mod_w_inv[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += g_dim * sizeof(double);

  Q_mod[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * state_dim * sizeof(double);

  q_mod[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  c_mod[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  assert(cum_size == Workspace::RegularizedLQRData::num_bytes(
                         state_dim, control_dim, num_stages, g_dim));

  return cum_size;
}

void Workspace::reserve(int state_dim, int control_dim, int num_stages,
                        int c_dim, int g_dim) {
  model_callback_output.reserve(state_dim, control_dim, num_stages, c_dim,
                                g_dim);
  model_callback_input.reserve(num_stages);

  const int x_dim = num_stages * (state_dim + control_dim) + state_dim;
  const int y_dim = (c_dim + state_dim) * (num_stages + 1);
  const int z_dim = g_dim * (num_stages + 1);

  gradient_f = new double[x_dim];
  c = new double[y_dim];
  g = new double[z_dim];

  lqr_workspace.reserve(state_dim, control_dim, num_stages);
  lqr_output.reserve(num_stages);

  regularized_lqr_data.reserve(state_dim, control_dim, num_stages, g_dim);

  sip_workspace.reserve(x_dim, z_dim, y_dim);
}

void Workspace::free(int num_stages) {
  model_callback_output.free(num_stages);
  model_callback_input.free();
  delete[] gradient_f;
  delete[] c;
  delete[] g;
  lqr_workspace.free(num_stages);
  lqr_output.free();
  regularized_lqr_data.free(num_stages);
  sip_workspace.free();
}

auto Workspace::mem_assign(int state_dim, int control_dim, int num_stages,
                           int c_dim, int g_dim, unsigned char *mem_ptr)
    -> int {
  const int x_dim = num_stages * (state_dim + control_dim) + state_dim;
  const int y_dim = (c_dim + state_dim) * (num_stages + 1);
  const int z_dim = g_dim * (num_stages + 1);

  int cum_size = 0;

  cum_size += model_callback_output.mem_assign(
      state_dim, control_dim, num_stages, c_dim, g_dim, mem_ptr + cum_size);
  cum_size += model_callback_input.mem_assign(num_stages, mem_ptr + cum_size);

  gradient_f = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += x_dim * sizeof(double);

  c = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += y_dim * sizeof(double);

  g = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += z_dim * sizeof(double);

  cum_size += lqr_workspace.mem_assign(state_dim, control_dim, num_stages,
                                       mem_ptr + cum_size);

  cum_size += lqr_output.mem_assign(num_stages, mem_ptr + cum_size);

  cum_size += regularized_lqr_data.mem_assign(
      state_dim, control_dim, num_stages, g_dim, mem_ptr + cum_size);

  cum_size += sip_workspace.mem_assign(x_dim, z_dim, y_dim, mem_ptr + cum_size);

  assert(cum_size == Workspace::num_bytes(state_dim, control_dim, num_stages,
                                          c_dim, g_dim));

  return cum_size;
}

} // namespace sip::optimal_control

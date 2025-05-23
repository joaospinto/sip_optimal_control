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

  for (int i = 0; i <= num_stages; ++i) {
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
  W = new double *[num_stages];
  K = new double *[num_stages];
  V = new double *[num_stages + 1];
  G_inv = new double *[num_stages];
  k = new double *[num_stages];
  v = new double *[num_stages + 1];
  mod_w_inv = new double *[num_stages + 1];

  for (int i = 0; i < num_stages; ++i) {
    W[i] = new double[state_dim * state_dim];
    K[i] = new double[control_dim * state_dim];
    V[i] = new double[state_dim * state_dim];
    G_inv[i] = new double[control_dim * control_dim];
    k[i] = new double[control_dim];
    v[i] = new double[state_dim];
    mod_w_inv[i] = new double[g_dim];
  }

  V[num_stages] = new double[state_dim * state_dim];
  v[num_stages] = new double[state_dim];
  mod_w_inv[num_stages] = new double[g_dim];

  G = new double[control_dim * control_dim];
  g = new double[control_dim];
  H = new double[control_dim * state_dim];
  h = new double[control_dim];
  F = new double[state_dim * state_dim];
  F_inv = new double[state_dim * state_dim];
  f = new double[state_dim];

  Q_mod = new double[state_dim * state_dim];
  M_mod = new double[state_dim * control_dim];
  R_mod = new double[control_dim * control_dim];
  q_mod = new double[state_dim];
  r_mod = new double[control_dim];
}

void Workspace::RegularizedLQRData::free(int num_stages) {
  delete[] G;
  delete[] g;
  delete[] H;
  delete[] h;
  delete[] F;
  delete[] F_inv;
  delete[] f;

  delete[] Q_mod;
  delete[] M_mod;
  delete[] R_mod;
  delete[] q_mod;
  delete[] r_mod;

  for (int i = 0; i < num_stages; ++i) {
    delete[] W[i];
    delete[] K[i];
    delete[] V[i];
    delete[] G_inv[i];
    delete[] k[i];
    delete[] v[i];
    delete[] mod_w_inv[i];
  }

  delete[] V[num_stages];
  delete[] v[num_stages];
  delete[] mod_w_inv[num_stages];

  delete[] W;
  delete[] K;
  delete[] V;
  delete[] G_inv;
  delete[] k;
  delete[] v;
  delete[] mod_w_inv;
}

auto Workspace::RegularizedLQRData::mem_assign(int state_dim, int control_dim,
                                               int num_stages, int g_dim,
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

  mod_w_inv = reinterpret_cast<double **>(mem_ptr + cum_size);
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

    mod_w_inv[i] = reinterpret_cast<double *>(mem_ptr + cum_size);
    cum_size += g_dim * sizeof(double);
  }

  V[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * state_dim * sizeof(double);

  v[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  mod_w_inv[num_stages] = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += g_dim * sizeof(double);

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

  F_inv = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * state_dim * sizeof(double);

  f = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  Q_mod = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * state_dim * sizeof(double);

  M_mod = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * control_dim * sizeof(double);

  R_mod = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += control_dim * control_dim * sizeof(double);

  q_mod = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += state_dim * sizeof(double);

  r_mod = reinterpret_cast<double *>(mem_ptr + cum_size);
  cum_size += control_dim * sizeof(double);

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

  regularized_lqr_data.reserve(state_dim, control_dim, num_stages, g_dim);

  sip_workspace.reserve(x_dim, z_dim, y_dim);
}

void Workspace::free(int num_stages) {
  model_callback_output.free(num_stages);
  model_callback_input.free();
  delete[] gradient_f;
  delete[] c;
  delete[] g;
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

  cum_size += regularized_lqr_data.mem_assign(
      state_dim, control_dim, num_stages, g_dim, mem_ptr + cum_size);

  cum_size += sip_workspace.mem_assign(x_dim, z_dim, y_dim, mem_ptr + cum_size);

  assert(cum_size == Workspace::num_bytes(state_dim, control_dim, num_stages,
                                          c_dim, g_dim));

  return cum_size;
}

} // namespace sip::optimal_control

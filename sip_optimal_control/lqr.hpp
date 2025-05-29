#pragma once

namespace sip::optimal_control {

class LQR {
public:
  struct Input {
    struct Dimensions {
      int state_dim;
      int control_dim;
      int num_stages;
    };

    double **Q;
    double **M;
    double **R;
    double **q;
    double **r;
    double **A;
    double **B;
    double **c;

    Dimensions dimensions;
  };

  struct Output {
    double **x;
    double **u;
    double **y;

    // To dynamically allocate the required memory.
    void reserve(int num_stages);
    void free();

    // For using pre-allocated (possibly statically allocated) memory.
    auto mem_assign(int num_stages, unsigned char *mem_ptr) -> int;

    // For knowing how much memory to pre-allocate.
    static constexpr auto num_bytes(int num_stages) -> int {
      return (3 * num_stages + 2) * sizeof(double *);
    }
  };

  struct Workspace {
    // NOTE: we need to store these for ALL stages.
    double **W;
    double **K;
    double **V;
    double **G_inv;
    double **k;
    double **v;

    // NOTE: we only need to store these for one stage at a time.
    double *G;
    double *g;
    double *H;
    double *h;
    double *F;
    double *f;

    // To dynamically allocate the required memory.
    void reserve(int state_dim, int control_dim, int num_stages);
    void free(int num_stages);

    // For using pre-allocated (possibly statically allocated) memory.
    auto mem_assign(int state_dim, int control_dim, int num_stages,
                    unsigned char *mem_ptr) -> int;

    // For knowing how much memory to pre-allocate.
    static constexpr auto num_bytes(int state_dim, int control_dim,
                                    int num_stages) -> int {
      const int n = state_dim;
      const int m = control_dim;
      const int T = num_stages;
      const int W_size = T * sizeof(double *) + T * n * n * sizeof(double);
      const int K_size = T * sizeof(double *) + T * m * n * sizeof(double);
      const int V_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * n * sizeof(double);
      const int G_inv_size = T * sizeof(double *) + T * m * m * sizeof(double);
      const int k_size = T * sizeof(double *) + T * m * sizeof(double);
      const int v_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
      const int G_size = m * m * sizeof(double);
      const int g_size = m * sizeof(double);
      const int H_size = m * n * sizeof(double);
      const int h_size = m * sizeof(double);
      const int F_size = n * n * sizeof(double);
      const int f_size = n * sizeof(double);
      return W_size + K_size + V_size + G_inv_size + k_size + v_size + G_size +
             g_size + H_size + h_size + F_size + f_size;
    }
  };

  LQR(const Input &data, Workspace &workspace);

  void factor(const double δ);
  void solve(const double δ, Output &output);

private:
  const Input &input_;
  Workspace &workspace_;
};

} // namespace sip::optimal_control

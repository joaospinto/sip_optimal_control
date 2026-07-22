#pragma once

namespace sip::optimal_control {

struct Topology {
  int num_edges = 0;
  int root = 0;
  const int *edge_parents = nullptr;
  const int *edge_children = nullptr;

  int num_nodes() const;

  void reserve(int num_edges);
  void free();
  int mem_assign(int num_edges, unsigned char *mem_ptr);
  static constexpr int num_bytes(int num_edges) {
    return 2 * num_edges * sizeof(int);
  }

  void set_chain();
  void set_tree(int root, const int *edge_parents, const int *edge_children);
};

struct Dimensions {
  // State and node-constraint dimensions are indexed by node ID. Control and
  // edge-constraint dimensions are indexed by edge ID.
  int theta_dim = 0;
  const int *state_dims = nullptr;
  const int *control_dims = nullptr;
  const int *node_c_dims = nullptr;
  const int *node_g_dims = nullptr;
  const int *edge_c_dims = nullptr;
  const int *edge_g_dims = nullptr;

  void reserve(int num_edges);
  void free();
  int mem_assign(int num_edges, unsigned char *mem_ptr);
  static constexpr int num_bytes(int num_edges) {
    return (3 * (num_edges + 1) + 3 * num_edges) * sizeof(int);
  }

  void set_uniform(int num_edges, int state_dim, int control_dim,
                   int node_c_dim, int node_g_dim, int edge_c_dim,
                   int edge_g_dim, int theta_dim = 0);

  int get_schur_dim() const;
  int get_state_dim(int node) const;
  int get_control_dim(int edge) const;
  int get_node_c_dim(int node) const;
  int get_node_g_dim(int node) const;
  int get_edge_c_dim(int edge) const;
  int get_edge_g_dim(int edge) const;
  int max_state_dim(int num_nodes) const;
  int max_control_dim(int num_edges) const;
  int max_node_c_dim(int num_nodes) const;
  int max_node_g_dim(int num_nodes) const;
  int max_edge_c_dim(int num_edges) const;
  int max_edge_g_dim(int num_edges) const;
  int get_stagewise_x_dim(int num_edges) const;
  int get_x_dim(int num_edges) const;
  int get_y_dim(int num_edges) const;
  int get_z_dim(int num_edges) const;
  int get_stagewise_kkt_dim(int num_edges) const;
};

class LQR {
public:
  enum class FactorStatus {
    SUCCESS = 0,
    INVALID_DELTA = 1,
    F_FACTORIZATION_FAILURE = 2,
    G_FACTORIZATION_FAILURE = 3,
    INVALID_TOPOLOGY = 4,
  };

  struct Input {
    double **Q;
    double **M;
    double **R;
    double **q;
    double **r;
    double **A;
    double **B;
    double **c;
    double **delta;

    const Dimensions &dimensions;
    const Topology &topology;
  };

  struct Output {
    double **x;
    double **u;
    double **y;

    // To dynamically allocate the required memory.
    void reserve(int num_edges);
    void free();

    // For using pre-allocated (possibly statically allocated) memory.
    auto mem_assign(int num_edges, unsigned char *mem_ptr) -> int;

    // For knowing how much memory to pre-allocate.
    static constexpr auto num_bytes(int num_edges) -> int {
      return (3 * num_edges + 2) * sizeof(double *);
    }
  };

  struct Workspace {
    // NOTE: we need to store these for ALL nodes.
    double **W;
    double **K;
    double **V;
    double **G_factor;
    double **F_factor;
    double **sqrt_delta;
    double **sqrt_delta_inv;
    double **k;
    double **v;

    // NOTE: we only need to store these for one edge at a time.
    double *G;
    double *g;
    double *H;
    double *h;
    double *F;
    double *f;

    int *child_offsets;
    int *child_edges;
    int *edge_parents;
    int *edge_children;
    int *preorder_nodes;
    int *postorder_nodes;
    int *node_marks;
    // To dynamically allocate the required memory.
    void reserve(int state_dim, int control_dim, int num_edges);
    void reserve(const Dimensions &dimensions, const Topology &topology);
    void free(int num_edges);

    // For using pre-allocated (possibly statically allocated) memory.
    auto mem_assign(const Dimensions &dimensions, const Topology &topology,
                    unsigned char *mem_ptr) -> int;

    // For knowing how much memory to pre-allocate.
    static constexpr auto num_bytes(int state_dim, int control_dim,
                                    int num_edges) -> int {
      const int n = state_dim;
      const int m = control_dim;
      const int T = num_edges;
      const int W_size = T * sizeof(double *) + T * n * n * sizeof(double);
      const int K_size = T * sizeof(double *) + T * m * n * sizeof(double);
      const int V_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * n * sizeof(double);
      const int G_factor_size =
          T * sizeof(double *) + T * m * m * sizeof(double);
      const int F_factor_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * n * sizeof(double);
      const int sqrt_delta_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
      const int sqrt_delta_inv_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
      const int k_size = T * sizeof(double *) + T * m * sizeof(double);
      const int v_size =
          (T + 1) * sizeof(double *) + (T + 1) * n * sizeof(double);
      const int G_size = m * m * sizeof(double);
      const int g_size = n * sizeof(double);
      const int H_size = m * n * sizeof(double);
      const int h_size = m * sizeof(double);
      const int F_size = n * n * sizeof(double);
      const int f_size = n * sizeof(double);
      const int child_offsets_size = (T + 2) * sizeof(int);
      const int child_edges_size = T * sizeof(int);
      const int edge_parents_size = T * sizeof(int);
      const int edge_children_size = T * sizeof(int);
      const int preorder_nodes_size = (T + 1) * sizeof(int);
      const int postorder_nodes_size = (T + 1) * sizeof(int);
      const int node_marks_size = (T + 1) * sizeof(int);
      return W_size + K_size + V_size + G_factor_size + F_factor_size +
             sqrt_delta_size + sqrt_delta_inv_size + k_size + v_size + G_size +
             g_size + H_size + h_size + F_size + f_size + child_offsets_size +
             child_edges_size + edge_parents_size + edge_children_size +
             preorder_nodes_size + postorder_nodes_size + node_marks_size;
    }
    static auto num_bytes(const Dimensions &dimensions,
                          const Topology &topology) -> int;
  };

  LQR(const Input &data, Workspace &workspace);

  auto compile_topology() -> FactorStatus;
  FactorStatus factor_with_status();
  bool factor();
  void solve(Output &output);

private:
  const Input &input_;
  Workspace &workspace_;
  FactorStatus traversal_status_;
};

} // namespace sip::optimal_control

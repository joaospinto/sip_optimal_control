#pragma once

namespace sip::optimal_control {

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
    // Optional rooted-tree topology. When omitted, edges use the chain
    // topology edge -> (edge, edge + 1). General DAGs should be condensed into
    // tree pieces plus separator Schur complements before calling this solver.
    struct Topology {
      const void *context = nullptr;
      int (*root)(const void *context) = nullptr;
      int (*edge_parent)(const void *context, int edge) = nullptr;
      int (*edge_child)(const void *context, int edge) = nullptr;
    };

    struct Dimensions {
      int state_dim;
      int control_dim;
      int num_stages;
      const int *state_dims = nullptr;
      const int *control_dims = nullptr;

      int num_edges() const { return num_stages; }
      int num_nodes() const { return num_stages + 1; }
      int get_state_dim(int node) const {
        return state_dims == nullptr ? state_dim : state_dims[node];
      }
      int get_control_dim(int edge) const {
        return control_dims == nullptr ? control_dim : control_dims[edge];
      }
      int max_state_dim() const {
        int result = 0;
        for (int node = 0; node < num_nodes(); ++node) {
          const int dim = get_state_dim(node);
          result = result < dim ? dim : result;
        }
        return result;
      }
      int max_control_dim() const {
        int result = 0;
        for (int edge = 0; edge < num_edges(); ++edge) {
          const int dim = get_control_dim(edge);
          result = result < dim ? dim : result;
        }
        return result;
      }
    };

    double **Q;
    double **M;
    double **R;
    double **q;
    double **r;
    double **A;
    double **B;
    double **c;
    double **delta;

    Dimensions dimensions;
    Topology topology = {};
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
    double **G_factor;
    double **F_factor;
    double **sqrt_delta;
    double **sqrt_delta_inv;
    double **k;
    double **v;

    // NOTE: we only need to store these for one stage at a time.
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
    bool topology_is_initialized;
    FactorStatus topology_status;
    int topology_state_dim;
    int topology_control_dim;
    int topology_num_stages;
    const int *topology_state_dims;
    const int *topology_control_dims;
    const void *topology_context;
    int (*topology_root)(const void *context);
    int (*topology_edge_parent)(const void *context, int edge);
    int (*topology_edge_child)(const void *context, int edge);

    // To dynamically allocate the required memory.
    void reserve(int state_dim, int control_dim, int num_stages);
    void reserve(const Input::Dimensions &dimensions);
    void free(int num_stages);

    // For using pre-allocated (possibly statically allocated) memory.
    auto mem_assign(int state_dim, int control_dim, int num_stages,
                    unsigned char *mem_ptr) -> int;
    auto mem_assign(const Input::Dimensions &dimensions,
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
             sqrt_delta_size + sqrt_delta_inv_size + k_size + v_size +
             G_size + g_size + H_size + h_size + F_size + f_size +
             child_offsets_size + child_edges_size + edge_parents_size +
             edge_children_size + preorder_nodes_size + postorder_nodes_size +
             node_marks_size;
    }
    static auto num_bytes(const Input::Dimensions &dimensions) -> int;
  };

  LQR(const Input &data, Workspace &workspace);

  auto compile_topology() -> FactorStatus;
  FactorStatus factor_with_status();
  bool factor();
  void solve(Output &output);

private:
  const Input &input_;
  Workspace &workspace_;
};

} // namespace sip::optimal_control

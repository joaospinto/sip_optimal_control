# sip_optimal_control
[![CI](https://github.com/joaospinto/sip_optimal_control/actions/workflows/ci.yml/badge.svg)](https://github.com/joaospinto/sip_optimal_control/actions/workflows/ci.yml)
[![Benchmarks](https://github.com/joaospinto/sip_optimal_control/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/joaospinto/sip_optimal_control/actions/workflows/benchmarks.yml)

This repository implements an optimal control front-end to the
[SIP](https://github.com/joaospinto/sip)
solver.

The stagewise nature of the optimal control problems allows us
to reduce the Newton-KKT linear system solves to
[regularized LQR](https://github.com/joaospinto/regularized_lqr_jax)
problems.

You can find a usage example in the
[SIP Examples](https://github.com/joaospinto/sip_examples)
repository.

## Initial model cache

`Input::initial_model_is_current` forwards SIP's initial-cache contract. Set it
only when the workspace's node/edge model values match the initial variables
and current problem data. The model callback receives `new_x=false`,
`new_y=new_z=true`, and `need_derivatives=true` on entry; these flags are also
forwarded on later evaluations. The adapter assembles the flattened constraint
arrays on the first callback even when node/edge values are reused. The default
is false, and caller-provided slacks are preserved.

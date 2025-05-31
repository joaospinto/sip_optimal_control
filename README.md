# sip_optimal_control

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

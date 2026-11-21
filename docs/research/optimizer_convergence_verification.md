# Convergence verification

Lean 4.35.0 and the pinned Mathlib passed the source scan, package check, compilation, axiom audit, and kernel replay. The required declarations appear in the checker's PASS record. They use only the standard axioms propext, Classical.choice, and Quot.sound.

The loss is bounded below by \(f_\star\). Iterates live in a real inner product space. The smoothness bridge uses completeness, which holds for the report's Euclidean parameter spaces. It derives the quadratic upper model from a Lipschitz gradient.

| Report claim | Lean declaration in OptimizerConvergence | Correspondence |
| --- | --- | --- |
| Lipschitz gradient implies the quadratic upper model | gradient_upper_model | Faithful. HasGradientAt ties the supplied vector to the derivative. The pointwise Lipschitz condition has the same constant \(L\). |
| Uphill reflection improves the gradient inner product and preserves or reduces squared step length | reflection_compatible | Faithful. The strict uphill condition guarantees that the gradient can be normalized. |
| Momentum with \(\eta L<1-\mu^2\), nonincreasing post-step kinetic energy, and exact gradients | momentum_descent, momentum_exact_rate, momentum_geometric_rate | Faithful for the finite-horizon and gradient-dominance bounds. The replicas form a finite type. Positive replica and update counts license division by \(PT\). |
| Momentum along each noisy trajectory | noise_cross_bound, momentum_pathwise_rate | Faithful. The noise can include bias and the attenuation can use the current batch. |
| Momentum under centered noise, including the gradient-dominance bound | history_cross_zero, momentum_expected_rates | Finite-sampling case. Weights are nonnegative and sum to one. Grouping outcomes by history gives the cross-term cancellation. The general conditional-expectation argument is in the report. |
| Guarded AdamW fallback at \(\eta L\le1\) and exact and inexact stationarity bounds | fallback_passes, guarded_rates | Faithful. GuardedStep uses the candidate exactly when both tests pass and uses \(-\eta G\) otherwise. Existence of an iterate below the bound is equivalent to the minimum bound for \(T>0\). AdamW supplies one allowable candidate. |
| Guarded loss contraction and the gradient-error floor under gradient dominance | guarded_geometric_rates | Faithful. The finite-horizon formula includes the transient term and its \(17\bar\varepsilon^2/m\) limiting upper bound. |
| Gravity flight, elastic reflection, and the damping-tick kinetic bound | gravity_flight_energy, elastic_reflection_energy, gravity_damping_rate | Partial for the continuous simulator. The formalization verifies the conservative operations and the sampled balance and rate given the feasible event sequence. |

The hypotheses admit \(f(x)=\|x\|^2/2\), \(L=m=1\), \(\mu=\eta=1/2\), exact gradients, and nonzero initial parameters. Then \(\eta L=1/2<1-\mu^2=3/4\). This quadratic also permits the guarded boundary \(\eta L=1\). The rates concern the mathematical updates and training objective. The saved NanoGPT run has no certified smoothness constant, and the proof does not verify the PyTorch implementation.

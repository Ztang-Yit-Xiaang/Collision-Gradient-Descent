# Quadratic exchange verification

Lean 4.35.0 with the pinned Mathlib passed compilation, the axiom audit, and kernel replay. The checked claims rely only on propext, Classical.choice, and Quot.sound.

| Proposition component | Lean declaration in QuadraticExchange | Correspondence |
| --- | --- | --- |
| Equal initial mean parameters and velocities give equal means at every step under arbitrary exchange schedules. | mean_trajectory | The parameter updates use the current inherited velocity. Each subsequent permutation acts only on the new velocities. A positive replica count makes equality of means equivalent to equality of totals. The statement applies to the affine gradient \(A\theta+b\). |
| The change in the sum of the pair's next losses is \(\mu(z_i-z_j)^\top A(u_j-u_i)\). | quadratic_step_expansion, pair_loss_difference | The quadratic uses a symmetric bilinear form \(B\). Taking \(B(x,y)=x^\top A y\) gives the stated matrix identity. The two anchors \(z_i,z_j\) may be arbitrary, so the identity applies to the specific anchors produced by the gradient step. |
| A two-replica run starting at \(a,-a\) with zero velocities can give a strictly larger next loss in both replicas. | reachable_loss_rise | Exact real arithmetic for \(\mu=\eta=1/2\) and \(a\ne0\). The first step is generated from the common initial state. Swapping the resulting velocity buffers gives losses \(a^2/8\) in both replicas, compared with zero in the retaining arm. |

The example uses the strongly convex, unit-smooth loss \(x^2/2\) and satisfies the sufficient condition \(\eta L<1-\mu^2\). The population mean is identical in both arms, and the example concerns the loss at the next step.

#set document(title: "Beyond Backprop — Phase 3 Crash Course", author: "")
#set page(margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show math.equation.where(block: true): it => pad(y: 0.5em, it)

#let note(body) = block(
  fill: luma(235),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  body
)

#let gotcha(body) = block(
  fill: rgb("fff3cd"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [⚠ *Gotcha:* ] + body
)

#let insight(body) = block(
  fill: rgb("d4edda"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [💡 *Insight:* ] + body
)

#let definition(title, body) = block(
  fill: rgb("e8f4f8"),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  [*Definition — #title:* ] + body
)

// ─── TITLE ───────────────────────────────────────────────────────────────────

#align(center)[
  #text(size: 22pt, weight: "bold")[Beyond Backprop]
  #v(0.3em)
  #text(size: 15pt)[Phase 3 — Spiking Neural Networks · Neural ODEs]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[Discrete events in continuous time — where computation meets physics]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

#outline(indent: 1em, depth: 2)

#v(0.5em)
#line(length: 100%)

// ─── INTRO ───────────────────────────────────────────────────────────────────

= Orientation

Phase 3 is the deepest departure from the DNN template in this course. Both architectures question something the previous phases took for granted: the nature of a "forward pass."

In every architecture so far — DNN, ELM, Transformer, RBM, ESN, SOM — the forward pass is a synchronous, discrete-time computation. You feed in an input. You compute activations, layer by layer (or step by step). You get an output. Time, if it appears at all, is a discrete index.

Phase 3 breaks this in two different directions:

- *SNNs* replace continuous activations with discrete spike events. Computation is *asynchronous* and *sparse*. A neuron does not output a number at every timestep — it either fires or it does not, and firing carries information through its *timing*, not its magnitude.

- *Neural ODEs* replace discrete layers with a continuous-time differential equation. There is no "layer 1, layer 2, layer 3." There is a single function $f$ evaluated continuously as a system evolves from $t=0$ to $t=1$. Depth becomes a real number.

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Module*], [*Core abstraction*], [*Gradients?*], [*Primary novelty*]),
  [SNN], [Discrete spikes in continuous time], [Surrogate (approximate)], [Event-driven, energy-efficient computation],
  [Neural ODE], [Continuous-depth dynamical system], [Adjoint method (exact)], [Infinite-depth with constant memory],
)

These two architectures are also the most mathematically demanding in the course. Take your time with the SNN's surrogate gradient derivation and the Neural ODE's adjoint method — both are genuinely subtle.


// ═══════════════════════════════════════════════════════════════════════════════
= Module 5 — Spiking Neural Networks (SNN)
// ═══════════════════════════════════════════════════════════════════════════════

== Core Idea

The brain does not pass floating-point numbers between neurons. It passes *spikes* — brief, stereotyped electrical pulses, all-or-nothing, roughly 1–2 ms in duration. A neuron integrates incoming spikes over time, and when its internal voltage (membrane potential) crosses a threshold, it emits a spike of its own and resets.

Information is encoded not in activation magnitudes but in *spike timing* and *spike rates*. A neuron that fires 80 times per second encodes something different from one that fires 20 times per second, even if both emit identical spikes.

An SNN formalizes this biologically inspired computation for machine learning. The payoff is not (primarily) accuracy — standard DNNs match or exceed SNNs on most benchmarks. The payoff is *efficiency*: on neuromorphic hardware, SNNs compute with orders of magnitude less energy because most neurons are silent most of the time (sparse computation), and the hardware performs addition (spike accumulation) rather than multiply-accumulate.

== The Leaky Integrate-and-Fire Neuron

The standard SNN neuron model is the *Leaky Integrate-and-Fire* (LIF) neuron. Its dynamics are described by a first-order ODE:

$ tau_m frac(d V(t), d t) = -(V(t) - V_"rest") + R I(t) $

where:
- $V(t)$ — membrane potential at time $t$
- $V_"rest"$ — resting potential (typically 0, by convention)
- $tau_m = R C$ — membrane time constant (resistance × capacitance)
- $R$ — membrane resistance
- $I(t)$ — input current at time $t$

The neuron integrates input current, but the membrane potential *leaks* back toward $V_"rest"$ with time constant $tau_m$. This leakage is the "leaky" part — the neuron forgets its history exponentially.

When $V(t)$ crosses a threshold $V_"th"$, the neuron *fires* (emits a spike) and $V(t)$ is reset:

$ "if" V(t) >= V_"th": quad V(t) <- V_"reset", quad "emit spike" $

After firing, the neuron may enter a *refractory period* during which it cannot fire again, modeling the biological absolute refractory period.

== Discrete-Time Formulation

For practical SNN implementations, the LIF ODE is discretized. Let $Delta t$ be the simulation timestep. Define the *decay factor* $beta = e^(-Delta t / tau_m)$. The discrete update rule is:

$ U[t] = beta U[t-1] + (1 - beta) I[t] $

where $U[t]$ is the membrane potential at timestep $t$ and $I[t]$ is the weighted sum of incoming spikes. The $(1-beta)$ factor normalizes the input so that the steady-state potential equals $I$ for constant input.

The spike output is:

$ S[t] = Theta(U[t] - V_"th") $

where $Theta$ is the Heaviside step function:

$ Theta(x) = cases(1 & "if" x >= 0, 0 & "if" x < 0) $

After firing, the potential resets. The *subtract-reset* formulation (preferred for gradient flow) subtracts the threshold rather than setting to a fixed value:

$ U[t] <- U[t] - V_"th" dot S[t] $

So the full update per timestep is:

$ U[t] &= beta (U[t-1] - V_"th" dot S[t-1]) + I[t] $
$ S[t] &= Theta(U[t] - V_"th") $

This can be seen as a *gated leaky integrator*: the neuron accumulates input, leaks toward zero, and resets by subtracting the threshold whenever it spikes.

== Network Architecture

An SNN processes inputs over $T$ timesteps. For a fully connected SNN with layers $ell = 1, dots, L$:

*Input current to layer $ell$ at time $t$:*

$ bold(I)^ell [t] = bold(W)^ell bold(S)^(ell-1)[t] $

where $bold(S)^(ell-1)[t] in {0,1}^(n_(ell-1))$ is the spike vector from the previous layer, and $bold(W)^ell in RR^(n_ell times n_(ell-1))$ are the synaptic weights.

*Membrane potential update:*

$ bold(U)^ell [t] = beta (bold(U)^ell [t-1] - V_"th" bold(S)^ell [t-1]) + bold(I)^ell [t] $

*Spike generation:*

$ bold(S)^ell [t] = Theta(bold(U)^ell [t] - V_"th") $

The output layer does not spike — it accumulates membrane potential over all $T$ timesteps and classifies based on which output neuron has the highest total potential (or highest spike count):

$ hat(bold(y)) = "softmax"(sum_(t=1)^T bold(U)^L [t]) $

== Rate Coding for MNIST

Static images must be converted into spike trains — temporal sequences of 0s and 1s. The standard approach for MNIST is *Poisson rate coding*:

For each pixel with intensity $p_i in [0, 1]$, at each timestep $t$, generate a spike independently with probability $p_i$:

$ S_i[t] ~ "Bernoulli"(p_i) $

A brighter pixel fires more often on average. Over $T$ timesteps, the expected spike count for pixel $i$ is $T dot p_i$. The network must learn to integrate these stochastic spike trains into a reliable classification.

#note[
  Rate coding is simple but inefficient — it requires many timesteps to build up reliable rate estimates. More advanced coding schemes exist: *temporal coding* (information in exact spike timing), *population coding* (information distributed over a population of neurons), and *burst coding* (information in bursts of spikes). For MNIST, rate coding with $T = 25$–$100$ timesteps is standard and sufficient.
]

== The Gradient Problem

Here is the fundamental obstacle to training SNNs with backpropagation. The spike function $S[t] = Theta(U[t] - V_"th")$ has derivative:

$ frac(d S, d U) = frac(d, d U) Theta(U - V_"th") = delta(U - V_"th") $

where $delta$ is the Dirac delta — zero everywhere except at the threshold, where it is infinite. In the discretized setting, $Theta$ is a step function whose derivative is zero almost everywhere and undefined at the threshold.

This means that standard backpropagation through the spike function gives zero gradient everywhere, making it impossible to learn by gradient descent. The network cannot know whether to increase or decrease a weight because the spike function provides no gradient signal.

This is the *dead neuron problem*, but more fundamental than in ReLU networks — here it is not just some neurons that are dead, it is the *entire spike generation mechanism*.

== Surrogate Gradients

The solution is to use a *surrogate gradient*: replace the true derivative of $Theta$ with a smooth approximation *during the backward pass only*. The forward pass uses the real spike function; the backward pass uses a proxy that provides useful gradient signal.

#definition("Surrogate Gradient")[
  A function $tilde(sigma)'(x)$ used in place of $Theta'(x)$ during backpropagation through spike generation, chosen to be smooth, bounded, and centered at the threshold.
]

The most common surrogate is the derivative of the fast sigmoid:

$ tilde(sigma)'(x) = frac(1, (1 + |x / k|)^2) dot frac(1, k) $

or equivalently, treating it as if the forward function were a shifted sigmoid with sharpness $k$:

$ "forward": S approx sigma(k(U - V_"th")) $
$ "backward": frac(d S, d U) approx sigma'(k(U - V_"th")) = frac(k e^(-k(U - V_"th"))}{(1 + e^{-k(U - V_"th")})^2) $

Another popular choice is the *piecewise linear* surrogate:

$ tilde(sigma)'(x) = max(0, 1 - |x|) $

which is nonzero only in a window of width 2 around the threshold.

The *straight-through estimator* (STE) is the simplest surrogate:

$ tilde(sigma)'(x) = bold(1)[|x| < 0.5] $

— pass the gradient through unchanged if the neuron is near threshold, zero otherwise.

#insight[
  The surrogate gradient is a deliberate lie to the optimizer. We tell backprop "pretend the spike function is smooth" when computing gradients, while actually using the hard threshold in the forward pass. This mismatch between forward and backward passes is theoretically inelegant but empirically effective. The optimizer learns weights that work with the real (hard) spike function, guided by gradients computed with the fake (smooth) one.
]

== Backpropagation Through Time

Because SNNs are recurrent in time (the membrane potential at $t$ depends on $t-1$), training uses *Backpropagation Through Time* (BPTT) — the same algorithm used for RNNs.

Unroll the network over all $T$ timesteps. Define the loss $cal(L)$ on the output (e.g., cross-entropy on $sum_t bold(U)^L[t]$). Gradients flow backward through time and through layers simultaneously.

The gradient of the loss with respect to the weights $bold(W)^ell$ accumulates contributions from all timesteps:

$ frac(diff cal(L), diff bold(W)^ell) = sum_(t=1)^T frac(diff cal(L), diff bold(I)^ell[t]) (bold(S)^(ell-1)[t])^top $

where $frac(diff cal(L), diff bold(I)^ell[t])$ propagates backward through the membrane dynamics and the surrogate spike function.

The membrane potential recurrence introduces a term analogous to the vanishing/exploding gradient problem in RNNs. The decay factor $beta < 1$ causes gradients to decay exponentially backward in time:

$ frac(diff bold(U)^ell[t], diff bold(U)^ell[t - k]) = beta^k prod_(s=1)^k (1 - V_"th" frac(diff bold(S)^ell[t-s+1], diff bold(U)^ell[t-s+1])) $

For large $k$ (many timesteps back), this product tends to zero unless the surrogate gradient terms compensate. In practice, $T$ is kept small (25–100) to avoid severe gradient decay.

== What to Observe

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Experiment*], [*What you learn*]),
  [Vary $T$: 10, 25, 50, 100 timesteps], [Accuracy vs. latency tradeoff — minimum $T$ for good performance],
  [Vary $beta$: 0.5, 0.9, 0.99], [Memory timescale — how far back the neuron integrates],
  [Compare surrogate functions], [Fast sigmoid vs piecewise linear vs STE — similar accuracy, different stability],
  [Spike raster plot], [Visualize which neurons fire when — sparsity is the key metric],
  [Measure sparsity: mean firing rate], [Fraction of neurons firing per timestep — should be low ($<$ 10%)],
  [Vary threshold $V_"th"$], [Higher threshold = sparser spikes = less computation, lower accuracy],
)

#gotcha[
  The number of timesteps $T$ is a compute multiplier — a 2-layer SNN with $T = 50$ does 100 forward passes through the layers (50 per layer). Training time scales linearly with $T$. Start with $T = 25$ to verify correctness, then increase. On GPU, the temporal loop is often the bottleneck because it is sequential — parallelization across the batch dimension helps, but the time dimension cannot be parallelized without changing the algorithm.
]

#gotcha[
  SNN accuracy on MNIST is typically 98–99% with enough timesteps and proper tuning — competitive with DNNs. But this does *not* mean the SNN is as efficient as it appears. On a GPU, SNNs are slower than DNNs of comparable accuracy because GPUs are optimized for dense matrix multiplications, not sparse event-driven computation. The efficiency argument holds on *neuromorphic hardware* (Intel Loihi, BrainScaleS), not on standard hardware. Always be clear about what hardware you are comparing on.
]

== Summary

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*Value*]),
  [Neuron model], [Leaky Integrate-and-Fire (LIF)],
  [Output], [Binary spike trains $S[t] in {0,1}$],
  [Training], [BPTT with surrogate gradients],
  [Gradient approximation], [Smooth surrogate for $Theta'$],
  [Input encoding], [Poisson rate coding (for static images)],
  [Key hyperparameters], [$T$ (timesteps), $beta$ (decay), $V_"th"$ (threshold), surrogate function],
  [Expected MNIST accuracy], [98–99% (with $T >= 50$)],
  [Efficiency payoff], [Energy, on neuromorphic hardware only],
)


// ═══════════════════════════════════════════════════════════════════════════════
= Module 6 — Neural Ordinary Differential Equations
// ═══════════════════════════════════════════════════════════════════════════════

== Core Idea

A standard ResNet computes:

$ bold(h)_(ell+1) = bold(h)_ell + f(bold(h)_ell, bold(theta)_ell) $

where $f$ is the residual block and $bold(h)_ell$ is the hidden state at layer $ell$. This is an *Euler discretization* of the ODE:

$ frac(d bold(h)(t), d t) = f(bold(h)(t), t, bold(theta)) $

A Neural ODE takes this observation seriously. Instead of stacking discrete residual blocks, it defines the hidden state dynamics as a continuous ODE and uses a numerical ODE solver to evolve the state from $t_0$ to $t_1$. The "depth" of the network becomes the integration interval $[t_0, t_1]$ — a continuous quantity.

#insight[
  Every residual network is implicitly a discretized ODE. The Neural ODE makes the continuous limit explicit and uses it to gain two properties: (1) the solver adapts its step size to the difficulty of the input, spending more computation where the dynamics are complex; (2) the memory cost of training is $O(1)$ in depth, not $O(L)$, via the adjoint method.
]

== Architecture

A Neural ODE replaces the sequence of layers with:

$ bold(h)(t_1) = bold(h)(t_0) + integral_(t_0)^(t_1) f(bold(h)(t), t, bold(theta)) d t $

where:
- $bold(h)(t_0) in RR^d$ — initial hidden state, produced by a shallow encoder applied to the input
- $f: RR^d times RR times RR^p -> RR^d$ — a small neural network parameterized by $bold(theta)$ that defines the dynamics
- $bold(h)(t_1)$ — final hidden state, passed to a decoder/classifier
- $bold(theta)$ — the *single set of weights* shared across all "depths" (times)

For MNIST: a linear encoder maps the 784-dimensional input to $bold(h)(t_0) in RR^d$ (e.g., $d = 64$). The ODE dynamics $f$ are a small 2-layer MLP. The final state $bold(h)(1)$ is passed to a linear classifier.

The ODE is solved numerically by a solver such as *Dormand-Prince* (RK45) or *Euler*. The solver makes adaptive decisions about how many function evaluations (NFEs) to use — this is the compute cost of one forward pass, and it varies per input.

== The Adjoint Method

The critical question: how do we train $bold(theta)$? We need $frac(diff cal(L), diff bold(theta))$, which requires differentiating through the ODE solver.

The naive approach — store all intermediate solver states and backpropagate through each step — has memory cost $O(N_"NFE")$ where $N_"NFE"$ is the number of function evaluations. For adaptive solvers, $N_"NFE"$ can be large and varies per input. This is prohibitive.

The *adjoint method* (Pontryagin, 1962; adapted to Neural ODEs by Chen et al., 2018) computes gradients with $O(1)$ memory by solving a *second ODE backward in time*.

=== The Adjoint State

Define the *adjoint state* (also called the adjoint variable or costate):

$ bold(a)(t) = frac(diff cal(L), diff bold(h)(t)) $

This is the gradient of the loss with respect to the hidden state at time $t$. We know $bold(a)(t_1) = frac(diff cal(L), diff bold(h)(t_1))$ from the output loss. We need $bold(a)(t_0)$ and $frac(diff cal(L), diff bold(theta))$.

=== The Adjoint ODE

The adjoint state satisfies its own ODE, running *backward in time*:

$ frac(d bold(a)(t), d t) = -bold(a)(t)^top frac(diff f(bold(h)(t), t, bold(theta)), diff bold(h)(t)) $

The term $frac(diff f, diff bold(h)) in RR^(d times d)$ is the Jacobian of the dynamics with respect to the hidden state — computable by autodiff applied to $f$.

This ODE tells us how the sensitivity of the loss to the hidden state evolves backward through time. Starting from $bold(a)(t_1)$ and integrating backward to $t_0$ gives us $bold(a)(t_0)$.

=== Gradient with Respect to Parameters

Simultaneously, the gradient with respect to $bold(theta)$ accumulates along the backward trajectory:

$ frac(diff cal(L), diff bold(theta)} = -integral_(t_1)^(t_0) bold(a)(t)^top frac(diff f(bold(h)(t), t, bold(theta)), diff bold(theta)) d t $

The term $frac(diff f, diff bold(theta)}$ is the Jacobian of the dynamics with respect to the parameters — also computable by autodiff.

=== The Augmented Backward ODE

In practice, the three quantities $bold(h)(t)$, $bold(a)(t)$, and $frac(diff cal(L), diff bold(theta)}$ are computed together by solving one *augmented ODE* backward from $t_1$ to $t_0$:

$ frac(d, d t) mat(bold(h)(t); bold(a)(t); frac(diff cal(L), diff bold(theta))) = mat(f(bold(h)(t), t, bold(theta)); -bold(a)(t)^top frac(diff f, diff bold(h)); -bold(a)(t)^top frac(diff f, diff bold(theta))) $

with initial conditions at $t_1$:

$ mat(bold(h)(t_1); bold(a)(t_1); bold(0)) $

Solving this augmented ODE backward (using the same numerical solver as the forward pass) gives all required gradients without storing any intermediate states.

#note[
  The backward pass requires knowing $bold(h)(t)$ at each solver step, since $f$ depends on $bold(h)(t)$. In the adjoint method, $bold(h)(t)$ is *recomputed* during the backward pass by solving the forward ODE again — this is why memory is $O(1)$: we never store the trajectory. The cost is that the forward ODE is solved twice (once forward, once during the backward augmented solve). This makes Neural ODE training slower than standard backprop but with dramatically lower memory.
]

== Why This Is Different from Standard Backprop

In a standard $L$-layer network, backpropagation follows the computation graph of the forward pass exactly — one backward step per forward step, storing all intermediate activations. Memory is $O(L)$ in the number of layers.

In a Neural ODE, the forward pass uses a variable number of solver steps (NFEs), but the backward pass does *not* retrace these steps. Instead it solves a new ODE. This decouples the memory cost from the forward computation cost.

The philosophical implication: the Neural ODE does not "remember" how it computed the answer — it only remembers the answer itself ($bold(h)(t_1)$) and then re-derives the gradients from scratch by integrating backward. This is closer to how physics computes sensitivities (via Lagrangian mechanics) than how computers typically compute gradients.

== Adaptive Computation

Because the ODE solver is adaptive, the number of function evaluations is not fixed. The solver monitors the local truncation error and adjusts step size accordingly:

- Simple, low-curvature inputs → few NFEs → fast inference
- Complex, high-curvature inputs → many NFEs → slow inference

This means the *compute time per sample varies*. This is fundamentally unlike any other architecture in this course — computation is allocated proportionally to difficulty. For MNIST, simple digits (clean 1s and 0s) use fewer NFEs than ambiguous ones (noisy 3s, 8s, 5s).

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Architecture*], [*Compute per sample*]),
  [DNN, ELM, Transformer], [Fixed (same graph for all inputs)],
  [ESN], [Fixed (same $T$ steps for all inputs)],
  [SNN], [Fixed (same $T$ timesteps, variable spikes)],
  [Neural ODE], [*Adaptive* — harder inputs use more compute],
)

== Continuous Normalizing Flows

A beautiful application of Neural ODEs (not for classification, but worth knowing): *Continuous Normalizing Flows* (CNFs). If we apply a Neural ODE to a random variable $bold(z)(t_0) ~ p_0$, the density evolves according to the *instantaneous change of variables* formula:

$ frac(diff log p(bold(z)(t)), diff t) = -"tr"(frac(diff f, diff bold(z)(t))) $

The trace of the Jacobian (the divergence of $f$) controls how probability density changes as the state flows. This allows exact likelihood computation for generative models — something intractable for discrete normalizing flows with arbitrary architectures. CNFs are a direct consequence of treating the network as a continuous dynamical system.

== Stability and the Dynamics of $f$

The function $f$ defines how the hidden state changes. The stability of the ODE depends on the eigenvalues of the Jacobian $frac(diff f, diff bold(h))$:

- Eigenvalues with negative real part → *stable* dynamics (state is attracted to a fixed point)
- Eigenvalues with positive real part → *unstable* dynamics (state diverges)
- Pure imaginary eigenvalues → *oscillatory* dynamics

For classification, we want stable dynamics — the hidden state should settle to a representation that is easy to classify, not oscillate or diverge. This can be encouraged by regularizing $"tr"(frac(diff f, diff bold(h)})$ or the Frobenius norm of the Jacobian during training.

Alternatively, the *adjoint sensitivity* to time horizon $[t_0, t_1]$ tells us something useful: if the dynamics are stable, increasing $t_1$ (integrating longer) refines the representation. This is the continuous analog of adding more layers to a ResNet.

== Regularization via NFE

The number of function evaluations is a proxy for the complexity of the learned dynamics. We can add an NFE regularization term to the loss:

$ cal(L)_"total" = cal(L)_"CE" + lambda dot N_"NFE" $

This penalizes the solver for using many steps, encouraging the network to learn simpler, smoother dynamics that require less computation. This is a form of *computational regularization* unique to Neural ODEs — there is no analog in discrete networks.

== What to Observe

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Experiment*], [*What you learn*]),
  [Plot NFE distribution across test set], [Harder inputs use more compute — adaptive computation in action],
  [Compare Euler solver vs RK45], [Accuracy vs. speed tradeoff; Euler is fast but inaccurate],
  [Vary integration time $t_1$: 0.5, 1.0, 2.0], [Longer integration = more "depth" — accuracy vs. cost],
  [Add NFE regularization], [Can the model learn simpler dynamics without losing accuracy?],
  [Visualize hidden state trajectory], [Plot $bold(h)(t)$ for $t in [0,1]$ — watch how it evolves],
  [Compare memory: Neural ODE vs ResNet (same NFE)], [Constant vs linear memory — the adjoint method payoff],
  [Perturb $bold(h)(t_0)$ slightly], [Stability of dynamics — do nearby inputs stay nearby?],
)

#gotcha[
  Neural ODE training is slow. A single epoch on MNIST can take 10–50× longer than a comparable ResNet because each forward and backward pass involves running an ODE solver (many NFEs) rather than one pass through a fixed graph. Start with the *Euler solver* (fixed step, 1 NFE per step, no adaptivity) to verify correctness and debug the model. Switch to RK45 only once the architecture is working. For the Euler solver, the number of steps is a fixed hyperparameter — start with 10 steps.
]

#gotcha[
  The adjoint method can be *numerically unstable* when the dynamics $f$ are stiff (large eigenvalues in the Jacobian). Stiff ODEs require very small step sizes in the backward pass, making training slow or causing NaN gradients. If you observe instability, try: (1) constraining the Jacobian norm of $f$, (2) using a stiffer solver (e.g., implicit methods), or (3) using *direct backpropagation* through the solver steps (more memory, but more stable) for debugging.
]

== Summary

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*Value*]),
  [Core abstraction], [Hidden state as solution to an ODE],
  [Parameters], [Weights of $f$ — shared across all depths],
  [Forward pass], [ODE solver (adaptive number of steps)],
  [Backward pass], [Adjoint method — solve augmented ODE backward],
  [Memory cost], [$O(1)$ in depth (vs $O(L)$ for ResNet)],
  [Compute cost], [Adaptive — proportional to input complexity],
  [Key hyperparameters], [Solver (Euler vs RK45), $t_1$, architecture of $f$, $lambda$ for NFE reg.],
  [Expected MNIST accuracy], [98–99% (with appropriate $f$ and solver)],
)


// ═══════════════════════════════════════════════════════════════════════════════
= Phase 3 Comparison
// ═══════════════════════════════════════════════════════════════════════════════

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*SNN*], [*Neural ODE*]),
  [Time axis], [Discrete timesteps $t = 1, dots, T$], [Continuous $t in [t_0, t_1]$],
  [State update], [Spike + membrane potential reset], [ODE integration],
  [Gradient flow], [Surrogate gradient (approximate)], [Adjoint method (exact)],
  [Compute per sample], [Fixed ($T$ steps)], [Adaptive (NFE varies)],
  [Memory cost (training)], [$O(T dot L)$ — BPTT unrolling], [$O(1)$ — adjoint method],
  [Core efficiency claim], [Sparse spikes → low energy], [Adaptive depth → optimal compute],
  [Interpretability], [Spike raster, firing rates], [Hidden state trajectory, NFE distribution],
  [Biological motivation], [Strong (LIF is a standard neuron model)], [Weak (ODEs as physics metaphor)],
  [Expected MNIST accuracy], [98–99%], [98–99%],
  [Primary bottleneck], [Sequential time loop (T steps)], [ODE solver overhead],
)

== The Unified Thread Across All Six Architectures

Looking across the full course, a single question has been asked in six different ways: *where does the useful computation come from?*

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Architecture*], [*Source of computation*], [*Training finds...*]),
  [ELM], [Random nonlinear projection], [Best linear readout],
  [Transformer], [Pairwise attention between tokens], [Which tokens to attend to],
  [RBM], [Energy landscape over visible-hidden pairs], [Low-energy configurations for real data],
  [ESN], [Chaotic recurrent dynamics], [Best linear readout of reservoir state],
  [SOM], [Competitive geometry on a 2D grid], [Topology-preserving prototype positions],
  [SNN], [Sparse spike timing across timesteps], [Weights that produce correct spike patterns],
  [Neural ODE], [Continuous dynamical flow in state space], [Dynamics $f$ that map inputs to classifiable states],
)

The DNN (your starting point) answers: *learned nonlinear transformations, layer by layer, trained by gradient descent end to end.* Every architecture in this course is an alternative answer — each illuminating a different aspect of what computation and learning can be.

== What Comes Next

This course covered the core architectures. Several important directions were not covered, each building on what you now know:

- *Deep Belief Networks*: stack multiple RBMs, fine-tune with backprop. The historical bridge between RBMs and modern deep learning.
- *Liquid State Machines*: spiking neuron equivalent of ESNs. Combines Phase 2 (reservoir computing) with Phase 3 (SNNs).
- *Continuous-time RNNs*: replace the discrete ESN update with a true ODE over hidden state — a Neural ODE applied to sequences.
- *Hamiltonian Neural Networks*: constrain the Neural ODE's dynamics to conserve energy, making them physically consistent for scientific applications.
- *Equilibrium Propagation*: train energy-based models (like RBMs) with a biologically plausible alternative to contrastive divergence, using two phases of settling dynamics.

Each of these is now within reach. The conceptual vocabulary — energy functions, dynamical systems, spike coding, adjoint methods, competitive learning — is in place.

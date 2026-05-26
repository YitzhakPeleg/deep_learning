#set document(title: "Beyond Backprop — Phase 2 Crash Course", author: "")
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

// ─── TITLE ───────────────────────────────────────────────────────────────────

#align(center)[
  #text(size: 22pt, weight: "bold")[Beyond Backprop]
  #v(0.3em)
  #text(size: 15pt)[Phase 2 — Echo State Networks · Self-Organizing Maps]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[Dynamics, chaos, and topology — without a single gradient]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

#outline(indent: 1em, depth: 2)

#v(0.5em)
#line(length: 100%)

// ─── INTRO ───────────────────────────────────────────────────────────────────

= Orientation

Phase 1 covered models that are recognizably "neural networks" — they map inputs to outputs, they have weights, and two of the three even use backpropagation. Phase 2 is a genuine paradigm shift.

Both architectures in this phase share a deep principle: *useful computation can emerge from structure, not from learned weights*. The ESN exploits chaotic recurrent dynamics. The SOM exploits competitive geometry. Neither uses gradient descent anywhere in their core training algorithm.

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Module*], [*Training mechanism*], [*Gradients?*], [*Primary novelty*]),
  [ESN], [Linear regression on readout only], [No], [Reservoir dynamics],
  [SOM], [Competitive + neighborhood update], [No], [Topology preservation],
)

A note on MNIST applicability: both architectures require some adaptation to handle static images, since their natural domain is either temporal sequences (ESN) or low-dimensional continuous data (SOM). These adaptations are themselves instructive — they reveal the inductive biases baked into each architecture.


// ═══════════════════════════════════════════════════════════════════════════════
= Module 4 — Echo State Network (ESN)
// ═══════════════════════════════════════════════════════════════════════════════

== Core Idea

An ESN is a large recurrent neural network where *all weights except the output layer are fixed at initialization and never trained*. The recurrent core — called the *reservoir* — is a randomly connected network of neurons whose dynamics transform the input into a rich, high-dimensional trajectory through state space. A simple linear readout is then trained to extract the answer from this trajectory.

The fundamental claim is remarkable: a random recurrent network, if initialized correctly, is already a powerful computational substrate. You do not need to train it — you just need to read from it correctly.

#insight[
  The reservoir is analogous to a physical system being used as a computer. A bucket of water, when perturbed, produces complex ripple patterns on its surface — patterns that can be read off by sensors to perform computation. This is not a metaphor; "liquid state machines" (the spiking neuron equivalent of ESNs) were literally inspired by this idea. The ESN formalizes it for rate-coded neurons.
]

== Architecture

An ESN has four components:

- *Input weights* $bold(W)_"in" in RR^(N times d)$: project the input into the reservoir. Fixed, random.
- *Reservoir weights* $bold(W)_"res" in RR^(N times N)$: the recurrent connections within the reservoir. Fixed, random, sparse.
- *Output weights* $bold(W)_"out" in RR^(C times N)$: read from the reservoir state. *The only trained weights.*
- *Reservoir state* $bold(x)(t) in RR^N$: the internal state of the reservoir at time $t$.

Typical sizes: $d = 784$ (MNIST pixels), $N = 500$–$5000$ reservoir neurons, $C = 10$ classes.

== Reservoir Dynamics

At each timestep $t$, the reservoir state updates as:

$ bold(x)(t) = (1 - alpha) bold(x)(t-1) + alpha, tanh(bold(W)_"res" bold(x)(t-1) + bold(W)_"in" bold(u)(t) + bold(b)) $

where:
- $bold(u)(t) in RR^d$ is the input at time $t$
- $alpha in (0, 1]$ is the *leak rate* — how fast the reservoir forgets its past state
- $bold(b) in RR^N$ is a fixed random bias
- $tanh$ is applied elementwise

When $alpha = 1$ this reduces to standard Elman RNN dynamics:

$ bold(x)(t) = tanh(bold(W)_"res" bold(x)(t-1) + bold(W)_"in" bold(u)(t) + bold(b)) $

The leak rate introduces an *exponential moving average* over time, controlling the effective memory timescale of the reservoir. Small $alpha$ = long memory (slow dynamics). Large $alpha$ = short memory (fast dynamics).

== The Echo State Property

The defining requirement of an ESN is the *echo state property* (ESP): the reservoir state $bold(x)(t)$ must be determined entirely by the input history $bold(u)(t), bold(u)(t-1), dots$ — not by the initial state $bold(x)(0)$.

Formally, for any two initial states $bold(x)_a(0)$ and $bold(x)_b(0)$, driven by the same input sequence:

$ ||bold(x)_a(t) - bold(x)_b(t)|| -> 0 quad "as" t -> infinity $

The initial conditions must be "washed out" by the input. If this holds, the reservoir is a well-defined function of input history — it *echoes* the input rather than its own initialization.

A sufficient condition (not tight, but widely used in practice): the *spectral radius* of $bold(W)_"res"$ satisfies:

$ rho(bold(W)_"res") < 1 $

where $rho(bold(W)_"res") = max_i |lambda_i|$ is the largest absolute eigenvalue.

#note[
  The condition $rho < 1$ guarantees the ESP for zero input, but the true boundary depends on input statistics. In practice, ESNs are initialized with a target spectral radius (typically $rho = 0.9$–$0.99$) by first generating a random sparse matrix and then rescaling it:
  $ bold(W)_"res" <- bold(W)_"res" dot frac(rho_"target", rho(bold(W)_"res")) $
  This rescaling is the only "design" choice for the reservoir — everything else is random.
]

== The Edge of Chaos

The spectral radius controls the regime of reservoir dynamics:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Spectral radius*], [*Regime*], [*Behavior*]),
  [$rho << 1$], [Stable / fading], [State collapses to fixed point. Short memory. Poor at temporal tasks.],
  [$rho approx 1$], [Edge of chaos], [Rich, non-repeating dynamics. Long memory. Best performance.],
  [$rho > 1$], [Chaotic], [ESP may fail. State diverges or oscillates wildly. Unstable.],
)

The "edge of chaos" at $rho approx 1$ is where the reservoir has the richest dynamics: it neither forgets immediately nor diverges. Long-range temporal dependencies can propagate through the network state. This is why $rho = 0.9$–$0.99$ is the standard initialization target.

#insight[
  The edge of chaos is not just an ESN concept — it appears throughout complex systems theory. Cellular automata near the phase transition between order and chaos are maximally computationally expressive (Langton, 1990). Cortical neural circuits are hypothesized to operate near criticality for the same reason. The ESN formalizes this intuition for recurrent neural networks.
]

== Reading MNIST as a Sequence

MNIST images are static — there is no natural time axis. We impose one by feeding the image pixel-by-pixel or row-by-row:

*Option A — Row-by-row* (recommended): feed one row of 28 pixels per timestep, for $T = 28$ timesteps. Input at each step: $bold(u)(t) in RR^(28)$.

*Option B — Pixel-by-pixel*: feed one pixel per timestep, for $T = 784$ timesteps. Input at each step: $bold(u)(t) in RR^1$. Longer sequence, more reservoir transients, slower.

After all $T$ timesteps, the reservoir has processed the entire image and its state $bold(x)(T)$ encodes a summary of the full sequence. We classify from this final state.

Optionally, concatenate all intermediate states into a matrix $bold(X) in RR^(T times N)$ and train the readout on the full trajectory, not just the endpoint. This typically improves accuracy.

== Readout Training

Collect reservoir states for all $N_"train"$ training images. If using only the final state, stack them into:

$ bold(X)_"states" in RR^(N_"train" times N) $

Train the output weights $bold(W)_"out" in RR^(N times C)$ by *ridge regression* — exactly as in ELM:

$ bold(W)_"out" = (bold(X)_"states"^top bold(X)_"states" + lambda bold(I))^(-1) bold(X)_"states"^top bold(Y) $

where $bold(Y) in RR^(N_"train" times C)$ is the one-hot label matrix and $lambda$ is the regularization coefficient.

Inference for a new image: run the reservoir for $T$ steps, collect final state $bold(x)(T)$, compute:

$ hat(bold(y)) = "softmax"(bold(W)_"out"^top bold(x)(T)) $

No backpropagation. No iterative optimization. The entire training is: run reservoir on all training images, collect states, solve one linear system.

== Reservoir Initialization Details

The reservoir matrix $bold(W)_"res"$ is sparse — typically connectivity $p = 0.1$ (10% of entries are nonzero). The initialization procedure:

+ Generate a random $N times N$ matrix with entries drawn from $cal(N)(0, 1)$, then zero out $1-p$ fraction of entries randomly.
+ Compute the spectral radius $rho(bold(W)_"res")$ via eigendecomposition.
+ Rescale: $bold(W)_"res" <- bold(W)_"res" dot rho_"target" / rho(bold(W)_"res")$.

The input weights $bold(W)_"in"$ are typically dense, with entries drawn uniformly from $[-sigma_"in", sigma_"in"]$ where $sigma_"in"$ controls input scaling — a hyperparameter that sets how strongly the input drives the reservoir.

== Hyperparameter Sensitivity

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Hyperparameter*], [*Effect*]),
  [Spectral radius $rho$], [Memory timescale. Most important hyperparameter.],
  [Reservoir size $N$], [Capacity. Bigger = better, up to computational limits.],
  [Leak rate $alpha$], [Temporal smoothing. Small = slow dynamics, large = fast.],
  [Input scaling $sigma_"in"$], [How strongly input drives reservoir vs. its own dynamics.],
  [Connectivity $p$], [Sparsity. Lower $p$ = less interference between neurons.],
  [Regularization $lambda$], [Readout overfitting. Tune by cross-validation.],
)

#gotcha[
  The spectral radius $rho$ and the input scaling $sigma_"in"$ interact. A strong input signal can push the reservoir into saturation (all neurons near $pm 1$), effectively killing the dynamics regardless of $rho$. If performance is poor, check whether the reservoir states are saturating by monitoring the distribution of $bold(x)(T)$ values — they should be spread across $(-1, 1)$, not clustered near the extremes.
]

#gotcha[
  The echo state property is a property of the *untrained* reservoir. After you collect reservoir states, the ESP is no longer relevant — you are just doing linear regression. But if the ESP does not hold during data collection, the states are initial-condition-dependent and your training data is inconsistent. Always verify the ESP before collecting states by running the same input through two different initializations of $bold(x)(0)$ and checking that the states converge.
]

== What to Observe

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Experiment*], [*What you learn*]),
  [Vary $rho$: 0.5, 0.9, 0.99, 1.1], [Edge of chaos in action — performance peaks near 1],
  [Final state vs full trajectory readout], [How much temporal info is in intermediate states],
  [Row-by-row vs pixel-by-pixel], [Tradeoff between sequence length and information granularity],
  [Vary $N$: 100, 500, 2000], [Reservoir capacity vs computation cost],
  [Visualize reservoir states], [High-dimensional trajectory through state space for each digit class],
)

== Summary

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*Value*]),
  [Learned parameters], [$bold(W)_"out"$ only],
  [Training cost], [One linear system solve, after one pass through data],
  [Core idea], [Random recurrent dynamics as a feature extractor],
  [Key hyperparameter], [Spectral radius $rho$ of reservoir],
  [Expected MNIST accuracy], [90–95% (depends heavily on $rho$, $N$, readout strategy)],
  [Natural domain], [Temporal sequences — MNIST requires sequentialization],
)


// ═══════════════════════════════════════════════════════════════════════════════
= Module 5 — Self-Organizing Map (SOM)
// ═══════════════════════════════════════════════════════════════════════════════

== Core Idea

A SOM is an unsupervised learning algorithm that maps high-dimensional input data onto a low-dimensional grid — typically 2D — while *preserving topology*: inputs that are similar in the original space should land near each other on the grid.

There is no loss function in the conventional sense. There is no output layer. There is no backpropagation. Training is a sequence of competitive updates: each input activates exactly one neuron (the winner), and that neuron plus its neighbors move toward the input.

The result is a *topographic map* of the data manifold, compressed onto a 2D grid. For MNIST, the map will spontaneously organize so that visually similar digits (3 and 8, 4 and 9, 1 and 7) cluster together.

== Architecture

A SOM consists of:

- A 2D grid of $K times K$ neurons (typical: $K = 10$–$30$ for MNIST)
- Each neuron $i$ has a *prototype vector* (also called weight vector or codebook vector): $bold(w)_i in RR^d$
- The set of all prototypes: ${ bold(w)_i }_{i=1}^{K^2}$

For MNIST: $d = 784$, so each prototype is a point in 784-dimensional space that can be reshaped and visualized as a 28×28 image.

There are no "layers" in the DNN sense. The grid is purely a topological structure — it defines which neurons are neighbors of which. The grid metric (who is close to whom) is what drives organization.

== Distance and the Best Matching Unit

For each input $bold(x) in RR^d$, find the *Best Matching Unit* (BMU) — the neuron whose prototype is closest to the input:

$ i^* = arg min_i ||bold(x) - bold(w)_i||_2^2 $

This is a nearest-neighbor search over all $K^2$ prototype vectors. The BMU "wins" the competition — it and its neighbors will update.

Note that $||bold(x) - bold(w)_i||$ is the *Euclidean distance in input space*, not a dot product. The SOM is based on distance, not activation. This is fundamentally different from a DNN neuron, which computes $sigma(bold(w)^top bold(x) + b)$.

== Neighborhood Function

The neighborhood function $h(i, i^*, t)$ defines how strongly neuron $i$ should update when neuron $i^*$ wins at time $t$:

$ h(i, i^*, t) = exp(- frac(||bold(r)_i - bold(r)_(i^*)||_"grid"^2, 2 sigma(t)^2)) $

where:
- $bold(r)_i in RR^2$ is the *grid position* of neuron $i$ (not its prototype — its position on the 2D map)
- $||bold(r)_i - bold(r)_(i^*)||_"grid"$ is the Euclidean distance between neurons $i$ and $i^*$ *on the grid*
- $sigma(t)$ is the *neighborhood radius* at time $t$, which decays over training

The neighborhood function is a Gaussian centered on the BMU in grid space. Neurons near the BMU on the grid update strongly; neurons far away update weakly or not at all.

== Weight Update Rule

For each training sample $bold(x)$, update all prototypes:

$ bold(w)_i <- bold(w)_i + eta(t) dot h(i, i^*, t) dot (bold(x) - bold(w)_i) $

where $eta(t)$ is the learning rate at time $t$.

Unpacking this:
- $(bold(x) - bold(w)_i)$ is the direction from the prototype toward the input — move the prototype closer to the input
- $h(i, i^*, t)$ scales the move: the BMU moves the most, neighbors move less, distant neurons barely move
- $eta(t)$ scales the overall step size

This is *not* gradient descent. There is no loss function whose gradient is $h(i, i^*, t)(bold(x) - bold(w)_i)$. It is a biologically motivated competitive rule: the winner and its neighbors learn to represent the current input.

#note[
  There *is* a related energy function whose minimization approximates SOM training under certain conditions. Kohonen (1991) showed that SOM minimizes an expected quantization error weighted by the neighborhood function. But this is a post-hoc justification — the algorithm was designed from competitive learning principles, not from the energy function. Unlike the RBM, there is no clean probabilistic interpretation of what the SOM is optimizing.
]

== Decay Schedules

Both $eta(t)$ and $sigma(t)$ must decay over training. Standard schedules:

$ eta(t) = eta_0 dot exp(- t / tau_eta) $

$ sigma(t) = sigma_0 dot exp(- t / tau_sigma) $

where $t$ is the current iteration (or epoch), and $tau_eta$, $tau_sigma$ are time constants.

*Phase interpretation:*

- *Early training* (large $sigma$, large $eta$): the neighborhood covers most of the map. Every input causes a large-scale reorganization. The map finds its global topological structure.
- *Late training* (small $sigma$, small $eta$): only the BMU and its immediate neighbors update. The map fine-tunes local quantization. Prototypes converge to local cluster centers.

The two-phase nature is crucial. If $sigma$ decays too fast, the map freezes before finding the right global topology — you get a "folded" map where topologically distant neurons end up representing similar inputs. If $sigma$ decays too slow, the map never converges to sharp local structure.

#gotcha[
  The two decay rates $tau_eta$ and $tau_sigma$ are the most consequential hyperparameters, and they are easy to get wrong. A common mistake is using a fixed number of epochs with linear decay, which works but is less robust than exponential decay matched to the dataset size. For MNIST with 60,000 samples and a $20 times 20$ grid, a reasonable starting point is $sigma_0 = 10$ (half the grid width), $sigma_"final" = 0.5$, $eta_0 = 0.5$, $eta_"final" = 0.01$, over 10–20 epochs.
]

== Topology Preservation — The Key Property

The SOM preserves topology in a precise sense. If inputs $bold(x)_a$ and $bold(x)_b$ are close in input space, their BMUs $i^*_a$ and $i^*_b$ will be close in grid space (after training).

This means the 2D grid is a *continuous map* of the data manifold. The grid "unfolds" the manifold and lays it flat. For MNIST:

- Digit classes form contiguous regions on the grid
- Visually similar classes (e.g., 3 and 8) are adjacent regions
- Visually distinct classes (e.g., 1 and 0) are far apart on the grid

This topology is not imposed — it emerges from the competitive dynamics.

== Using SOM for MNIST Classification

The SOM is trained *without labels*. After training, we assign a class label to each neuron:

$ "label"(i) = arg max_c sum_(n : i^*_n = i) bold(1)[y_n = c] $

Each neuron is labeled with the majority class among all training samples that mapped to it as BMU. At inference:

+ Find the BMU $i^*$ for the new input $bold(x)$
+ Return $"label"(i^*)$

This is the simplest approach. More sophisticated: use $k$ nearest BMUs and vote, or use a softmax over distances to nearby labeled neurons.

#note[
  Some neurons may receive no training samples as their BMU — they are *dead neurons*. This happens when the initialization places a prototype in a low-density region. Dead neurons do not contribute to classification and effectively reduce the map's capacity. Monitoring the fraction of dead neurons is a useful diagnostic. If many neurons are dead, increase $sigma_0$ or reduce the map size.
]

== Quantization Error and Topographic Error

Two standard metrics for evaluating a trained SOM (independent of classification accuracy):

*Quantization error*: average distance from each training sample to its BMU:

$ E_Q = frac(1, N) sum_(n=1)^N ||bold(x)^((n)) - bold(w)_(i^*_n)||_2 $

Measures how well the prototypes represent the data. Lower is better.

*Topographic error*: fraction of training samples for which the BMU and second-BMU are not adjacent on the grid:

$ E_T = frac(1, N) sum_(n=1)^N bold(1)["BMU and 2nd-BMU of" bold(x)^((n)) "are not grid-adjacent"] $

Measures topology preservation. Lower is better. A well-trained SOM should have $E_T < 0.05$.

These two metrics are in tension: a larger map reduces $E_Q$ (more neurons = better quantization) but can increase $E_T$ if the map folds. The right map size balances both.

== Visualizing the Trained Map

The SOM produces two standard visualizations:

*U-matrix (Unified Distance Matrix)*: for each neuron $i$, compute the average distance to its grid neighbors:

$ U_i = frac(1, |cal(N)(i)|) sum_(j in cal(N)(i)) ||bold(w)_i - bold(w)_j||_2 $

High $U_i$ values = the neuron sits on a boundary between clusters. Low $U_i$ values = the neuron is deep inside a cluster. The U-matrix is a topographic map of the data clusters.

*Component planes*: for each input dimension $k$, plot the value of $w_(i,k)$ for all neurons $i$ as a heatmap. For MNIST, each component plane is one pixel's weight across the entire map — collectively they show which pixels drive which regions.

*Prototype images*: reshape each $bold(w)_i in RR^(784)$ to $28 times 28$ and display it. The grid of prototype images shows what each neuron has learned to represent. This is the most intuitive visualization — you will see recognizable digit shapes arranged spatially.

#insight[
  The prototype image visualization is the SOM's unique gift. Unlike a DNN (where weights are in a high-dimensional space with no obvious visual interpretation) or a Transformer (where attention maps show relationships but not learned representations), the SOM's prototypes *are* the data. Each neuron is a learned exemplar — a point in input space that the neuron has learned to represent. This makes the SOM one of the most interpretable neural network architectures.
]

== What to Observe

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Experiment*], [*What you learn*]),
  [Visualize prototype images on grid], [Digit classes form contiguous spatial regions],
  [Plot U-matrix], [Cluster boundaries emerge as ridges],
  [Vary map size: $10 times 10$, $20 times 20$, $30 times 30$], [Tradeoff between resolution and dead neurons],
  [Fast vs slow $sigma$ decay], [Global topology vs local convergence tradeoff],
  [Count dead neurons], [Map utilization — are all neurons contributing?],
  [Plot class regions (color each neuron by majority label)], [Topology of the digit manifold],
  [Compare $E_Q$ and $E_T$ across runs], [Quantization vs topology preservation tradeoff],
)

#gotcha[
  SOM training is *not reproducible* without fixing the random seed, and even then it is sensitive to the order in which samples are presented. Two runs with the same hyperparameters can produce maps that are mirror images, rotations, or different folds of the same underlying topology. This is not a bug — the map can organize in any orientation. When comparing runs, use $E_Q$ and $E_T$ as metrics, not visual appearance.
]

== Summary

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*Value*]),
  [Model type], [Unsupervised topology-preserving map],
  [Training], [Competitive + neighborhood update (no gradients)],
  [Core "flow"], [Nearest-neighbor search + prototype update],
  [Unique capability], [Visualizable 2D map of high-dimensional data],
  [Classification], [Post-hoc majority voting per neuron],
  [Expected MNIST accuracy], [85–92% (not the point — visualization is)],
  [Key hyperparameters], [Map size $K$, $sigma_0$, $tau_sigma$, $eta_0$, $tau_eta$],
)


// ═══════════════════════════════════════════════════════════════════════════════
= Phase 2 Comparison
// ═══════════════════════════════════════════════════════════════════════════════

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*ESN*], [*SOM*]),
  [Training mechanism], [Linear regression on reservoir states], [Competitive neighborhood update],
  [Gradients?], [No], [No],
  [Supervised?], [Yes (readout trained with labels)], [No (labels only used post-hoc)],
  [Core abstraction], [Temporal dynamics as features], [Topographic map of data manifold],
  [What is learned?], [Output weights only], [Prototype positions in input space],
  [Interpretability], [Reservoir state trajectories], [Prototype images + U-matrix],
  [Natural domain], [Temporal sequences], [Low-to-medium dimensional continuous data],
  [MNIST adaptation], [Serialize image as sequence], [Direct (flatten to vector)],
  [Expected accuracy], [90–95%], [85–92%],
  [Main limitation], [Requires sequentialization for static data], [Accuracy lower than discriminative models],
)

== The Conceptual Thread

Phase 1 asked: *what kind of objective should we optimize?* (analytic least squares, cross-entropy, approximate likelihood).

Phase 2 asks a different question: *what kind of structure should the computation have?*

- The *ESN* answer: structure the computation as a dynamical system. Let chaos and recurrence do the heavy lifting. Read off the answer from the system's state trajectory.

- The *SOM* answer: structure the computation as a competitive geometry. Let inputs fight over neurons, and let neighbors share what they learn. Read off the answer from the topology of the resulting map.

Neither architecture is asking "what is the gradient of a loss?" They are asking "what kind of computation naturally produces useful representations?" This is the shift that defines Phase 2 — and it is preparation for Phase 3, where even the notion of what a "neuron" is will change.

== Looking Ahead to Phase 3

Phase 3 introduces two architectures that depart even further from the DNN template:

- *SNNs*: neurons communicate via discrete events in time (spikes), not continuous activations. The "computation" is distributed across time, not across layers.
- *Neural ODEs*: the network is not a sequence of discrete layers but a continuous-time dynamical system. "Depth" becomes a real number, not an integer.

Both Phase 3 architectures build directly on Phase 2 intuitions: the ESN's dynamical systems perspective is a stepping stone to Neural ODEs; the SOM's event-driven competitive dynamics is a stepping stone to understanding spike-based computation.

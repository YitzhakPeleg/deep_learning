#set document(title: "Beyond Backprop — Introduction", author: "")
#set page(margin: (x: 2.5cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt)
#set math.equation(numbering: "(1)")
#show math.equation.where(block: true): it => pad(y: 0.5em, it)

#let note(body) = block(
  fill: luma(235),
  inset: 10pt,
  radius: 4pt,
  width: 100%,
  body
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
  #text(size: 26pt, weight: "bold")[Beyond Backprop]
  #v(0.4em)
  #text(size: 13pt, fill: luma(80))[A math-first crash course in alternative neural network architectures]
  #v(0.8em)
  #line(length: 60%)
  #v(0.8em)
  #text(size: 11pt)[Goal: train every major non-standard architecture on MNIST, from scratch, understanding the math behind each one.]
]

#v(2em)

// ─── MOTIVATION ──────────────────────────────────────────────────────────────

= Why Look Beyond Gradient Descent?

Most practical deep learning today converges on a single recipe: a differentiable architecture, a loss function, and stochastic gradient descent with backpropagation. This recipe is powerful — but it is one point in a much larger space of possible learning machines.

Looking beyond it is valuable for three reasons.

*Understanding.* Gradient descent and backpropagation are not fundamental laws of learning — they are engineering choices. Understanding what alternatives exist, and why they work, sharpens intuition about what learning actually is. Why does a randomly initialized network already contain useful structure? What does it mean to learn a probability distribution rather than a classifier? How can useful computation emerge from chaos?

*Efficiency.* The brain performs roughly $10^{15}$ synaptic operations per second on roughly 20 watts. A GPU performing a comparable number of floating-point operations consumes kilowatts. The gap is not just engineering — it reflects a fundamentally different computational substrate. Spiking networks, reservoir computers, and energy-based models are attempts to close this gap by rethinking what a "computation" is.

*History and context.* Many ideas now considered peripheral — energy-based models, competitive learning, dynamical systems — were mainstream before the deep learning era and are returning in new forms. Knowing them gives you the ability to read the literature across five decades, not just the last ten years.

// ─── PREREQUISITES ───────────────────────────────────────────────────────────

= Prerequisites

This course assumes you are already comfortable with two architectures:

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Architecture*], [*What you should know*]),
  [DNN], [Forward pass, backpropagation, SGD, cross-entropy loss, activation functions],
  [Hopfield Network], [Energy function, Hebbian weight rule, attractor dynamics, pattern retrieval],
)

You should also be comfortable with: linear algebra (matrix multiplication, eigenvalues, pseudoinverse), basic probability (Bayes' rule, Bernoulli distributions, expectations), and calculus (chain rule, partial derivatives, ODEs at an introductory level).

The ODE content in Phase 3 goes deeper into differential equations — a brief refresher is included there.

// ─── THE BENCHMARK ───────────────────────────────────────────────────────────

= The Common Benchmark: MNIST

Every architecture in this course is applied to the same dataset: *MNIST*, 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels.

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Split*], [*Size*]),
  [Training], [60,000 images],
  [Test], [10,000 images],
  [Classes], [10 (digits 0–9)],
  [Input dimension], [784 (28×28, flattened)],
  [Pixel values], [$[0, 1]$ after normalization],
)

MNIST is intentionally simple — a state-of-the-art DNN reaches 99.7%+ accuracy. This simplicity is a feature, not a limitation. When the benchmark is easy, architectural differences are isolated from dataset difficulty. You can focus on *how* each model learns, not on tuning it to convergence.

Some architectures require adaptation: the ESN needs MNIST serialized as a temporal sequence; the SOM works unsupervised and assigns labels post-hoc; the RBM is generative and requires binarization. Each adaptation is documented in the relevant phase and is itself instructive — it reveals the inductive biases of the architecture.

// ─── COURSE STRUCTURE ────────────────────────────────────────────────────────

= Course Structure

The course is organized into three phases, ordered by conceptual distance from the DNN+backprop baseline.

#v(0.5em)

*Phase 1 — Familiar math, new training rules*

The models in Phase 1 are structurally similar to DNNs but replace gradient descent with a different training mechanism. The forward pass is recognizable; the learning algorithm is not.

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Architecture*], [*Training*], [*Key idea*]),
  [Extreme Learning Machine], [Analytic least squares], [Random projection + linear readout],
  [Transformer], [Backprop (but coded from scratch)], [Self-attention between input tokens],
  [Restricted Boltzmann Machine], [Contrastive divergence], [Energy-based generative model],
)

#v(0.5em)

*Phase 2 — Different computational paradigm*

Phase 2 models do not optimize a loss function in any conventional sense. Computation emerges from dynamics (ESN) or geometry (SOM). Neither uses gradients anywhere in core training.

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Architecture*], [*Training*], [*Key idea*]),
  [Echo State Network], [Linear regression on reservoir states], [Chaotic recurrence as a feature extractor],
  [Self-Organizing Map], [Competitive neighborhood update], [Topology-preserving map of data manifold],
)

#v(0.5em)

*Phase 3 — Rethinking what a neuron and a layer are*

Phase 3 questions the most basic assumptions: what is an activation (SNN), and what is a layer (Neural ODE). These are the deepest departures from the DNN template and carry the most mathematical weight.

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Architecture*], [*Training*], [*Key idea*]),
  [Spiking Neural Network], [BPTT + surrogate gradients], [Binary spikes encode information in timing],
  [Neural ODE], [Adjoint method], [Continuous-depth dynamical system],
)

// ─── READING GUIDE ───────────────────────────────────────────────────────────

= How to Read Each Module

Every module in Phases 1–3 follows the same structure:

+ *Core idea* — one-paragraph intuition, no math. Read this first and sit with it before continuing.
+ *Architecture* — what the network looks like, what its components are.
+ *Math* — the full derivation of the training rule and forward pass.
+ *Applied to MNIST* — how the architecture handles static image data, including any necessary adaptations.
+ *What to observe* — specific experiments to run and what they reveal.
+ *Gotchas* — the things that are easy to get wrong and that most tutorials skip.
+ *Summary table* — key properties at a glance for later reference.

The math sections do not skip steps. If a derivation feels obvious, move quickly. If it feels hard, slow down — the difficulty is usually pointing at something genuinely important.

#insight[
  The most useful thing to track across all modules is the answer to one question: *where does the useful representation come from?* In DNNs, it comes from end-to-end gradient descent shaping every layer. In ELMs, from random projection. In RBMs, from energy minimization. In ESNs, from chaotic dynamics. In SOMs, from competitive geometry. In SNNs, from spike timing. In Neural ODEs, from continuous flow. Each answer is a different theory of what a good representation is and how to find one.
]

// ─── OVERVIEW TABLE ──────────────────────────────────────────────────────────

= Full Architecture Overview

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 7pt,
  align: left,
  table.header([*Architecture*], [*Phase*], [*Gradients?*], [*Generative?*], [*Expected MNIST*]),
  [Extreme Learning Machine],  [1], [No],        [No],  [96–97%],
  [Transformer],               [1], [Yes],       [No],  [98.5–99%],
  [Restricted Boltzmann Machine],[1],[Approx.], [Yes], [~98% (DBN)],
  [Echo State Network],        [2], [No],        [No],  [90–95%],
  [Self-Organizing Map],       [2], [No],        [No],  [85–92%],
  [Spiking Neural Network],    [3], [Surrogate], [No],  [98–99%],
  [Neural ODE],                [3], [Adjoint],   [No],  [98–99%],
)

#note[
  Accuracy figures are indicative, not competitive benchmarks. The goal is not to maximize MNIST accuracy — a simple CNN already does that. The goal is to understand each architecture well enough to implement it, tune it, and reason about why it works.
]

// ─── NOTATION ────────────────────────────────────────────────────────────────

= Notation Reference

The following conventions are used consistently across all three phase documents.

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Symbol*], [*Meaning*]),
  [$bold(X) in RR^(N times d)$], [Data matrix: $N$ samples, $d$ features],
  [$bold(y), bold(Y)$], [Labels: vector or one-hot matrix],
  [$bold(W), bold(beta)$], [Weight matrices (subscripted per role)],
  [$bold(h), bold(z), bold(x)$], [Hidden states or activations (context-dependent)],
  [$sigma(dot)$], [Sigmoid function $1/(1+e^{-x})$, or generic activation],
  [$Theta(dot)$], [Heaviside step function (SNN)],
  [$rho(bold(A))$], [Spectral radius of matrix $bold(A)$],
  [$|| dot ||_F$], [Frobenius norm],
  [$|| dot ||_2$], [Euclidean norm],
  [$bold(A)^dagger$], [Moore–Penrose pseudoinverse of $bold(A)$],
  [$cal(L)$], [Loss function],
  [$eta$], [Learning rate],
  [$EE[dot]$], [Expectation],
  [$bold(1)[dot]$], [Indicator function],
  [$f(dot; bold(theta))$], [A neural network parameterized by $bold(theta)$],
)

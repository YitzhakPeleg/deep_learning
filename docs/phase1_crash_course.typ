#set document(title: "Beyond Backprop — Phase 1 Crash Course", author: "")
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
  #text(size: 15pt)[Phase 1 — ELM · Transformer · RBM]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[A math-first crash course toward MNIST on every architecture]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

#outline(indent: 1em, depth: 2)

#v(0.5em)
#line(length: 100%)

// ─── INTRO ───────────────────────────────────────────────────────────────────

= Orientation

You already know two architectures: *Hopfield networks* (energy minimization, Hebbian weights) and standard *DNNs* (gradient descent + backpropagation). This course builds outward from that foundation.

Phase 1 covers three architectures ordered by conceptual distance from what you know:

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Module*], [*Training mechanism*], [*Uses gradients?*], [*Conceptual jump*]),
  [ELM], [Least-squares (analytic)], [No], [Small],
  [Transformer], [Backprop, but structured], [Yes], [Medium],
  [RBM], [Contrastive divergence], [Approximate], [Large],
)

All three will be applied to *MNIST*: 60,000 grayscale 28×28 images of handwritten digits, 10 classes.

Throughout this document, we use the following conventions:

- $bold(X) in RR^(N times d)$ — data matrix ($N$ samples, $d$ features)
- $bold(y) in RR^(N times C)$ — one-hot label matrix ($C$ classes)
- $bold(W)$ — weight matrix (subscripted per layer/role)
- $sigma(dot)$ — a generic activation function


// ═══════════════════════════════════════════════════════════════════════════════
= Module 1 — Extreme Learning Machine (ELM)
// ═══════════════════════════════════════════════════════════════════════════════

== Core Idea

An ELM is a single-hidden-layer feedforward network with one radical simplification: *the hidden layer weights are set randomly and never changed*. Only the output layer is trained, and it is trained analytically — not iteratively.

This sounds absurd. Why would random projections be useful?

The key insight is the *universal approximation* perspective: a sufficiently wide random projection into a nonlinear space almost surely creates a representation from which the target function can be linearly separated. You are essentially doing *random kitchen sink feature mapping*, then fitting a linear model on top.

#insight[
  ELMs are the neural network equivalent of the kernel trick. You map data into a high-dimensional nonlinear space (the hidden layer), then fit a hyperplane. The difference from SVMs is that the mapping is random rather than kernel-induced — but with enough neurons, random works surprisingly well.
]

== Architecture

For MNIST, flatten each image to a vector $bold(x) in RR^(784)$. The network has three components:

1. *Input layer*: 784 neurons (one per pixel)
2. *Hidden layer*: $L$ neurons, weights $bold(W)_1 in RR^(784 times L)$ and biases $bold(b) in RR^L$ drawn randomly and *frozen*
3. *Output layer*: 10 neurons (one per class), weights $bold(beta) in RR^(L times 10)$ — the only learned parameters

== Forward Pass

The hidden layer computes a random nonlinear projection:

$ bold(H) = sigma(bold(X) bold(W)_1 + bold(1) bold(b)^top) $

where $bold(H) in RR^(N times L)$ is the *hidden layer output matrix*, $bold(1) in RR^N$ is a column of ones (for broadcasting the bias), and $sigma$ is applied elementwise.

The output is:

$ hat(bold(Y)) = bold(H) bold(beta) $

== Training — The Analytic Solution

We want to find $bold(beta)$ that minimizes the squared loss:

$ cal(L)(bold(beta)) = ||bold(H) bold(beta) - bold(Y)||_F^2 $

where $|| dot ||_F$ is the Frobenius norm (sum of squared entries). This is a standard *linear least-squares* problem. Taking the derivative and setting it to zero:

$ frac(diff cal(L), diff bold(beta)) = 2 bold(H)^top (bold(H) bold(beta) - bold(Y)) = 0 $

$ bold(H)^top bold(H) bold(beta) = bold(H)^top bold(Y) $

This is the *normal equation*. Its solution is:

$ bold(beta) = bold(H)^dagger bold(Y) $

where $bold(H)^dagger = (bold(H)^top bold(H))^(-1) bold(H)^top$ is the *Moore–Penrose pseudoinverse* of $bold(H)$.

#note[
  *Why pseudoinverse and not regular inverse?* $bold(H)$ is $N times L$ and generally not square. The pseudoinverse gives the minimum-norm least-squares solution — the $bold(beta)$ with the smallest $||bold(beta)||$ among all solutions that minimize the squared error. When $N > L$ (more samples than neurons), the system is overdetermined and there may be no exact solution; the pseudoinverse gives the best approximation. When $N < L$, it is underdetermined and the pseudoinverse picks the smoothest one.
]

In practice, computing $(bold(H)^top bold(H))^(-1)$ can be numerically unstable when $bold(H)^top bold(H)$ is near-singular. A regularized version (ridge regression) is preferred:

$ bold(beta) = (bold(H)^top bold(H) + lambda bold(I))^(-1) bold(H)^top bold(Y) $

The scalar $lambda > 0$ is a regularization coefficient that improves conditioning and reduces overfitting. This is equivalent to adding an $L_2$ penalty $lambda ||bold(beta)||^2$ to the loss.

== Why Random Weights Work — The Theory

Let $bold(w)_i in RR^d$ be the $i$-th column of $bold(W)_1$, drawn i.i.d. from any continuous distribution. The function computed by the $i$-th hidden neuron is:

$ h_i(bold(x)) = sigma(bold(w)_i^top bold(x) + b_i) $

Huang et al. (2006) proved that for any target function $f$ and any $epsilon > 0$, there exists an $L$ large enough such that an ELM with $L$ hidden neurons approximates $f$ to within $epsilon$ — *with probability 1 over the random weights*. The hidden neurons form a basis for the function space, and $bold(beta)$ finds the right linear combination.

== Inference

Given a new sample $bold(x)_"new"$:

$ hat(bold(y)) = sigma(bold(x)_"new" bold(W)_1 + bold(b)) bold(beta) $

This is two matrix-vector products — nothing else. No iterations, no solver, no numerical integration.

== Sensitivity to Hyperparameters

The hidden size $L$ and the activation function $sigma$ matter enormously:

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Choice*], [*Effect*]),
  [$sigma = $ ReLU], [Random dead neurons are permanent — often underperforms],
  [$sigma = $ sigmoid or tanh], [Smoother random projections — works better empirically],
  [$L$ too small], [Underfitting — random features can't span the function space],
  [$L$ too large], [Overfitting if $lambda = 0$; regularize with $lambda > 0$],
  [Weight distribution], [Uniform $[-1,1]$ vs Gaussian — mild effect, Gaussian slightly better],
)

#gotcha[
  ReLU seems like the natural choice coming from DNNs, but in ELMs it causes problems. A randomly initialized ReLU neuron fires on roughly half the input space. With random weights, the pattern of dead neurons is fixed — you cannot fix it during training. Sigmoid and tanh create dense, smooth random features that span the space more uniformly.
]

== Summary

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*Value*]),
  [Learned parameters], [$bold(beta)$ only ($L times 10$ values)],
  [Training cost], [One pseudoinverse: $O(N L^2)$ or $O(L^3)$],
  [Inference cost], [Two matrix-vector products],
  [Expected MNIST accuracy], [96–97% (with $L approx 1000$, sigmoid)],
  [Training time], [Milliseconds to seconds],
)


// ═══════════════════════════════════════════════════════════════════════════════
= Module 2 — Transformer
// ═══════════════════════════════════════════════════════════════════════════════

== Core Idea

A Transformer is a DNN where the primary operation is *self-attention*: every element of a sequence looks at every other element and decides how much to "attend" to it. There are no convolutions, no recurrence. Position is injected explicitly via encodings.

For MNIST, we treat the 28×28 image as a sequence of 28 row-vectors, each of length 28 — so a sequence of length $T = 28$ where each *token* is one row of pixels.

The Transformer then learns relationships between rows: does the top of a "7" attend to the bottom? Does the middle of a "0" attend to its own circumference?

== Tokenization for MNIST

Let the image $bold(X)_"img" in RR^(28 times 28)$. We define tokens:

$ bold(x)_t in RR^(28), quad t = 1, dots, 28 $

where $bold(x)_t$ is the $t$-th row. These are projected to the model dimension $d_"model"$:

$ bold(z)_t = bold(x)_t bold(W)_e, quad bold(W)_e in RR^(28 times d_"model") $

giving a sequence $bold(Z) in RR^(T times d_"model")$ where $T = 28$.

== Positional Encoding

Self-attention is *permutation equivariant* — if you shuffle the rows, the output is shuffled the same way. The model has no inherent sense of which row comes first. We add a *positional encoding* $bold(P) in RR^(T times d_"model")$ to inject order:

$ bold(P)_(t, 2i) &= sin(t / 10000^(2i \/ d_"model")) $
$ bold(P)_(t, 2i+1) &= cos(t / 10000^(2i \/ d_"model")) $

for position $t$ and dimension index $i$. The sinusoidal pattern has a key property: the encoding of position $t + k$ can be expressed as a *linear function* of the encoding at position $t$, for any offset $k$. This lets the model learn relative positions easily.

The input to the Transformer is then:

$ bold(Z) <- bold(Z) + bold(P) $

== Scaled Dot-Product Attention

This is the core operation. Given a sequence $bold(Z) in RR^(T times d_"model")$, we project it three times:

$ bold(Q) = bold(Z) bold(W)^Q, quad bold(K) = bold(Z) bold(W)^K, quad bold(V) = bold(Z) bold(W)^V $

where $bold(W)^Q, bold(W)^K, bold(W)^V in RR^(d_"model" times d_k)$ are learned projection matrices, and $d_k$ is the key/query dimension.

- *Queries* $bold(Q)$: what this token is looking for
- *Keys* $bold(K)$: what this token has to offer
- *Values* $bold(V)$: what this token will actually contribute if attended to

The attention scores measure query-key compatibility:

$ "Attention"(bold(Q), bold(K), bold(V)) = "softmax"((bold(Q) bold(K)^top) / sqrt(d_k)) bold(V) $

The matrix $bold(Q) bold(K)^top in RR^(T times T)$ is the *attention matrix* — entry $(i, j)$ measures how much token $i$ should attend to token $j$. The softmax normalizes each row to a probability distribution. Multiplying by $bold(V)$ produces a weighted combination of value vectors.

#note[
  *Why divide by $sqrt(d_k)$?* The dot products $bold(Q) bold(K)^top$ grow in magnitude with $d_k$. Large values push the softmax into saturated regions where gradients vanish (the distribution becomes close to a one-hot vector). Dividing by $sqrt(d_k)$ keeps the variance of the dot products approximately constant regardless of dimension.
]

== Multi-Head Attention

Running attention once gives one "view" of the relationships. *Multi-head attention* runs $h$ attention operations in parallel, each with different projections:

$ "head"_i = "Attention"(bold(Z) bold(W)_i^Q, bold(Z) bold(W)_i^K, bold(Z) bold(W)_i^V) $

$ "MultiHead"(bold(Z)) = "Concat"("head"_1, dots, "head"_h) bold(W)^O $

where each $bold(W)_i^Q, bold(W)_i^K, bold(W)_i^V in RR^(d_"model" times d_k)$ with $d_k = d_"model" / h$, and $bold(W)^O in RR^(d_"model" times d_"model")$ projects the concatenated heads back to model dimension.

Different heads learn to attend to different types of relationships simultaneously — one head might track vertical alignment, another might track digit stroke direction.

== Feed-Forward Sublayer

After attention, each token is processed *independently* through a small two-layer network:

$ "FFN"(bold(z)) = "GELU"(bold(z) bold(W)_1 + bold(b)_1) bold(W)_2 + bold(b)_2 $

where $bold(W)_1 in RR^(d_"model" times d_"ff")$, $bold(W)_2 in RR^(d_"ff" times d_"model")$, and $d_"ff" = 4 d_"model"$ by convention. GELU (Gaussian Error Linear Unit) is preferred over ReLU in Transformers:

$ "GELU"(x) = x dot Phi(x) $

where $Phi(x)$ is the standard normal CDF. It smoothly gates activations, outperforming ReLU on most Transformer tasks.

== Transformer Block

One block combines both sublayers with *Pre-LayerNorm* residual connections:

$ bold(Z)' &= bold(Z) + "MultiHead"("LayerNorm"(bold(Z))) $
$ bold(Z)'' &= bold(Z)' + "FFN"("LayerNorm"(bold(Z)')) $

LayerNorm normalizes across the feature dimension for each token independently:

$ "LayerNorm"(bold(z)) = (bold(z) - mu) / sqrt(sigma^2 + epsilon) dot bold(gamma) + bold(delta) $

where $mu$ and $sigma^2$ are the mean and variance of the elements of $bold(z)$, and $bold(gamma), bold(delta) in RR^(d_"model")$ are learned scale and shift parameters.

#note[
  *Pre-norm vs Post-norm*: The original "Attention Is All You Need" paper placed LayerNorm *after* the residual addition (post-norm). Pre-norm (normalizing the input before the sublayer) trains more stably because the residual stream is never normalized away — gradients flow cleanly through the additions. Modern practice uses pre-norm almost universally.
]

== The CLS Token

After $N_"blocks"$ Transformer blocks, we have $T = 28$ output vectors. We need one classification vector. Two options:

1. *Mean pooling*: average all 28 output tokens
2. *CLS token*: prepend a special learnable token $bold(c) in RR^(d_"model")$ to the sequence, making it length $T + 1 = 29$. After processing, take only the CLS output for classification.

The CLS approach is standard (BERT, ViT). The CLS token has no positional meaning of its own; it learns to aggregate global information via attention to all other tokens.

$ bold(z)^("cls") in RR^(d_"model") arrow.r "Linear"(bold(z)^("cls")) in RR^(10) arrow.r "softmax" arrow.r hat(bold(y)) $

== Full Architecture — Data Flow

$ bold(X)_"img" in RR^(28 times 28) $
$ arrow.b quad "tokenize: rows as tokens" $
$ bold(Z) in RR^(28 times d_"model") quad "after linear projection" $
$ arrow.b quad "prepend CLS, add positional encoding" $
$ bold(Z) in RR^(29 times d_"model") $
$ arrow.b quad times N_"blocks" "Transformer blocks" $
$ bold(Z)^"out" in RR^(29 times d_"model") $
$ arrow.b quad "take CLS token, linear + softmax" $
$ hat(bold(y)) in RR^(10) $

== Training

Standard cross-entropy loss and Adam optimizer — backpropagation flows through attention, FFN, and LayerNorm just as in any DNN. The only architectural novelty is that attention scores form part of the computation graph, so gradients flow through the softmax and the $bold(Q) bold(K)^top$ product.

The gradient of the attention output with respect to $bold(Q)$ involves the Jacobian of the softmax, which is:

$ frac(diff, diff bold(Q)) ["softmax"(bold(Q) bold(K)^top / sqrt(d_k)) bold(V)] $

This is non-trivial but computed automatically by any autodiff library. The point is that *nothing fundamentally different from standard backprop is happening* — the graph is just structured differently.

== What to Observe

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Experiment*], [*What you learn*]),
  [Remove positional encoding], [Does row order matter for MNIST? (Surprisingly little)],
  [Visualize attention matrix], [Which rows attend to which — the core interpretability win],
  [1 head vs 8 heads], [Diminishing returns on simple tasks],
  [Vary $d_"model"$: 32, 64, 128], [Transformers are overparameterized for MNIST],
  [Row tokens vs 4×4 patch tokens (49 tokens)], [ViT-style: more tokens, finer spatial info],
)

#gotcha[
  Transformers are sensitive to learning rate. Too high and attention scores collapse to near-uniform (the model stops differentiating between tokens). Too low and training stalls. For MNIST, $lr = 3 times 10^(-4)$ with Adam and a linear warmup over the first 5–10% of steps is a reliable starting point. The warmup matters because in early training, the attention weights are near-uniform and gradients are small — a large initial LR destabilizes the softmax.
]

== Summary

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*Value*]),
  [Core operation], [Scaled dot-product self-attention],
  [Positional information], [Injected via sinusoidal encoding],
  [Training], [Backprop + Adam, with LR warmup],
  [Interpretability], [Attention maps are directly visualizable],
  [Expected MNIST accuracy], [98.5–99% (2 blocks, 4 heads, $d = 64$)],
  [Key hyperparameters], [$d_"model"$, $h$ (heads), $N_"blocks"$, LR schedule],
)


// ═══════════════════════════════════════════════════════════════════════════════
= Module 3 — Restricted Boltzmann Machine (RBM)
// ═══════════════════════════════════════════════════════════════════════════════

== Core Idea

An RBM is a fundamentally different kind of model. It is *generative* and *undirected*. There is no forward pass in the DNN sense. Instead, the model defines a *probability distribution* over input data — it learns what MNIST digits look like, not just how to classify them.

The "restricted" in RBM means there are no connections within the visible layer or within the hidden layer — only between them. This restriction is what makes training tractable.

== Architecture and Energy

An RBM has:
- *Visible layer*: $bold(v) in {0, 1}^d$ — the data (binarized pixel values for MNIST)
- *Hidden layer*: $bold(h) in {0, 1}^m$ — latent features
- *Weights*: $bold(W) in RR^(d times m)$
- *Visible biases*: $bold(b) in RR^d$
- *Hidden biases*: $bold(c) in RR^m$

The model defines an *energy function*:

$ E(bold(v), bold(h)) = - bold(b)^top bold(v) - bold(c)^top bold(h) - bold(v)^top bold(W) bold(h) $

Low energy = high probability. The *joint probability* of a visible-hidden configuration is:

$ P(bold(v), bold(h)) = (1) / (Z) e^(-E(bold(v), bold(h))) $

where $Z$ is the *partition function* — a normalizing constant that sums over all possible configurations:

$ Z = sum_(bold(v), bold(h)) e^(-E(bold(v), bold(h))) $

For binary units with $d = 784$ and $m = 500$, $Z$ involves a sum over $2^(784 + 500)$ terms — astronomically large and completely intractable to compute exactly.

== Conditional Distributions

The key tractability of RBMs: because there are no within-layer connections, the hidden units are *conditionally independent* given the visible units, and vice versa.

*Probability of a single hidden unit being active, given visible:*

$ P(h_j = 1 | bold(v)) = sigma(c_j + sum_i v_i W_(i j)) = sigma(c_j + bold(v)^top bold(W)_j) $

where $sigma(x) = 1/(1 + e^(-x))$ is the sigmoid function and $bold(W)_j$ is the $j$-th column of $bold(W)$.

*Probability of a single visible unit being active, given hidden:*

$ P(v_i = 1 | bold(h)) = sigma(b_i + sum_j h_j W_(i j)) = sigma(b_i + bold(W)_i bold(h)) $

These can be written in vectorized form:

$ P(bold(h) = 1 | bold(v)) = sigma(bold(c) + bold(W)^top bold(v)) $
$ P(bold(v) = 1 | bold(h)) = sigma(bold(b) + bold(W) bold(h)) $

This bidirectionality is what makes RBMs "undirected" — information flows both ways, unlike a DNN.

== The Learning Objective

We want to maximize the *log-likelihood* of the training data:

$ cal(L)(bold(theta)) = sum_(n=1)^N log P(bold(v)^((n))) = sum_(n=1)^N log sum_(bold(h)) P(bold(v)^((n)), bold(h)) $

The gradient of the log-likelihood with respect to $bold(W)$ is:

$ frac(diff log P(bold(v)), diff bold(W)) = underbrace(EE_(bold(h) | bold(v))[bold(v) bold(h)^top], "positive phase") - underbrace(EE_(bold(v), bold(h))[bold(v) bold(h)^top], "negative phase") $

The *positive phase* is tractable: given a data sample $bold(v)$, we can compute $P(bold(h)|bold(v))$ exactly because the hidden units are conditionally independent.

The *negative phase* is the problem. $EE_(bold(v), bold(h))[bold(v) bold(h)^top]$ is an expectation under the *model distribution* — to compute it, we would need to sample from $P(bold(v), bold(h))$, which requires dealing with the intractable $Z$.

== Contrastive Divergence

Hinton (2002) proposed *Contrastive Divergence* (CD-$k$) as a practical approximation. Instead of drawing samples from the true model distribution (which requires running a Markov chain to convergence), we run only $k$ steps of *Gibbs sampling* starting from a real data point.

*CD-1 algorithm (one Gibbs step):*

+ *Positive phase*: Given data $bold(v)^((0))$, sample $bold(h)^((0)) ~ P(bold(h)|bold(v)^((0)))$
+ *Reconstruction*: Sample $bold(v)^((1)) ~ P(bold(v)|bold(h)^((0)))$
+ *Negative phase*: Sample $bold(h)^((1)) ~ P(bold(h)|bold(v)^((1)))$
+ *Update*:

$ Delta bold(W) = eta (bold(v)^((0)) (bold(h)^((0)))^top - bold(v)^((1)) (bold(h)^((1)))^top) $
$ Delta bold(b) = eta (bold(v)^((0)) - bold(v)^((1))) $
$ Delta bold(c) = eta (bold(h)^((0)) - bold(h)^((1))) $

The intuition: the positive phase *raises* the probability of real data; the negative phase *lowers* the probability of reconstructions (which are what the model currently thinks data looks like). The weights adjust to make real data more probable and hallucinations less probable.

#note[
  CD-$k$ is not gradient descent on the log-likelihood. It is a biased estimator — the "negative phase" statistics come from a chain started at data, not from the true model distribution. In practice, CD-1 works well for learning good features even though it does not maximize the likelihood exactly. CD-$k$ with larger $k$ is a better approximation but more expensive. Persistent CD (PCD) maintains a set of persistent chains across mini-batches, which is a better approximation than CD-1.
]

== Gibbs Sampling

Gibbs sampling exploits the conditional independence structure. Because all hidden units are independent given the visible:

$ bold(h) | bold(v): quad h_j ~ "Bernoulli"(sigma(c_j + bold(v)^top bold(W)_j)) quad "independently for all" j $

and all visible units are independent given the hidden:

$ bold(v) | bold(h): quad v_i ~ "Bernoulli"(sigma(b_i + bold(W)_i bold(h))) quad "independently for all" i $

So one Gibbs step = two vectorized sigmoid operations + Bernoulli sampling. This is the "flow" of the RBM — not gradient descent, but stochastic sampling alternating between two layers.

== Using RBM for MNIST Classification

An RBM alone is a *density model*, not a classifier. There are two approaches to classification:

*Approach 1 — Generative classifier*: Train one RBM per digit class. At inference, compute $P(bold(v))$ under each RBM and pick the class with highest probability.

*Approach 2 — Deep Belief Network (DBN)*: Stack two RBMs. Train greedily: first RBM trains on raw pixels, second RBM trains on the hidden activations of the first. Then add a softmax classifier on top and fine-tune the whole stack with backpropagation. This was the key architecture that reinvigorated deep learning around 2006–2007.

== What the Hidden Units Learn

After training, the columns of $bold(W)$ (each of dimension 784) can be reshaped to 28×28 and visualized as images. They are *Gabor-like filters* — edge detectors, stroke detectors, curve detectors. The hidden units are feature detectors that activate when the corresponding pattern is present in the input.

This is unsupervised feature learning: the model discovers what structure exists in the data without any labels.

== What to Observe

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Experiment*], [*What you learn*]),
  [Visualize weight columns $bold(W)_j$ as images], [Features learned = Gabor-like detectors],
  [Reconstruct a digit: $bold(v) -> bold(h) -> bold(v)'$], [How well the model models MNIST],
  [Sample new digits: random $bold(h) -> bold(v)$], [The generative capability],
  [CD-1 vs CD-10 vs PCD], [Approximation quality vs cost],
  [Monitor reconstruction error], [Proxy for training progress (not true likelihood)],
)

#gotcha[
  The training loss (reconstruction error) is *not* the log-likelihood. It is a convenient proxy that decreases as training improves, but it is not what we are optimizing. You cannot compare RBM training loss to DNN cross-entropy — they are different quantities. A better monitor is held-out reconstruction quality, or for small models, AIS (Annealed Importance Sampling) to estimate the true partition function.
]

#gotcha[
  MNIST pixels are in $[0, 1]$ after normalization. The RBM assumes *binary* visible units $bold(v) in {0,1}^d$. For MNIST you have two options: (a) binarize by thresholding at 0.5, or (b) treat pixel values as probabilities and use them directly as the "mean" of the Bernoulli, without sampling. Option (b) (called "mean-field" visible units) works better in practice and avoids information loss from binarization.
]

== Summary

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*Value*]),
  [Model type], [Undirected generative model],
  [Training], [Contrastive divergence (CD-$k$)],
  [Core "flow"], [Gibbs sampling between visible and hidden],
  [Unique capability], [Generates new samples, learns features unsupervised],
  [Classification], [Via DBN stacking + fine-tuning],
  [Expected MNIST accuracy (DBN)], [~98% after fine-tuning],
  [Key hyperparameters], [Hidden size $m$, learning rate $eta$, $k$ in CD-$k$],
)


// ═══════════════════════════════════════════════════════════════════════════════
= Phase 1 Comparison
// ═══════════════════════════════════════════════════════════════════════════════

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Property*], [*ELM*], [*Transformer*], [*RBM*]),
  [Training paradigm], [Analytic (pseudoinverse)], [Backprop + Adam], [Contrastive divergence],
  [Uses gradients?], [No], [Yes], [Approximately],
  [Discriminative?], [Yes], [Yes], [No (generative)],
  [Can generate data?], [No], [No], [Yes],
  [Training time], [Seconds], [Minutes], [Minutes–hours],
  [Inference cost], [O(1) matrix multiply], [O($T^2 d$) per block], [O($d m$) per step],
  [Interpretability], [Weight columns], [Attention maps], [Weight filters],
  [MNIST accuracy], [96–97%], [98.5–99%], [~98% (DBN)],
  [Main limitation], [Limited capacity], [Data hungry], [Intractable likelihood],
)

== The Conceptual Progression

These three models represent three different answers to the question *"what does learning mean?"*

- *ELM*: Learning is finding the best linear combination of random features. The structure of the solution space is simple enough to be solved analytically.

- *Transformer*: Learning is finding which parts of the input to pay attention to. The model builds a structured, interpretable computation graph where relationships are explicit.

- *RBM*: Learning is modeling the probability distribution of the data. The model internalizes what the data *is*, not just how to classify it — a fundamentally different objective.

Phase 2 (ESN, SOM) will push further into dynamics and topology. Phase 3 (SNN, Neural ODE) will question what computation itself looks like.

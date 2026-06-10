#set document(
  title: "Time Series Forecasting with Transformers",
  author: "Notes",
)

#set page(
  paper: "a4",
  margin: (x: 2.5cm, y: 3cm),
  numbering: "1",
)

#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en",
)

#set heading(numbering: "1.1")

#set par(
  justify: true,
  leading: 0.75em,
  spacing: 1.2em,
)

#show heading.where(level: 1): it => {
  v(1.5em)
  text(size: 16pt, weight: "bold", it)
  v(0.5em)
}

#show heading.where(level: 2): it => {
  v(1em)
  text(size: 13pt, weight: "bold", it)
  v(0.3em)
}

#show raw.where(block: true): it => {
  block(
    fill: luma(240),
    inset: 10pt,
    radius: 4pt,
    width: 100%,
    text(font: "New Computer Modern Mono", size: 9.5pt, it),
  )
}

#show raw.where(block: false): it => {
  box(
    fill: luma(240),
    inset: (x: 3pt, y: 1pt),
    radius: 2pt,
    text(font: "New Computer Modern Mono", size: 9.5pt, it),
  )
}

// ── Title block ──────────────────────────────────────────────────────────────

#align(center)[
  #v(2em)
  #text(size: 24pt, weight: "bold")[
    Time Series Forecasting \ with Transformers
  ]
  #v(0.8em)
  #text(size: 12pt, fill: luma(80))[
    Architecture, Design Decisions, and Practical Considerations
  ]
  #v(0.5em)
  #text(size: 10pt, fill: luma(120))[#datetime.today().display()]
  #v(2em)
  #line(length: 100%, stroke: 0.5pt + luma(180))
  #v(1em)
]

// ── Abstract ─────────────────────────────────────────────────────────────────

#block(
  inset: (x: 1.5cm),
)[
  *Abstract.* Transformer models, originally developed for natural language
  processing, have been successfully adapted for time series forecasting. This
  document traces that adaptation step by step — from the structure of a time
  series input, through tokenization and positional encoding, to the output head
  and loss function — and surveys the key architectural variants that have emerged.
]

#v(1em)
#line(length: 100%, stroke: 0.5pt + luma(200))
#v(0.5em)

// ── TOC ──────────────────────────────────────────────────────────────────────

#outline(depth: 2, indent: 1.5em)

#pagebreak()

// ── 1. Background ─────────────────────────────────────────────────────────────

= Background and Motivation

Time series forecasting is one of the oldest problems in machine learning:
given a sequence of observations $x_1, x_2, dots, x_N$, predict $x_{N+1}$
(or a horizon of $H$ future steps). Classical approaches — ARIMA, exponential
smoothing, state-space models — rely on explicit assumptions about stationarity
and linear dynamics.

Deep learning relaxed these assumptions. LSTMs and GRUs could model nonlinear
dynamics and long-range dependencies, but suffered from vanishing gradients and
sequential computation that prevented parallelism.

Transformers solve both problems. Their self-attention mechanism captures
arbitrary-range dependencies in $O(1)$ layers, and their fully parallel
computation makes training on long sequences practical.

== The core analogy

The adaptation from NLP to time series is almost mechanical:

#block(
  fill: luma(245),
  inset: 10pt,
  radius: 4pt,
)[
  #table(
    columns: (auto, auto, auto, auto),
    stroke: none,
    inset: 6pt,
    align: left,
    table.header(
      text(weight: "bold")[Modality],
      text(weight: "bold")[Input token],
      text(weight: "bold")[Tokenization],
      text(weight: "bold")[Output],
    ),
    [Text], [word / subword], [BPE vocabulary], [softmax over vocab],
    [Image (ViT)], [patch $16 times 16$], [linear projection], [CLS → classifier],
    [*Time series*], [*timestep vector*], [*linear projection*], [*regression head*],
  )
]

All three use the same transformer backbone. The differences are entirely
in the input projection and output head.

// ── 2. Problem formulation ────────────────────────────────────────────────────

= Problem Formulation

== Input

A multivariate time series is a sequence of vectors:

$ bold(X) = [bold(x)_1, bold(x)_2, dots, bold(x)_N] in RR^(N times d) $

where $N$ is the sequence (look-back) length and $d$ is the number of features
(channels). Each $bold(x)_i in RR^d$ represents all measured variables at
timestep $i$.

== Forecasting objective

Given $bold(X)$, predict the next $H$ timesteps:

$ hat(bold(Y)) = [hat(bold(x))_{N+1}, dots, hat(bold(x))_{N+H}] in RR^(H times d) $

$H$ is called the *forecast horizon*. Single-step forecasting has $H=1$;
multi-step forecasting has $H > 1$.

== Loss function

Since the output is continuous, the standard loss is mean squared error (MSE):

$ cal(L) = 1/(H dot d) sum_(h=1)^H sum_(j=1)^d (x_{N+h, j} - hat(x)_{N+h, j})^2 $

Mean absolute error (MAE) is also common, and more robust to outliers.
Probabilistic models output distribution parameters $(mu, sigma)$ and optimize
negative log-likelihood instead.

// ── 3. Architecture ───────────────────────────────────────────────────────────

= Transformer Architecture for Time Series

== Input projection

Unlike text, time series has no discrete vocabulary. Each timestep vector
$bold(x)_i in RR^d$ is mapped into the model's working dimension via a
learned linear projection:

$ bold(e)_i = bold(x)_i bold(W)_("proj") + bold(b) quad bold(W)_("proj") in RR^(d times d_"model") $

This is equivalent to the patch projection in ViT — a trainable interface
between raw observation space and the transformer's internal representation.

== Positional encoding

Transformers are permutation-invariant without explicit position information.
Two options are common:

*Sinusoidal encoding* (original Transformer):
$ "PE"(i, 2k) = sin(i / 10000^(2k \/ d_"model")) $
$ "PE"(i, 2k+1) = cos(i / 10000^(2k \/ d_"model")) $

*Timestamp encoding:* replace sinusoidal with actual temporal features —
hour of day, day of week, month, year. Especially useful when data has
strong seasonal patterns or irregular sampling intervals.

The final token embedding is:

$ bold(e)_i = bold(x)_i bold(W)_"proj" + bold(p)_i $

== Self-attention (causal)

For autoregressive forecasting, future timesteps must not be visible.
A causal mask is applied before softmax:

$ bold(A) = "softmax"((bold(Q) bold(K)^top) / sqrt(d_"head") + bold(M)) bold(V) $

where $bold(M)_{i,j} = -infinity$ if $j > i$, else $0$.

This is identical to the causal mask in GPT — each position can only attend
to itself and past positions.

== Feed-forward network

Each transformer block includes a position-wise FFN:

$ "FFN"(bold(z)) = "ReLU"(bold(z) bold(W)_1 + bold(b)_1) bold(W)_2 + bold(b)_2 $

The FFN operates independently on each timestep — it does not mix information
across time. Only attention mixes across the sequence.

== Output head

After $L$ transformer blocks, the output at the last observed position
$bold(z)_N$ is passed through a linear regression head:

$ hat(bold(Y)) = bold(z)_N bold(W)_"out" + bold(b)_"out" quad bold(W)_"out" in RR^(d_"model" times H dot d) $

reshaped to $[H times d]$ to give the full forecast horizon.

Alternatively, a *direct multi-step* head predicts all $H$ steps simultaneously
from the full sequence output $[bold(z)_1, dots, bold(z)_N]$.

// ── 4. Architectural variants ─────────────────────────────────────────────────

= Key Architectural Variants

== Vanilla Transformer (2017)

The original encoder-decoder architecture can be applied directly:
encoder processes the look-back window, decoder autoregressively generates
the forecast. Simple and strong baseline.

*Limitation:* $O(N^2)$ attention complexity makes very long sequences expensive.

== Informer (2021)

Introduced *ProbSparse attention* — observing that in practice, most attention
weights are near-zero, with only a few queries dominating. Informer selects
the top-$u$ queries and computes attention only for those:

$ O(N^2) arrow.r O(N log N) $

Also introduced a *distilling operation* (halving sequence length between
encoder layers) to further reduce complexity.

*Best for:* very long input sequences ($N > 1000$).

== Autoformer (2021)

Replaced standard attention with *Auto-Correlation* — computing correlations
between the series and its time-lagged copies using FFT:

$ cal(R)_(bold(Q) bold(K))(tau) = cal(F)^(-1)(cal(F)(bold(Q)) dot overline(cal(F)(bold(K)))) $

Aggregates top-$k$ period-based dependencies. Also introduced series
decomposition (trend + seasonal) as an internal block.

*Best for:* data with strong periodicity.

== PatchTST (2023)

Applied the ViT patch idea to time series: instead of one token per timestep,
group $P$ consecutive timesteps into a patch:

$ N "timesteps" arrow.r floor(N / P) "patch tokens" $

Benefits:
- Shorter sequence → less attention computation
- Each token sees a local temporal context, not a single point
- Channel-independence: each feature treated as a separate univariate series

*Best for:* general forecasting, strong empirical results across benchmarks.

== Temporal Fusion Transformer — TFT (2020)

Designed specifically for real-world forecasting with heterogeneous inputs:

- *Static covariates* (e.g. store ID, location) — processed by learned embeddings
- *Known future inputs* (e.g. day of week, promotions) — fed to decoder
- *Past observed inputs* — fed to encoder
- *Variable selection networks* — learn which features matter per timestep
- *Quantile output* — produces prediction intervals, not just point estimates

TFT is less a pure transformer and more a carefully engineered forecasting
system that uses attention as one component.

*Best for:* production forecasting with rich metadata and uncertainty requirements.

// ── 5. Discrete vs continuous ─────────────────────────────────────────────────

= Discrete vs Continuous Output

== Point forecasting (continuous)

Standard approach. Output is $hat(bold(x)) in RR^d$, loss is MSE or MAE.
Simple but gives no uncertainty estimate.

== Probabilistic forecasting

Output distribution parameters instead of point estimates:

$ (mu_h, sigma_h) = "head"(bold(z)_N) $

Assume Gaussian: $x_{N+h} tilde cal(N)(mu_h, sigma_h^2)$

Loss: negative log-likelihood $-log p(bold(x) | mu, sigma)$

At inference, sample multiple trajectories to get prediction intervals.

== Discretization (token-based)

Inspired by NLP: discretize continuous values into bins, treat forecasting
as next-token classification:

$ x in [x_"min", x_"max"] arrow.r "bin index" in {0, 1, dots, B-1} $

Now cross-entropy loss applies, and the output is a distribution over bins.
TimesFM (Google, 2024) uses this approach with $B = 4096$ bins, enabling
a foundation model trained on billions of time series datapoints.

*Tradeoff:* quantization error vs clean classification formulation and ability
to pre-train at scale like an LLM.

// ── 6. Key design decisions ───────────────────────────────────────────────────

= Practical Design Decisions

== Channel strategy

*Channel mixing:* all $d$ features processed together — attention can capture
cross-feature dependencies. Higher capacity, more data needed.

*Channel independence:* each feature treated as a separate univariate series —
$d$ separate transformer passes. Simpler, often generalizes better when
cross-feature correlations are noisy or spurious.

PatchTST showed empirically that channel independence often outperforms
channel mixing on standard benchmarks, which was a surprising result.

== Normalization

Input normalization is critical. Two common strategies:

*Instance normalization:* normalize each sample's look-back window to
zero mean and unit variance at inference, then denormalize the output.
Removes distribution shift between training and test.

*Reversible instance normalization (RevIN):* learnable affine parameters
on top of instance normalization, applied and reversed around the model.

== Look-back window length

Longer look-back ($N$) gives more context but increases attention cost
as $O(N^2)$. With patching, the effective sequence length is
$floor(N / P)$, making longer windows tractable.

Empirically, models often benefit from $N = 512$ or $N = 1024$ with
patches of size $P = 16$ or $P = 32$.

// ── 7. Comparison ─────────────────────────────────────────────────────────────

= Comparison of Models

#block(
  fill: luma(245),
  inset: 10pt,
  radius: 4pt,
)[
  #table(
    columns: (auto, auto, auto, auto),
    stroke: none,
    inset: 6pt,
    align: left,
    table.header(
      text(weight: "bold")[Model],
      text(weight: "bold")[Complexity],
      text(weight: "bold")[Key idea],
      text(weight: "bold")[Best for],
    ),
    [Transformer], [$O(N^2)$], [baseline], [short sequences],
    [Informer], [$O(N log N)$], [sparse attention], [very long sequences],
    [Autoformer], [$O(N log N)$], [auto-correlation], [periodic data],
    [PatchTST], [$O((N/P)^2)$], [patch tokens], [general, strong baseline],
    [TFT], [$O(N^2)$], [multi-input, quantile], [production forecasting],
    [TimesFM], [$O(N^2)$], [discretized tokens], [foundation model],
  )
]

// ── 8. Summary ────────────────────────────────────────────────────────────────

= Summary

Time series forecasting with transformers follows the same blueprint as
vision and language:

+ *Tokenize* — project each timestep (or patch of timesteps) into $d_"model"$
+ *Encode position* — sinusoidal or timestamp features
+ *Apply transformer blocks* — self-attention + FFN, with causal mask
+ *Read out* — regression head (continuous) or classification head (discretized)
+ *Train* — MSE / MAE for point forecasts, NLL for probabilistic

The main open questions are:

- *Does cross-channel attention help or hurt?* Evidence leans toward
  channel independence for general benchmarks, but real-world datasets
  with genuine cross-feature structure may benefit from mixing.

- *How long a look-back is useful?* Patching makes this tractable;
  the right window depends on the dominant frequencies in the data.

- *Foundation models for time series?* TimesFM and Moirai suggest
  large-scale pretraining on diverse time series is viable, following
  the GPT playbook, but the discrete tokenization required introduces
  quantization error that continuous models avoid.

The transformer's core strength — attending to arbitrary-range dependencies
in parallel — transfers cleanly to temporal data. The engineering choices
around tokenization, normalization, and output representation determine
whether that strength translates into practical forecasting accuracy.

// ── References ────────────────────────────────────────────────────────────────

= References

#set par(hanging-indent: 1.5em)

Vaswani, A. et al. (2017). *Attention is All You Need.* NeurIPS.

Zhou, H. et al. (2021). *Informer: Beyond Efficient Transformer for Long
Sequence Time-Series Forecasting.* AAAI.

Wu, H. et al. (2021). *Autoformer: Decomposition Transformers with
Auto-Correlation for Long-Term Series Forecasting.* NeurIPS.

Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term
Forecasting with Transformers.* ICLR. (PatchTST)

Lim, B. et al. (2020). *Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting.* International Journal of Forecasting.

Das, A. et al. (2024). *A Decoder-Only Foundation Model for Time-Series
Forecasting.* ICML. (TimesFM)

Woo, G. et al. (2024). *Unified Training of Universal Time Series
Forecasting Transformers.* ICML. (Moirai)

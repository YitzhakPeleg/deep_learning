#set document(title: "Time-Series Forecasting: Senior R&D Curriculum", author: "")
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.8cm, right: 2.8cm),
  numbering: "1",
  number-align: right,
)
#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.75em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 15pt, weight: "bold", fill: rgb("#1a3a5c"))[#it]
  v(0.4em)
  line(length: 100%, stroke: 0.5pt + rgb("#1a3a5c"))
  v(0.3em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12.5pt, weight: "bold", fill: rgb("#2c5f8a"))[#it]
  v(0.2em)
}
#show heading.where(level: 3): it => {
  v(0.5em)
  text(size: 11pt, weight: "bold", fill: rgb("#444444"))[#it]
  v(0.1em)
}

// Color boxes
#let note-box(title, body, accent: rgb("#e8f0f8"), border: rgb("#2c5f8a")) = block(
  width: 100%,
  inset: (x: 12pt, y: 10pt),
  radius: 4pt,
  fill: accent,
  stroke: (left: 3pt + border),
)[
  #text(weight: "bold", fill: border)[#title] \
  #body
]

#let week-box(label, body) = block(
  width: 100%,
  inset: (x: 12pt, y: 10pt),
  radius: 4pt,
  fill: rgb("#f5f9f0"),
  stroke: (left: 3pt + rgb("#4a8c3f")),
)[
  #text(weight: "bold", fill: rgb("#4a8c3f"))[#label] \
  #body
]

#let code-block(body) = block(
  width: 100%,
  inset: (x: 10pt, y: 8pt),
  radius: 3pt,
  fill: rgb("#f4f4f4"),
  stroke: 0.5pt + rgb("#cccccc"),
)[#text(font: "New Computer Modern Mono", size: 9.5pt)[#body]]

// ─── TITLE PAGE ───────────────────────────────────────────────────────────────
#align(center)[
  #v(2cm)
  #text(size: 26pt, weight: "bold", fill: rgb("#1a3a5c"))[
    Time-Series Forecasting
  ]
  #v(0.3em)
  #text(size: 16pt, fill: rgb("#2c5f8a"))[
    A Senior R&D Curriculum
  ]
  #v(0.6em)
  #line(length: 60%, stroke: 1pt + rgb("#2c5f8a"))
  #v(0.6em)
  #text(size: 11pt, fill: rgb("#666666"))[
    From Classical Baselines to Foundation Models \
    Interview-Ready in 4–6 Weeks of Part-Time Study
  ]
  #v(2cm)
]

#note-box("Prerequisites & Framing")[
  This curriculum assumes strong foundations in probabilistic modeling, stochastic processes,
  Bayesian inference, and production deep learning (CNN-based). The framing deliberately
  connects new material to those foundations rather than re-teaching fundamentals.
  Target: contribute meaningfully in a Senior R&D role; speak with precision about methods,
  tradeoffs, and failure modes.
]

#v(0.5em)

// ─── OVERVIEW TABLE ───────────────────────────────────────────────────────────
#align(center)[
  #table(
    columns: (2fr, 4fr, 1.5fr),
    stroke: 0.5pt + rgb("#cccccc"),
    fill: (col, row) => if row == 0 { rgb("#1a3a5c") } else if calc.odd(row) { rgb("#f7f9fc") } else { white },
    inset: (x: 10pt, y: 7pt),
    align: (left, left, center),
    text(fill: white, weight: "bold")[Phase],
    text(fill: white, weight: "bold")[Topic],
    text(fill: white, weight: "bold")[Weeks],
    [Phase 1], [Classical Baselines: ARIMA, ETS, State Space], [1],
    [Phase 2], [Core Deep Learning: LSTM, TCN, Temporal Fusion Transformer], [2],
    [Phase 3], [Foundation Models: TimesFM, Chronos, Moirai], [0.5],
    [Phase 4], [Evaluation, Backtesting, Failure Modes], [0.5],
    [Phase 5], [Hands-On Mini-Project], [1–2],
  )
]

#pagebreak()

// ─── PHASE 1 ──────────────────────────────────────────────────────────────────
= Phase 1 — Classical Baselines (Week 1)

The goal is not to master classical methods but to understand the problem structure they
encode, the inductive biases DL methods need to replicate or supersede, and the vocabulary
shared across the field.

== 1.1 The State Space View (Your Entry Point)

Classical time-series models are most naturally understood as state space models (SSMs),
which you already know from Bayesian filtering.

A general linear Gaussian SSM:

$
z_t = A z_(t-1) + epsilon_t, quad epsilon_t ~ cal(N)(0, Q) \
y_t = C z_t + eta_t, quad eta_t ~ cal(N)(0, R)
$

The *Kalman filter* computes the optimal posterior $p(z_t | y_(1:t))$. All the classical
methods below are special cases with restricted $A$, $C$, $Q$, $R$.

#note-box("Connection to What You Know")[
  ARIMA ↔ polynomial SSM with unit-root dynamics. \
  Exponential smoothing ↔ SSM with a single latent level (and optionally trend/season). \
  The Kalman smoother solves the same backward pass as BPTT — both are instances of
  the forward-backward algorithm on a chain graphical model.
]

== 1.2 ARIMA

*Intuition.* Differencing ($nabla^d$) removes non-stationarity (unit roots); the AR terms
capture autocorrelation structure; the MA terms capture shock propagation.

$
nabla^d y_t = c + phi_1 nabla^d y_(t-1) + dots + phi_p nabla^d y_(t-p)
            + epsilon_t + theta_1 epsilon_(t-1) + dots + theta_q epsilon_(t-q)
$

*What to know:*
- Augmented Dickey-Fuller test for unit roots (stationarity check before fitting)
- ACF/PACF plots → (p, q) identification; this is manual spectral analysis
- ARIMA is a *conditional mean* model — it ignores heteroscedasticity (GARCH adds that)
- `statsmodels.tsa.arima.model.ARIMA` in Python; `auto_arima` from `pmdarima`

*Failure modes:*
- Assumes linearity and stationarity (after differencing)
- Point forecast only; prediction intervals assume Gaussian errors
- Breaks on multiple seasonalities (weekly + annual)

== 1.3 Exponential Smoothing & ETS

ETS (Error, Trend, Seasonality) decomposes the series and propagates weighted averages
of past observations. The Holt-Winters variant handles both trend and seasonality.

Level update (additive ETS):
$
l_t = alpha (y_t - s_(t-m)) + (1 - alpha)(l_(t-1) + b_(t-1)) \
b_t = beta (l_t - l_(t-1)) + (1 - beta) b_(t-1)
$

*What to know:*
- ETS has a state space representation with exact likelihood → parameter estimation via MLE
- Additive vs. multiplicative seasonality (multiplicative better for series with increasing amplitude)
- `statsmodels.tsa.holtwinters.ExponentialSmoothing`

== 1.4 When Classical Wins

Classical methods remain competitive (and often preferred) when:
- Series are short ($n < 200$), making DL overparameterized
- Interpretability and uncertainty quantification are requirements
- Strong, regular seasonality (ETS handles this analytically)
- Forecasting horizon is 1–3 steps ahead (DL advantage grows at longer horizons)

#week-box("Week 1 Practice")[
  Dataset: M4 Competition (available via `datasetsforecast` package). \
  Task: Fit ARIMA and ETS on 10 heterogeneous series (different frequencies). Compute
  sMAPE and MASE. Examine residuals with Ljung-Box test. \
  Note where and why each model breaks. This calibrates your baseline intuition.
]

#pagebreak()

// ─── PHASE 2 ──────────────────────────────────────────────────────────────────
= Phase 2 — Core Deep Learning Approaches (Weeks 2–3)

== 2.1 Problem Reformulation

DL forecasting reframes the problem as supervised sequence-to-sequence (seq2seq) learning:

$
hat(y)_(t+1:t+H) = f_theta (y_(t-L+1:t), x_(t-L+1:t+H))
$

where $L$ is the lookback window, $H$ the forecast horizon, and $x$ are covariates
(known-future: calendar features, promotions; past-observed: weather, demand).

This is a *function approximation* problem, not an explicit probabilistic model —
though DL methods can output distributional parameters.

== 2.2 LSTM

=== Architecture

The LSTM addresses the vanishing gradient problem of vanilla RNNs through gated state:

$
f_t &= sigma(W_f [h_(t-1), x_t] + b_f) & "forget gate" \
i_t &= sigma(W_i [h_(t-1), x_t] + b_i) & "input gate" \
tilde(c)_t &= tanh(W_c [h_(t-1), x_t] + b_c) & "cell candidate" \
c_t &= f_t dot.circle c_(t-1) + i_t dot.circle tilde(c)_t & "cell update" \
o_t &= sigma(W_o [h_(t-1), x_t] + b_o) & "output gate" \
h_t &= o_t dot.circle tanh(c_t) & "hidden state"
$

The cell state $c_t$ acts as a *controlled integrator*: the forget gate allows gradients
to flow through long sequences, analogous to how the Kalman gain controls how much
new information overwrites the posterior.

=== Training

- Teacher forcing: feed ground truth $y_t$ as input during training (vs. model output)
  → creates train/inference mismatch (exposure bias)
- Scheduled sampling bridges this gap
- For multi-step: either direct (predict all $H$ steps at once) or recursive (step-by-step)

=== When to Use LSTM

- Moderate-length sequences ($L < 500$)
- Strong temporal dependencies that are irregular (not fixed-period)
- Baseline for sequential DL — always worth including

=== Failure Modes

- Slow training (sequential, not parallelizable)
- Poor on very long sequences (TCN and Transformer outperform)
- Sensitive to input scaling; always normalize

== 2.3 Temporal Convolutional Network (TCN)

=== Architecture

TCN replaces recurrence with *dilated causal convolutions*, allowing parallel training
and exponentially large receptive fields:

$
"receptive field" = 1 + 2(k-1)(2^l - 1)
$

for kernel size $k$ and $l$ dilation layers ($d = 1, 2, 4, dots, 2^(l-1)$).

Causality is enforced by zero-padding the left side of each convolution — the network
cannot see future values. Residual connections stabilize deep stacks:

$
y = "Conv"(x) + x quad "(with 1×1 conv on x if channels differ)"
$

=== Connection to CNNs

You already understand dilated convolutions from DeepLab/segmentation contexts. TCN
applies exactly the same idea on 1D temporal signals. The key difference is the
*causal* constraint and the exponential dilation schedule.

=== When to Use TCN

- Long sequences (TCN's $O(1)$ depth vs. LSTM's $O(L)$ sequential bottleneck)
- Fixed-period patterns (dilation picks up periodicity naturally)
- When training speed is a priority (fully parallelizable)

=== Failure Modes

- Receptive field is fixed at architecture design time — adapting requires re-architecture
- Less natural handling of irregular time (LSTM's hidden state is more flexible)

#note-box("TCN vs. LSTM Rule of Thumb")[
  If the effective receptive field you need is $< 200$ steps and the patterns are
  irregular: LSTM. If you need $> 500$ steps and patterns are periodic: TCN.
  In practice, run both and let validation metrics decide.
]

== 2.4 Temporal Fusion Transformer (TFT)

TFT (Lim et al., 2021) is the most practically important architecture for *tabular*
multi-horizon forecasting with mixed covariates.

=== Key Innovations

*Variable Selection Network (VSN).* Soft attention over input features per timestep.
Produces feature-level importance weights — useful for interpretability and debugging.

$
xi_t = sum_j tilde(alpha)_(t,j) dot h_(t,j)
$

*Gated Residual Network (GRN).* The basic computation unit throughout TFT:

$
"GRN"(a, c) = "LayerNorm"(a + "GLU"(eta_1)) \
eta_1 = W_1 eta_2 + b_1, quad eta_2 = "ELU"(W_2 a + W_3 c + b_2)
$

where $c$ is an optional context vector. GLU (gated linear unit) provides adaptive
depth — the gate can zero out the transformation when linear pass-through is sufficient.

*Static covariate encoders.* Entity-level embeddings (store ID, location) produced
by GRNs and injected at multiple points in the network as context $c$.

*Multi-head attention with interpretable weights.*

$
"Attention"(Q, K, V) = A(Q, K) V, quad A(Q, K) = "softmax"(Q K^T / sqrt(d_k))
$

TFT uses a *shared* $V$ weight across heads, which enables attention weight
averaging across heads to produce interpretable temporal importance patterns.

*Quantile outputs.* TFT outputs $hat(y)_(tau)$ for quantiles
$tau in {0.1, 0.5, 0.9}$ by optimizing the pinball (quantile) loss:

$
cal(L)_tau (y, hat(y)) = tau (y - hat(y))_+ + (1 - tau)(hat(y) - y)_+
$

=== When to Use TFT

- You have multiple time series (panel data) with entity-level covariates
- You have known-future features (promotions, holidays, calendar)
- Interpretability is required (feature importance, temporal attention)
- Forecast horizon $H$ is large (TFT handles multi-horizon natively)

=== Failure Modes

- Complex to tune; many hyperparameters
- Needs substantial data per entity; underperforms on short individual series
- Slower than TCN; not suitable for very high-frequency inference

#week-box("Weeks 2–3 Practice")[
  Dataset: Electricity dataset (UCL, 370 clients × 26,304 hourly steps — standard benchmark). \
  Tasks:
  - Implement LSTM baseline with PyTorch `nn.LSTM` and direct multi-step head
  - Implement TCN with dilated causal convolutions from scratch (good interview signal)
  - Fine-tune TFT via `pytorch-forecasting` (the reference implementation); study the
    `TemporalFusionTransformer` class internals — don't just call `.fit()`
  - Compare on MAE and coverage of 80% prediction interval \
  Key question to answer: At what horizon does LSTM start losing to TCN/TFT on this dataset?
]

#pagebreak()

// ─── PHASE 3 ──────────────────────────────────────────────────────────────────
= Phase 3 — Foundation Models for Time Series (Week 4, Part 1)

== 3.1 The Foundation Model Paradigm

LLM-style pretraining on massive corpora, zero/few-shot transfer. The analogy:
BERT is to NLP as TimesFM/Chronos are to forecasting.

#table(
  columns: (1.4fr, 1.2fr, 1.5fr, 1.5fr, 1.5fr),
  stroke: 0.5pt + rgb("#cccccc"),
  fill: (col, row) => if row == 0 { rgb("#2c5f8a") } else if calc.odd(row) { rgb("#f7f9fc") } else { white },
  inset: (x: 8pt, y: 6pt),
  align: (left, center, left, left, left),
  text(fill: white, weight: "bold")[Model],
  text(fill: white, weight: "bold")[Source],
  text(fill: white, weight: "bold")[Architecture],
  text(fill: white, weight: "bold")[Key Idea],
  text(fill: white, weight: "bold")[Strengths],
  [TimesFM], [Google, 2024], [Transformer (patched)], [Patch tokenization of time series], [Long horizon, fast inference],
  [Chronos], [Amazon, 2024], [T5 (seq2seq)], [Quantize values → language tokens], [Probabilistic, uncertainty-aware],
  [Moirai], [Salesforce, 2024], [Transformer (unified)], [Any-variate patching], [Multi-variate, multi-freq],
  [Lag-Llama], [Open, 2024], [LLaMA-based], [Lag features as tokens], [Fully open, fine-tunable],
)

== 3.2 TimesFM (Google)

Tokenizes time series into *patches* (non-overlapping windows of length $p$), projects
each to an embedding, and applies a standard decoder-only Transformer. The key design
choice is *output patch size > 1*: the model predicts a patch of future values per
autoregressive step, reducing the sequence length at inference.

Training data: ~100B real-world time points across domains.

*When to use:* Zero-shot forecasting on a new domain where you have no training data.
Strong on long-horizon tasks. Not probabilistic by default.

== 3.3 Chronos (Amazon)

Treats forecasting as a *language modeling* problem by quantizing real-valued time
series into discrete tokens (via mean/scale normalization + uniform binning):

$
hat(y)_t ~ "Categorical"("softmax"(z_t)), quad z_t in RR^B
$

where $B$ is the vocabulary size (bins). Outputs a full predictive distribution
naturally — sample multiple draws for ensemble/interval estimation. Architecture is T5
(encoder-decoder Transformer).

*When to use:* When you need calibrated prediction intervals with zero labeled data.
Fine-tuning on domain data improves performance substantially.

*Connection to your background:*
Chronos is a learned, nonparametric approximation to Bayesian forecasting. The
token distribution approximates the posterior predictive; sampling is Monte Carlo
integration over the learned model.

== 3.4 Moirai (Salesforce)

Designed for *multi-variate* zero-shot forecasting across frequencies. Uses a
*any-variate* attention mechanism: channels are treated as tokens within the sequence,
so the model sees correlations between variates. Frequency-specific tokenization.

*When to use:* Multi-variate panels with cross-series correlations, especially when
labeled data is scarce.

== 3.5 When Are Foundation Models Worth Using?

#table(
  columns: (2.5fr, 1fr, 1fr),
  stroke: 0.5pt + rgb("#cccccc"),
  fill: (col, row) => if row == 0 { rgb("#2c5f8a") } else if calc.odd(row) { rgb("#f7f9fc") } else { white },
  inset: (x: 8pt, y: 6pt),
  align: (left, center, center),
  text(fill: white, weight: "bold")[Scenario],
  text(fill: white, weight: "bold")[Use FM],
  text(fill: white, weight: "bold")[Use TFT/TCN],
  [No labeled training data], [✓], [],
  [Short series ($n < 50$)], [✓], [],
  [Large panel, rich covariates], [], [✓],
  [Custom loss / business constraints], [], [✓],
  [Strict latency / edge deployment], [], [✓],
  [Quick prototype / POC], [✓], [],
  [Production system requiring calibration], [(fine-tune)], [✓],
)

#pagebreak()

// ─── PHASE 4 ──────────────────────────────────────────────────────────────────
= Phase 4 — Evaluation, Backtesting, Failure Modes (Week 4, Part 2)

== 4.1 Metrics

=== Point Forecast Metrics

*MAE (Mean Absolute Error).* $"MAE" = 1/n sum |y_t - hat(y)_t|$. Robust to outliers;
interpretable in original units.

*RMSE (Root Mean Squared Error).* Penalizes large errors more heavily than MAE.
Use when large errors are disproportionately costly.

*MAPE (Mean Absolute Percentage Error).* $"MAPE" = 100/n sum |e_t / y_t|$. Undefined
when $y_t = 0$; biased toward under-forecasting. Avoid in practice; use sMAPE or MASE.

*MASE (Mean Absolute Scaled Error).* $"MASE" = "MAE" / overline("MAE")_"naive"$.
Scales by the in-sample MAE of the naive seasonal forecast. Scale-free and comparable
across series with different units. Preferred for M-competition-style evaluation.

*sMAPE.* $"sMAPE" = 200/n sum |y_t - hat(y)_t| / (|y_t| + |hat(y)_t|)$. Bounded
but still problematic near zero. Used historically; MASE is generally preferred.

=== Probabilistic Forecast Metrics

*CRPS (Continuous Ranked Probability Score).* Proper scoring rule for predictive
distributions. Generalizes MAE to distributions:

$
"CRPS"(F, y) = integral_(-oo)^(oo) [F(z) - bb("1")(z >= y)]^2 d z
$

For Gaussian: $"CRPS"(cal(N)(mu, sigma), y) = sigma [z Phi(z) + phi(z) - 1/2]$ where
$z = (y - mu)/sigma$.

*Quantile (Pinball) Loss.* $cal(L)_tau(y, hat(q)_tau) = (y - hat(q)_tau)(tau - bb("1")(y < hat(q)_tau))$.
Average over $tau$ gives the quantile score.

*Coverage.* Fraction of actuals inside the $[q_(alpha/2), q_(1-alpha/2)]$ interval.
A well-calibrated 80% interval should achieve ~80% empirical coverage.

*Calibration.* Plot reliability diagrams: predicted quantile $tau$ vs. empirical
coverage. Perfectly calibrated → diagonal line.

== 4.2 Backtesting Strategy

*Walk-forward validation (expanding window):*

$
"Train"_k = [1, t_k], quad "Test"_k = [t_k + 1, t_k + H], quad k = 1, dots, K
$

Use at least $K = 5$ folds. The origin $t_k$ steps by stride $s$ (commonly $s = H/2$).

#note-box("Critical: No Look-Ahead Contamination")[
  All feature engineering — scalers, lag statistics, moving averages — must be fit
  *only on the training window* and applied to test. Fitting a StandardScaler on the
  full series before splitting is a common, silent error that inflates performance.
]

*Blocked cross-validation.* Adds a gap between train and test to prevent temporal
leakage through autocorrelation. Gap size ≥ forecast horizon.

== 4.3 Common Failure Modes

=== Data Leakage

- Normalizing with future statistics (scaler fit on full series)
- Including future covariates at inference time that were labeled as "known future"
  but aren't actually known
- Overfitting evaluation: running many experiments and selecting the best split
  (use a held-out test set that you touch exactly once)

=== Distribution Shift

Time series frequently violate stationarity assumptions. Monitor:
- Rolling residual analysis (should be white noise; ACF of residuals should be flat)
- Regime change detection (structural breaks): Chow test, CUSUM
- Concept drift in real deployments: retrain triggers based on sliding window performance

=== Overfitting to Seasonality

DL models can memorize seasonal patterns from training data that don't generalize.
Symptom: strong in-sample performance; poor out-of-sample on the same season next year.
Fix: ensure multiple seasonal cycles in training; use seasonal normalization
(subtract and divide by historical seasonal index before modeling).

=== Intermittent / Zero-Inflated Series

Standard regression losses fail on series with many zeros (slow-moving inventory,
rare events). Use: Croston's method, zero-inflated distributions, or a two-stage
classification-then-regression model.

#pagebreak()

// ─── PHASE 5 ──────────────────────────────────────────────────────────────────
= Phase 5 — Hands-On Mini-Project (Weeks 5–6)

== 5.1 Project Specification

*Dataset:* ETT (Electricity Transformer Temperature) — `ETTh1.csv` or `ETTm1.csv`.
Publicly available; 7 variates, hourly/15-min frequency, 2 years.
Standard benchmark in the Informer, PatchTST, and iTransformer papers.

*Forecasting Task:* Predict `OT` (oil temperature) for horizons $H in {24, 96, 336, 720}$
steps ahead. This range deliberately stresses short- and long-horizon behavior.

*Deliverable:* A reproducible PyTorch codebase + analysis notebook demonstrating:

+ Baseline: ARIMA on the univariate `OT` series; compute MASE and CRPS
+ LSTM: custom seq2seq with direct multi-step output
+ TCN: dilated causal conv stack (implement from scratch)
+ TFT: via `pytorch-forecasting`; extract and interpret temporal attention weights
+ Zero-shot: Chronos on $H = 96$; compare against trained models
+ Evaluation: walk-forward backtest with $K = 5$ folds; reliability diagram for TFT's
  quantile outputs

== 5.2 Codebase Structure

#code-block[
```
ts_forecast/
├── data/
│   ├── download.py          # fetch ETT from HuggingFace datasets
│   └── dataset.py           # TimeSeriesDataset with proper windowing
├── models/
│   ├── lstm.py              # Encoder-Decoder LSTM
│   ├── tcn.py               # Dilated causal conv (TemporalBlock + TCN)
│   └── tft_wrapper.py       # pytorch-forecasting TFT interface
├── baselines/
│   └── arima_baseline.py    # statsmodels ARIMA per-entity
├── evaluation/
│   ├── metrics.py           # MAE, MASE, CRPS, pinball loss, coverage
│   └── backtest.py          # WalkForwardCV with gap
├── foundation/
│   └── chronos_eval.py      # Chronos zero-shot inference
└── notebooks/
    └── analysis.ipynb       # Results, plots, reliability diagrams
```
]

== 5.3 Key Implementation Details

=== TimeSeriesDataset

#code-block[
```python
class TimeSeriesDataset(Dataset):
    def __init__(self, data: np.ndarray, lookback: int, horizon: int):
        self.x, self.y = [], []
        for i in range(len(data) - lookback - horizon + 1):
            self.x.append(data[i : i + lookback])
            self.y.append(data[i + lookback : i + lookback + horizon])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
```
]

=== TCN TemporalBlock

#code-block[
```python
class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal padding
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                                padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                                padding=pad, dilation=dilation)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.net = nn.Sequential(self.conv1, nn.ReLU(), nn.Dropout(0.2),
                                  self.conv2, nn.ReLU(), nn.Dropout(0.2))

    def forward(self, x):
        # Slice off the right-padding to maintain causal property
        out = self.net(x)[..., :x.size(-1)]
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)
```
]

=== Walk-Forward Backtest

#code-block[
```python
def walk_forward_cv(model_fn, data, n_splits=5, horizon=96, gap=0):
    n = len(data)
    fold_size = (n - horizon - gap) // n_splits
    results = []
    for k in range(n_splits):
        train_end = fold_size * (k + 1)
        test_start = train_end + gap
        test_end   = test_start + horizon
        train = data[:train_end]
        test  = data[test_start:test_end]
        model = model_fn(train)          # fit on train only
        preds = model.predict(horizon)   # forecast
        results.append(evaluate(preds, test))
    return aggregate(results)
```
]

== 5.4 Interview Angle

The project is designed to demonstrate:

- *Breadth:* classical → DL → foundation model comparison in one pipeline
- *Rigor:* correct backtesting with no leakage; probabilistic evaluation
- *Depth:* TCN implemented from scratch shows architectural understanding
- *Judgment:* interpreting TFT attention weights; knowing when Chronos zero-shot
  wins or loses relative to trained models

Expected talking points:
- Why MASE is preferred over MAPE for this dataset
- What the temporal attention weights reveal about the data's structure
- When you would and would not deploy a foundation model in production
- How you would handle a distribution shift detected in production

#pagebreak()

// ─── RESOURCES ────────────────────────────────────────────────────────────────
= Reference: Key Resources

== Papers (Read in Order)

+ *TFT:* Lim et al. (2021) — "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" — _NeurIPS 2021_
+ *TCN:* Bai et al. (2018) — "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" — _arXiv:1803.01271_
+ *Chronos:* Ansari et al. (2024) — "Chronos: Learning the Language of Time Series" — _arXiv:2403.07815_
+ *TimesFM:* Das et al. (2024) — "A Decoder-Only Foundation Model for Time-Series Forecasting" — _ICML 2024_
+ *Moirai:* Woo et al. (2024) — "Unified Training of Universal Time Series Forecasting Transformers" — _ICML 2024_
+ *N-BEATS:* Oreshkin et al. (2020) — useful contrast: purely DL basis expansion, no recurrence
+ *PatchTST:* Nie et al. (2023) — patches applied to Transformers before FM era; bridges Phase 2–3

== Libraries

#table(
  columns: (1.5fr, 3fr, 2fr),
  stroke: 0.5pt + rgb("#cccccc"),
  fill: (col, row) => if row == 0 { rgb("#2c5f8a") } else if calc.odd(row) { rgb("#f7f9fc") } else { white },
  inset: (x: 8pt, y: 6pt),
  text(fill: white, weight: "bold")[Library],
  text(fill: white, weight: "bold")[Purpose],
  text(fill: white, weight: "bold")[Install],
  [`pytorch-forecasting`], [TFT reference implementation], [`pip install pytorch-forecasting`],
  [`statsmodels`], [ARIMA, ETS, Kalman filter], [`pip install statsmodels`],
  [`datasetsforecast`], [M4, M5, ETT datasets], [`pip install datasetsforecast`],
  [`chronos-forecasting`], [Amazon Chronos inference], [`pip install chronos-forecasting`],
  [`neuralforecast`], [NHITS, NBEATS, TFT alternatives], [`pip install neuralforecast`],
  [`properscoring`], [CRPS implementation], [`pip install properscoring`],
)

== Datasets by Phase

#table(
  columns: (1fr, 2fr, 2fr),
  stroke: 0.5pt + rgb("#cccccc"),
  fill: (col, row) => if row == 0 { rgb("#2c5f8a") } else if calc.odd(row) { rgb("#f7f9fc") } else { white },
  inset: (x: 8pt, y: 6pt),
  text(fill: white, weight: "bold")[Phase],
  text(fill: white, weight: "bold")[Dataset],
  text(fill: white, weight: "bold")[Why],
  [1 — Baselines], [M4 Competition], [Heterogeneous; many frequencies],
  [2 — DL], [Electricity (ECL)], [Panel; standard benchmark],
  [3 — Foundation], [ETT (ETTh1)], [Standard; used in all FM papers],
  [5 — Project], [ETT (ETTh1)], [Reproducible; rich literature comparison],
)

#v(1em)
#note-box("One-Line Summary per Method", accent: rgb("#fef9ec"), border: rgb("#c47c00"))[
  *ARIMA:* linear model of autocorrelation after removing trends and unit roots. \
  *ETS/Holt-Winters:* state space model with adaptive level, trend, and seasonality components. \
  *LSTM:* gated recurrent network; flexible memory; sequential bottleneck limits scaling. \
  *TCN:* dilated causal convolutions; parallelizable; fixed receptive field; great for long periodic series. \
  *TFT:* Transformer with covariate handling, interpretable attention, and quantile outputs; the practical DL workhorse for tabular panel forecasting. \
  *TimesFM / Chronos / Moirai:* pretrained Transformers on massive corpora; zero-shot or few-shot deployment; trade customizability for immediacy. \
]

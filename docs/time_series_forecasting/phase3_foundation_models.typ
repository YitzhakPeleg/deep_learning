#set document(title: "Time-Series Forecasting — Phase 3: Foundation Models", author: "")
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
  #text(size: 22pt, weight: "bold")[Time-Series Forecasting]
  #v(0.3em)
  #text(size: 15pt)[Phase 3 — Foundation Models (Week 4, Part 1)]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[TimesFM · Chronos · Moirai · Lag-Llama]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

#outline(indent: 1em, depth: 2)

#v(0.5em)
#line(length: 100%)

// ─── OVERVIEW ────────────────────────────────────────────────────────────────

= Overview

Foundation models for time series are pretrained Transformers that generalize across domains, frequencies, and series lengths — analogous to GPT for text or ViT for images. The promise: zero-shot or few-shot forecasting on a new dataset without any task-specific training. The reality: they work remarkably well in some regimes and fail quietly in others.

Phase 3 is deliberately shorter than Phases 1 and 2. The reason is that foundation models are primarily _used_, not implemented from scratch — the architectures are standard Transformers with domain-specific tokenization schemes. The depth you need here is architectural understanding (so you can reason about failure modes) and practical judgment (so you know when to reach for them vs. a trained TFT).

*By the end of Week 4, Part 1 you should be able to:*

- Explain what makes time series tokenization hard and how each model solves it differently
- Trace a Chronos forward pass end-to-end, including quantization and sampling
- Describe the pretraining data strategy and why domain coverage matters
- Run zero-shot inference with Chronos and TimesFM on a new dataset
- Decide when a foundation model should replace or complement a trained model
- Fine-tune Chronos on domain-specific data
- Articulate the failure modes specific to each model

// ─── SECTION 1 ───────────────────────────────────────────────────────────────

= The Foundation Model Paradigm

== Why This is Hard for Time Series

Language models pretrain on text, which has a natural discrete token vocabulary. Images have pixels in $[0, 255]$. Time series are *continuous, scale-varying, domain-heterogeneous* — there is no natural unit of tokenization.

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Challenge*], [*Why it matters*], [*Solution space*]),
  [Continuous values],
  [No natural vocabulary],
  [Discretize (Chronos) or patch-embed (TimesFM, Moirai)],
  [Wildly different scales],
  [A temperature series ($-20$ to $40$) and a sales series ($0$ to $1$M) can't share token embeddings naively],
  [Instance normalization before tokenization],
  [Variable frequency],
  [Hourly, daily, monthly series have different temporal structures],
  [Frequency embeddings; frequency-specific tokenization (Moirai)],
  [Variable length],
  [Context windows vary across tasks],
  [Patching reduces effective sequence length; masking handles variable-length contexts],
  [Multi-variate vs. univariate],
  [Some series have 1 channel, some have 100],
  [Univariate modeling (Chronos, TimesFM) or any-variate attention (Moirai)],
  [No semantic tokens],
  [In NLP, co-occurrence patterns across the corpus teach word meaning; in TS, there's no analogous signal],
  [Pretrain on statistical patterns across diverse real datasets],
)

== The Pretraining Recipe (Common to All)

All current TS foundation models follow roughly the same recipe:

1. *Collect a massive, diverse corpus* of time series across domains (finance, energy, retail, weather, traffic, health, sensor data)
2. *Normalize each series* to remove scale (instance normalization or mean/std scaling)
3. *Tokenize* the normalized series (discretize or patch-embed)
4. *Pretrain* a Transformer with a forecasting objective: given a context window, predict the next patch/token(s)
5. *Evaluate zero-shot* on held-out datasets from domains not seen during training

The Transformer architecture used is largely standard — the innovation is in steps 1–3.

== Connection to Your Background

The probabilistic framing connects directly to Bayesian forecasting:

- *Prior:* the model encodes a prior over temporal patterns learned from the pretraining corpus
- *Posterior update:* the context window $bold(y)_(1:T)$ is the observed data; the model conditions on it
- *Posterior predictive:* sampling from the model's output distribution gives draws from $p(bold(y)_(T+1:T+H) | bold(y)_(1:T))$

Chronos makes this explicit: the softmax over discrete bins approximates a discretized posterior predictive. The key limitation vs. true Bayesian forecasting: the prior is implicit and opaque, learned from data, not specified by a domain expert.

#insight[
  *Transfer learning analogy:* A pretrained ResNet learns general visual features (edges, textures, shapes) that transfer to new visual tasks. A pretrained TS foundation model learns general temporal features (trend shapes, seasonal patterns, noise structures) that transfer to new TS tasks. The analogy isn't perfect — TS domains vary more than visual domains — but the fine-tuning workflow is identical.
]

// ─── SECTION 2 ───────────────────────────────────────────────────────────────

= Model Landscape

#table(
  columns: (auto, auto, auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 7pt,
  align: left,
  table.header(
    [*Model*], [*Lab*], [*Year*], [*Architecture*], [*Tokenization*],
    [*Output*], [*Parameters*], [*Key paper*],
  ),
  [TimesFM], [Google], [2024],
  [Decoder-only Transformer], [Patch embedding],
  [Point (+ quantiles via head)], [200M], [Das et al., ICML 2024],
  [Chronos], [Amazon], [2024],
  [T5 (enc-dec Transformer)], [Value quantization],
  [Categorical (full distribution)], [8M–710M], [Ansari et al., arXiv 2403.07815],
  [Moirai], [Salesforce], [2024],
  [Encoder Transformer], [Patch embedding (any-variate)],
  [Mixture distribution], [14M–311M], [Woo et al., ICML 2024],
  [Lag-Llama], [Academic], [2024],
  [LLaMA (decoder-only)], [Lag features as tokens],
  [Student-t distribution], [7M], [Rasul et al., arXiv 2310.08278],
  [MOIRAI-MoE], [Salesforce], [2024],
  [Mixture of Experts], [Patch embedding],
  [Mixture distribution], [—], [Follow-up to Moirai],
  [UniTS], [Academic], [2024],
  [Unified Transformer], [Patch + task tokens],
  [Task-dependent], [—], [Gao et al.],
)

For practical purposes: *Chronos* is the most accessible (Apache 2.0, pip-installable, HuggingFace), *TimesFM* is strong for point forecasts and long horizons, *Moirai* is the best choice for multi-variate zero-shot.

// ─── SECTION 3 ───────────────────────────────────────────────────────────────

= TimesFM (Google)

== Core Architecture

TimesFM is a *decoder-only Transformer* — the same family as GPT. Its key innovation is *patching*: instead of tokenizing individual timesteps, it groups $p$ consecutive timesteps into a single token.

```
Input series: y_1, y_2, ..., y_T

Patching (p=32):
  Patch 1: [y_1,  ..., y_32]     -> linear projection -> token e_1 in R^d
  Patch 2: [y_33, ..., y_64]     -> linear projection -> token e_2 in R^d
  ...
  Patch k: [y_{T-31}, ..., y_T]  -> linear projection -> token e_k in R^d

Transformer: autoregressively predicts future patches
  [e_1, ..., e_k] -> predict e_{k+1} (patch of p future values)
```

*Why patching?* Two reasons:

1. *Sequence length reduction:* a 512-step series becomes 16 tokens with patch size 32. Transformer attention is $O(n^2)$ in sequence length — patching makes long-horizon forecasting computationally tractable.
2. *Local pattern capture:* each patch embeds a local temporal structure (a week's worth of hourly data, a month of daily data) as a single representation. The Transformer then operates on these higher-level patterns.

*Output patch size > 1:* TimesFM predicts an entire patch of future values per autoregressive step. For a patch size of 128 and horizon of 512, only 4 autoregressive steps are needed. This substantially reduces the error accumulation problem of step-by-step autoregressive decoding.

== Pretraining Data

Trained on ~100 billion real-world time points from:
- Google-internal datasets (search trends, query logs)
- Public datasets: M4, ETT, Weather, Traffic, Electricity
- Synthetic data generated from statistical models (ARIMA, ETS) to ensure broad coverage of structural patterns

#note[The synthetic data component is important: real-world data is biased toward certain domains (finance, retail, energy). Synthetic generation ensures the model sees diverse trend shapes, seasonality types, and noise structures.]

== Inference

```python
# pip install timesfm
import timesfm

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend='gpu',          # or 'cpu'
        per_core_batch_size=32,
        horizon_len=96,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id='google/timesfm-1.0-200m',
    ),
)
tfm.initialize()

import numpy as np
context = np.array([...])   # your time series, shape (context_len,)
frequency_input = [0]        # 0=high-freq (hourly/minutely), 1=medium (daily/weekly), 2=low (monthly+)

point_forecast, quantile_forecast = tfm.forecast(
    inputs=[context],
    freq=frequency_input,
)
# point_forecast:   (1, horizon_len)
# quantile_forecast:(1, horizon_len, n_quantiles)
```

#gotcha[*Frequency input matters:* TimesFM uses a learned frequency embedding. Providing the correct frequency class (high/medium/low) significantly affects forecast quality because the model internally adjusts what temporal patterns it expects.]

== Strengths and Weaknesses

*Strengths:*
- State-of-the-art zero-shot accuracy on standard benchmarks (ETT, Weather, Traffic)
- Long-horizon capability due to patch-based autoregressive decoding
- Fast inference once loaded (model is only 200M parameters)

*Weaknesses:*
- Not natively probabilistic (quantile outputs require a separate head, less well-calibrated than Chronos)
- No covariate support — univariate context only
- Closed-source training data (Google-internal); limited reproducibility
- No straightforward fine-tuning API (as of mid-2024)

// ─── SECTION 4 ───────────────────────────────────────────────────────────────

= Chronos (Amazon)

Chronos is the architecturally most interesting model to understand deeply, because it makes a strong and explicit probabilistic commitment. It is also the most practical: fully open-source, pip-installable, fine-tunable, and available at multiple model sizes.

== The Tokenization Idea

Chronos treats forecasting as *language modeling over quantized values*. The pipeline:

*Step 1: Instance normalization*

$ mu = "mean"(y_(1:T)), quad sigma = "std"(y_(1:T)), quad tilde(y)_t = (y_t - mu) / sigma $

This maps any series to approximately zero-mean unit-variance — making the vocabulary (the bins) reusable across series with wildly different scales.

*Step 2: Uniform quantization*

```
bins = linspace(-15, 15, B+1)    # B+1 boundaries -> B bins
token(ỹ_t) = argmax_b  [ bins[b] <= ỹ_t < bins[b+1] ]
```

The range $[-15, 15]$ covers $plus.minus 15$ standard deviations — essentially all real-world values after normalization. Chronos uses $B = 4096$ bins (vocabulary size 4096), giving resolution of $30 \/ 4096 approx 0.0073$ standard deviations per bin.

*Step 3: Language model over tokens*

$ p("token"_(T+1), dots, "token"_(T+H) | "token"_1, dots, "token"_T) quad "— T5 seq2seq" $

The T5 model (encoder-decoder Transformer) takes the tokenized past as input and autoregressively predicts future tokens. The output at each step is a softmax distribution over $B = 4096$ bins — a discrete approximation of the conditional predictive distribution.

*Step 4: De-quantization and de-normalization*

$
  tilde(y)_(T+h) &tilde "Categorical"("softmax"(bold(z)_(T+h))) quad "(sample a bin)" \
  y_(T+h) &= "bin\_midpoint"(tilde(y)_(T+h)) dot sigma + mu quad "(de-normalize)"
$

Draw multiple samples → get Monte Carlo estimates of any statistic (mean, quantiles, intervals).

== Why This Works (and Why It's Elegant)

The tokenization trick is not obvious — it is worth understanding why it succeeds.

*It converts a regression problem to classification.* The model doesn't predict a real number; it predicts a distribution over discrete bins. This sidesteps the need to specify a parametric distribution (Gaussian? Student-t? Log-normal?) — the model learns the shape of the predictive distribution from data.

*The vocabulary is universal.* After normalization, all series live in the same scale space. The same bin 1024 means "approximately $-1.5$ standard deviations above mean" regardless of whether the series is electricity demand or stock price. The model learns what comes next in normalized value-space — a form of pattern matching over shapes rather than magnitudes.

*Uncertainty is native.* A language model already outputs a distribution over the next token. For Chronos, that distribution is the predictive distribution of the next time step's value. No separate uncertainty head needed.

*Connection to Bayesian forecasting:*
- The T5 encoder computes a representation $h(bold(y)_(1:T))$ — a sufficient statistic of the context
- The decoder computes $p(y_(T+h) | y_(T+h-1), dots, y_(T+1), h(bold(y)_(1:T)))$ at each step
- Full predictive distribution via Monte Carlo: draw $N$ samples from the autoregressive decoder

$ p(bold(y)_(T+1:T+H) | bold(y)_(1:T)) approx frac(1, N) sum_i delta(bold(y)^((i))_(T+1:T+H)) $

The model approximates the posterior predictive by learning a prior over temporal patterns. If your series structure matches what the model saw in pretraining, the posterior predictive is well-calibrated. If it doesn't (novel domain, unusual patterns), the prior dominates and calibration degrades.

== Model Sizes

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Model*], [*Parameters*], [*Context length*], [*Notes*]),
  [`chronos-t5-tiny`],  [8M],   [512], [Fast CPU inference],
  [`chronos-t5-mini`],  [20M],  [512], [Good balance],
  [`chronos-t5-small`], [46M],  [512], [Recommended default],
  [`chronos-t5-base`],  [200M], [512], [Best zero-shot accuracy],
  [`chronos-t5-large`], [710M], [512], [Marginal improvement over base],
)

For most use cases, `chronos-t5-small` is the right default — strong zero-shot performance at low inference cost. Use `chronos-t5-base` for project benchmarking.

== Zero-Shot Inference

```python
# pip install chronos-forecasting
import torch
from chronos import ChronosPipeline
import numpy as np

pipeline = ChronosPipeline.from_pretrained(
    'amazon/chronos-t5-small',
    device_map='cuda',          # or 'cpu'
    torch_dtype=torch.bfloat16,
)

context = torch.tensor(series_values, dtype=torch.float32)  # (context_len,)

forecast = pipeline.predict(
    context=context.unsqueeze(0),   # (1, context_len) — batch of 1
    prediction_length=96,
    num_samples=100,                # draw 100 Monte Carlo samples
    temperature=1.0,
    top_k=50,
    top_p=1.0,
)
# forecast: (1, num_samples, prediction_length)

forecast_np = forecast[0].numpy()          # (100, 96)
median_fc   = np.quantile(forecast_np, 0.5, axis=0)
lower_80    = np.quantile(forecast_np, 0.1, axis=0)
upper_80    = np.quantile(forecast_np, 0.9, axis=0)
```

*Batched inference (for large panels):*

```python
contexts = [torch.tensor(s) for s in list_of_series]  # variable-length OK
forecasts = pipeline.predict(
    context=contexts,
    prediction_length=96,
    num_samples=100,
)
# forecasts: (n_series, num_samples, prediction_length)
```

== Fine-Tuning on Domain Data

Fine-tuning Chronos on your domain data is the highest-leverage operation available. Even a small fine-tuning set (a few hundred series, a few epochs) can substantially close the gap between zero-shot and a fully trained model.

```python
from chronos import ChronosPipeline
from torch.utils.data import Dataset, DataLoader
import torch

class TimeSeriesDataset(Dataset):
    """Dataset for fine-tuning: returns (context, target) pairs."""
    def __init__(self, series_list: list, context_len: int, horizon: int):
        self.samples = []
        for series in series_list:
            series = torch.tensor(series, dtype=torch.float32)
            for start in range(0, len(series) - context_len - horizon, horizon // 2):
                ctx = series[start : start + context_len]
                tgt = series[start + context_len : start + context_len + horizon]
                self.samples.append((ctx, tgt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def finetune_chronos(
    train_series: list,
    val_series: list,
    model_name: str = 'amazon/chronos-t5-small',
    context_len: int = 512,
    horizon: int = 96,
    n_epochs: int = 5,
    lr: float = 1e-4,
    batch_size: int = 16,
):
    pipeline  = ChronosPipeline.from_pretrained(model_name, device_map='cuda',
                                                torch_dtype=torch.bfloat16)
    model     = pipeline.model
    tokenizer = pipeline.tokenizer

    train_loader = DataLoader(
        TimeSeriesDataset(train_series, context_len, horizon),
        batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * len(train_loader)
    )

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for ctx, tgt in train_loader:
            ctx, tgt = ctx.cuda(), tgt.cuda()

            ctx_tokens, ctx_scale = tokenizer.context_input_transform(ctx)
            tgt_tokens, _         = tokenizer.label_input_transform(tgt, ctx_scale)

            outputs = model(
                context_ids=ctx_tokens,
                target_ids=tgt_tokens,
                context_length=torch.full((ctx.size(0),), ctx.size(1)),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += outputs.loss.item()

        print(f"Epoch {epoch+1}: loss = {total_loss / len(train_loader):.4f}")

    return pipeline
```

*Fine-tuning best practices:*
- Use a small learning rate (`1e-5` to `1e-4`); large LR will catastrophically forget pretraining
- Fine-tune for 3–10 epochs; monitor validation CRPS to stop early
- Use the full context window (512 tokens) to avoid distribution mismatch with pretraining
- If your domain data is small ($<$ 100 series), consider frozen encoder fine-tuning: only train the decoder

== Chronos Failure Modes

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Failure*], [*Cause*], [*Symptom*], [*Fix*]),
  [Poor calibration on novel domain],
  [Prior doesn't match domain structure],
  [Coverage far from nominal],
  [Fine-tune on domain data],
  [Context length exceeded],
  [Series longer than 512 steps],
  [Truncation → model misses long-range patterns],
  [Subsample context; use TimesFM for long contexts],
  [Slow inference at large scale],
  [100 samples × many series],
  [Inference time too long],
  [Reduce `num_samples` to 20; use `chronos-t5-tiny`],
  [Overconfident intervals],
  [Low temperature + limited sample diversity],
  [Intervals too narrow],
  [Increase `temperature`, `top_k`],
  [Underperformance on multi-variate],
  [Univariate model; no cross-series info],
  [Treats each series independently],
  [Use Moirai for multi-variate tasks],
  [Scale anomaly],
  [A single outlier distorts $mu$, $sigma$ used for normalization],
  [Bizarre forecasts],
  [Winsorize context before passing to model],
)

// ─── SECTION 5 ───────────────────────────────────────────────────────────────

= Moirai (Salesforce)

== The Any-Variate Design

Moirai (Woo et al., 2024) solves the problem that Chronos and TimesFM side-step: how to handle *multi-variate time series* where cross-series correlations matter.

The insight: instead of treating each variate as a separate univariate series, represent the entire multi-variate series as a *sequence of (variate, patch) tokens*. Attention operates across both time and variates simultaneously.

```
Multi-variate input: Y in R^{T x V}   (T timesteps, V variates)

Tokenization:
  For each variate v = 1..V, for each time patch p = 1..P:
    token_{v,p} = PatchEmbed(Y_{p*S:(p+1)*S, v})
                + variate_embed(v) + time_embed(p)

Sequence length = V x P    (variates x patches)

Transformer attention: each token attends to all other tokens
  -> captures both temporal dependencies AND cross-variate dependencies
```

This is the "any-variate" property: the model handles 1 variate, 10 variates, or 100 variates with the same architecture. The number of tokens scales with $V times P$, so attention cost scales as $O((V P)^2)$ — manageable for moderate $V$.

== Output Distribution: Mixture of Distributions

Moirai outputs a *mixture of distributions* at each forecast step:

$ p(y_(T+h) | "context") = sum_k pi_k dot "Distribution"_k (mu_k, sigma_k) $

where the mixture weights $pi_k$ and component parameters $(mu_k, sigma_k)$ are predicted by the model. The component distributions include Student-$t$ (heavy tails), Normal, and log-Normal (positive-valued series). This is more expressive than a single Gaussian or discrete bins — the model can represent multimodal predictive distributions.

== Frequency-Specific Tokenization

A weekly series and an hourly series have fundamentally different temporal structures — a patch of 32 weekly points spans 8 months, while a patch of 32 hourly points spans 32 hours. Moirai uses *frequency-specific patch sizes*:

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Frequency*], [*Patch size*]),
  [High (sub-hourly, hourly)], [32],
  [Medium (daily, weekly)], [2],
  [Low (monthly, quarterly)], [1],
)

This ensures each patch covers approximately the same "temporal coverage" regardless of the underlying frequency.

== Inference

```python
# pip install uni2ts
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import torch

module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-small')

predictor = MoiraiForecast(
    module=module,
    prediction_length=96,
    context_length=512,
    patch_size='auto',                  # inferred from frequency
    num_samples=100,
    target_dim=1,                       # 1 for univariate, V for V-variate
    feat_dynamic_real_dim=0,            # number of known-future covariates
    past_feat_dynamic_real_dim=0,
)

# Univariate
forecast = predictor(
    past_target=torch.tensor(series).unsqueeze(0).unsqueeze(-1),  # (1, T, 1)
    past_observed_target=torch.ones(1, len(series), 1, dtype=torch.bool),
)
# forecast: (1, num_samples, prediction_length, 1)

# Multi-variate (V=3)
forecast_mv = predictor.change_prediction_params(target_dim=3)(
    past_target=torch.tensor(mv_series).unsqueeze(0),             # (1, T, 3)
    past_observed_target=torch.ones(1, T, 3, dtype=torch.bool),
)
```

== When Moirai Outperforms Chronos and TimesFM

- *Multi-variate series* with genuine cross-series dependencies (e.g., electricity demand across interconnected grid regions, or related product demands in retail)
- *Missing data* — the `past_observed_target` mask explicitly handles observations missing at arbitrary positions
- *Mixed-frequency panels* — the frequency-specific tokenization handles diverse input frequencies in one model

// ─── SECTION 6 ───────────────────────────────────────────────────────────────

= Lag-Llama

Lag-Llama (Rasul et al., 2024) takes a different approach: instead of patching or quantizing, it uses *lag features* as the token representation.

```
For timestep t, the input token is:
  x_t = [y_t, y_{t-1}, y_{t-7}, y_{t-14}, y_{t-30}, y_{t-365}, ...]
  (current value + selected lags capturing daily, weekly, monthly, annual patterns)
```

This lag vector is projected to a token embedding. The model is a LLaMA-architecture decoder-only Transformer, making it straightforwardly fine-tunable with the HuggingFace `transformers` ecosystem.

*When to use Lag-Llama:*
- You want a fully open-source model (Apache 2.0) that you can modify freely
- You want the fine-tuning workflow to be identical to LLM fine-tuning (LoRA, PEFT, etc.)
- You are building on top of the LLaMA ecosystem

#gotcha[The lag features encode seasonality at fixed lags — if your series has a seasonal period not in the predefined lag set, the model misses it. Patching (TimesFM, Moirai) is more flexible.]

// ─── SECTION 7 ───────────────────────────────────────────────────────────────

= Practical Comparison: What Actually Matters

== Zero-Shot Benchmark Performance

On the standard ETTh1 benchmark (horizon=96):

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Model*], [*MAE*], [*Zero-shot?*], [*Notes*]),
  [ARIMA], [$approx 0.45$], [—], [Fitted per-series],
  [LSTM (trained)], [$approx 0.40$], [—], [Trained on ETTh1 train set],
  [TFT (trained)], [$approx 0.38$], [—], [Trained on ETTh1 train set],
  [TimesFM], [$approx 0.39$], [✓], [No ETTh1 training data],
  [Chronos-small], [$approx 0.41$], [✓], [No ETTh1 training data],
  [Moirai-small], [$approx 0.38$], [✓], [No ETTh1 training data],
)

#note[These numbers are illustrative; exact results depend on implementation details, preprocessing, and evaluation protocol. The key takeaway is that zero-shot FMs are competitive with _trained_ LSTM and TFT — not consistently better, but on the same order. On out-of-distribution datasets (where the FM training corpus is relevant), FMs can substantially outperform trained models.]

== The Regime Map

The key question isn't "which model is best overall?" but "which model is best for this specific situation?"

```
                    Amount of labeled data available

                    None / tiny         Moderate           Large
                   +------------------+-----------------+------------------+
Univariate,        | Chronos/TimesFM  | Chronos         | LSTM / TCN       |
no covariates      | zero-shot        | fine-tune or    |                  |
                   |                  | LSTM            |                  |
                   +------------------+-----------------+------------------+
Univariate,        | Chronos          | TFT with        | TFT              |
with covariates    | zero-shot        | covariates      |                  |
                   | (ignores covars) |                 |                  |
                   +------------------+-----------------+------------------+
Multi-variate,     | Moirai           | Moirai          | TFT or custom    |
with structure     | zero-shot        | fine-tune or    | multi-variate DL |
                   |                  | TFT             |                  |
                   +------------------+-----------------+------------------+
```

*The practical protocol for a new forecasting problem:*

1. Run Chronos zero-shot → establishes a strong baseline in $<$ 1 hour
2. Train ARIMA/ETS → establishes classical baseline
3. If Chronos $>>$ ARIMA: domain is well-covered by pretraining; consider fine-tuning Chronos
4. If ARIMA $approx$ Chronos: series is short or has unusual structure; train TFT with features
5. If ARIMA $>>$ Chronos: your series has idiosyncratic structure the FM prior doesn't cover; train from scratch

== Inference Speed vs. Accuracy Tradeoff

```python
import time

def benchmark_inference(pipeline, context, n_series=100, horizon=96):
    contexts = [torch.tensor(context)] * n_series

    start = time.perf_counter()
    forecasts = pipeline.predict(contexts, prediction_length=horizon, num_samples=20)
    elapsed = time.perf_counter() - start

    print(f"{n_series} series, horizon={horizon}, num_samples=20")
    print(f"Total time: {elapsed:.2f}s  |  Per series: {elapsed/n_series*1000:.1f}ms")

# Rough benchmarks on A100 GPU:
# chronos-t5-tiny  (8M):   ~2ms/series
# chronos-t5-small (46M):  ~5ms/series
# chronos-t5-base  (200M): ~15ms/series
# TimesFM (200M):          ~10ms/series (fewer autoregressive steps due to larger patches)
```

For production systems with latency constraints, `chronos-t5-tiny` or `chronos-t5-mini` are often sufficient. Accuracy degrades ~10–15% vs. `base`, which may be acceptable.

// ─── SECTION 8 ───────────────────────────────────────────────────────────────

= Critical Failure Modes Shared Across All Foundation Models

== Distribution Shift from Pretraining

FMs are trained on a fixed corpus. If your series has structural features not present in that corpus, the model's implicit prior is wrong.

*High-risk domains:*
- Highly specialized sensor data (industrial equipment, rare scientific instruments)
- Series with unusual noise distributions (heavy tails, zero-inflation)
- Series with non-standard seasonal periods (13-month fiscal years, Islamic calendar)
- Series with abrupt regime changes (policy interventions, market microstructure changes)

#note[*Detection:* compare FM zero-shot CRPS to a simple ETS baseline. If ETS wins, the FM prior is misspecified for your domain.]

== The Context Length Wall

All current models have a maximum context window (Chronos: 512; TimesFM: 512; Moirai: 512). For hourly data, 512 steps covers only ~3 weeks — missing monthly and annual seasonality entirely.

*Mitigation strategies:*
1. *Subsample* the context (e.g., take every 24th observation to capture daily-resolution patterns at ~1-year lookback)
2. *Decompose* the series with STL, forecast the residual with the FM, add back trend and seasonal components
3. *Use TimesFM* which has larger effective context via patching (512 patches $times$ patch_size timesteps)

```python
from statsmodels.tsa.seasonal import STL

stl = STL(series, period=24).fit()
residual = stl.resid

# Forecast residual with Chronos
residual_forecast = chronos_forecast(residual, horizon=96)

# Extrapolate trend and seasonal
trend_forecast    = extrapolate_trend(stl.trend, horizon=96)
seasonal_forecast = stl.seasonal[-24:].tolist() * 4   # repeat last cycle

final_forecast = residual_forecast + trend_forecast + seasonal_forecast
```

== Covariate Blindness

Chronos, TimesFM, and Lag-Llama are univariate — they see no covariates. If your domain has strong exogenous drivers (promotions, weather, economic indicators), the FM cannot use this information zero-shot.

*Options:*
1. Use Moirai (supports known-future covariates via `feat_dynamic_real_dim`)
2. Fine-tune Chronos on residuals after removing covariate effect with a linear model
3. Ensemble: FM for the "base" pattern + linear model for the covariate effect

== Silent Miscalibration

FMs can produce confident-looking prediction intervals that are systematically miscalibrated. Unlike a trained TFT where you can observe the training loss, FM calibration is dataset-dependent and not visible until you evaluate.

*Always check calibration empirically:*

```python
def calibration_check(samples: np.ndarray, actuals: np.ndarray,
                       quantiles: list = [0.1, 0.2, 0.5, 0.8, 0.9]) -> dict:
    """
    samples: (n_samples, horizon)
    actuals: (horizon,)
    Returns empirical coverage at each nominal quantile level.
    """
    results = {}
    for q in quantiles:
        predicted_q = np.quantile(samples, q, axis=0)
        coverage = np.mean(actuals <= predicted_q)
        results[q] = {'nominal': q, 'empirical': coverage, 'gap': coverage - q}
    return results

# Good calibration:   empirical ~= nominal across all quantiles
# Over-confident:     empirical <  nominal (intervals too narrow)
# Under-confident:    empirical >  nominal (intervals too wide)
```

A reliability diagram (nominal quantile vs. empirical coverage) should be a diagonal line. Systematic deviation indicates the FM's uncertainty is miscalibrated for your domain.

// ─── SECTION 9 ───────────────────────────────────────────────────────────────

= Week 4 (Part 1) Practice Exercises

*Dataset: ETT (Electricity Transformer Temperature)*

```python
import pandas as pd
import numpy as np

df = pd.read_csv('ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

ot = df['OT'].values          # oil temperature, hourly

n         = len(ot)           # 17420 hours
train_end = 12 * 30 * 24      # first 12 months
val_end   = 16 * 30 * 24      # next 4 months
# test: remaining 4 months
```

== Exercise 1: Zero-Shot Chronos vs. Baselines (2–3 hours)

Evaluate Chronos-small zero-shot on ETTh1 OT, horizon=96:

```python
from chronos import ChronosPipeline
import torch

pipeline = ChronosPipeline.from_pretrained('amazon/chronos-t5-small',
                                           device_map='cpu',
                                           torch_dtype=torch.float32)

def chronos_wf_eval(series, train_end, horizon=96, n_folds=5, context_len=512):
    results = []
    test_starts = np.linspace(train_end, len(series) - horizon, n_folds, dtype=int)

    for start in test_starts:
        ctx_start = max(0, start - context_len)
        context   = torch.tensor(series[ctx_start:start], dtype=torch.float32)
        actuals   = series[start : start + horizon]

        forecast = pipeline.predict(
            context=context.unsqueeze(0),
            prediction_length=horizon,
            num_samples=100,
        ).numpy()[0]   # (100, 96)

        median_fc = np.median(forecast, axis=0)
        mae  = np.mean(np.abs(actuals - median_fc))
        crps = compute_crps(forecast, actuals)
        cov  = np.mean((actuals >= np.quantile(forecast, 0.1, 0)) &
                       (actuals <= np.quantile(forecast, 0.9, 0)))

        results.append({'mae': mae, 'crps': crps, 'coverage_80': cov})

    return pd.DataFrame(results)
```

Compare against: ARIMA zero-shot (fitted on same context), seasonal naive.

*Questions to answer:*
- Does Chronos beat ARIMA on this dataset? By how much?
- Is the 80% interval empirically covering ~80% of actuals?
- How does performance vary across the 5 folds?

== Exercise 2: TimesFM Comparison (1–2 hours)

Run TimesFM on the same walk-forward protocol:

```python
import timesfm

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(backend='cpu', horizon_len=96),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id='google/timesfm-1.0-200m',
    ),
)
tfm.initialize()

point_fc, quantile_fc = tfm.forecast(inputs=[context_np], freq=[0])
```

#note[Chronos tends to be better calibrated; TimesFM tends to have lower MAE on point forecasts. Compare both on MAE and CRPS.]

== Exercise 3: Calibration Analysis (1 hour)

Generate a reliability diagram for Chronos:

```python
import matplotlib.pyplot as plt

def reliability_diagram(all_samples: np.ndarray, all_actuals: np.ndarray):
    """
    all_samples: (n_folds * horizon, num_samples)
    all_actuals: (n_folds * horizon,)
    """
    quantile_levels    = np.linspace(0.05, 0.95, 19)
    empirical_coverages = []

    for q in quantile_levels:
        predicted_q = np.quantile(all_samples, q, axis=1)
        empirical_coverages.append(np.mean(all_actuals <= predicted_q))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(quantile_levels, empirical_coverages, 'o-', label='Chronos-small')
    plt.xlabel('Nominal quantile level')
    plt.ylabel('Empirical coverage')
    plt.title('Reliability diagram: ETTh1 OT, H=96')
    plt.legend()
    plt.grid(True)
```

*What to look for:* If the curve is above the diagonal, intervals are too wide (conservative). Below: too narrow (overconfident). Report the mean calibration gap across quantile levels.

== Exercise 4: Fine-Tuning Chronos (Optional, 2–4 hours)

Fine-tune `chronos-t5-small` on the ETTh1 training set for 3 epochs:

```python
fine_tuned_pipeline = finetune_chronos(
    train_series=[ot[:train_end]],
    val_series=[ot[train_end:val_end]],
    model_name='amazon/chronos-t5-small',
    context_len=512,
    horizon=96,
    n_epochs=3,
    lr=5e-5,
)
```

*Expected:* modest improvement, ~5–15% MAE reduction for a single series — larger gains with more domain data.

== CRPS Implementation for Sampled Forecasts

```python
def compute_crps(samples: np.ndarray, actuals: np.ndarray) -> float:
    """
    Energy score form of CRPS for sample-based forecasts.
    samples: (n_samples, horizon)
    actuals: (horizon,)
    """
    n = samples.shape[0]
    term1 = np.mean(np.abs(samples - actuals[None, :]))
    idx = np.random.choice(n, min(50, n), replace=False)
    s1, s2 = samples[idx], samples[np.random.choice(n, len(idx))]
    term2 = 0.5 * np.mean(np.abs(s1 - s2))
    return term1 - term2
```

// ─── SECTION 10 ──────────────────────────────────────────────────────────────

= Interview Fluency

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Term*], [*Definition*]),
  [Patch tokenization],
  [Grouping $p$ consecutive timesteps into a single token via linear projection; reduces sequence length and captures local temporal structure],
  [Value quantization],
  [Discretizing continuous time series values into $B$ bins after normalization; converts regression to classification],
  [Zero-shot forecasting],
  [Applying a pretrained model to a new dataset without any task-specific training],
  [Few-shot / fine-tuning],
  [Updating a pretrained model's weights on domain-specific data to improve performance],
  [Any-variate attention],
  [Attention over tokens representing both time patches and variates; enables cross-series dependency modeling in a single Transformer],
  [Posterior predictive],
  [$p(bold(y)_"future" | bold(y)_"past")$ — the distribution over future values given past observations; what a well-calibrated forecast approximates],
  [Calibration],
  [Alignment between a model's stated confidence (e.g., 80% interval) and empirical accuracy; measured by coverage],
  [Reliability diagram],
  [Plot of nominal quantile level vs. empirical coverage; a diagonal line indicates perfect calibration],
  [CRPS],
  [Continuous Ranked Probability Score; proper scoring rule for predictive distributions; equals MAE when applied to empirical samples],
  [Context length wall],
  [The maximum history a foundation model can use; series history beyond this window is inaccessible to the model],
)

*"How does Chronos turn time series forecasting into a language modeling problem?"*

#note[Chronos normalizes each series to zero-mean unit-variance (making scales comparable), then quantizes the normalized values into discrete bins — effectively creating a vocabulary of "value tokens." A T5 encoder-decoder Transformer then models the sequence of these tokens, just as a language model models word sequences. At inference, the decoder outputs a softmax distribution over bins at each future step, which is a discrete approximation to the predictive distribution. Sampling from these distributions gives Monte Carlo draws of the forecast, enabling interval estimation without any distributional assumption.]

*"When would you use a foundation model instead of training TFT from scratch?"*

#note[The main trigger is *data scarcity*: when you have few labeled observations, few series, or need a forecast immediately without training time. Foundation models are also the right first step when entering a new domain — they give you a calibrated baseline in under an hour, telling you whether the problem is solvable with standard methods before you invest in a full training pipeline. The cases where I'd stick with TFT: when I have rich covariate information the FM can't use, when I need to encode business constraints in the loss, when the series structure is highly domain-specific, or when the deployment requires guaranteed latency.]

*"What is the context length wall and how do you work around it?"*

#note[All current foundation models cap at ~512 timesteps of context. For hourly data, that's only 3 weeks — not enough to capture monthly or annual seasonality. The cleanest workaround is STL decomposition: decompose the full history into trend, seasonal, and residual components. The seasonal and trend components can be extrapolated analytically. You then feed only the residual (which is roughly stationary) to the FM within its context window, and add the extrapolated components back at the output. An alternative is subsampling the context (e.g., daily averages of hourly data) to get a longer effective window at lower resolution.]

*"How do you evaluate whether a foundation model's uncertainty is calibrated?"*

#note[Draw $N=100$ samples from the model and check empirical coverage at multiple quantile levels. At the 80th percentile prediction, the actual value should fall below the predicted quantile roughly 80% of the time across many evaluation windows. Plot these as a reliability diagram (nominal vs. empirical coverage) — a calibrated model gives a diagonal line. Systematic deviation indicates the model's uncertainty is wrong for your domain. This is crucial to check because FMs can produce confident-looking intervals that are systematically too narrow or too wide on out-of-distribution data, and this won't be obvious without explicit evaluation.]

*"What is patching and why does it help for long-horizon forecasting?"*

#note[Patching groups $p$ consecutive timesteps into a single token via linear projection. This reduces the effective sequence length by a factor of $p$ — a 512-step series becomes 16 tokens with $p=32$. For Transformer attention ($O(n^2)$ in sequence length), this is a significant speedup. More importantly, the model predicts an output patch of $p$ values per autoregressive step rather than one step at a time. To forecast 512 steps ahead, only 16 autoregressive steps are needed instead of 512, drastically reducing error accumulation. Each patch also captures a local temporal structure (a day's worth of hourly data, a week of daily data) as a holistic embedding — a better unit for the Transformer than individual timesteps.]

// ─── SECTION 11 ──────────────────────────────────────────────────────────────

= Summary: Where Foundation Models Fit in the Stack

```
New forecasting problem
         |
         v
+----------------------------------------------------------+
| Step 1: Chronos zero-shot  (< 1 hour)                    |
| -> Establishes FM baseline; checks domain coverage       |
+-------------------------------+--------------------------+
                                |
                    +-----------+----------+
                    |                      |
              FM >> baseline          FM ~= baseline
                    |                      |
                    v                      v
           Fine-tune Chronos       Train TFT / TCN
           on domain data          with features
                    |                      |
                    v                      v
+----------------------------------------------------------+
| Production ensemble: FM + trained model + classical      |
| Weights optimized on validation set                      |
+----------------------------------------------------------+
         |
         v
Monitor calibration on live data
Retrain FM fine-tune / TFT when CRPS degrades
```

*One-sentence summary per model:*

- *TimesFM:* patch-based decoder Transformer; best zero-shot point forecast accuracy at long horizons; not natively probabilistic
- *Chronos:* quantization-based T5; best zero-shot calibrated intervals; fine-tunable; univariate
- *Moirai:* any-variate patch Transformer; only model doing genuine multi-variate zero-shot; mixture distribution output
- *Lag-Llama:* LLaMA with lag features; most open (Apache 2.0); most fine-tunable; niche

#line(length: 100%)

_Next: Phase 4 — Evaluation, Backtesting, Failure Modes_

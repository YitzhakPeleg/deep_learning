#set document(title: "Time-Series Forecasting — Phase 2: Core DL Approaches", author: "")
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
  #text(size: 15pt)[Phase 2 — Core Deep Learning Approaches (Weeks 2–3)]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[LSTM · TCN · Temporal Fusion Transformer]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

#outline(indent: 1em, depth: 2)

#v(0.5em)
#line(length: 100%)

// ─── OVERVIEW ────────────────────────────────────────────────────────────────

= Overview

Phase 1 gave you the problem structure: trend, seasonality, autocorrelation, noise. DL methods don't model these components explicitly — they learn functions that approximate the mapping from a lookback window to a forecast horizon, and the question is always: _what architectural bias helps that learning on temporal data?_

Three architectures dominate practical DL forecasting, each encoding a different inductive bias:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Architecture*], [*Inductive bias*], [*Core mechanism*]),
  [LSTM],
  [Sequential ordering; long-range memory via gating],
  [Recurrent state],
  [TCN],
  [Local temporal patterns; causality; parallelism],
  [Dilated causal convolution],
  [TFT],
  [Heterogeneous covariates; multi-horizon; interpretability],
  [Attention + gating + covariate routing],
)

*By the end of Week 3 you should be able to:*

- Derive the LSTM gate equations from first principles and explain what each gate does
- Implement a TCN `TemporalBlock` from scratch in PyTorch
- Trace an input tensor through a TFT forward pass, naming each sub-module and its purpose
- Choose between the three architectures given a problem specification
- Diagnose training failures (gradient issues, teacher forcing mismatch, receptive field misses)
- Run a proper walk-forward comparison on a real benchmark dataset

// ─── SECTION 1 ───────────────────────────────────────────────────────────────

= Problem Reformulation: From SSMs to Supervised Seq2Seq

Classical methods modeled the series generatively: specify a stochastic process, fit parameters, compute the posterior predictive. DL methods reframe forecasting as *supervised function approximation*:

$ hat(bold(y))_(t+1 : t+H) = f_theta (bold(y)_(t-L+1 : t), bold(x)_(t-L+1 : t+H)) $

where:
- $L$ = lookback window (context length)
- $H$ = forecast horizon
- $bold(y)_(t-L+1:t)$ = past observed target values
- $bold(x)_(t-L+1:t+H)$ = covariates, partitioned into:
  - *Past-observed only:* weather, sales of related products, sensor readings
  - *Known future:* calendar features (day-of-week, holidays), scheduled promotions, future prices

The distinction between past-only and known-future covariates is architecturally important — TFT treats them differently; vanilla LSTM and TCN require you to handle this explicitly at the data layer.

== What the Model Must Learn Implicitly

Everything ARIMA and ETS encoded explicitly must now be learned from data:

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Classical encoding*], [*DL must learn*]),
  [Differencing ($d$)], [That recent deviations from mean matter],
  [AR($p$) weights], [Which past timesteps are predictive],
  [Seasonal period $m$], [That observations $m$ steps back are strongly correlated],
  [Level $l_t$], [A representation of the current "baseline"],
  [Trend $b_t$], [The direction and rate of drift],
)

This means DL models need enough data to learn these structures, and they need architectural mechanisms (memory, receptive field, attention) that make this learning tractable.

== Normalization is Non-Negotiable

DL optimizers assume roughly unit-scale activations. Time series raw values are often on wildly different scales (electricity in kWh, temperature in °C, sales in dollars). Always apply *reversible instance normalization* or at minimum per-series standard normalization before feeding to any DL model:

```python
# RevIN (reversible instance normalization) — normalize per sample, denormalize output
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        # x: (batch, seq_len, features)
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std  = x.std(dim=1, keepdim=True, unbiased=False).detach() + self.eps
            x = (x - self.mean) / self.std
            x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self.std + self.mean
        return x
```

RevIN (Kim et al., 2022) normalizes each instance (each lookback window) independently, so the model sees a zero-mean unit-variance signal and the original scale is restored at output. This handles non-stationarity implicitly.

// ─── SECTION 2 ───────────────────────────────────────────────────────────────

= LSTM

== Why Vanilla RNNs Fail

A vanilla RNN:

$ bold(h)_t = tanh(bold(W)_h bold(h)_(t-1) + bold(W)_x bold(x)_t + bold(b)) $

The gradient of the loss with respect to $bold(h)_0$ involves a product of $T$ Jacobian matrices $partial bold(h)_t \/ partial bold(h)_(t-1)$. If the dominant singular value of $bold(W)_h$ is $< 1$, gradients vanish exponentially; if $> 1$, they explode. Gradient clipping addresses explosion; vanishing is the fundamental problem — the model literally cannot receive training signal from distant past timesteps.

== The LSTM Solution: Gated State

The LSTM introduces a *cell state* $bold(c)_t$ — a highway through time that gradients can flow along almost unimpeded. Three gates control what information enters, what leaves, and what gets output. With $[bold(h)_(t-1), bold(x)_t]$ denoting concatenation:

$
  bold(f)_t &= sigma(bold(W)_f [bold(h)_(t-1), bold(x)_t] + bold(b)_f)
    &&quad "forget gate: what to erase from" bold(c)_(t-1) \
  bold(i)_t &= sigma(bold(W)_i [bold(h)_(t-1), bold(x)_t] + bold(b)_i)
    &&quad "input gate: fraction of candidate to write" \
  tilde(bold(c))_t &= tanh(bold(W)_c [bold(h)_(t-1), bold(x)_t] + bold(b)_c)
    &&quad "cell candidate: proposed new content" \
  bold(c)_t &= bold(f)_t dot.o bold(c)_(t-1) + bold(i)_t dot.o tilde(bold(c))_t
    &&quad "cell update: controlled integration" \
  bold(o)_t &= sigma(bold(W)_o [bold(h)_(t-1), bold(x)_t] + bold(b)_o)
    &&quad "output gate: what to expose" \
  bold(h)_t &= bold(o)_t dot.o tanh(bold(c)_t)
    &&quad "hidden state: actual output"
$

*Gradient flow through the cell:* $partial bold(c)_t \/ partial bold(c)_(t-1) = bold(f)_t$ (element-wise). If the forget gate stays near 1, the gradient flows through $bold(c)_t$ unattenuated for arbitrarily many steps. The forget gate learning to stay open for relevant information is the core of LSTM's long-range memory.

#insight[
  *Kalman filter analogy:* $bold(c)_t = bold(f)_t dot.o bold(c)_(t-1) + bold(i)_t dot.o tilde(bold(c))_t$ is structurally identical to the Kalman filter state update $bold(mu)_t = (bold(I) - bold(K)_t bold(C)) bold(mu)_(t-1) + bold(K)_t bold(y)_t$. The forget gate $approx (bold(I) - bold(K)_t bold(C))$, the input gate $approx bold(K)_t$. The difference: Kalman gain is computed analytically from the model; LSTM gates are learned nonlinearly from data.
]

== GRU: The Streamlined Variant

The Gated Recurrent Unit (GRU) merges the cell and hidden state and uses two gates instead of three:

$
  bold(r)_t &= sigma(bold(W)_r [bold(h)_(t-1), bold(x)_t]) \
  bold(z)_t &= sigma(bold(W)_z [bold(h)_(t-1), bold(x)_t]) \
  tilde(bold(h))_t &= tanh(bold(W)_h [bold(r)_t dot.o bold(h)_(t-1), bold(x)_t]) \
  bold(h)_t &= (bold(1) - bold(z)_t) dot.o bold(h)_(t-1) + bold(z)_t dot.o tilde(bold(h))_t
$

GRU has fewer parameters (~75% of LSTM for same hidden size), trains faster, and often performs comparably. Use GRU when computation is a bottleneck; use LSTM when you need the additional representational capacity of the separated cell state.

== Multi-Step Forecasting Strategies

Given a trained LSTM encoder that produces $bold(h)_T$ from the lookback, you need to produce $H$ forecast steps. Three strategies:

*Direct (recommended for most cases):*

$ hat(bold(y))_(T+1:T+H) = "Linear"(bold(h)_T) quad "(single forward pass, H outputs)" $

Pros: no error accumulation; fully parallelizable; simple. Cons: doesn't model inter-step dependencies.

*Recursive (autoregressive):*

```
for h in 1..H:
    ŷ_{T+h}, h_{T+h} = LSTM(ŷ_{T+h-1}, h_{T+h-1})
```

Pros: explicitly models step-to-step dynamics. Cons: errors compound; training/inference mismatch (teacher forcing).

*MIMO (Multi-Input Multi-Output):* a middle ground — predict blocks of steps directly, then chain blocks. Reduces error accumulation vs. fully recursive while partially capturing inter-step dependencies.

#note[For interviews: direct is the practical default. Recursive only makes sense when inter-step conditional structure matters (e.g., next-day demand conditioned on this-day demand in a realistic causal chain).]

== Teacher Forcing and Exposure Bias

During training with recursive decoding, you can feed the *ground truth* $y_(T+h-1)$ as input to step $h$ instead of the model's prediction $hat(y)_(T+h-1)$. This is *teacher forcing* — it stabilizes training by preventing error cascades, but it creates a mismatch: at inference, the model has never seen its own (imperfect) outputs as inputs.

*Scheduled sampling* (Bengio et al., 2015): interpolate between teacher forcing and free-running during training. With probability $epsilon_i$ (decaying over training), feed ground truth; otherwise feed model output.

```python
def scheduled_sample(y_true, y_pred, epsilon):
    """epsilon: probability of using ground truth (start ~1.0, decay toward 0)"""
    mask = torch.bernoulli(torch.full_like(y_true, epsilon)).bool()
    return torch.where(mask, y_true, y_pred.detach())
```

For direct forecasting, teacher forcing doesn't apply — the model outputs all $H$ steps in one shot and the mismatch doesn't exist. Another argument for direct over recursive.

== Full Implementation: Encoder-Decoder LSTM

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,       # number of input features (1 if univariate)
        hidden_size: int,      # LSTM hidden dimension
        num_layers: int,       # stacked LSTM depth
        horizon: int,          # forecast steps H
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Direct multi-step head: maps final hidden state to H outputs
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, input_size)
        returns: (batch, horizon)
        """
        _, (h_n, _) = self.encoder(x)   # h_n: (num_layers, batch, hidden_size)
        h_last = h_n[-1]                 # take top layer: (batch, hidden_size)
        return self.head(h_last)         # (batch, horizon)


class LSTMTrainer:
    def __init__(self, model, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        preds, targets = [], []
        for x, y in loader:
            x = x.to(self.device)
            preds.append(self.model(x).cpu())
            targets.append(y)
        preds   = torch.cat(preds)
        targets = torch.cat(targets)
        mae  = (preds - targets).abs().mean().item()
        rmse = ((preds - targets)**2).mean().sqrt().item()
        return {'mae': mae, 'rmse': rmse}
```

== LSTM Failure Modes and Diagnostics

#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Failure*], [*Symptom*], [*Diagnosis*], [*Fix*]),
  [Vanishing gradient],
  [Loss stops decreasing early; distant timesteps ignored],
  [Gradient norm near zero in early layers],
  [Gradient clipping; reduce `num_layers`; use GRU],
  [Exploding gradient],
  [Loss spikes or NaN],
  [Gradient norm $>> 1$],
  [`clip_grad_norm_` with `max_norm=1.0`],
  [Teacher forcing mismatch],
  [Good train loss, poor inference on long rollouts],
  [Recursive inference much worse than direct],
  [Switch to direct; or scheduled sampling],
  [Sequence length bottleneck],
  [Performance degrades beyond $L approx 300$],
  [Performance plateau at long lookbacks],
  [Switch to TCN or Transformer],
  [Forgetting short-term patterns],
  [Model tracks trend but misses local spikes],
  [Attention concentrated at distant lags],
  [Reduce hidden size; use bidirectional encoder],
  [Scale sensitivity],
  [Loss diverges or saturates early],
  [Raw values in thousands or tiny fractions],
  [Apply RevIN or per-series normalization],
)

// ─── SECTION 3 ───────────────────────────────────────────────────────────────

= Temporal Convolutional Network (TCN)

== The Core Idea: Dilated Causal Convolution

TCN applies 1D convolutions along the time axis with two critical modifications.

*Causal:* the convolution at position $t$ can only see positions $<= t$. Enforced by left-padding only (no right-padding), so future values cannot leak into the output.

*Dilated:* the filter samples the input at intervals of $d$ rather than every step, giving a larger effective receptive field without increasing the number of parameters or the depth of the network.

A dilated causal convolution with dilation $d$ and kernel size $k$ at position $t$:

$ (bold(x) *_d bold(w))[t] = sum_(j=0)^(k-1) bold(w)[j] dot bold(x)[t - d dot j] $

The exponential dilation schedule $d = 1, 2, 4, 8, dots, 2^(L-1)$ gives a receptive field of:

$ "RF" = 1 + 2(k-1)(2^L - 1) $

For $k=3$, $L=8$: RF $= 1 + 2 dot 2 dot (256-1) = 1021$ steps from 8 layers and $2 times 8 times 2$ parameters per filter. Compare to an LSTM that would need 1021 sequential steps to reach the same history.

#note[
  *Connection to your CNN background:* This is DeepLab's atrous/dilated convolution, applied causally to a 1D temporal signal. The key difference is the strict left-only causality (no symmetric dilation) and the exponential schedule that covers the receptive field densely.
]

== Receptive Field Calculation and Planning

Before designing a TCN, calculate the required receptive field:

```python
def tcn_receptive_field(kernel_size: int, n_levels: int) -> int:
    """Receptive field of a TCN with exponential dilation doubling."""
    return 1 + 2 * (kernel_size - 1) * (2**n_levels - 1)

def tcn_levels_needed(target_rf: int, kernel_size: int) -> int:
    import math
    return math.ceil(math.log2(target_rf / (2 * (kernel_size - 1)) + 1))

# Example: need 512-step RF with k=3
print(tcn_levels_needed(512, 3))   # -> 8 levels, each doubling dilation
print(tcn_receptive_field(3, 8))   # -> 1021 (more than enough)
```

Plan your TCN architecture by answering: what is the longest meaningful dependency in my series? For hourly electricity with daily+weekly patterns, that's 168 steps (7 days × 24 hours). For monthly M4 data with annual seasonality, that's 12 steps — a very shallow TCN suffices.

== TemporalBlock: The Building Unit

Each TCN layer is a *TemporalBlock*: two dilated causal convolutions with weight normalization, ReLU, and dropout, wrapped in a residual connection.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class TemporalBlock(nn.Module):
    """
    One TCN residual block: two dilated causal convolutions + residual.
    Causal padding: pad (k-1)*d on the LEFT only, then trim the right.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation   # left-only causal padding

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        ))

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len)"""
        out = self.conv1(x)
        out = out[:, :, : x.size(2)]   # remove right-side padding -> causal
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]   # causal again
        out = F.relu(out)
        out = self.dropout2(out)

        return F.relu(out + self.downsample(x))
```

#note[
  *Why `weight_norm` instead of `BatchNorm`?* BatchNorm over the time dimension computes statistics across the batch AND the sequence, which conflates different positions. For causal temporal models it also creates implicit look-ahead (batch statistics include future-position samples). `weight_norm` reparameterizes weight matrices as $bold(w) = g dot bold(v) \/ norm(bold(v))$, keeping training stable without temporal mixing.
]

== Full TCN

```python
class TCN(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_channels: list[int],   # e.g. [64, 64, 64, 64] — one entry per level
        kernel_size: int,
        horizon: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        n_levels = len(num_channels)

        for i in range(n_levels):
            dilation  = 2 ** i
            in_ch  = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.head    = nn.Linear(num_channels[-1], horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, input_size)
        returns: (batch, horizon)
        """
        x = x.permute(0, 2, 1)          # -> (batch, input_size, lookback)
        features = self.network(x)       # -> (batch, num_channels[-1], lookback)
        last_step = features[:, :, -1]   # take the last timestep's representation
        return self.head(last_step)       # -> (batch, horizon)
```

== Verifying Causality

This is a common interview question and a common implementation bug. After building your TCN, verify that future timesteps cannot affect earlier outputs:

```python
def verify_causality(model, seq_len=100, input_size=1):
    model.eval()
    x = torch.randn(1, seq_len, input_size, requires_grad=False)

    x_perturbed = x.clone()
    x_perturbed[0, -1, :] += 100.0   # large perturbation at the last step

    with torch.no_grad():
        out_orig    = model(x)
        out_perturb = model(x_perturbed)

    print("Max output change from future perturbation:",
          (out_orig - out_perturb).abs().max().item())
    # Should be near 0 if model is causal; large if there is look-ahead leakage
```

== TCN Failure Modes

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Failure*], [*Symptom*], [*Fix*]),
  [Receptive field too small],
  [Performance plateaus; adding more lookback data doesn't help],
  [Add more TCN levels or increase kernel size],
  [Receptive field over-designed],
  [Good accuracy but slow; more parameters than needed],
  [Reduce levels; profile inference time],
  [Right-padding leak],
  [Causal violation — model uses future],
  [Verify with causality check; ensure trim `[:, :, :T]`],
  [Weight explosion],
  [Training diverges],
  [Ensure `weight_norm` applied; add gradient clipping],
  [Periodic structure at non-power-of-2 period],
  [Model misses weekly (168-step) cycle],
  [Add a long-dilation layer targeting that period; or preprocess with STL],
)

// ─── SECTION 4 ───────────────────────────────────────────────────────────────

= Temporal Fusion Transformer (TFT)

TFT (Lim et al., 2021, NeurIPS) is the most important architecture to understand deeply for interview purposes. It solves the *practical* forecasting problem: multiple heterogeneous time series, mixed covariate types, multiple horizons, and a need for interpretable outputs. Every design choice is motivated — understanding the _why_ is more important than memorizing the architecture.

== The Problem TFT Solves

Vanilla LSTM and TCN assume a single input stream. Real production forecasting involves:

- *Static metadata:* store ID, product category, geography — time-invariant entity descriptors
- *Past-observed covariates:* competitor prices, weather, past demand of related products — observed up to $t$, unknown beyond
- *Known-future covariates:* promotions planned for next week, calendar features, scheduled events — known for the entire horizon

LSTM handles none of this natively. You'd have to concatenate everything manually, with no mechanism to distinguish covariate types or learn which are important. TFT is explicitly designed around this structure.

== Architecture Overview

TFT is best understood as a pipeline of five components:

```
Input Features
     |
     v
[1] Variable Selection Networks (VSN)     — which features matter, per timestep
     |
     v
[2] Static Covariate Encoders            — entity context vectors (c_s, c_e, c_h, c_c)
     |
     v
[3] Sequence Processing (LSTM encoder)   — local temporal patterns
     |
     v
[4] Multi-Head Attention                 — long-range temporal dependencies
     |
     v
[5] Point-wise Feed-Forward + Quantile Output
```

== Gated Residual Network (GRN): The Fundamental Unit

Every sub-component of TFT uses the GRN, so understand it first. Given input $bold(a)$ and optional context $bold(c)$:

$
  bold(eta)_2 &= "ELU"(bold(W)_2 bold(a) + bold(W)_3 bold(c) + bold(b)_2) \
  bold(eta)_1 &= bold(W)_1 bold(eta)_2 + bold(b)_1 \
  "GRN"(bold(a), bold(c)) &= "LayerNorm"(bold(a) + "GLU"(bold(eta)_1))
$

where the Gated Linear Unit (GLU) splits its input in half along the feature dimension:

$ "GLU"(bold(x)) = bold(x)_1 dot.o sigma(bold(x)_2) $

*Why this matters:* The sigmoid in GLU can zero out the entire nonlinear path, making the GRN reduce to a linear pass-through when the transformation isn't needed. This is *adaptive depth* — the model learns how much nonlinearity each sub-task requires. It also provides a gradient highway via the residual connection.

```python
class GRN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int = None, context_size: int = None,
                 dropout: float = 0.1):
        super().__init__()
        output_size = output_size or input_size

        self.W2 = nn.Linear(input_size, hidden_size)
        self.W3 = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.W1 = nn.Linear(hidden_size, output_size * 2)   # *2 for GLU split
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, a: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        eta2 = F.elu(self.W2(a) + (self.W3(c) if self.W3 is not None and c is not None else 0))
        eta1 = self.W1(eta2)
        eta1, gate = eta1.chunk(2, dim=-1)
        eta1 = eta1 * torch.sigmoid(gate)
        return self.layer_norm(self.skip(a) + self.dropout(eta1))
```

== Variable Selection Network (VSN)

The VSN learns which input features are relevant at each timestep. For a set of $D$ input features each of dimension $d_"model"$:

1. For each feature $j$: process through GRN → embedding $bold(xi)_(t,j) in RR^(d_"model")$
2. Concatenate all embeddings → GRN → softmax → selection weights $tilde(bold(a))_t in RR^D$
3. Selected representation:

$ bold(xi)_t = sum_j tilde(a)_(t,j) dot bold(xi)_(t,j) $

The weights $tilde(bold(a))_t$ are interpretable: after training you can plot them to see which features the model relied on at each timestep.

```python
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_sizes: dict[str, int], hidden_size: int,
                 context_size: int = None, dropout: float = 0.1):
        super().__init__()
        self.input_sizes = input_sizes
        self.hidden_size = hidden_size

        self.var_grns = nn.ModuleDict({
            name: GRN(size, hidden_size, hidden_size, context_size, dropout)
            for name, size in input_sizes.items()
        })
        total_size = sum(input_sizes.values())
        self.weight_grn = GRN(total_size, hidden_size, len(input_sizes), context_size, dropout)

    def forward(self, inputs: dict[str, torch.Tensor], context: torch.Tensor = None):
        embeddings = [self.var_grns[name](x, context)
                      for name, x in inputs.items()]

        flat = torch.cat([x for x in inputs.values()], dim=-1)
        weights = torch.softmax(self.weight_grn(flat, context), dim=-1)

        stacked = torch.stack(embeddings, dim=-2)          # (..., D, hidden_size)
        selected = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        return selected, weights
```

== Static Covariate Encoders

Static features (e.g., store ID, product category) are time-invariant. TFT encodes them once and injects the resulting context vector at four points:

$
  bold(c)_s &= "GRN"("static embedding") quad & "(context for variable selection)" \
  bold(c)_e &= "GRN"("static embedding") quad & "(initial cell state for LSTM)" \
  bold(c)_h &= "GRN"("static embedding") quad & "(initial hidden state for LSTM)" \
  bold(c)_c &= "GRN"("static embedding") quad & "(context for static enrichment)"
$

This is meaningful: the static context shapes what the VSN pays attention to, how the LSTM initializes (entity-specific priors), and how the final temporal representation is enriched. A different store starts with a different LSTM hidden state.

== Temporal Processing: LSTM + Attention

*LSTM encoder-decoder* processes the sequence after VSN selection. The encoder runs over the lookback window (past observed + known past covariates); the decoder runs over the forecast horizon (known future covariates only). The LSTM is initialized with $(bold(c)_h, bold(c)_e)$ from static encoders.

*Multi-head attention* then operates on the LSTM decoder outputs. Standard multi-head attention:

$
  "Attention"(bold(Q), bold(K), bold(V)) &= "softmax"(bold(Q) bold(K)^top \/ sqrt(d_k)) bold(V) \
  "head"_i &= "Attention"(bold(Q) bold(W)_(Q,i),\ bold(K) bold(W)_(K,i),\ bold(V) bold(W)_(V,i))
$

TFT's *interpretable attention* uses shared $bold(W)_V$ across all $N_h$ heads:

$
  "head"_i &= "Attention"(bold(Q) bold(W)_(Q,i), bold(K) bold(W)_(K,i), bold(V) bold(W)_V) \
  overline(bold(A)) &= frac(1, N_h) sum_i "softmax"(bold(Q) bold(W)_(Q,i) dot (bold(K) bold(W)_(K,i))^top \/ sqrt(d_k)) \
  "output" &= overline(bold(A)) dot bold(V) bold(W)_V
$

With separate V matrices per head, averaging attention weights produces a weighted average of different representations — the averaged weights have no clean semantic interpretation. With shared V, averaging the weights *before* multiplying gives a single attention pattern over a shared representation. This is what makes the attention plots interpretable.

== Quantile Output and Pinball Loss

TFT predicts quantiles $tau in {0.1, 0.5, 0.9}$ by optimizing the *pinball loss*:

$
  cal(L)_tau (y, hat(y))
  = tau dot max(y - hat(y), 0) + (1 - tau) dot max(hat(y) - y, 0)
  = cases(
      tau (y - hat(y)) & "if" y >= hat(y),
      (1-tau)(hat(y) - y) & "if" y < hat(y)
    )
$

For $tau = 0.9$: severely penalizes under-forecasting (9× the over-forecast penalty). The model learns the 90th percentile. Total loss = mean pinball loss across all quantiles.

This is a proper scoring rule for quantiles: the optimal predictor is exactly the $tau$-th quantile of the true conditional distribution. No distributional assumption required.

```python
def pinball_loss(y_pred: torch.Tensor, y_true: torch.Tensor,
                 quantiles: list[float] = [0.1, 0.5, 0.9]) -> torch.Tensor:
    """
    y_pred: (batch, horizon, n_quantiles)
    y_true: (batch, horizon)
    """
    y_true = y_true.unsqueeze(-1)  # (batch, horizon, 1)
    q = torch.tensor(quantiles, device=y_pred.device)
    errors = y_true - y_pred
    loss = torch.max(q * errors, (q - 1) * errors)
    return loss.mean()
```

== Using pytorch-forecasting

Don't reimplement TFT from scratch for the project. `pytorch-forecasting` has the reference implementation — your job is to understand its internals well enough to configure it correctly and interpret its outputs.

```python
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

df['time_idx'] = (df['ds'] - df['ds'].min()).dt.days  # integer time index

max_encoder_length = 168   # 7 days lookback
max_prediction_length = 24  # 1 day horizon
training_cutoff = df['time_idx'].max() - max_prediction_length

dataset = TimeSeriesDataSet(
    df[df['time_idx'] <= training_cutoff],
    time_idx='time_idx',
    target='demand',
    group_ids=['store_id'],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=['store_id', 'product_category'],
    static_reals=['store_size'],
    time_varying_known_reals=['time_idx', 'price', 'day_of_week', 'is_holiday'],
    time_varying_unknown_reals=['demand'],
    target_normalizer=GroupNormalizer(groups=['store_id'], transformation='softplus'),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

train_loader = dataset.to_dataloader(train=True, batch_size=64, num_workers=4)
val_loader   = TimeSeriesDataSet.from_dataset(
    dataset, df, predict=True, stop_randomization=True
).to_dataloader(train=False, batch_size=64, num_workers=4)

tft = TemporalFusionTransformer.from_dataset(
    dataset,
    learning_rate=3e-3,
    hidden_size=32,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,               # number of quantiles to predict
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)
print(f"Parameters: {sum(p.numel() for p in tft.parameters()):,}")
```

```python
import lightning.pytorch as pl

trainer = pl.Trainer(
    max_epochs=30,
    accelerator='auto',
    gradient_clip_val=0.1,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=5),
        pl.callbacks.LearningRateMonitor(),
    ],
)
trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

== Interpreting TFT Outputs

The interpretability features are TFT's core differentiator. Always extract and examine them:

```python
best_model = TemporalFusionTransformer.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)
best_model.eval()

raw_predictions, x = best_model.predict(val_loader, mode='raw', return_x=True)
interpretation = best_model.interpret_output(raw_predictions, reduction='sum')

best_model.plot_interpretation(interpretation)
# encoder_variables: importance of past covariates in the encoder
# decoder_variables: importance of future covariates in the decoder
# static_variables:  importance of static features

best_model.plot_prediction(x, raw_predictions, idx=0, add_loss_to_title=True)
best_model.plot_partial_dependence('price')
```

*What to look for in attention plots:*
- Attention concentrated at the same time-of-day across past days → model is picking up daily seasonality
- Attention at lag-7 → weekly seasonality identified
- Diffuse attention → the model is uncertain or the pattern is irregular
- Attention concentrated near $t-1$ → mostly using recent history; long lookback may be unnecessary

*What to look for in variable importance:*
- If `time_idx` dominates over business features → model is fitting a calendar trend, not causal relationships
- If a feature you know to be important has near-zero weight → check encoding (continuous vs. categorical, normalization)
- High static variable importance → entity-level heterogeneity matters; the model is using store/product identity heavily

== TFT Failure Modes

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Failure*], [*Symptom*], [*Fix*]),
  [`hidden_size` too small],
  [Train and val loss both high],
  [Increase `hidden_size`, `attention_head_size`],
  [Overfitting],
  [Train loss $<<$ val loss],
  [Increase dropout; reduce `hidden_size`; add more data],
  [Poor calibration],
  [Coverage of 80% interval far from 80%],
  [Check `GroupNormalizer`; inspect residuals by quantile],
  [One feature dominates VSN],
  [All weight on `time_idx`],
  [Check if other features are properly encoded and scaled],
  [Static context unused],
  [Static variable importance near zero],
  [Ensure `static_categoricals` actually have variance across groups],
  [Slow convergence],
  [Loss decreasing but very slowly after epoch 10],
  [`ReduceLROnPlateau` too conservative; try cosine annealing],
  [OOM on long sequences],
  [CUDA out of memory],
  [Reduce `max_encoder_length` or `batch_size`; use gradient checkpointing],
)

// ─── SECTION 5 ───────────────────────────────────────────────────────────────

= Architecture Decision Guide

== Decision Tree

```
Do you have multiple series with shared patterns?
+-- NO  -> ARIMA or ETS (Phase 1); if DL needed, single-series LSTM
+-- YES ↓

Do you have known-future covariates (promotions, calendar, planned events)?
+-- NO  -> LSTM or TCN (simpler; less to configure)
+-- YES ↓

Is interpretability required (feature importance, attention)?
+-- NO  -> TCN (fast, parallelizable) or LSTM (if short, irregular patterns)
+-- YES -> TFT

Is the lookback window > 300 steps?
+-- YES -> TCN or TFT; avoid pure LSTM
+-- NO  -> LSTM viable

Is training speed or inference latency critical?
+-- YES -> TCN (fully parallelizable; no attention quadratic cost)
+-- NO  -> TFT (best accuracy on complex covariate problems)
```

== Comparison Table

#table(
  columns: (1fr, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Dimension*], [*LSTM*], [*TCN*], [*TFT*]),
  [Parallelizable], [✗ (sequential)], [✓], [Partially],
  [Long sequences ($>500$)], [✗], [✓], [✓],
  [Known-future covariates], [Manual], [Manual], [Native],
  [Static metadata], [Manual], [Manual], [Native],
  [Multi-horizon], [Direct head], [Direct head], [Native],
  [Probabilistic output], [Manual (NLL head)], [Manual], [Native (quantiles)],
  [Interpretability], [✗], [✗], [✓ (VSN, attention)],
  [Data requirement], [Low], [Low], [Medium–High],
  [Implementation complexity], [Low], [Medium], [High],
  [Typical parameters], [10K–1M], [50K–5M], [100K–10M],
)

== The Ensemble Perspective

In production, the strongest forecasting systems are almost always ensembles. The M4 and M5 competition winners all used ensembles. A principled ensemble:

```python
def ensemble_forecast(series, covariates, weights=[0.2, 0.3, 0.5]):
    arima_fc = fit_arima(series).forecast(H)
    lstm_fc  = train_lstm(series, covariates).predict(series, covariates)
    tft_fc   = train_tft(series, covariates).predict(series, covariates)

    # Weights optimized on validation set
    return weights[0]*arima_fc + weights[1]*lstm_fc + weights[2]*tft_fc
```

The classical model regularizes the DL ensemble: on short series or low-data entities, the ARIMA prediction keeps the ensemble grounded. On high-data entities with complex covariate structure, TFT dominates.

// ─── SECTION 6 ───────────────────────────────────────────────────────────────

= Weeks 2–3 Practice Exercises

*Dataset: Electricity (ECL)* — Standard benchmark: 370 clients, hourly electricity consumption, 2012–2014 (26,304 steps per client).

```python
from datasets import load_dataset
ds = load_dataset('monash_tsf', 'electricity_hourly')
```

== Exercise 1: LSTM Baseline (Days 1–2)

Build and train `LSTMForecaster` on a single client (`MT_001`), horizon $H=24$ (1 day ahead):

1. Build `TimeSeriesDataset` with `lookback=168` (1 week), `horizon=24`
2. Apply RevIN normalization
3. Train with `ReduceLROnPlateau`; use gradient clipping
4. Evaluate MAE and MASE on a held-out final month (walk-forward, 5 folds)
5. Plot predictions vs. actuals for 3 representative weeks

*Questions to answer:*
- What happens to MAE when you increase lookback from 48 → 168 → 336?
- Does the model capture the daily pattern? The weekly pattern?
- What is the MASE relative to seasonal naive (repeat last week)?

== Exercise 2: TCN from Scratch (Days 2–3)

Implement `TCN` using only `TemporalBlock` as defined above. Do NOT use any TCN library.

1. Calculate the receptive field you need: 168 steps (weekly pattern)
2. Design the level/channel configuration: how many levels? what kernel size?
3. Verify causality with the `verify_causality` function
4. Train on the same data as LSTM; compare training time per epoch
5. Ablate: vary dilation schedule (`[1,2,4,8,...]` vs `[1,3,9,27,...]`)

*Questions to answer:*
- How does training time compare to LSTM (should be substantially faster)?
- At what receptive field does adding more levels stop helping?
- Is the exponential dilation schedule better than arithmetic ($1, 2, 3, 4, dots$)?

== Exercise 3: TFT with Covariates (Days 4–5)

Extend to the full 370-client dataset with covariates:

1. Add features: hour of day, day of week, month, `is_weekend` (known future)
2. Configure `TimeSeriesDataSet` with `static_categoricals=['client_id']`
3. Train TFT; track convergence
4. Extract variable importance: does `hour_of_day` or `day_of_week` dominate?
5. Plot attention patterns for two clients with different consumption profiles

*Questions to answer:*
- What does the VSN attention distribution look like before vs. after 10 epochs?
- Do clients cluster in their static variable importance?
- Compare TFT's 80% interval coverage to LSTM's (LSTM needs a Gaussian head for this)

== Exercise 4: Horizon Stress Test (Day 5)

At what horizon does LSTM lose to TCN and TFT?

```python
horizons = [1, 6, 12, 24, 48, 96, 168]
results = {}

for H in horizons:
    lstm_mae = train_and_eval(LSTMForecaster(..., horizon=H), H)
    tcn_mae  = train_and_eval(TCN(..., horizon=H), H)
    results[H] = {'lstm': lstm_mae, 'tcn': tcn_mae}
```

Expected pattern: LSTM competitive at $H <= 24$; TCN pulls ahead at $H = 48$–$96$ on this dataset; TFT best at $H = 96+$ when covariates are included.

// ─── SECTION 7 ───────────────────────────────────────────────────────────────

= Interview Fluency

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Term*], [*Definition*]),
  [Vanishing gradient],
  [Gradient signal attenuated exponentially through BPTT; prevents learning from distant steps],
  [Forget gate],
  [$bold(f)_t = sigma(dots)$; multiplicative mask on previous cell state; values near 1 preserve memory, near 0 erase],
  [Causal convolution],
  [Convolution where output at $t$ depends only on inputs $<= t$; enforced by left-only padding],
  [Dilated convolution],
  [Convolution with step size $d$ between filter taps; receptive field grows without adding parameters],
  [Receptive field],
  [The number of input timesteps that can influence a single output position],
  [Teacher forcing],
  [Feeding ground truth $y_(t-1)$ as input during training of autoregressive models],
  [Exposure bias],
  [Train/inference mismatch from teacher forcing; model has never seen its own errors as inputs],
  [Pinball loss],
  [$cal(L)_tau (y, hat(y)) = tau (y - hat(y))_+ + (1-tau)(hat(y) - y)_+$; optimal solution is the $tau$-th quantile],
  [VSN],
  [Variable Selection Network; softmax-weighted GRN mixture of per-feature embeddings; provides feature importance],
  [GRN],
  [Gated Residual Network; ELU nonlinearity + GLU gate + LayerNorm skip; provides adaptive depth],
)

*"Why does LSTM handle long-range dependencies better than vanilla RNN?"*

#note[
  The cell state creates a gradient highway through time. The gradient of $bold(c)_t$ with respect to $bold(c)_(t-k)$ is approximately $product_i bold(f)_i$. When the forget gate learns to stay near 1, gradients flow backward through many timesteps without attenuation. In contrast, the vanilla RNN gradient involves products of the Jacobian of $tanh$, which saturates and produces magnitudes $< 1$, compounding to near-zero over long sequences.
]

*"How does TCN compare to LSTM for sequential modeling?"*

#note[
  TCN's key advantage is parallelism: all positions in the sequence are processed simultaneously, so training is much faster on GPUs. The receptive field grows exponentially with depth, so TCN can cover very long histories efficiently. The disadvantage is that the receptive field is fixed at architecture design time — you can't adapt to variable-length dependencies without re-architecture. LSTM's hidden state is more flexible for irregular, non-periodic temporal structure. In practice on standard benchmarks, TCN tends to match or exceed LSTM at longer horizons.
]

*"Why does TFT use shared V weights in its attention?"*

#note[
  Standard multi-head attention averages the attention weights after multiplying by different $bold(W)_V$ matrices per head. The averaged weights don't correspond to a single representation, so they lack interpretability. TFT shares $bold(W)_V$ across heads, which means averaging the attention weights *before* the V projection is equivalent. This gives a single, averaged attention pattern over a shared representation, which can be interpreted as the model's view of temporal importance.
]

*"What is the pinball loss and why is it correct for quantile regression?"*

#note[
  The pinball loss $cal(L)_tau (y, hat(y)) = tau dot max(y - hat(y), 0) + (1-tau) dot max(hat(y) - y, 0)$ is a proper scoring rule for quantiles — the unique loss function whose expected value is minimized by the true $tau$-th quantile of the conditional distribution. It is asymmetric: for $tau = 0.9$, over-forecasting is penalized with weight 0.1 and under-forecasting with weight 0.9, so the model learns to produce a value that is exceeded by the true observation only 10% of the time.
]

*"When would you NOT use TFT in production?"*

#note[
  When you have very few series per entity (TFT's shared parameters need cross-series data to generalize), when inference latency is critical (attention is quadratic in sequence length), when you have no covariate information (TFT's complexity isn't justified without rich features), or when the series are short and ETS already achieves MASE $< 1$. The full TFT training and tuning pipeline is also significant engineering overhead — for a quick-turnaround project, a well-tuned TCN often delivers 80% of the performance at 20% of the complexity.
]

// ─── SECTION 8 ───────────────────────────────────────────────────────────────

= Summary Mental Model

```
Lookback window y_{t-L:t} + covariates
         |
         v
[Normalization — RevIN per instance]
         |
         +---- LSTM ────────────────────────────────────────────────────────+
         |     Sequential; gated cell state; flexible memory                |
         |     Best: irregular patterns, moderate L (<300), fast prototyping |
         |                                                                   |
         +---- TCN ─────────────────────────────────────────────────────────+
         |     Parallel; dilated causal conv; fixed RF                       |
         |     Best: long L, periodic patterns, training speed               |
         |                                                                   |
         +---- TFT ─────────────────────────────────────────────────────────+
               VSN -> LSTM encoder/decoder -> shared-V attention -> quantiles |
               Best: mixed covariates, multi-horizon, interpretability        |
                                                                              v
                                       Direct head:  ŷ_{t+1:t+H} in R^H
                                       Quantile head: ŷ_τ for τ in {0.1, 0.5, 0.9}
                                                      |
                                       [Denormalize — reverse RevIN]
                                                      |
                                              Final forecast
```

#line(length: 100%)

_Next: Phase 3 — Foundation Models: TimesFM, Chronos, Moirai_

#set document(title: "Time-Series Forecasting — Phase 5: Hands-On Mini-Project", author: "")
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
  #text(size: 15pt)[Phase 5 — Hands-On Mini-Project (Weeks 5–6)]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[ETTh1 · Full Pipeline · ARIMA → LSTM → TCN → TFT → Chronos]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

#outline(indent: 1em, depth: 2)

#v(0.5em)
#line(length: 100%)

// ─── OVERVIEW ────────────────────────────────────────────────────────────────

= Overview

This phase integrates everything from Phases 1–4 into a single, reproducible project. The goal is not to build the most accurate forecaster on ETTh1 — the benchmark literature already has that. The goal is to build a project that demonstrates breadth (classical through foundation models), rigor (correct backtesting, no leakage, probabilistic evaluation), and judgment (interpreting results, knowing when each model wins and why).

*By the end of Week 6 you should have:*
- A clean, reproducible PyTorch codebase covering all five model families
- Walk-forward backtest results across four horizons with all key metrics
- A reliability diagram confirming (or refuting) TFT's calibration
- A horizon stress-test plot showing where each architecture breaks
- An analysis notebook with written interpretations of every result
- A set of prepared talking points for interview discussion

// ─── SECTION 1 ───────────────────────────────────────────────────────────────

= Dataset and Task Specification

== Dataset: ETTh1 (Electricity Transformer Temperature)

*Source:* Introduced by the Informer paper (Zhou et al., NeurIPS 2021). Now a standard benchmark in virtually every TS forecasting paper.

```
URL: https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
Frequency: Hourly
Duration: 2016-07-01 to 2018-06-26 (17,420 timesteps)
Variates: 7
  - HUFL: High Useful Load (transformer load)
  - HULL: High Useless Load
  - MUFL: Medium Useful Load
  - MULL: Medium Useless Load
  - LUFL: Low Useful Load
  - LULL: Low Useless Load
  - OT:   Oil Temperature (target)
```

*Why ETTh1:*
- Long enough (2 years, hourly) to show daily + weekly + seasonal patterns
- Short enough to train on a laptop CPU
- Used in the Informer, PatchTST, FEDformer, iTransformer, TimesFM, and Moirai papers — your numbers are directly comparable to published benchmarks
- Rich enough (7 covariates) to show TFT's covariate-handling advantage

== Primary Task

Predict `OT` (oil temperature) at four forecast horizons:

$ H in {24, 96, 336, 720} quad "(1 day, 4 days, 2 weeks, 1 month)" $

This range is deliberate. At $H=24$, ARIMA and LSTM are competitive. At $H=720$, only TCN, TFT, and foundation models remain viable. The horizon stress test is the central analytical result.

== Standard Data Splits

The Informer paper defined these splits; use them exactly for comparability with the literature:

```python
import pandas as pd
import numpy as np

df = pd.read_csv('ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()

n = len(df)   # 17,420

# Standard ETT splits
TRAIN_END = 12 * 30 * 24      # 8,640  — first 12 months
VAL_END   = 16 * 30 * 24      # 11,520 — next 4 months
TEST_END  = n                  # 17,420 — final ~5.5 months

train = df.iloc[:TRAIN_END]
val   = df.iloc[TRAIN_END:VAL_END]
test  = df.iloc[VAL_END:]

ot_train = train['OT'].values
ot_val   = val['OT'].values
ot_test  = test['OT'].values
```

#note[The *walk-forward backtest* in evaluation uses only the training split for rolling origins, consistent with the leakage rules from Phase 4. The fixed test split is touched once at the end for final reporting.]

// ─── SECTION 2 ───────────────────────────────────────────────────────────────

= Codebase Structure

```
ts_forecast/
├── data/
│   ├── download.py          # fetch and cache ETTh1
│   └── dataset.py           # TimeSeriesDataset, RevIN, covariate builders
├── models/
│   ├── lstm.py              # LSTMForecaster (direct multi-step)
│   ├── tcn.py               # TemporalBlock + TCN (from scratch)
│   └── tft_wrapper.py       # pytorch-forecasting TFT setup
├── baselines/
│   └── arima_baseline.py    # SARIMA per-fold, MASE-aware
├── evaluation/
│   ├── metrics.py           # MAE, MASE, CRPS, coverage, Winkler
│   └── backtest.py          # WalkForwardCV with gap enforcement
├── foundation/
│   └── chronos_eval.py      # Chronos zero-shot + calibration
├── train.py                 # unified training entry point
├── evaluate.py              # unified evaluation entry point
├── config.py                # all hyperparameters in one place
├── requirements.txt
└── notebooks/
    └── analysis.ipynb       # results, plots, interpretation
```

// ─── SECTION 3 ───────────────────────────────────────────────────────────────

= `config.py` — All Hyperparameters in One Place

```python
# config.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfig:
    csv_path: str = 'data/ETTh1.csv'
    target: str = 'OT'
    covariates: List[str] = field(default_factory=lambda: [
        'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'
    ])
    train_end: int = 8_640
    val_end: int   = 11_520

@dataclass
class BacktestConfig:
    n_splits: int = 5
    gap: int = 96        # always gap >= max horizon you're evaluating
    window: str = 'expanding'
    min_train: int = 2_000

@dataclass
class LSTMConfig:
    lookback: int = 336
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5

@dataclass
class TCNConfig:
    lookback: int = 512
    num_channels: List[int] = field(default_factory=lambda: [64, 64, 64, 64, 64, 64, 64, 64])
    kernel_size: int = 3
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5

@dataclass
class TFTConfig:
    max_encoder_length: int = 168   # 1 week lookback
    hidden_size: int = 32
    attention_head_size: int = 4
    dropout: float = 0.1
    lr: float = 3e-3
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])

@dataclass
class ChronosConfig:
    model_name: str = 'amazon/chronos-t5-small'
    num_samples: int = 100
    context_len: int = 512

HORIZONS = [24, 96, 336, 720]
```

// ─── SECTION 4 ───────────────────────────────────────────────────────────────

= `data/download.py`

```python
# data/download.py
import os
import urllib.request
import pandas as pd

ETT_URL = (
    'https://raw.githubusercontent.com/zhouhaoyi/ETDataset/'
    'main/ETT-small/ETTh1.csv'
)

def download_ett(dest: str = 'data/ETTh1.csv') -> str:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if not os.path.exists(dest):
        print(f'Downloading ETTh1 -> {dest}')
        urllib.request.urlretrieve(ETT_URL, dest)
    else:
        print(f'Found cached {dest}')
    return dest

def load_ett(path: str = 'data/ETTh1.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    return df

if __name__ == '__main__':
    path = download_ett()
    df = load_ett(path)
    print(df.shape)
    print(df.describe())
```

// ─── SECTION 5 ───────────────────────────────────────────────────────────────

= `data/dataset.py`

```python
# data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


class RevIN(torch.nn.Module):
    """Reversible instance normalization (Kim et al., 2022).
    Normalizes each lookback window independently; stores stats for denorm.
    """
    def __init__(self, num_features: int = 1, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine_weight = torch.nn.Parameter(torch.ones(num_features))
        self.affine_bias   = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x: (batch, seq_len, features)
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std  = x.std(dim=1, keepdim=True, unbiased=False).detach() + self.eps
            x = (x - self._mean) / self._std
            x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
        return x


class TimeSeriesDataset(Dataset):
    """Sliding window dataset for supervised forecasting.

    Args:
        data:     np.ndarray of shape (T,) or (T, F)
        lookback: context window length L
        horizon:  forecast window length H
        stride:   step between windows (default 1)
    """
    def __init__(self, data: np.ndarray, lookback: int, horizon: int,
                 stride: int = 1):
        if data.ndim == 1:
            data = data[:, None]   # (T, 1)
        self.data     = data
        self.lookback = lookback
        self.horizon  = horizon
        self.stride   = stride

        max_start = len(data) - lookback - horizon
        self.indices = list(range(0, max_start + 1, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        start = self.indices[idx]
        x = self.data[start            : start + self.lookback]
        y = self.data[start + self.lookback : start + self.lookback + self.horizon]
        return (
            torch.tensor(x, dtype=torch.float32),         # (lookback, F)
            torch.tensor(y[:, 0], dtype=torch.float32),   # (horizon,)
        )


def build_calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Known-future calendar covariates for TFT."""
    feats = np.stack([
        index.hour.values / 23.0,
        index.dayofweek.values / 6.0,
        index.day.values / 31.0,
        index.month.values / 12.0,
        (index.dayofweek >= 5).astype(float),   # is_weekend
    ], axis=1)
    return feats.astype(np.float32)   # (T, 5)
```

// ─── SECTION 6 ───────────────────────────────────────────────────────────────

= `baselines/arima_baseline.py`

```python
# baselines/arima_baseline.py
import numpy as np
import warnings
from pmdarima import auto_arima
from typing import Optional


class ARIMABaseline:
    """SARIMA fitted fresh per walk-forward fold."""

    def __init__(self, seasonal: bool = True, m: int = 24):
        self.seasonal = seasonal
        self.m = m
        self._model = None

    def fit(self, series: np.ndarray) -> 'ARIMABaseline':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._model = auto_arima(
                series,
                seasonal=self.seasonal,
                m=self.m,
                information_criterion='aic',
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=3, max_q=3,
                max_P=2, max_Q=2,
            )
        return self

    def predict(self, horizon: int) -> np.ndarray:
        assert self._model is not None, 'Call fit() first'
        fc, conf = self._model.predict(horizon, return_conf_int=True, alpha=0.2)
        return fc, conf   # (horizon,), (horizon, 2) for 80% interval

    def predict_point(self, horizon: int) -> np.ndarray:
        fc, _ = self.predict(horizon)
        return fc


def run_arima_backtest(series: np.ndarray, cv, m: int = 24) -> list:
    """Run ARIMA through a WalkForwardCV and collect results."""
    from evaluation.metrics import ForecastMetrics

    results = []
    metrics = ForecastMetrics()

    for fold, (train_idx, test_idx) in enumerate(cv.split(len(series))):
        train  = series[train_idx]
        actual = series[test_idx]
        h      = len(test_idx)

        model = ARIMABaseline(seasonal=True, m=m)
        model.fit(train)
        point_fc, conf_int = model.predict(h)

        # 80% interval -> approx sigma: 80% interval = +/- 1.28 sigma
        sigma = (conf_int[:, 1] - conf_int[:, 0]) / (2 * 1.28)
        samples = np.random.normal(
            point_fc[None, :], sigma[None, :], size=(200, h)
        )  # (200, h)

        result = {
            'fold':        fold,
            'train_end':   train_idx[-1],
            'MAE':         metrics.mae(actual, point_fc),
            'MASE':        metrics.mase(actual, point_fc, train, m=m),
            'CRPS':        metrics.crps_sample(samples.T, actual),
            'Coverage_80': metrics.coverage(samples.T, actual, alpha=0.2),
            'predictions': point_fc,
            'samples':     samples,
            'actuals':     actual,
        }
        results.append(result)
        print(f'  Fold {fold}: MAE={result["MAE"]:.4f}  MASE={result["MASE"]:.4f}')

    return results
```

// ─── SECTION 7 ───────────────────────────────────────────────────────────────

= `models/lstm.py`

```python
# models/lstm.py
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import TimeSeriesDataset, RevIN


class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, horizon: int, dropout: float = 0.1):
        super().__init__()
        self.revin   = RevIN(num_features=input_size)
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, features)
        x = self.revin(x, 'norm')
        _, (h_n, _) = self.encoder(x)
        h_last = h_n[-1]                   # (batch, hidden_size)
        out = self.head(h_last)            # (batch, horizon)
        out = out.unsqueeze(-1)            # (batch, horizon, 1)
        out = self.revin(out, 'denorm')
        return out.squeeze(-1)             # (batch, horizon)


def train_lstm(
    train_series: np.ndarray,
    val_series: np.ndarray,
    horizon: int,
    cfg,
    device: str = 'cpu',
) -> LSTMForecaster:
    train_ds = TimeSeriesDataset(train_series, cfg.lookback, horizon, stride=1)
    val_ds   = TimeSeriesDataset(
        np.concatenate([train_series[-cfg.lookback:], val_series]),
        cfg.lookback, horizon, stride=horizon
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                               num_workers=0)

    model = LSTMForecaster(
        input_size=1, hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers, horizon=horizon, dropout=cfg.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5, verbose=False
    )

    best_val, best_state, patience_counter = float('inf'), None, 0

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += nn.functional.mse_loss(pred, y).item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_lstm(model: LSTMForecaster, context: np.ndarray,
                 horizon: int, device: str = 'cpu',
                 lookback: int = 336) -> np.ndarray:
    model.eval()
    ctx = context[-lookback:]
    x   = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    return model(x).squeeze(0).cpu().numpy()   # (horizon,)
```

// ─── SECTION 8 ───────────────────────────────────────────────────────────────

= `models/tcn.py`

```python
# models/tcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
from torch.utils.data import DataLoader
from data.dataset import TimeSeriesDataset


class TemporalBlock(nn.Module):
    """Single TCN residual block: two dilated causal convolutions + residual."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation   # left-only causal pad

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        ))
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        ))
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        T = x.size(2)
        out = self.conv1(x)[:, :, :T]    # causal trim
        out = self.drop1(F.relu(out))
        out = self.conv2(out)[:, :, :T]  # causal trim
        out = self.drop2(F.relu(out))
        return F.relu(out + self.downsample(x))


class TCN(nn.Module):
    def __init__(self, input_size: int, num_channels: list,
                 kernel_size: int, horizon: int, dropout: float = 0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size,
                dilation=2**i, dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
        self.head    = nn.Linear(num_channels[-1], horizon)

        rf = 1 + 2 * (kernel_size - 1) * (2**len(num_channels) - 1)
        print(f'TCN receptive field: {rf} steps')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, 1)
        x = x.permute(0, 2, 1)           # -> (batch, 1, lookback)
        features = self.network(x)        # -> (batch, channels[-1], lookback)
        return self.head(features[:, :, -1])  # -> (batch, horizon)


def train_tcn(
    train_series: np.ndarray,
    val_series: np.ndarray,
    horizon: int,
    cfg,
    device: str = 'cpu',
) -> TCN:
    train_ds = TimeSeriesDataset(train_series, cfg.lookback, horizon)
    val_ds   = TimeSeriesDataset(
        np.concatenate([train_series[-cfg.lookback:], val_series]),
        cfg.lookback, horizon, stride=horizon
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    model = TCN(
        input_size=1, num_channels=cfg.num_channels,
        kernel_size=cfg.kernel_size, horizon=horizon, dropout=cfg.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs
    )

    best_val, best_state, patience_counter = float('inf'), None, 0

    for epoch in range(cfg.max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                val_loss += F.mse_loss(model(x.to(device)), y.to(device)).item()
        val_loss /= max(len(val_loader), 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model
```

// ─── SECTION 9 ───────────────────────────────────────────────────────────────

= `models/tft_wrapper.py`

```python
# models/tft_wrapper.py
import pandas as pd
import numpy as np
import torch
import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss


def build_tft_dataset(df_full: pd.DataFrame, cfg, horizon: int,
                       train_end: int, split: str = 'train'):
    """
    Build pytorch-forecasting TimeSeriesDataSet.
    df_full must have columns: date (DatetimeIndex), OT, HUFL, HULL, ...,
    plus integer time_idx and string entity_id.
    """
    df = df_full.copy()
    df['time_idx']  = np.arange(len(df))
    df['entity_id'] = 'ett'    # single entity for univariate task

    df['hour']       = df.index.hour
    df['dayofweek']  = df.index.dayofweek
    df['month']      = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    cutoff = train_end - horizon

    dataset = TimeSeriesDataSet(
        df[df['time_idx'] <= cutoff] if split == 'train' else df,
        time_idx='time_idx',
        target='OT',
        group_ids=['entity_id'],
        min_encoder_length=cfg.max_encoder_length // 2,
        max_encoder_length=cfg.max_encoder_length,
        min_prediction_length=horizon,
        max_prediction_length=horizon,
        static_categoricals=['entity_id'],
        time_varying_known_reals=['time_idx', 'hour', 'dayofweek', 'month', 'is_weekend'],
        time_varying_unknown_reals=['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
        target_normalizer=GroupNormalizer(groups=['entity_id']),
        add_relative_time_idx=True,
        add_target_scales=True,
    )
    return dataset


def train_tft(df_train: pd.DataFrame, df_val: pd.DataFrame,
              cfg, horizon: int) -> TemporalFusionTransformer:
    train_ds = build_tft_dataset(
        pd.concat([df_train, df_val]), cfg, horizon, len(df_train), split='train'
    )
    val_ds = TimeSeriesDataSet.from_dataset(
        train_ds, pd.concat([df_train, df_val]), predict=True, stop_randomization=True
    )

    train_loader = train_ds.to_dataloader(
        train=True, batch_size=cfg.batch_size, num_workers=0
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=cfg.batch_size, num_workers=0
    )

    tft = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=cfg.lr,
        hidden_size=cfg.hidden_size,
        attention_head_size=cfg.attention_head_size,
        dropout=cfg.dropout,
        hidden_continuous_size=16,
        output_size=len(cfg.quantiles),
        loss=QuantileLoss(quantiles=cfg.quantiles),
        log_interval=20,
        reduce_on_plateau_patience=3,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator='cpu',
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.patience,
                                        mode='min'),
        ],
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_path = trainer.checkpoint_callback.best_model_path
    tft = TemporalFusionTransformer.load_from_checkpoint(best_path)
    return tft, trainer


def interpret_tft(tft: TemporalFusionTransformer, val_loader) -> dict:
    """Extract attention weights and variable importance."""
    raw_preds, x = tft.predict(val_loader, mode='raw', return_x=True)
    interpretation = tft.interpret_output(raw_preds, reduction='sum')
    return {
        'interpretation': interpretation,
        'raw_predictions': raw_preds,
        'x': x,
    }
```

// ─── SECTION 10 ──────────────────────────────────────────────────────────────

= `foundation/chronos_eval.py`

```python
# foundation/chronos_eval.py
import numpy as np
import torch
from chronos import ChronosPipeline


def load_chronos(model_name: str = 'amazon/chronos-t5-small') -> ChronosPipeline:
    return ChronosPipeline.from_pretrained(
        model_name,
        device_map='cpu',
        torch_dtype=torch.float32,
    )


def chronos_forecast(
    pipeline: ChronosPipeline,
    context: np.ndarray,
    horizon: int,
    num_samples: int = 100,
    context_len: int = 512,
) -> np.ndarray:
    """
    Zero-shot forecast with Chronos.
    Returns samples: (num_samples, horizon)
    """
    ctx = context[-context_len:]
    ctx_tensor = torch.tensor(ctx, dtype=torch.float32).unsqueeze(0)

    forecast = pipeline.predict(
        context=ctx_tensor,
        prediction_length=horizon,
        num_samples=num_samples,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    )
    return forecast[0].numpy()   # (num_samples, horizon)


def run_chronos_backtest(
    series: np.ndarray,
    cv,
    pipeline: ChronosPipeline,
    horizon: int,
    num_samples: int = 100,
) -> list:
    from evaluation.metrics import ForecastMetrics
    metrics = ForecastMetrics()
    results = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(len(series))):
        context = series[train_idx]
        actual  = series[test_idx]

        samples   = chronos_forecast(pipeline, context, horizon, num_samples)
        samples_T = samples.T   # (horizon, num_samples)

        point_fc = np.median(samples, axis=0)
        result = {
            'fold':        fold,
            'MAE':         metrics.mae(actual, point_fc),
            'MASE':        metrics.mase(actual, point_fc, series[train_idx], m=24),
            'CRPS':        metrics.crps_sample(samples_T, actual),
            'Coverage_80': metrics.coverage(samples_T, actual, alpha=0.2),
            'predictions': point_fc,
            'samples':     samples,
            'actuals':     actual,
        }
        results.append(result)
        print(f'  Fold {fold}: MAE={result["MAE"]:.4f}  MASE={result["MASE"]:.4f}  '
              f'Cov80={result["Coverage_80"]:.2f}')

    return results
```

// ─── SECTION 11 ──────────────────────────────────────────────────────────────

= `evaluation/metrics.py`

This is the complete `ForecastMetrics` class from Phase 4 — copy it verbatim. Key methods needed for the project:

```python
# evaluation/metrics.py
import numpy as np
from scipy import stats

class ForecastMetrics:
    @staticmethod
    def mae(y_true, y_pred): ...
    @staticmethod
    def rmse(y_true, y_pred): ...
    @staticmethod
    def mase(y_true, y_pred, y_train, m=24): ...
    @staticmethod
    def crps_sample(samples, y_true): ...   # samples: (n_obs, n_samples)
    @staticmethod
    def coverage(samples, y_true, alpha=0.2): ...
    @staticmethod
    def winkler(samples, y_true, alpha=0.2): ...
    @classmethod
    def report(cls, y_true, y_pred_point, y_pred_samples=None, y_train=None, m=24): ...
```

See Phase 4 for the full implementations.

// ─── SECTION 12 ──────────────────────────────────────────────────────────────

= `train.py` — Unified Training Entry Point

```python
# train.py
"""
Train all models for a given horizon and save results.

Usage:
  python train.py --horizon 96
  python train.py --horizon 96 --models lstm tcn tft
"""
import argparse
import pickle
import os
import numpy as np
import pandas as pd

from config import DataConfig, LSTMConfig, TCNConfig, TFTConfig, BacktestConfig
from data.download import download_ett, load_ett
from data.dataset import build_calendar_features
from evaluation.backtest import WalkForwardCV
from evaluation.metrics import ForecastMetrics
from baselines.arima_baseline import run_arima_backtest
from models.lstm import train_lstm, predict_lstm
from models.tcn import train_tcn
from foundation.chronos_eval import load_chronos, run_chronos_backtest


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--horizon', type=int, default=96)
    p.add_argument('--models', nargs='+',
                   default=['arima', 'lstm', 'tcn', 'chronos'],
                   choices=['arima', 'lstm', 'tcn', 'tft', 'chronos'])
    p.add_argument('--device', default='cpu')
    p.add_argument('--output_dir', default='results')
    return p.parse_args()


def main():
    args = parse_args()
    H    = args.horizon
    os.makedirs(args.output_dir, exist_ok=True)

    path = download_ett()
    df   = load_ett(path)
    data_cfg = DataConfig()
    ot   = df['OT'].values

    cv = WalkForwardCV(
        n_splits=BacktestConfig.n_splits,
        horizon=H,
        gap=H,
        min_train_size=BacktestConfig.min_train,
        window='expanding',
    )

    all_results = {}

    if 'arima' in args.models:
        print(f'\n=== ARIMA  H={H} ===')
        results = run_arima_backtest(ot[:data_cfg.train_end], cv, m=24)
        all_results['arima'] = results

    if 'lstm' in args.models:
        print(f'\n=== LSTM  H={H} ===')
        lstm_cfg = LSTMConfig()
        lstm_results = []
        metrics = ForecastMetrics()

        for fold, (train_idx, test_idx) in enumerate(cv.split(data_cfg.train_end)):
            train_s = ot[train_idx]
            val_s   = ot[data_cfg.train_end:data_cfg.val_end]
            actual  = ot[test_idx]

            model = train_lstm(train_s, val_s[:H*2], H, lstm_cfg, args.device)
            preds = predict_lstm(model, train_s, H, args.device)

            r = {
                'fold':        fold,
                'MAE':         metrics.mae(actual, preds),
                'MASE':        metrics.mase(actual, preds, train_s, m=24),
                'predictions': preds,
                'actuals':     actual,
            }
            lstm_results.append(r)
            print(f'  Fold {fold}: MAE={r["MAE"]:.4f}  MASE={r["MASE"]:.4f}')

        all_results['lstm'] = lstm_results

    if 'tcn' in args.models:
        print(f'\n=== TCN  H={H} ===')
        tcn_cfg = TCNConfig()
        tcn_results = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(data_cfg.train_end)):
            train_s = ot[train_idx]
            val_s   = ot[data_cfg.train_end:data_cfg.val_end]
            actual  = ot[test_idx]

            model = train_tcn(train_s, val_s[:H*2], H, tcn_cfg, args.device)
            import torch
            ctx   = torch.tensor(train_s[-tcn_cfg.lookback:],
                                  dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                preds = model(ctx).squeeze(0).numpy()

            r = {
                'fold':        fold,
                'MAE':         ForecastMetrics.mae(actual, preds),
                'MASE':        ForecastMetrics.mase(actual, preds, train_s, m=24),
                'predictions': preds,
                'actuals':     actual,
            }
            tcn_results.append(r)
            print(f'  Fold {fold}: MAE={r["MAE"]:.4f}  MASE={r["MASE"]:.4f}')

        all_results['tcn'] = tcn_results

    if 'chronos' in args.models:
        print(f'\n=== Chronos  H={H} ===')
        pipeline = load_chronos()
        results  = run_chronos_backtest(
            ot[:data_cfg.train_end], cv, pipeline, H, num_samples=100
        )
        all_results['chronos'] = results

    out_path = os.path.join(args.output_dir, f'results_H{H}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved -> {out_path}')


if __name__ == '__main__':
    main()
```

// ─── SECTION 13 ──────────────────────────────────────────────────────────────

= `evaluate.py` — Results Aggregation and Summary

```python
# evaluate.py
"""
Load all saved results and print a formatted summary table.

Usage:
  python evaluate.py
  python evaluate.py --horizons 24 96 336 720
"""
import argparse
import pickle
import os
import numpy as np
import pandas as pd


def load_results(output_dir: str, horizon: int) -> dict:
    path = os.path.join(output_dir, f'results_H{horizon}.pkl')
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def summarize(results: dict,
              metrics: list = ('MAE', 'MASE', 'CRPS', 'Coverage_80')) -> pd.DataFrame:
    rows = []
    for model_name, fold_results in results.items():
        for metric in metrics:
            vals = [r[metric] for r in fold_results if metric in r]
            if vals:
                rows.append({
                    'Model':  model_name.upper(),
                    'Metric': metric,
                    'Mean':   np.mean(vals),
                    'Std':    np.std(vals),
                    'Min':    np.min(vals),
                    'Max':    np.max(vals),
                })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--horizons', nargs='+', type=int, default=[24, 96, 336, 720])
    p.add_argument('--output_dir', default='results')
    args = p.parse_args()

    print('=' * 70)
    for H in args.horizons:
        results = load_results(args.output_dir, H)
        if not results:
            print(f'No results found for H={H}')
            continue

        df = summarize(results)
        print(f'\n--- Horizon H={H} ---')

        pivot_rows = []
        for model in df['Model'].unique():
            row = {'Model': model}
            for metric in df['Metric'].unique():
                sub = df[(df['Model'] == model) & (df['Metric'] == metric)]
                if not sub.empty:
                    m, s = sub.iloc[0]['Mean'], sub.iloc[0]['Std']
                    row[metric] = f'{m:.4f} +/- {s:.4f}'
            pivot_rows.append(row)

        pivot = pd.DataFrame(pivot_rows).set_index('Model')
        print(pivot.to_string())

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
```

// ─── SECTION 14 ──────────────────────────────────────────────────────────────

= Analysis Notebook Guide

The notebook is where the project lives as a readable story. Structure it as follows.

== Section 1: Data Exploration

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from data.download import download_ett, load_ett

df = load_ett(download_ett())
ot = df['OT']

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(ot, lw=0.5)
axes[0].set_title('OT: Full series (2 years)')

week = ot.iloc[1000:1168]
axes[1].plot(week)
axes[1].set_title('OT: One week (hourly pattern visible)')

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(ot.values, lags=168, ax=axes[2])
axes[2].set_title('ACF: Spikes at 24h and 168h confirm daily + weekly seasonality')
```

*What to write:* describe what you see. The daily cycle is obvious in the zoom. The ACF spikes at 24 and 168 tell you the minimum lookback required to capture both seasonalities — this motivates the `lookback=168` choice for LSTM and the TCN receptive field design.

== Section 2: Horizon Stress Test — The Central Plot

This is the most important result. After running `train.py` for all four horizons:

```python
import pickle, numpy as np, matplotlib.pyplot as plt

horizons = [24, 96, 336, 720]
models   = ['ARIMA', 'LSTM', 'TCN', 'CHRONOS']
colors   = {'ARIMA': '#666', 'LSTM': '#2196F3', 'TCN': '#4CAF50', 'CHRONOS': '#FF9800'}

mae_table = {m: [] for m in models}

for H in horizons:
    with open(f'results/results_H{H}.pkl', 'rb') as f:
        results = pickle.load(f)
    for m in models:
        key = m.lower()
        if key in results:
            vals = [r['MAE'] for r in results[key]]
            mae_table[m].append(np.mean(vals))
        else:
            mae_table[m].append(np.nan)

fig, ax = plt.subplots(figsize=(9, 5))
for model, maes in mae_table.items():
    ax.plot(horizons, maes, 'o-', label=model, color=colors[model], linewidth=2)

ax.set_xlabel('Forecast horizon H (hours)')
ax.set_ylabel('Mean MAE (5-fold walk-forward)')
ax.set_title('ETTh1 OT: MAE vs. Horizon - All Models')
ax.legend()
ax.set_xticks(horizons)
ax.grid(True, alpha=0.3)
```

*What to write:*
- At $H=24$: ARIMA within ~10% of LSTM/TCN — classical methods remain viable at short horizons
- At $H=96$: TCN pulls ahead of LSTM; the sequential bottleneck starts to hurt
- At $H=336$: ARIMA degrades sharply (uncertainty compounds over 2 weeks); TCN and Chronos hold up
- At $H=720$: ARIMA unusable; only TCN and Chronos remain competitive; 1-month-ahead forecasting is fundamentally hard

This is the core interview talking point. You've empirically demonstrated why TCN's receptive field design matters and at what horizon each model breaks.

== Section 3: TFT Variable Importance and Attention

```python
from models.tft_wrapper import train_tft, interpret_tft, build_tft_dataset

tft_results = interpret_tft(tft, val_loader)
tft.plot_interpretation(tft_results['interpretation'])
```

*What to write:*
- Which covariates dominate the VSN weights? `hour` and `dayofweek` should be near the top — the model is learning to use time-of-day and day-of-week structure
- Does `time_idx` dominate? If so, the model is fitting a linear trend — check if removing it changes performance
- Plot attention patterns for two specific weeks: one regular, one anomalous. Does the attention pattern change?

== Section 4: Chronos Calibration

```python
all_samples = []   # (total_obs, num_samples)
all_actuals = []   # (total_obs,)

with open('results/results_H96.pkl', 'rb') as f:
    results = pickle.load(f)

for fold_r in results['chronos']:
    s = fold_r['samples'].T     # (horizon, num_samples)
    a = fold_r['actuals']       # (horizon,)
    all_samples.append(s)
    all_actuals.append(a)

all_samples = np.vstack(all_samples)   # (5*96, 100)
all_actuals = np.concatenate(all_actuals)

from evaluation.metrics import reliability_diagram
fig, ax = plt.subplots(figsize=(6, 6))
reliability_diagram(all_samples, all_actuals,
                    title='Chronos-small: ETTh1 OT, H=96', ax=ax)
```

*What to write:*
- Is Chronos calibrated on this dataset?
- Report the 80% coverage: if it's between 75–85%, calibration is acceptable
- Compare the Winkler score vs. TFT: which gives sharper intervals at the same coverage?

== Section 5: Per-Horizon Summary Table

```bash
python evaluate.py --horizons 24 96 336 720
```

Format this as a clean table in the notebook. The expected shape of results:

#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Model*], [*H=24 MAE*], [*H=96 MAE*], [*H=336 MAE*], [*H=720 MAE*]),
  [ARIMA],   [~0.35], [~0.45], [~0.65], [~0.90],
  [LSTM],    [~0.33], [~0.42], [~0.58], [~0.82],
  [TCN],     [~0.32], [~0.40], [~0.52], [~0.74],
  [Chronos], [~0.36], [~0.41], [~0.55], [~0.76],
)

#note[These are illustrative. Your actual numbers depend on hyperparameters, random seeds, and preprocessing. The pattern (TCN/Chronos win at long horizons; all converge at short horizons) should be robust.]

// ─── SECTION 15 ──────────────────────────────────────────────────────────────

= Week-by-Week Schedule

== Week 5

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Day*], [*Task*], [*Time*]),
  [Mon], [Environment setup, data download, EDA notebook], [2h],
  [Mon], [Run ARIMA backtest for all 4 horizons], [1h],
  [Tue], [Implement and train LSTM for H=24 and H=96], [3h],
  [Wed], [Implement TCN from scratch; verify causality; train H=24, H=96], [3h],
  [Thu], [Train LSTM and TCN for H=336 and H=720], [2h],
  [Thu], [Run Chronos zero-shot for all 4 horizons], [1h],
  [Fri], [Horizon stress test plot + written analysis], [2h],
)

== Week 6

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Day*], [*Task*], [*Time*]),
  [Mon], [Set up and train TFT for H=96], [3h],
  [Tue], [Extract TFT variable importance + attention plots; write interpretation], [2h],
  [Wed], [Chronos calibration analysis + reliability diagram], [1h],
  [Wed], [Full summary table across all horizons and models], [1h],
  [Thu], [Residual analysis on ARIMA and LSTM (Phase 4 protocol)], [1h],
  [Thu], [Polish notebook: prose, captions, clean figures], [2h],
  [Fri], [Interview prep: practice explaining each result out loud], [2h],
)

// ─── SECTION 16 ──────────────────────────────────────────────────────────────

= Requirements

```
# requirements.txt
torch>=2.0.0
lightning>=2.0.0
pytorch-forecasting>=1.0.0
statsmodels>=0.14.0
pmdarima>=2.0.4
chronos-forecasting>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

```bash
pip install -r requirements.txt
```

// ─── SECTION 17 ──────────────────────────────────────────────────────────────

= Interview Talking Points

The project is designed to generate five specific talking points. Practice each one as a 2–3 minute answer.

== The Horizon Stress Test

#insight[At $H=24$, ARIMA is within 10% of the best DL model — the problem is simple enough at short horizons that classical inductive biases suffice. At $H=96$, the gap opens: TCN outperforms LSTM because TCN's receptive field covers the full weekly pattern without the sequential bottleneck. At $H=336$, ARIMA degrades sharply; TCN and Chronos hold up. At $H=720$, Chronos zero-shot is competitive with TCN despite seeing no training data — for month-ahead forecasting, patterns are general enough that pretraining from other domains transfers. Model selection should be driven by the deployment horizon, not a single benchmark number.]

== Why TCN Outperforms LSTM at Long Horizons

#insight[With 8 levels of exponential dilation ($d = 1, 2, 4, dots, 128$) and kernel size 3, the TCN's receptive field exceeds 1000 steps. On ETTh1 hourly data, the weekly pattern is at lag 168 — well within the receptive field. The LSTM processes this sequence recursively; signal from 168 steps ago is weaker than from 1 step ago even with forget gates. The TCN processes the full lookback in parallel with direct connections to all lags in its receptive field. At $H=336$ and $H=720$, TCN MAE is roughly 10–15% lower than LSTM. Causality can be verified explicitly: perturb the last timestep and confirm earlier outputs are unchanged — a correctness check for any production setting.]

== TFT Variable Importance Interpretation

#insight[TFT's Variable Selection Network gave feature importance weights after training. The dominant encoder features were `OT` (lagged target) and `hour` — oil temperature has a strong daily cycle driven by load patterns. `HUFL` (high useful load) had moderate importance, confirming cross-variate correlation. In the decoder, `hour` and `dayofweek` dominated — the model uses time-of-day and day-of-week structure to anchor forecasts. Attention patterns concentrated at lags that are multiples of 24 — the model learned daily periodicity from data without being told about it. When `time_idx` was removed from known-future features, performance barely changed, confirming it was used as a trend proxy rather than for causal inference.]

== Chronos Zero-Shot vs. Trained Models

#insight[Chronos zero-shot — seeing no ETTh1 data during inference — was competitive with my trained LSTM at all horizons and with TCN at $H=720$. The calibration analysis was the interesting part: the reliability diagram showed 79% empirical coverage at the nominal 80% interval — well calibrated. However, the Winkler score was worse than TFT's — Chronos achieves similar coverage with wider intervals. TFT's intervals were sharper because it trained specifically on this domain. Practical implication: in a new domain with no training data, Chronos gives a usable probabilistic forecast immediately. Once you have training data, a properly-trained TFT gives sharper, domain-adapted intervals.]

== Backtesting Rigor

#insight[Walk-forward with $K=5$ folds, expanding window, and a gap equal to the forecast horizon. Every scaler fitted on the training window only. The ARIMA is re-fitted fresh per fold, not cached. For TFT I used the standard ETT train/val/test splits and reported on the test set once. Residual analysis: the LSTM residuals show significant ACF at lag 24 at $H=24$ — the model is leaving daily structure on the table; increasing the lookback or adding explicit calendar features would help. The TCN residuals are flatter, consistent with its broader receptive field.]

// ─── SECTION 18 ──────────────────────────────────────────────────────────────

= Common Mistakes and How to Avoid Them

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Mistake*], [*How it manifests*], [*Prevention*]),
  [Fitting RevIN on full series],
  [Test performance looks too good; degrades on deployment],
  [RevIN is applied per-sample inside the model — it is not a global transform],
  [Using `center=True` in rolling features],
  [Leakage — future values enter the feature],
  [Always use `.shift(1).rolling(w).mean()`],
  [Re-using best fold's model for final report],
  [Optimistic bias — you selected the lucky fold],
  [Average across all folds; report mean ± std],
  [LSTM lookback shorter than seasonal period],
  [Model can't see the weekly pattern it needs],
  [Lookback ≥ 168 for hourly data with weekly seasonality],
  [Chronos context > 512],
  [Silently truncated; long-range patterns lost],
  [Always check `len(context[-context_len:])`],
  [TFT `time_idx` as dominant feature],
  [Model fitting a time-trend, not causal structure],
  [Check VSN weights; ablate by removing `time_idx`],
  [Not checking TCN causality],
  [Silent look-ahead bug; inflated results],
  [Run `verify_causality()` before any training],
  [Reporting MAPE on OT (near zero)],
  [Misleading metric or undefined values],
  [OT has near-zero values; always use MASE],
)

// ─── SECTION 19 ──────────────────────────────────────────────────────────────

= Suggested Extensions (If Time Permits)

After completing the core project, these extensions add significant depth.

*Extension 1: NLL head for LSTM.* Replace the MSE head with a Gaussian NLL head (predict $mu$ and $log sigma$). This makes LSTM probabilistic, enabling fair CRPS comparison with TFT and Chronos. Expected: CRPS competitive but slightly worse than TFT; intervals less calibrated than Chronos.

*Extension 2: STL-LSTM (decomposition + residual modeling).* Decompose OT with STL, model only the residual with LSTM, and add back the seasonal and trend components. This often substantially improves long-horizon performance by separating the "easy" structured signal from the "hard" residual.

*Extension 3: Horizon-specific model selection.* Build a meta-model: train a classifier on features of the series (autocorrelation structure, stationarity statistics, variance) that predicts which model will be best at each horizon. At test time, the meta-model selects the forecaster.

*Extension 4: Fine-tune Chronos on ETTh1.* Use the fine-tuning code from Phase 3. Fine-tune `chronos-t5-small` on the ETTh1 train split for 5 epochs. Compare against zero-shot Chronos and trained TCN. Expected: 5–15% MAE improvement; better calibration on domain-specific patterns.

#line(length: 100%)

_This completes the curriculum. Phases 1–5 cover classical baselines through foundation models with full implementation, evaluation, and interview preparation._

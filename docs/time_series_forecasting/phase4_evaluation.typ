#set document(title: "Time-Series Forecasting — Phase 4: Evaluation, Backtesting, and Failure Modes", author: "")
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
  #text(size: 15pt)[Phase 4 — Evaluation, Backtesting, and Failure Modes (Week 4, Part 2)]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[Metrics · Walk-Forward CV · Leakage · Distribution Shift · Production Readiness]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

#outline(indent: 1em, depth: 2)

#v(0.5em)
#line(length: 100%)

// ─── OVERVIEW ────────────────────────────────────────────────────────────────

= Overview

Evaluation is where forecasting projects most commonly fail silently. A model that looks excellent on a poorly designed backtest can be worthless — or actively harmful — in production. Phase 4 is about developing the rigorous, skeptical mindset that separates an engineer who can build a model from one who can trust a model.

This phase is shorter than Phases 2 and 3 but denser in judgment. The concepts are not technically difficult; the difficulty is in applying them consistently under deadline pressure, when the temptation is to declare victory and move on.

*By the end of Week 4, Part 2 you should be able to:*

- Choose the right metric for a given forecasting context and explain why
- Derive the CRPS for a Gaussian predictive distribution from first principles
- Design a walk-forward backtest that is free of leakage, appropriate for the horizon, and statistically meaningful
- Identify all five major forms of leakage in a given pipeline
- Detect and respond to distribution shift, structural breaks, and concept drift
- Know what "good enough" looks like for a production system and when to stop tuning

// ─── SECTION 1 ───────────────────────────────────────────────────────────────

= Metrics

Metrics encode assumptions about what kinds of errors matter. Choosing the wrong metric produces a model optimized for the wrong thing. Know the math, know the assumptions, know when each one lies to you.

== Point Forecast Metrics

=== MAE — Mean Absolute Error

$ "MAE" = frac(1, n) sum_t abs(y_t - hat(y)_t) $

*Properties:*
- Robust to outliers (linear in error, not quadratic)
- Interpretable in original units (same scale as the target)
- Optimal predictor: the *conditional median* of $p(y_t | "past")$ — not the mean
- Symmetric: over-forecasting penalized equally to under-forecasting

*When MAE is the right metric:* when errors are roughly symmetric in cost and you don't have strong reason to penalize large errors more than small ones. Retail demand, energy consumption.

*When MAE misleads:* when you are optimizing with MSE loss (which targets the conditional mean) but evaluating with MAE (which targets the conditional median). For asymmetric distributions, these diverge — a model trained with MSE may have worse MAE than one trained with MAE loss.

=== RMSE — Root Mean Squared Error

$ "RMSE" = sqrt(frac(1, n) sum_t (y_t - hat(y)_t)^2) $

*Properties:*
- Penalizes large errors quadratically — a single error of 10 contributes as much as 100 errors of 1
- Optimal predictor: the *conditional mean* of $p(y_t | "past")$
- Not robust to outliers; units are the same as the target

*When RMSE is the right metric:* when large errors are disproportionately costly. Grid management (one large spike is a crisis), surgical supply chains.

*RMSE vs MAE ratio:* for a Gaussian error distribution, $"RMSE"\/"MAE" = sqrt(pi\/2) approx 1.25$. If your empirical ratio $>> 1.25$, your errors have heavy tails — a sign the model is systematically wrong on a small number of hard cases.

=== MASE — Mean Absolute Scaled Error

$ "MASE" = frac("MAE"_"model", "MAE"_"naive"), quad "MAE"_"naive" = frac(1, T-m) sum_(t=m+1)^T |y_t - y_(t-m)| $

$m$ is the seasonal period ($m=1$ for non-seasonal naive). The denominator is the in-sample one-step MAE of the seasonal naive forecast.

*Properties:*
- Scale-free: can compare performance across series with different units and magnitudes
- MASE $= 1.0$ means the model equals the naive seasonal baseline
- MASE $< 1.0$ means the model beats naive; MASE $> 1.0$ means it's worse
- Robust to zero values (denominator uses differences, not ratios)
- Defined even for series with zeros (unlike MAPE)

*Why MASE is the standard for M-competition evaluation:* it makes the question "is this model useful?" concrete and comparable across thousands of heterogeneous series.

```python
def mase(y_true: np.ndarray, y_pred: np.ndarray,
         y_train: np.ndarray, m: int = 1) -> float:
    """
    y_true:  actual future values, shape (horizon,)
    y_pred:  forecast values, shape (horizon,)
    y_train: in-sample training series, shape (T,)
    m:       seasonal period (1 for non-seasonal)
    """
    mae_model   = np.mean(np.abs(y_true - y_pred))
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    mae_naive   = np.mean(naive_errors)
    if mae_naive == 0:
        return np.nan   # constant series — metric undefined
    return mae_model / mae_naive
```

=== MAPE and sMAPE — Why They're Problematic

$ "MAPE" = frac(100, n) sum_t frac(abs(y_t - hat(y)_t), abs(y_t)) $

Problems: (1) undefined at zero; (2) asymmetric bias — overforecasting ($hat(y) > y$) is bounded at 100% but underforecasting is unbounded, so the optimal predictor under MAPE systematically incentivizes under-forecasting; (3) scale-dependent in a subtle way.

$ "sMAPE" = frac(200, n) sum_t frac(abs(y_t - hat(y)_t), abs(y_t) + abs(hat(y)_t)) $

Bounded in $[0, 200%]$ and more symmetric than MAPE, but still problematic near zero. Used in M4 competition. *Prefer MASE in any new work* — sMAPE is a historical artifact.

=== Choosing Between MAE and MASE

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Situation*], [*Use*]),
  [Single series, single scale], [MAE (interpretable in original units)],
  [Multiple series, different scales/units], [MASE (scale-free comparison)],
  [Reporting to stakeholders], [MAE (units are meaningful to them)],
  [Academic benchmark comparison], [MASE (standard in M-competition literature)],
  [Series with zeros], [MASE (MAPE undefined; MAE still valid but not comparable)],
)

== Probabilistic Forecast Metrics

Point metrics evaluate a single predicted value. Probabilistic metrics evaluate the *entire predictive distribution* — rewarding both accuracy (is the distribution centered near the true value?) and calibration (does the stated 80% interval actually contain the true value 80% of the time?).

=== CRPS — Continuous Ranked Probability Score

CRPS is the canonical proper scoring rule for predictive distributions. It generalizes MAE to distributions.

*Definition:*

$ "CRPS"(F, y) = integral_(-oo)^(oo) [F(z) - bb(1)(z >= y)]^2 d z $

where $F$ is the CDF of the predictive distribution and $y$ is the true observation. The term $bb(1)(z >= y)$ is the CDF of a point mass at $y$. CRPS measures the integrated squared difference between the predictive CDF and the "oracle" CDF (a step function at the truth).

*Why it's a proper scoring rule:* the expected CRPS $EE_y ["CRPS"(F, y)]$ is minimized when $F$ equals the true data-generating distribution. A model cannot improve its expected CRPS by misreporting its uncertainty — honesty is optimal.

*Closed form for Gaussian predictive distribution:*

$ "CRPS"(cal(N)(mu, sigma), y) = sigma [z dot Phi(z) + phi.alt(z) - 1\/sqrt(pi)], quad z = (y - mu)\/sigma $

where $Phi$ is the standard normal CDF and $phi.alt$ is the standard normal PDF. Derivation: substitute $F(z) = Phi((z-mu)\/sigma)$ into the integral definition, change variables to the standardized residual, and integrate term by term. The closed form is useful for models with Gaussian output heads (e.g., LSTM + NLL loss).

*Energy score form (for sample-based forecasts like Chronos):*

$ "CRPS"(F, y) = EE_F [|X - y|] - frac(1, 2) EE_F [|X - X'|], quad X, X' tilde F "independently" $

```python
def crps_sample(samples: np.ndarray, y: float) -> float:
    """
    samples: (n_samples,) — draws from the predictive distribution
    y: scalar true observation
    """
    n = len(samples)
    term1 = np.mean(np.abs(samples - y))
    idx1 = np.random.choice(n, min(200, n), replace=True)
    idx2 = np.random.choice(n, min(200, n), replace=True)
    term2 = 0.5 * np.mean(np.abs(samples[idx1] - samples[idx2]))
    return term1 - term2

def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    """Closed-form CRPS for Gaussian predictive distribution."""
    from scipy.special import ndtr
    from scipy.stats import norm

    z = (y - mu) / sigma
    crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * norm.pdf(z) - 1/np.sqrt(np.pi))
    return np.mean(crps)
```

#note[*CRPS vs. log-likelihood:* both are proper scoring rules. CRPS is in the original units (like MAE); log-likelihood is in nats/bits. CRPS is more robust to misspecification of distribution tails. Log-likelihood is dominated by extreme events. CRPS is almost always preferred in practice for reporting.]

=== Quantile (Pinball) Loss

$ cal(L)_tau (y, hat(q)_tau) = tau dot max(y - hat(q)_tau, 0) + (1-tau) dot max(hat(q)_tau - y, 0) = cases(tau (y - hat(q)_tau) & "if" y >= hat(q)_tau, (1-tau)(hat(q)_tau - y) & "if" y < hat(q)_tau) $

The optimal predictor under $cal(L)_tau$ is the $tau$-th quantile of the true conditional distribution. This is a proper scoring rule for quantiles.

*Winkler score* (interval version) for a $(1-alpha)$ prediction interval $[l, u]$:

$ W_alpha (l, u, y) = (u - l) + frac(2, alpha) dot max(l - y, 0) + frac(2, alpha) dot max(y - u, 0) $

The interval width $(u - l)$ rewards sharpness; the penalty terms reward coverage. A calibrated, sharp interval minimizes the expected Winkler score.

```python
def winkler_score(lower: np.ndarray, upper: np.ndarray,
                  y: np.ndarray, alpha: float = 0.2) -> float:
    """Winkler score for (1-alpha) prediction interval. alpha=0.2 -> 80% interval."""
    width = upper - lower
    under = np.maximum(lower - y, 0)
    over  = np.maximum(y - upper, 0)
    return np.mean(width + (2/alpha) * (under + over))
```

=== Coverage and Calibration

*Coverage* is the fraction of true observations falling within the stated interval:

$ "Coverage"(q_"lo", q_"hi") = "mean"(q_"lo" <= y <= q_"hi") $

A well-calibrated 80% interval should achieve 80% empirical coverage. But coverage alone is insufficient — a trivially wide interval achieves 100% coverage with no skill.

*Reliability diagram* (calibration plot): for each quantile level $tau$, compute the empirical fraction of observations below the predicted $tau$-th quantile. Plot nominal $tau$ (x-axis) vs. empirical fraction (y-axis).

```python
import matplotlib.pyplot as plt
import numpy as np

def reliability_diagram(samples: np.ndarray, actuals: np.ndarray,
                         title: str = '', ax=None):
    """
    samples: (n_windows * horizon, n_samples)
    actuals: (n_windows * horizon,)
    """
    ax = ax or plt.gca()
    quantile_levels = np.linspace(0.02, 0.98, 49)
    empirical = []

    for q in quantile_levels:
        q_val = np.quantile(samples, q, axis=1)
        empirical.append(np.mean(actuals <= q_val))

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax.plot(quantile_levels, empirical, 'o-', markersize=3, label='Model')
    gap = np.mean(np.abs(np.array(empirical) - quantile_levels))
    ax.fill_between(quantile_levels, quantile_levels, empirical,
                    alpha=0.15, color='red', label=f'Calibration gap (mean={gap:.3f})')
    ax.set_xlabel('Nominal quantile')
    ax.set_ylabel('Empirical coverage')
    ax.set_title(title or 'Reliability Diagram')
    ax.legend()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return ax


def calibration_summary(samples: np.ndarray, actuals: np.ndarray) -> dict:
    quantile_levels = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    results = {}
    for q in quantile_levels:
        predicted = np.quantile(samples, q, axis=1)
        empirical = np.mean(actuals <= predicted)
        results[f'q{int(q*100)}'] = {'nominal': q, 'empirical': empirical, 'gap': empirical - q}
    lo = np.quantile(samples, 0.1, axis=1)
    hi = np.quantile(samples, 0.9, axis=1)
    results['coverage_80'] = np.mean((actuals >= lo) & (actuals <= hi))
    return results
```

*Diagnosing calibration failures:*

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Pattern in reliability diagram*], [*Interpretation*]),
  [Curve above diagonal (empirical > nominal)], [Intervals too wide; model over-cautious],
  [Curve below diagonal (empirical < nominal)], [Intervals too narrow; model overconfident],
  [S-curve (crosses diagonal at 0.5)], [Distribution shape wrong (e.g., symmetric model on skewed data)],
  [Perfect diagonal but bad Winkler score], [Calibrated but not sharp (intervals unnecessarily wide)],
  [Flat line near 0 or 1], [Degenerate output; model collapsed to near-deterministic],
)

== Complete Metrics Implementation

```python
import numpy as np
from scipy import stats

class ForecastMetrics:
    """Unified metrics computation for point and probabilistic forecasts."""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray,
             y_train: np.ndarray, m: int = 1) -> float:
        mae_model = np.mean(np.abs(y_true - y_pred))
        denom = np.mean(np.abs(y_train[m:] - y_train[:-m]))
        return mae_model / denom if denom > 0 else np.nan

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denom = np.abs(y_true) + np.abs(y_pred)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(denom > 0, 200 * np.abs(y_true - y_pred) / denom, 0)
        return np.mean(ratio)

    @staticmethod
    def crps_sample(samples: np.ndarray, y_true: np.ndarray) -> float:
        """samples: (n_obs, n_samples),  y_true: (n_obs,)"""
        term1 = np.mean(np.abs(samples - y_true[:, None]))
        n = samples.shape[1]
        if n <= 100:
            diff  = np.abs(samples[:, :, None] - samples[:, None, :])
            term2 = 0.5 * np.mean(diff)
        else:
            idx   = np.random.choice(n, 100, replace=False)
            term2 = 0.5 * np.mean(np.abs(samples[:, idx, None] - samples[:, None, idx]))
        return term1 - term2

    @staticmethod
    def crps_gaussian(mu: np.ndarray, sigma: np.ndarray,
                      y_true: np.ndarray) -> float:
        z = (y_true - mu) / (sigma + 1e-8)
        crps = sigma * (z * (2 * stats.norm.cdf(z) - 1)
                        + 2 * stats.norm.pdf(z) - 1/np.sqrt(np.pi))
        return np.mean(crps)

    @staticmethod
    def coverage(samples: np.ndarray, y_true: np.ndarray,
                 alpha: float = 0.2) -> float:
        lo = np.quantile(samples, alpha/2, axis=1)
        hi = np.quantile(samples, 1 - alpha/2, axis=1)
        return np.mean((y_true >= lo) & (y_true <= hi))

    @staticmethod
    def winkler(samples: np.ndarray, y_true: np.ndarray,
                alpha: float = 0.2) -> float:
        lo = np.quantile(samples, alpha/2, axis=1)
        hi = np.quantile(samples, 1 - alpha/2, axis=1)
        return np.mean(hi - lo + (2/alpha) * (np.maximum(lo - y_true, 0)
                                             + np.maximum(y_true - hi, 0)))

    @staticmethod
    def pinball(y_pred_q: np.ndarray, y_true: np.ndarray,
                quantiles: list) -> float:
        """y_pred_q: (n_obs, n_quantiles),  y_true: (n_obs,)"""
        q   = np.array(quantiles)
        err = y_true[:, None] - y_pred_q
        return float(np.mean(np.maximum(q * err, (q - 1) * err)))

    @classmethod
    def report(cls, y_true, y_pred_point, y_pred_samples=None,
               y_train=None, m=1, quantiles=None) -> dict:
        result = {'MAE': cls.mae(y_true, y_pred_point),
                  'RMSE': cls.rmse(y_true, y_pred_point)}
        if y_train is not None:
            result['MASE'] = cls.mase(y_true, y_pred_point, y_train, m)
        if y_pred_samples is not None:
            result['CRPS']        = cls.crps_sample(y_pred_samples, y_true)
            result['Coverage_80'] = cls.coverage(y_pred_samples, y_true, alpha=0.2)
            result['Winkler_80']  = cls.winkler(y_pred_samples, y_true, alpha=0.2)
        return result
```

== Metric Selection Guide

```
Is your output a point forecast or a distribution?
+-- Point -> MAE (default) or RMSE (if large errors are costly)
|           Always also compute MASE for cross-series comparability
+-- Distribution -> CRPS (primary) + coverage + reliability diagram

Is the target ever zero or near-zero?
+-- YES -> MASE (MAPE undefined), CRPS
+-- NO  -> any metric is valid

Are you comparing models across series with different scales?
+-- YES -> MASE, CRPS (scale-independent)
+-- NO  -> MAE, RMSE (interpretable in original units)

Is there asymmetric cost to over- vs. under-forecasting?
+-- YES -> pinball loss at the appropriate quantile (e.g., tau=0.9 for 90th pctile)
|          and Winkler score for the interval
+-- NO  -> MAE, CRPS (symmetric)
```

// ─── SECTION 2 ───────────────────────────────────────────────────────────────

= Backtesting Strategy

A backtest is not just a train/test split. It is a simulation of how the model would have performed in production. The closer the simulation is to reality, the more you can trust the results.

== Why Random Train/Test Split Fails for Time Series

If you randomly sample test points from a time series, two things happen:

1. *Temporal leakage:* the model trains on data from both before and after the "test" point, so it effectively has future information
2. *IID violation:* error terms are correlated across time; the standard statistical guarantees of cross-validation require IID samples

The correct primitive is the *walk-forward (time-series cross-validation)* design.

== Walk-Forward Validation Designs

=== Expanding Window (most common)

Training window grows; test window is fixed size and slides forward:

```
Fold 1: TRAIN [t1 .... t4]           TEST [t5 t6]
Fold 2: TRAIN [t1 ......... t6]      TEST [t7 t8]
Fold 3: TRAIN [t1 .............. t8] TEST [t9 t10]
```

Later folds have more training data and are more recent — closer to the actual deployment distribution. Recommended as the default.

=== Rolling Window (fixed-size training)

Training window size is fixed; both windows slide:

```
Fold 1: TRAIN [t1 .. t4]      TEST [t5 t6]
Fold 2: TRAIN [t3 .. t6]      TEST [t7 t8]
Fold 3: TRAIN [t5 .. t8]      TEST [t9 t10]
```

Use when the series is non-stationary and older data is actively harmful (regime change, trend break), or when training cost prohibits expanding windows.

=== The Gap

*Critical:* always add a gap between the train end and the test start equal to at least the forecast horizon:

```
TRAIN [.........] GAP [///] TEST [t_{T+H+1} ... t_{T+2H}]
                 |-- H steps --|
```

Without the gap, a model with $H=24$ can implicitly use observations at lags 1–23 that it wouldn't have access to in production.

== Complete Walk-Forward Implementation

```python
import numpy as np
import pandas as pd
from typing import Callable
from dataclasses import dataclass

@dataclass
class WalkForwardResult:
    fold:        int
    train_end:   int
    test_start:  int
    test_end:    int
    metrics:     dict
    predictions: np.ndarray
    actuals:     np.ndarray

class WalkForwardCV:
    """
    Walk-forward cross-validation for time series.
    Handles both expanding and rolling window designs.
    Enforces gap between train and test.
    """
    def __init__(
        self,
        n_splits:    int   = 5,
        horizon:     int   = 24,
        gap:         int   = 0,
        min_train_size: int = None,
        stride:      int   = None,
        window:      str   = 'expanding',
        rolling_size: int  = None,
    ):
        self.n_splits     = n_splits
        self.horizon      = horizon
        self.gap          = gap
        self.min_train_size = min_train_size
        self.stride       = stride or horizon
        self.window       = window
        self.rolling_size = rolling_size

    def split(self, n: int):
        """Yield (train_indices, test_indices) for each fold."""
        min_train  = self.min_train_size or n // (self.n_splits + 1)
        max_origin = n - self.gap - self.horizon
        origins    = np.linspace(min_train, max_origin, self.n_splits, dtype=int)

        for origin in origins:
            train_start = (max(0, origin - self.rolling_size)
                           if self.window == 'rolling' and self.rolling_size else 0)
            train_idx = np.arange(train_start, origin)
            test_idx  = np.arange(origin + self.gap,
                                  origin + self.gap + self.horizon)
            yield train_idx, test_idx

    def evaluate(
        self,
        series:        np.ndarray,
        model_fn:      Callable,
        predict_fn:    Callable,
        metric_fn:     Callable = None,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
    ) -> list[WalkForwardResult]:
        results = []
        for fold, (train_idx, test_idx) in enumerate(self.split(len(series))):
            train = series[train_idx]
            test  = series[test_idx]

            # Preprocessing fitted on train only — CRITICAL
            if preprocess_fn is not None:
                train_proc, scaler = preprocess_fn(train)
            else:
                train_proc, scaler = train, None

            model = model_fn(train_proc)
            preds = predict_fn(model, self.horizon)

            if postprocess_fn is not None and scaler is not None:
                preds = postprocess_fn(preds, scaler)

            metrics = metric_fn(test, preds) if metric_fn else {}
            results.append(WalkForwardResult(
                fold=fold, train_end=train_idx[-1],
                test_start=test_idx[0], test_end=test_idx[-1],
                metrics=metrics, predictions=preds, actuals=test,
            ))
        return results

    def summary(self, results: list[WalkForwardResult]) -> pd.DataFrame:
        rows = [{'fold': r.fold, **r.metrics} for r in results]
        return pd.DataFrame(rows).describe().loc[['mean', 'std', 'min', 'max']]
```

```python
cv = WalkForwardCV(n_splits=5, horizon=96, gap=96, window='expanding')

results = cv.evaluate(
    ot_series,
    model_fn   = lambda train: auto_arima(train, seasonal=True, m=24),
    predict_fn = lambda model, h: model.predict(h),
    metric_fn  = lambda act, pred: {
        'MAE':  np.mean(np.abs(act - pred)),
        'RMSE': np.sqrt(np.mean((act - pred)**2)),
    },
)
print(cv.summary(results))
```

== How Many Folds?

- *More folds → lower variance* in the performance estimate
- *Fewer folds → more training data per fold* (closer to the production model)

Practical rule: $K = 5$ is the default. Use $K >= 10$ when the test period shows high temporal variability. Use $K = 3$ when training cost is high and the series is long enough.

#note[*Statistical significance:* with $K=5$ folds you have low power — be humble about claiming one model is definitively better unless the gap is large. Run a paired t-test or Wilcoxon signed-rank test on per-fold differences.]

```python
from scipy import stats

def compare_models(results_a: list, results_b: list, metric: str = 'MAE') -> dict:
    scores_a = [r.metrics[metric] for r in results_a]
    scores_b = [r.metrics[metric] for r in results_b]
    diff = np.array(scores_a) - np.array(scores_b)

    _, p_t  = stats.ttest_rel(scores_a, scores_b)
    _, p_wx = stats.wilcoxon(diff)
    return {
        'mean_diff':        np.mean(diff),
        'std_diff':         np.std(diff),
        'p_value_t':        p_t,
        'p_value_wilcoxon': p_wx,
        'a_better':         np.mean(diff < 0),   # fraction of folds where A wins
    }
```

// ─── SECTION 3 ───────────────────────────────────────────────────────────────

= Failure Modes: Leakage

Leakage is the presence of information in the training or evaluation pipeline that would not be available at inference time. It is the single most common source of inflated backtest results. There are five distinct forms.

== Target Leakage (Scale/Normalization Leakage)

*What it is:* fitting a scaler, normalizer, or any statistics-based transform on the full dataset before splitting.

```python
# WRONG — StandardScaler fit on full series including test
scaler = StandardScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).ravel()
train, test = series_scaled[:train_end], series_scaled[train_end:]
```

The scaler's mean and standard deviation are computed from the test period. The model trains on normalized data that already "knows" the future mean and variance.

```python
# CORRECT — StandardScaler fit on training window only
scaler = StandardScaler()
train_raw, test_raw = series[:train_end], series[train_end:]
train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).ravel()
test_scaled  = scaler.transform(test_raw.reshape(-1, 1)).ravel()
```

This applies to any statistics derived from data: mean, std, min, max, quantiles, rolling statistics, STL decomposition parameters, and PCA/embedding transforms.

== Feature Leakage (Future Covariate Leakage)

*What it is:* including a covariate at time $t$ that contains information about $y_t$ or $y_(t+h)$ but would not be available at time $t$ in production.

```python
# WRONG — rolling mean uses future values
df['rolling_mean_7d'] = df['demand'].rolling(window=7, center=True).mean()
#                                                        ^^^^^^^^^^^
#                                                        center=True uses future!

# CORRECT — rolling mean uses only past values
df['rolling_mean_7d'] = df['demand'].shift(1).rolling(window=7).mean()
#                                    ^^^^^^^
#                                    shift(1) ensures we don't use current timestep
```

*Asymmetric covariates:* TFT distinguishes "past-observed" from "known-future" covariates. This distinction must be exactly right. If you label a covariate as "known future" but it's actually observed only in the past, you're leaking.

== Evaluation Leakage (Multiple Comparison / Test Set Reuse)

*What it is:* using the test set more than once in the model selection process.

Every time you evaluate on the test set and then make a design decision based on the result, you are overfitting the evaluation. With enough iterations, any model will appear to outperform purely by chance.

*The correct protocol:*

```
DATA: [======= TRAIN =======] [= VAL =] [= TEST =]
                                  |            |
                         Tune here      Report here (once)
```

In walk-forward CV: the "test" windows used during development are analogous to validation. Keep a true held-out final test period (e.g., the last 20% of your data) that you report on after all decisions are made.

== Pipeline Leakage (Cross-Entity Contamination)

*What it is:* for panel data (multiple time series), using information from other entities during evaluation in a way that wouldn't be available at inference time.

#gotcha[When fitting a global model on all 370 clients in the Electricity dataset, ensure the temporal split is consistent across entities: all entities' test windows start at the same calendar time. Never use future observations of any entity during the training phase.]

== Label Leakage (Preprocessing Contamination)

*What it is:* applying a global transformation to the target that encodes future information.

```python
# WRONG — percentile computed over full dataset including test
clip_val = np.percentile(series, 99)
series_clipped = np.clip(series, None, clip_val)

# CORRECT — computed on training data only
clip_val = np.percentile(series[:train_end], 99)
series_clipped = np.clip(series, None, clip_val)
```

The same principle applies to: log-transform minimum values, outlier winsorization thresholds, Box-Cox lambda estimation.

== Leakage Audit Checklist

Before finalizing a pipeline, go through this list:

```
[ ] All scalers/normalizers fitted on training window only
[ ] Rolling/expanding window features use only past values (no center=True)
[ ] No target-derived features with insufficient lag
[ ] All "known future" covariates are genuinely known at forecast time
[ ] Temporal split is consistent across entities in panel datasets
[ ] Outlier thresholds and clip values derived from training data only
[ ] STL / decomposition parameters fitted on training window only
[ ] Test set touched exactly once for final reporting
[ ] Hyperparameter selection done on validation, not test
[ ] No selection of "best" test fold from walk-forward (report average across all folds)
```

// ─── SECTION 4 ───────────────────────────────────────────────────────────────

= Distribution Shift and Structural Breaks

Even a correctly evaluated model degrades in production. The time series you trained on and the time series you're forecasting diverge over time. Understanding how to detect and respond to this is essential for production readiness.

== Types of Distribution Shift

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Type*], [*Description*], [*Example*]),
  [Covariate shift],
  [Input distribution changes; relationship $p(y|bold(x))$ stable],
  [Demand volume grows, but price elasticity unchanged],
  [Concept drift],
  [Relationship $p(y|bold(x))$ changes; inputs may be stable],
  [COVID changes consumer behavior; price elasticity shifts],
  [Seasonal shift],
  [Seasonal pattern changes over years],
  ["Seasonality" in retail as e-commerce grows],
  [Structural break],
  [Sudden, permanent level shift],
  [Policy change, regulatory intervention, product discontinuation],
  [Volatility regime change],
  [Error variance changes],
  [Financial crisis, supply chain shock],
)

== Detecting Structural Breaks

=== Chow Test

Tests whether the regression parameters differ before and after a hypothesized break point $k$:

$
  H_0: bold(beta)_1 = bold(beta)_2 quad "(same parameters before and after" k")" \
  H_1: bold(beta)_1 != bold(beta)_2
$

$ F = frac(("RSS"_"full" - "RSS"_1 - "RSS"_2) \/ K, ("RSS"_1 + "RSS"_2) \/ (n - 2K)) $

where RSS is residual sum of squares, $K$ is the number of parameters, and $n$ is total observations.

```python
from statsmodels.regression.linear_model import OLS
import numpy as np

def chow_test(y: np.ndarray, break_point: int) -> dict:
    """Simple Chow test: does a linear trend differ before and after break_point?"""
    n = len(y)
    t = np.arange(n)
    X_full = np.column_stack([np.ones(n), t])

    rss_full = np.sum((OLS(y,            X_full           ).fit().resid)**2)
    rss1     = np.sum((OLS(y[:break_point], X_full[:break_point]).fit().resid)**2)
    rss2     = np.sum((OLS(y[break_point:], X_full[break_point:]).fit().resid)**2)

    K = X_full.shape[1]
    F = ((rss_full - rss1 - rss2) / K) / ((rss1 + rss2) / (n - 2*K))

    from scipy import stats
    p_value = 1 - stats.f.cdf(F, K, n - 2*K)
    return {'F_stat': F, 'p_value': p_value, 'break_point': break_point}
```

=== CUSUM Test

CUSUM (Cumulative Sum) detects gradual parameter instability by tracking the cumulative sum of recursive residuals. A crossing of the bounds indicates instability.

```python
from statsmodels.stats.diagnostic import breaks_cusumolsresid
from statsmodels.regression.linear_model import OLS

X = np.column_stack([np.ones(len(y)), np.arange(len(y))])
model = OLS(y, X).fit()

cusum_stat, cusum_pvalue, cusum_crit = breaks_cusumolsresid(model.resid)
print(f"CUSUM p-value: {cusum_pvalue:.4f}")
# p < 0.05: significant parameter instability detected
```

=== Rolling Window Performance Monitoring

The most practical production approach: monitor rolling MAE over time and trigger retraining when it degrades beyond a threshold.

```python
def rolling_performance_monitor(
    actuals: np.ndarray,
    predictions: np.ndarray,
    window: int = 30,
    threshold_multiplier: float = 1.5,
    baseline_window: int = 90,
):
    n = len(actuals)
    baseline_mae = np.mean(np.abs(actuals[:baseline_window] - predictions[:baseline_window]))

    rolling_mae = np.array([
        np.mean(np.abs(actuals[i-window:i] - predictions[i-window:i]))
        for i in range(window, n)
    ])

    alert_threshold = baseline_mae * threshold_multiplier
    alert_indices   = np.where(rolling_mae > alert_threshold)[0] + window

    return {
        'rolling_mae':     rolling_mae,
        'baseline_mae':    baseline_mae,
        'alert_threshold': alert_threshold,
        'alert_indices':   alert_indices,
        'current_mae':     rolling_mae[-1] if len(rolling_mae) > 0 else None,
        'degraded':        len(alert_indices) > 0 and alert_indices[-1] > n - window//2,
    }
```

== Responding to Distribution Shift

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Strategy*], [*When to use*], [*Tradeoff*]),
  [Full retraining on new data], [Gradual shift; sufficient new data], [Expensive; may lose historical patterns],
  [Rolling window retraining], [Deliberate forgetting of old data], [Fast; loses long-run patterns],
  [Continual learning (incremental update)], [Frequent shifts; latency constraints], [Complex; risk of catastrophic forgetting],
  [Regime-switching model], [Abrupt, discrete regime changes], [Requires regime detection; model complexity],
  [Intervention dummies], [Known one-time events], [Simple; only works for identified events],
  [ETS (inherently adaptive)], [Moderate drift; no DL overhead], [Limited; assumes smooth adaptation],
)

```python
# In production: check daily
monitor_result = rolling_performance_monitor(recent_actuals, recent_preds)

if monitor_result['degraded']:
    if monitor_result['current_mae'] > 2.0 * monitor_result['baseline_mae']:
        retrain_model(window='rolling', size=365)   # severe: immediate retrain
    else:
        schedule_retrain(priority='normal')          # moderate: schedule retrain
```

// ─── SECTION 5 ───────────────────────────────────────────────────────────────

= Special Series Types

== Intermittent / Zero-Inflated Demand

Many practical series (slow-moving inventory, rare events, spare parts demand) have a large fraction of zeros. Standard regression models fail because MSE and MAE loss is dominated by the non-zero observations, the conditional mean is not a useful forecast for a zero-inflated distribution, and MAPE is undefined.

*Croston's Method:* separately tracks the demand size (when demand is non-zero) and the inter-demand interval (time between non-zero observations), each with exponential smoothing:

```python
def croston(series: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Croston's method for intermittent demand; returns forecast of mean demand per period."""
    q = series[series > 0]
    if len(q) == 0:
        return np.zeros(len(series))

    z, x = q[0], 1.0   # smoothed demand size and inter-demand interval
    forecasts, last_nonzero = [], 0

    for t, y in enumerate(series):
        if y > 0:
            interval = t - last_nonzero
            z = alpha * y + (1 - alpha) * z
            x = alpha * interval + (1 - alpha) * x
            last_nonzero = t
        forecasts.append(z / x)

    return np.array(forecasts)
```

*Two-stage model (classification + regression):*

```python
class IntermittentForecaster:
    """
    Stage 1: predict P(demand > 0) via binary classifier
    Stage 2: predict E[demand | demand > 0] via regressor
    Final:   P(demand > 0) * E[demand | demand > 0]
    """
    def __init__(self, classifier, regressor):
        self.clf = classifier
        self.reg = regressor

    def fit(self, X, y):
        binary_y = (y > 0).astype(float)
        nonzero_mask = binary_y == 1
        self.clf.fit(X, binary_y)
        self.reg.fit(X[nonzero_mask], y[nonzero_mask])
        return self

    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1] * self.reg.predict(X)
```

*Metrics for intermittent demand:* MASE with $m=1$ (non-seasonal naive) as denominator. Avoid RMSE (dominated by rare large spikes) and MAPE (undefined at zeros).

== Hierarchical Time Series

Many production systems require forecasts to be *coherent* across a hierarchy: national → regional → store → SKU. Forecasts are incoherent if the regional forecast doesn't equal the sum of the store forecasts.

*Approaches:*
- *Bottom-up:* forecast the lowest level; aggregate up. Coherent by construction. Noisy at the bottom.
- *Top-down:* forecast the top level; disaggregate using historical proportions. Smooth at the top; loses bottom-level idiosyncrasy.
- *Middle-out:* forecast at an intermediate level; aggregate up and disaggregate down.
- *MinT reconciliation (optimal):* forecast all levels independently; project the forecast vector onto the constraint space defined by the summing matrix:

$ hat(bold(y))_"reconciled" = bold(S) (bold(S)^top bold(W)^(-1) bold(S))^(-1) bold(S)^top bold(W)^(-1) hat(bold(y))_"base" $

where $bold(S)$ is the summing matrix and $bold(W)$ is the covariance of base forecast errors.

```python
from hierarchicalforecast.methods import MinTrace

reconciler = MinTrace(method='wls_var')   # variance-weighted least squares
reconciler.reconcile(S, base_forecasts, W)
```

== Bounded / Positive Series

Some targets are bounded: occupancy rates in $[0, 1]$, probabilities, percentages. Standard regression can predict outside the bounds.

*Transformations:*
- Logit transform for $(0, 1)$ bounded series: $"logit"(y) = ln(y\/(1-y))$
- Log transform for strictly positive series: $ln(y)$
- Box-Cox for general power transformation

```python
from scipy.special import expit, logit

# For proportion targets in (0, 1):
y_transformed = logit(y.clip(1e-6, 1 - 1e-6))   # model predicts logit
y_forecast    = expit(raw_forecast)                # inverse transform to (0, 1)
```

// ─── SECTION 6 ───────────────────────────────────────────────────────────────

= Production Readiness

== What "Good Enough" Looks Like

There is no universal threshold for "good" forecast accuracy. The right question is: *does the model improve the decision that depends on the forecast?*

Practical calibration points:
- *MASE $< 1.0$:* the model beats the seasonal naive baseline — a minimum bar for deployment
- *MASE $< 0.8$:* model provides meaningful lift over naive; typically worth the operational overhead
- *80% interval coverage within $plus.minus 5%$ of 80%:* calibration acceptable for most use cases
- *RMSE/MAE ratio $<= 1.5$:* error distribution is not heavily tailed; model isn't systematically failing on edge cases

== The Benchmark Hierarchy

Always evaluate against this hierarchy, in order:

1. Naive seasonal baseline (repeat last season's value)
2. ETS / ARIMA (strong classical model)
3. Your DL model
4. Foundation model zero-shot (Chronos / TimesFM)
5. Ensemble (weighted combination of above)

#note[A DL model that doesn't beat ETS is a symptom, not a failure — it tells you either that the problem is too simple for DL, or that something is wrong in your pipeline. Find out which before deploying.]

== Residual Analysis Protocol

After fitting any model, examine residuals $epsilon_t = y_t - hat(y)_t$:

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

def residual_analysis(actuals: np.ndarray, preds: np.ndarray,
                      horizon: int = 1, m: int = 24):
    residuals = actuals - preds

    print("=== Residual Analysis ===")
    print(f"Mean:     {residuals.mean():.4f}  (should be ~0)")
    print(f"Std:      {residuals.std():.4f}")
    print(f"Skewness: {stats.skew(residuals):.4f}  (should be ~0 for Gaussian)")
    print(f"Kurtosis: {stats.kurtosis(residuals):.4f}  (0 = Gaussian)")

    lb = acorr_ljungbox(residuals, lags=[m, 2*m], return_df=True)
    print(f"\nLjung-Box p-values:")
    print(lb[['lb_stat', 'lb_pvalue']])
    print("  p > 0.05: residuals look like white noise")
    print("  p < 0.05: significant autocorrelation remains")

    _, sw_pval = stats.shapiro(residuals[:500])
    print(f"\nShapiro-Wilk p-value: {sw_pval:.4f}")

    q33, q67 = np.percentile(actuals, [33, 67])
    low_mask  = actuals <= q33
    mid_mask  = (actuals > q33) & (actuals <= q67)
    high_mask = actuals > q67
    print(f"\nMean residual by actual value range:")
    print(f"  Low:  {residuals[low_mask].mean():.4f}")
    print(f"  Mid:  {residuals[mid_mask].mean():.4f}")
    print(f"  High: {residuals[high_mask].mean():.4f}")
    print("  All should be near 0; systematic bias by range -> heteroscedasticity")
```

*What bad residuals look like and what they mean:*

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Pattern*], [*Meaning*], [*Fix*]),
  [Mean $!= 0$], [Systematic bias], [Check normalization; add bias term],
  [ACF significant at lag $m$], [Seasonality not captured], [Add seasonal component; increase model capacity],
  [ACF significant at lags 1–3], [Remaining autocorrelation], [Increase AR order; add recurrent layer],
  [Variance grows with level], [Heteroscedasticity], [Log transform target; multiplicative ETS],
  [Heavy tails (excess kurtosis $> 3$)], [Outliers not captured], [Robust loss (Huber); Student-$t$ output distribution],
  [Bias in high-value range], [Underforecasting peaks], [Quantile regression at high $tau$; asymmetric loss],
)

== Overfitting Seasonality: A Specific DL Failure

DL models with enough capacity can memorize the seasonal pattern from the training data rather than generalizing it. Symptoms:
- Strong in-sample performance; poor performance on the same season in a different year
- Attention weights in TFT concentrated at exact multiples of the seasonal period
- Performance degrades sharply at horizons beyond one seasonal cycle

*Fixes:*
1. Ensure at least 3 full seasonal cycles in training
2. Apply seasonal normalization: divide each series by its seasonal index before modeling
3. Use STL-LSTM: decompose with STL, model only the residual, re-add seasonal component
4. Increase dropout; reduce model size

// ─── SECTION 7 ───────────────────────────────────────────────────────────────

= Week 4 (Part 2) Practice Exercises

== Exercise 1: Metric Sensitivity Analysis (1–2 hours)

On the ETTh1 OT series, compute all metrics for your best model from Phases 2–3:

```python
models = {'ARIMA': arima_preds, 'LSTM': lstm_preds,
          'TCN':   tcn_preds,   'TFT':  tft_preds,
          'Chronos': chronos_preds}

report = {}
for name, preds in models.items():
    report[name] = ForecastMetrics.report(
        y_true=test_actuals,
        y_pred_point=preds['point'],
        y_pred_samples=preds.get('samples'),
        y_train=train_series,
        m=24,   # hourly data with daily seasonality
    )

print(pd.DataFrame(report).T.to_string())
```

*Questions to answer:*
- Does the ranking of models change between MAE and MASE?
- Which model has the best CRPS? Is it the same as the best MAE model?
- What is the 80% coverage for TFT's quantile output? Is it near 80%?

== Exercise 2: Leakage Hunt (1 hour)

Take the following deliberately broken pipeline and identify all leakage instances:

```python
# BROKEN PIPELINE — find all leakage instances
df = pd.read_csv('ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])

# Feature engineering
df['rolling_mean_7d'] = df['OT'].rolling(24*7, center=True).mean()  # Bug 1?
df['rolling_std_3d']  = df['OT'].rolling(24*3, center=True).std()   # Bug 2?

# Clip outliers
clip_val = np.percentile(df['OT'], 99)                               # Bug 3?
df['OT_clipped'] = df['OT'].clip(None, clip_val)

# Scale
scaler = StandardScaler()
df['OT_scaled'] = scaler.fit_transform(df[['OT_clipped']])           # Bug 4?

# Train/test split
train_end = int(0.8 * len(df))
train, test = df.iloc[:train_end], df.iloc[train_end:]

# Hyperparameter search on test set (reported as "validation")
for lr in [1e-3, 1e-4, 5e-4]:
    model = train_lstm(train, lr=lr)
    score = evaluate(model, test)                                     # Bug 5?
    print(f"lr={lr}: MAE={score:.4f}")
best_lr = 1e-4  # selected based on test set performance
```

Identify which lines have bugs, explain the leakage mechanism, and write the corrected version.

== Exercise 3: Walk-Forward Design (1 hour)

Design and implement a walk-forward backtest for the ETTh1 dataset with all of the following:
- 5 folds, expanding window
- Horizon = 96 (4 days ahead)
- Gap = 96 (equal to horizon)
- Min training size = 3000 ($approx$ 4 months of hourly data)
- Scaler fitted on train window only
- Report mean and std of MAE and MASE across folds
- Statistical comparison between LSTM and TCN using Wilcoxon signed-rank test

== Exercise 4: Residual Diagnosis (30 minutes)

Run `residual_analysis` on the ARIMA residuals from Phase 1 and on the LSTM residuals from Phase 2. Compare:
- Are LSTM residuals better (more white-noise-like) than ARIMA residuals?
- Is there remaining seasonality in the residuals (check ACF at lags 24, 48, 168)?
- Is there a systematic bias in the high-value range for either model?

== Exercise 5: Reliability Diagram for TFT (30 minutes)

Generate a reliability diagram for your TFT model's quantile forecasts across all 5 walk-forward folds. Compute the mean calibration gap. If coverage at the 80% interval is outside $[75%, 85%]$, identify whether the model is overconfident or conservative and propose a recalibration strategy.

// ─── SECTION 8 ───────────────────────────────────────────────────────────────

= Interview Fluency

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Term*], [*Definition*]),
  [MASE],
  [MAE normalized by in-sample seasonal naive MAE; scale-free; MASE $< 1$ beats naive],
  [CRPS],
  [$integral [F(z) - bb(1)(z >= y)]^2 d z$; proper scoring rule for predictive CDFs; generalizes MAE to distributions],
  [Proper scoring rule],
  [A loss function whose expected value is minimized by the true data-generating distribution; honesty is optimal],
  [Walk-forward validation],
  [Temporal CV where the train window ends before the test window starts; simulates production deployment],
  [Gap],
  [Steps excluded between train end and test start; prevents exploiting short-range autocorrelation],
  [Target leakage],
  [Scaler or transform fitted on data that includes the test period; encodes future statistics],
  [Feature leakage],
  [Covariate computed using future observations; provides information unavailable at inference],
  [Evaluation leakage],
  [Using test set results to make model design decisions; overfits the evaluation],
  [Structural break],
  [Sudden, permanent change in the data-generating process; makes models trained before the break obsolete],
  [Calibration],
  [Alignment between stated confidence and empirical frequency; measured by reliability diagram],
)

*"Why is CRPS preferred over log-likelihood for evaluating probabilistic forecasts?"*

#note[Both are proper scoring rules, but CRPS has two practical advantages. First, it's in the original units of the series (like MAE), making it interpretable and comparable across datasets with different scales. Second, it's more robust to distribution misspecification at the tails — log-likelihood is dominated by extreme events and can swing dramatically from a single tail observation. CRPS integrates over the entire CDF, giving more stable estimates. For sample-based models like Chronos, CRPS can be computed directly from samples via the energy score formula without fitting a parametric distribution.]

*"What is the gap in walk-forward validation and why is it necessary?"*

#note[The gap is a buffer of steps excluded between the training window and the test window. Without it, a model forecasting $H$ steps ahead still has access to observations at lags 1 through $H-1$ via autocorrelation — information unavailable in production where you're genuinely forecasting $H$ steps into the future. For example, if forecasting tomorrow's electricity demand with no gap, the backtest implicitly lets the model use today's demand as an input because it's within $H=24$ steps of the test period. A gap of size $H$ ensures the test condition is realistic.]

*"Walk me through all the ways a forecasting pipeline can have data leakage."*

#note[There are five distinct forms. *Target leakage:* fitting scalers on the full dataset, encoding future mean and variance. *Feature leakage:* computing rolling features with `center=True`, or covariates that use future information. *Evaluation leakage:* using the test set to select hyperparameters or architecture choices. *Pipeline leakage:* in panel data, allowing future observations of one entity to influence the model through shared representations. *Label leakage:* applying global transforms like outlier clipping where parameters are derived from the full dataset. The fix is the same principle throughout: any statistic derived from data must be computed exclusively from the training window.]

*"What would you do if your model's performance degrades in production?"*

#note[First, quantify the degradation: compute rolling MAE over time and compare to the baseline window. If gradual, distinguish between covariate shift and concept drift — the former may be addressable with better feature engineering, the latter requires retraining. If sudden, run a Chow test or CUSUM to confirm a structural break. Depending on severity: mild degradation → schedule incremental retrain; severe → immediate full retrain on recent data with a rolling window. Also check residual ACF — if seasonality reappears, the seasonal structure may have shifted. The most important thing is to have the monitoring pipeline in place *before* deployment, not after degradation is noticed by downstream stakeholders.]

*"How do you test whether one model is statistically better than another?"*

#note[With $K=5$ walk-forward folds, compute the per-fold metric difference between model A and B. A paired t-test tests whether the mean difference is zero; Wilcoxon signed-rank is the non-parametric alternative (robust to non-Gaussian differences). With only 5 folds, statistical power is low — you need a fairly large effect size to reach significance. The honest answer when the p-value is borderline is that the models are competitive and the gap could be noise. Also look at fold-by-fold consistency: if one model wins on 4 out of 5 folds, that's more convincing than one fold with a large margin and four folds roughly tied.]

// ─── SECTION 9 ───────────────────────────────────────────────────────────────

= Summary: The Evaluation Mindset

```
Model development
        |
        +-- Am I computing metrics correctly?
        |   +-- MASE for cross-series; CRPS for probabilistic; coverage for intervals
        |
        +-- Is my backtest simulating production accurately?
        |   +-- Walk-forward (never random split)
        |   +-- Gap >= horizon
        |   +-- Scaler fitted on train window only
        |   +-- Test set touched exactly once
        |
        +-- Have I audited the pipeline for leakage?
        |   +-- Run the 10-point checklist before reporting any number
        |
        +-- Does my model beat naive seasonal? Does it beat ETS?
        |   +-- If not: pipeline problem, not model problem — investigate first
        |
        +-- Are the residuals white noise?
        |   +-- If not: remaining structure -> model is misspecified
        |
        +-- Is the distribution stable over the backtest period?
        |   +-- Plot rolling MAE; run CUSUM; check for structural breaks
        |
        +-- In production: monitor -> detect -> retrain -> verify
```

#line(length: 100%)

_Next: Phase 5 — Hands-On Mini-Project (ETT dataset, full pipeline)_

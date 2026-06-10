#set document(title: "Time-Series Forecasting — Phase 1: Classical Baselines", author: "")
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
  #text(size: 15pt)[Phase 1 — Classical Baselines]
  #v(0.3em)
  #text(size: 10pt, fill: luma(100))[ARIMA · ETS · State Space Models · Evaluation Framework]
]

#v(1em)
#line(length: 100%)
#v(0.5em)

#outline(indent: 1em, depth: 2)

#v(0.5em)
#line(length: 100%)

// ─── OVERVIEW ────────────────────────────────────────────────────────────────

= Overview

Classical methods matter not because you'll deploy them in production (though sometimes you will), but because they make the problem structure explicit. Every inductive bias that DL methods must learn implicitly — trend, seasonality, autocorrelation structure, noise modeling — is baked in analytically here. Understanding what these models assume, and where those assumptions break, gives you the vocabulary and the intuition to diagnose DL failures later.

*By the end of Week 1 you should be able to:*

- Identify whether a series needs differencing (ADF test, ACF/PACF)
- Fit ARIMA and ETS and interpret their parameters
- Read residual diagnostics and know what a bad fit looks like
- Articulate, precisely, the failure mode of each model
- Know when to reach for a classical model over a neural one in production

// ─── SECTION 1 ───────────────────────────────────────────────────────────────

= The Unifying Framework: State Space Models

Before touching ARIMA or ETS individually, it's worth seeing the roof they both live under. Every classical forecasting method can be written as a *linear Gaussian state space model (LG-SSM)*:

$
  bold(z)_t &= bold(A) bold(z)_(t-1) + bold(epsilon)_t, quad bold(epsilon)_t tilde cal(N)(bold(0), bold(Q)) \
  bold(y)_t &= bold(C) bold(z)_t + bold(eta)_t, quad bold(eta)_t tilde cal(N)(bold(0), bold(R))
$

where:

- $bold(z)_t$ is the latent state (level, trend, seasonal components — whatever the model tracks)
- $bold(A)$ is the state transition matrix (how the latent state evolves)
- $bold(C)$ is the observation matrix (how the state maps to observations)
- $bold(Q)$, $bold(R)$ are the process and observation noise covariances

The *Kalman filter* gives the exact posterior $p(bold(z)_t | bold(y)_(1:t)) = cal(N)(bold(mu)_t, bold(Sigma)_t)$ with the update:

$
  bold(mu)_(t|t-1) &= bold(A) bold(mu)_(t-1) \
  bold(Sigma)_(t|t-1) &= bold(A) bold(Sigma)_(t-1) bold(A)^top + bold(Q)
$

$
  bold(K)_t &= bold(Sigma)_(t|t-1) bold(C)^top (bold(C) bold(Sigma)_(t|t-1) bold(C)^top + bold(R))^(-1) quad "[Kalman gain]" \
  bold(mu)_t &= bold(mu)_(t|t-1) + bold(K)_t (bold(y)_t - bold(C) bold(mu)_(t|t-1)) \
  bold(Sigma)_t &= (bold(I) - bold(K)_t bold(C)) bold(Sigma)_(t|t-1)
$

The Kalman gain $bold(K)_t$ balances how much to trust the new observation vs. the prior — exactly the same trade-off exponential smoothing makes with its $alpha$ parameter.

*ARIMA* corresponds to an SSM with polynomial state dynamics and unit roots. *ETS* corresponds to an SSM with a structured decomposition into level, trend, and seasonality. The Kalman filter is the optimal estimator for both.

#insight[
  The Kalman smoother (backward pass) solves the same mathematical problem as BPTT — both are instances of the forward-backward algorithm on a chain graphical model. The difference is that Kalman operates in closed form on a linear Gaussian model, while BPTT handles nonlinear models via automatic differentiation.
]

// ─── SECTION 2 ───────────────────────────────────────────────────────────────

= Stationarity

Almost every classical method assumes *weak stationarity*: constant mean, constant variance, and autocovariance that depends only on lag (not on time).

Formally, a process ${y_t}$ is weakly stationary if:

$
  EE[y_t] = mu & quad "(constant mean)" \
  "Var"(y_t) = sigma^2 & quad "(constant variance)" \
  "Cov"(y_t, y_(t-k)) = gamma(k) & quad "(autocovariance depends only on lag" k")"
$

Real series violate this in structured ways:

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Violation*], [*Visual signature*], [*Fix*]),
  [Trend],
  [Drifting mean],
  [Differencing ($nabla y_t = y_t - y_(t-1)$)],
  [Changing variance],
  [Variance grows with level],
  [Log transform, then difference],
  [Seasonality],
  [Periodic oscillation],
  [Seasonal differencing ($nabla_m y_t = y_t - y_(t-m)$)],
  [Unit root],
  [Random walk (persistent shocks)],
  [Differencing],
)

== Augmented Dickey-Fuller (ADF) Test

Null hypothesis: the series has a unit root (non-stationary). The ADF tests whether $phi = 1$ in the regression:

$ nabla y_t = alpha + beta t + (phi - 1) y_(t-1) + sum_(j=1)^p delta_j nabla y_(t-j) + epsilon_t $

- *p-value < 0.05*: reject the null → series is stationary (after any transforms)
- *p-value > 0.05*: fail to reject → series likely has a unit root → difference

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(series, autolag='AIC')
print(f"ADF statistic: {result[0]:.4f}")
print(f"p-value:       {result[1]:.4f}")
print(f"Lags used:     {result[2]}")
# result[4] is the dict of critical values {'1%': ..., '5%': ..., '10%': ...}
```

#gotcha[
  ADF has low power for near-unit-root processes. Use KPSS (null: stationary) in conjunction for robustness. Disagreement between ADF and KPSS signals a process that's hard to classify — be conservative and difference.
]

== KPSS Test

```python
from statsmodels.tsa.stattools import kpss

stat, p_value, lags, crit = kpss(series, regression='c')  # 'c' = constant, 'ct' = constant + trend
# p-value < 0.05: reject stationarity
```

// ─── SECTION 3 ───────────────────────────────────────────────────────────────

= ACF and PACF: The Diagnostic Pair

Before fitting any ARIMA model, plot the autocorrelation function (ACF) and partial autocorrelation function (PACF) of the stationarized series.

*ACF at lag $k$:*

$ rho(k) = "Cov"(y_t, y_(t-k)) / "Var"(y_t) $

Measures total correlation between $y_t$ and $y_(t-k)$ — including indirect paths through intermediate lags.

*PACF at lag $k$:* Measures the correlation between $y_t$ and $y_(t-k)$ after partialing out the effects of lags 1 through $k-1$. Equivalently, it is the last coefficient in an $"AR"(k)$ regression.

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Pattern*], [*Implication*]),
  [ACF decays slowly (geometric)],
  [AR process — add AR terms],
  [ACF cuts off sharply at lag $q$],
  [$"MA"(q)$ process],
  [PACF cuts off sharply at lag $p$],
  [$"AR"(p)$ process],
  [Both decay slowly],
  [ARMA — need both],
  [Spike at lag $m, 2m, 3m$ in ACF],
  [Seasonality at period $m$],
  [ACF doesn't decay (stays near 1)],
  [Non-stationarity — difference first],
)

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(series_diff, lags=40, ax=ax1, title="ACF (after differencing)")
plot_pacf(series_diff, lags=40, ax=ax2, title="PACF (after differencing)")
plt.tight_layout()
```

The blue shaded region is the 95% confidence band: $plus.minus 1.96 \/ sqrt(n)$. Spikes outside this band are statistically significant.

// ─── SECTION 4 ───────────────────────────────────────────────────────────────

= ARIMA

== The Model

ARIMA$(p, d, q)$ is built from three components.

*AR$(p)$ — Autoregressive:*

$ y_t = c + phi_1 y_(t-1) + phi_2 y_(t-2) + dots + phi_p y_(t-p) + epsilon_t $

Each observation is a weighted sum of the $p$ previous observations plus noise.

*I$(d)$ — Integrated:* Apply the differencing operator $nabla$ a total of $d$ times to achieve stationarity:

$
  nabla y_t &= y_t - y_(t-1) quad "(d=1, removes linear trend)" \
  nabla^2 y_t &= nabla y_t - nabla y_(t-1) quad "(d=2, removes quadratic trend)"
$

*MA$(q)$ — Moving Average:*

$ y_t = c + epsilon_t + theta_1 epsilon_(t-1) + theta_2 epsilon_(t-2) + dots + theta_q epsilon_(t-q) $

The current observation depends on past _shock_ terms $epsilon$, not past _observations_. This is the mechanism by which transient shocks propagate and then dissipate.

*Combined ARIMA$(p, d, q)$:*

$
  nabla^d y_t = c &+ phi_1 nabla^d y_(t-1) + dots + phi_p nabla^d y_(t-p) \
                  &+ epsilon_t + theta_1 epsilon_(t-1) + dots + theta_q epsilon_(t-q)
$

*SARIMA$(p,d,q)(P,D,Q)_m$* adds seasonal AR, differencing, and MA terms at period $m$ (e.g., $m=12$ for monthly data with annual seasonality):

$ Phi_P (B^m) phi_p (B) nabla_m^D nabla^d y_t = c + Theta_Q (B^m) theta_q (B) epsilon_t $

where $B$ is the backshift operator ($B y_t = y_(t-1)$).

== Parameter Selection

*Manual (ACF/PACF reading):*
1. Plot original series → decide $d$ (0, 1, or 2 usually)
2. Plot ACF/PACF of differenced series → read off $p$ and $q$
3. Fit → check residuals → adjust

*Automatic (information criterion minimization):*

```python
from pmdarima import auto_arima

model = auto_arima(
    series,
    start_p=0, max_p=5,
    start_q=0, max_q=5,
    d=None,             # auto-select via ADF
    seasonal=True,
    m=12,               # seasonal period
    information_criterion='aic',
    stepwise=True,      # greedy search (faster)
    trace=True,
)
print(model.summary())
```

$"AIC" = 2k - 2 ln(L)$ where $k$ = number of parameters. $"BIC" = k ln(n) - 2 ln(L)$. BIC penalizes complexity more strongly — prefer it to avoid overfitting on short series.

== Fitting with statsmodels

```python
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Assume series is a pd.Series with DatetimeIndex
model = ARIMA(series, order=(p, d, q))
result = model.fit()

print(result.summary())

# Forecast
forecast = result.get_forecast(steps=24)
mean_fc  = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.2)  # 80% interval
```

== Residual Diagnostics

A correctly specified ARIMA should produce *white noise residuals*. Check:

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt

residuals = result.resid

# 1. Plot residuals
residuals.plot(title="Residuals")

# 2. ACF of residuals — should be flat (all within confidence bands)
plot_acf(residuals, lags=40)

# 3. Ljung-Box test (null: residuals are white noise)
lb_result = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
print(lb_result)
# p-values > 0.05: fail to reject -> residuals look like white noise
# p-values < 0.05: reject -> remaining autocorrelation -> model misspecified

# 4. Q-Q plot (check Gaussian assumption)
from scipy import stats
stats.probplot(residuals, plot=plt)
```

== ARIMA Failure Modes

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Failure*], [*Symptom*], [*What to do*]),
  [Non-stationarity missed],
  [Residuals trend; ACF of residuals decays slowly],
  [Increase $d$, or log-transform first],
  [Seasonality unmodeled],
  [Spikes in residual ACF at lags $m, 2m$],
  [Switch to SARIMA],
  [Structural break],
  [Model fit degrades suddenly],
  [Detect break (Chow test), model separately or add intervention],
  [Heteroscedasticity],
  [Residual variance grows over time],
  [Log-transform (multiplicative noise) or fit ARCH/GARCH],
  [Long memory],
  [ACF decays as power law, not exponentially],
  [ARFIMA (fractional differencing)],
  [Non-linearity],
  [Residuals correlate with lagged squared residuals],
  [Threshold AR, GARCH, or switch to DL],
)

// ─── SECTION 5 ───────────────────────────────────────────────────────────────

= Exponential Smoothing and ETS

== The Core Idea

ETS stands for *Error, Trend, Seasonality*. It decomposes a time series into these components, each updated by a weighted average of new information and prior estimates. The weights decay exponentially into the past — hence "exponential smoothing."

The parameter $alpha$ controls how fast the weights decay:

$ "Effective weight on observation" t-k: quad alpha (1 - alpha)^k $

- $alpha approx 1$: almost all weight on the most recent observation (fast adaptation)
- $alpha approx 0$: weight spread broadly over many past observations (smooth, slow adaptation)

== Simple Exponential Smoothing (SES) — No Trend, No Season

$
  l_t &= alpha y_t + (1 - alpha) l_(t-1) \
  hat(y)_(t+h) &= l_t quad "(constant for all" h")"
$

The forecast is just the current level, which is a geometric weighted average of all past observations.

*State space form (additive error):*

$
  y_t &= l_(t-1) + epsilon_t \
  l_t &= l_(t-1) + alpha epsilon_t
$

The Kalman filter applied to this SSM recovers SES exactly.

== Holt's Linear Method — Trend, No Season

$
  l_t &= alpha y_t + (1 - alpha)(l_(t-1) + b_(t-1)) \
  b_t &= beta (l_t - l_(t-1)) + (1 - beta) b_(t-1) \
  hat(y)_(t+h) &= l_t + h dot b_t
$

- $beta$ controls how quickly the trend estimate adapts
- Forecast grows linearly — problematic for long horizons (trend may not persist)
- *Damped trend* (Taylor, 2003): multiply $h$ by $phi + phi^2 + dots + phi^h$ where $0 < phi < 1$ → trend eventually flattens. This is almost always better in practice.

== Holt-Winters — Trend + Seasonality

*Additive seasonality* (seasonal variation constant in magnitude):

$
  l_t &= alpha (y_t - s_(t-m)) + (1-alpha)(l_(t-1) + b_(t-1)) \
  b_t &= beta (l_t - l_(t-1)) + (1-beta) b_(t-1) \
  s_t &= gamma (y_t - l_(t-1) - b_(t-1)) + (1-gamma) s_(t-m) \
  hat(y)_(t+h) &= l_t + h dot b_t + s_(t + h - m(floor((h-1)\/m)+1))
$

*Multiplicative seasonality* (seasonal variation proportional to level):

$
  l_t &= alpha (y_t \/ s_(t-m)) + (1-alpha)(l_(t-1) + b_(t-1)) \
  s_t &= gamma (y_t \/ (l_(t-1) + b_(t-1))) + (1-gamma) s_(t-m) \
  hat(y)_(t+h) &= (l_t + h dot b_t) dot s_(t + h - m(floor((h-1)\/m)+1))
$

#note[
  *Multiplicative is correct* when the amplitude of seasonal fluctuations grows proportionally with the level of the series (e.g., retail sales with growing trend — the Christmas spike is bigger when average sales are higher).
]

== The Full ETS Taxonomy

ETS is parameterized as ETS(error, trend, season):

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Component*], [*Options*]),
  [Error], [A (additive), M (multiplicative)],
  [Trend], [N (none), A (additive), $A_d$ (additive damped)],
  [Season], [N (none), A (additive), M (multiplicative)],
)

That's $2 times 3 times 3 = 18$ possible models. `statsmodels` selects automatically via AIC.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    series,
    trend='add',           # 'add', 'mul', or None
    damped_trend=True,
    seasonal='add',        # 'add', 'mul', or None
    seasonal_periods=12,
)
result = model.fit(optimized=True)  # MLE for alpha, beta, gamma, phi

# Forecast
forecast = result.forecast(24)

# Component decomposition
result.level      # l_t sequence
result.trend      # b_t sequence
result.season     # s_t sequence
```

== ETS Failure Modes

#table(
  columns: (auto, auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Failure*], [*Symptom*], [*What to do*]),
  [Multiple seasonalities],
  [Weekly + annual pattern not captured],
  [TBATS, or seasonal decomposition + residual modeling],
  [Very long seasonal period],
  [$m=52$ (weekly) → too many seasonal parameters],
  [Fourier terms as regressors],
  [Level shifts],
  [Sudden permanent change in mean],
  [Detect and re-initialize],
  [Non-Gaussian errors],
  [Multiplicative errors → log-normal intervals],
  [Use ETS(M,\*,\*) or log-transform],
)

// ─── SECTION 6 ───────────────────────────────────────────────────────────────

= Model Selection and Comparison Framework

== Metrics

*MASE (Mean Absolute Scaled Error)* — preferred:

$ "MASE" = "MAE"_"model" / "MAE"_"naive" $

where naive = seasonal naive ($hat(y)_(t+h) = y_(t+h-m)$). Scale-free, comparable across series, robust to scale and units. MASE $< 1$ means the model beats the naive.

*sMAPE* — used in M4 competition; be aware:

$ "sMAPE" = frac(200, n) sum abs(y_t - hat(y)_t) / (abs(y_t) + abs(hat(y)_t)) $

Bounded in $[0, 200%]$ but still has pathologies near zero. MASE is generally preferred.

*CRPS* — for probabilistic evaluation:

$ "CRPS"(F, y) = integral_(-oo)^(oo) [F(z) - bb(1)(z >= y)]^2 d z $

Proper scoring rule; rewards calibration and sharpness jointly. For Gaussian predictive distribution $cal(N)(mu, sigma)$:

$ "CRPS" = sigma [z dot Phi(z) + phi.alt(z) - 1\/2], quad z = (y - mu) / sigma $

== Backtesting (Walk-Forward Validation)

*Never* split a time series with random sampling. Use expanding-window walk-forward:

```python
def walk_forward_eval(model_fn, series, n_splits=5, horizon=12, gap=0):
    """
    model_fn: function(train_series) -> fitted model with .forecast(h) method
    gap: number of steps to drop between train end and test start
         (prevents leakage through autocorrelation)
    """
    n = len(series)
    min_train = n // (n_splits + 1)
    step = (n - min_train - horizon - gap) // n_splits

    results = []
    for k in range(n_splits):
        train_end = min_train + k * step
        test_start = train_end + gap
        test_end = test_start + horizon

        if test_end > n:
            break

        train = series[:train_end]
        test  = series[test_start:test_end]

        model = model_fn(train)
        forecast = model.forecast(horizon)

        mae  = np.mean(np.abs(test.values - forecast.values))
        mase = mae / np.mean(np.abs(np.diff(train.values)))  # naive MAE approx
        results.append({'fold': k, 'mae': mae, 'mase': mase})

    return pd.DataFrame(results)
```

#note[
  *Gap matters:* If your series has autocorrelation at lag 1–6 and your test window starts immediately after training, the model exploits that autocorrelation. A gap of `horizon` steps prevents this and simulates true production latency.
]

// ─── SECTION 7 ───────────────────────────────────────────────────────────────

= Classical vs. DL: When to Use Each

== Classical Wins When

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Condition*], [*Why classical is sufficient*]),
  [Short series ($n < 200$)],
  [DL has too many parameters; overfits],
  [Single series, no covariates],
  [DL's cross-series learning provides no benefit],
  [Strong, regular seasonality],
  [ETS handles it analytically and exactly],
  [Short horizon ($h <= 3$)],
  [DL advantage grows at longer horizons],
  [Interpretability required],
  [ARIMA coefficients and ETS components are auditable],
  [Fast production inference],
  [Kalman filter is $O(n dot d^3)$; no GPU required],
  [Cold start (new series, no data)],
  [ETS can be initialized with very few observations],
)

== DL Wins When

#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Condition*], [*Why DL is needed*]),
  [Large panel (many series)],
  [DL learns shared patterns across series],
  [Rich covariates (prices, calendar, promotions)],
  [ARIMA can't incorporate these natively],
  [Long horizon ($h > 12$)],
  [DL captures longer-range dependencies],
  [Non-linear dynamics],
  [ARIMA is linear by construction],
  [Multiple interacting frequencies],
  [DL handles this implicitly],
  [Distribution has heavy tails or is multimodal],
  [Parametric Gaussian assumption breaks],
)

#insight[
  For a new forecasting problem, start with a strong classical baseline — ARIMA or ETS depending on seasonality structure — before touching DL. It's fast, interpretable, and often surprisingly competitive. The classical model reveals the series structure (trend, seasonality, noise level) that informs DL architecture choices. And if the classical model already beats your DL model, that's a sign you have a data or training problem, not a model problem.
]

// ─── SECTION 8 ───────────────────────────────────────────────────────────────

= Week 1 Practice Exercises

*Dataset: M4 Competition*

```python
from datasetsforecast.m4 import M4, M4Info, M4Evaluation

# Load a subset
train, test = M4.load(directory='data/m4', group='Monthly')
# train: DataFrame with columns [unique_id, ds, y]
# test:  same structure, last H observations per series
```

== Exercise 1: Stationarity Analysis (Day 1)

Pick 5 series from the monthly group (different industries). For each:
1. Plot the raw series
2. Run ADF test on raw series
3. Apply log transform if variance is non-constant
4. Difference once ($nabla y_t$)
5. Re-run ADF; plot ACF and PACF of the differenced series
6. Identify $p$ and $q$ from the plots

#note[
  Some series need $d=1$, some need seasonal differencing, some are already stationary. This variance is what `auto_arima` handles — but you need to be able to do it by hand to know what the automation is doing.
]

== Exercise 2: Fitting ARIMA and ETS (Days 2–3)

For the same 5 series:

```python
from pmdarima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing

results = {}
for uid in selected_ids:
    s = train[train['unique_id'] == uid].set_index('ds')['y']
    h = len(test[test['unique_id'] == uid])

    # ARIMA
    arima = auto_arima(s, seasonal=True, m=12, information_criterion='aic',
                       stepwise=True, suppress_warnings=True)
    arima_fc = arima.predict(h)

    # ETS
    ets = ExponentialSmoothing(s, trend='add', damped_trend=True,
                               seasonal='add', seasonal_periods=12).fit()
    ets_fc = ets.forecast(h)

    actuals = test[test['unique_id'] == uid]['y'].values

    mase_arima = np.mean(np.abs(actuals - arima_fc)) / np.mean(np.abs(np.diff(s.values)))
    mase_ets   = np.mean(np.abs(actuals - ets_fc))   / np.mean(np.abs(np.diff(s.values)))

    results[uid] = {'ARIMA_MASE': mase_arima, 'ETS_MASE': mase_ets,
                    'ARIMA_order': arima.order, 'ARIMA_seasonal': arima.seasonal_order}
```

== Exercise 3: Residual Diagnostics (Day 3)

For each fitted ARIMA:
1. Plot residuals over time — are there obvious patterns?
2. Plot ACF of residuals — any significant spikes?
3. Run Ljung-Box test at lags 10 and 20
4. Plot histogram of residuals — does it look Gaussian?
5. Note any series where residuals look bad and hypothesize why

== Exercise 4: Failure Hunting (Days 4–5)

Deliberately find the failure modes:
1. Find a series with *multiple seasonalities* — fit SARIMA and watch it fail
2. Find a series with a *structural break* — fit ARIMA across the break and examine the residuals
3. Find a series with *growing variance* — fit ARIMA without log-transform and note the heteroscedastic residuals, then fix it

*What to write down (for your notes / interview prep):* After each exercise, answer:

- What order did `auto_arima` select, and does it match what you'd have guessed from ACF/PACF?
- Which series was hardest to model, and why?
- Did ARIMA or ETS win more often? What feature of the series predicted which would win?
- What residual patterns remain? What would you try next?

// ─── SECTION 9 ───────────────────────────────────────────────────────────────

= Key Concepts for Interview Fluency

#table(
  columns: (auto, 1fr),
  stroke: 0.5pt,
  inset: 8pt,
  align: left,
  table.header([*Term*], [*Precise definition*]),
  [Stationarity],
  [Time-invariant first and second moments],
  [Unit root],
  [AR coefficient $phi=1$; shocks are permanent (random walk)],
  [$d$ in ARIMA],
  [Number of times differenced to achieve stationarity],
  [ACF],
  [Correlation between $y_t$ and $y_(t-k)$ (total, including indirect paths)],
  [PACF],
  [Correlation between $y_t$ and $y_(t-k)$ after removing effect of lags $1 dots k-1$],
  [Information criterion],
  [Model selection statistic trading off fit (log-likelihood) vs. complexity (parameters)],
  [Ljung-Box test],
  [Portmanteau test for autocorrelation in residuals],
  [MASE],
  [MAE normalized by in-sample seasonal naive MAE; scale-free; MASE $< 1$ beats naive],
  [CRPS],
  [Proper scoring rule for predictive distributions; penalizes both calibration and sharpness],
  [Kalman gain],
  [The update weight in the Kalman filter; analogous to $alpha$ in exponential smoothing],
)

*"When would you use ARIMA over ETS?"*

#note[ARIMA is more flexible when the autocorrelation structure is complex (high $p$, non-trivial $q$); ETS is more robust when the series has a clear level+trend+season decomposition and you want guaranteed positive forecasts with multiplicative models. In practice, run both and compare via cross-validation.]

*"Why is MASE preferred over MAPE?"*

#note[MAPE is undefined when $y_t = 0$, asymmetric (penalizes over-forecasting more than under-forecasting), and scale-dependent in ways that make cross-series comparison misleading. MASE normalizes by the in-sample one-step naive error, making it scale-free and meaningful across series with different units and scales.]

*"What does a unit root mean economically?"*

#note[Shocks are permanent. In a stationary AR(1) process, a shock to $y_t$ decays geometrically and the series mean-reverts. With a unit root, a shock shifts the level permanently — the series "remembers" every shock forever. GDP and asset prices are often modeled as unit root processes (random walks with drift).]

*"Why can't you use ARIMA for multiple seasonalities?"*

#note[ARIMA's seasonal component handles one fixed period. If your series has both weekly and annual seasonality (as electricity demand does), you'd need two seasonal difference operators and seasonal AR/MA terms at both periods — the model complexity explodes and estimation becomes unstable. TBATS or DL are the correct tools.]

*"What's the connection between exponential smoothing and the Kalman filter?"*

#note[Simple exponential smoothing is the Kalman filter applied to a specific SSM: the state is a single latent level, the transition is identity, and the Kalman gain simplifies to a constant $alpha$. More complex ETS models correspond to SSMs with richer state vectors. This means ETS parameters can be estimated via the Kalman likelihood — giving exact MLE rather than heuristic grid search.]

// ─── SECTION 10 ──────────────────────────────────────────────────────────────

= Summary: Mental Model for Phase 1

```
Raw series
    |
    +--- Trend? Variance non-constant?
    |         |
    |    Log transform -> difference -> ADF -> stationary series
    |
    +--- ACF / PACF
    |         |
    |    AR(p): PACF cuts off at p
    |    MA(q): ACF cuts off at q
    |    ARMA: both decay geometrically
    |    Seasonal spike at lag m: add SARIMA seasonal terms
    |
    +--- Fit ARIMA (AIC/BIC selection)
    |         |
    |    Check: residuals white noise? (Ljung-Box, ACF plot)
    |    If not: increase p/q, add seasonal terms, or transform
    |
    +--- Fit ETS (auto-select from 18 models via AIC)
    |         |
    |    Additive or multiplicative seasonality?
    |    Damped trend? (almost always yes)
    |
    +--- Walk-forward backtest (5 folds, horizon = production horizon)
    |         |
    |    Metrics: MASE (point), CRPS (probabilistic)
    |
    +--- Residual failure modes -> inform Phase 2 DL architecture choices
              |
         Non-linearity, multiple seasonalities, covariates -> need DL
         Strong regular seasonality, short series -> classical may suffice
```

#line(length: 100%)

_Next: Phase 2 — Core Deep Learning Approaches (LSTM, TCN, TFT)_

# %% Imports
import warnings

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from time_series.load_data import Freq, Split, load_data

warnings.filterwarnings("ignore", category=FutureWarning)
# KPSS p-value saturates at 0.01 when the stat is off its lookup table — not a code problem.
warnings.filterwarnings("ignore", message="The test statistic is outside of the range")

# %% Load hourly data
df_train = load_data(Freq.HOURLY, Split.TRAIN)
df_test = load_data(Freq.HOURLY, Split.TEST)

print(f"Train: {df_train.shape[0]} series, up to {df_train.shape[1] - 1} time steps")
print(f"Test:  {df_test.shape[0]} series, {df_test.shape[1] - 1} forecast steps (horizon=48)")
print(df_train.head(3))

# %% Extract sample series


SAMPLE_IDS = ["H1", "H2", "H3", "H5", "H10"]


def extract_series(df: pl.DataFrame, m4id: str) -> np.ndarray:
    """Return the non-null observations for one series as a float32 array."""
    row = df.filter(pl.col("m4id") == m4id)
    values = row.select(pl.col("^t_.*$")).row(0)
    arr = np.array(values, dtype=np.float32)
    return arr[~np.isnan(arr)]


series_dict: dict[str, np.ndarray] = {sid: extract_series(df_train, sid) for sid in SAMPLE_IDS}

for sid, s in series_dict.items():
    print(f"{sid}: length={len(s)}, mean={s.mean():.1f}, std={s.std():.1f}")

# %% Plot raw series
fig = make_subplots(rows=len(SAMPLE_IDS), cols=1, shared_xaxes=False, subplot_titles=SAMPLE_IDS)
for i, (sid, s) in enumerate(series_dict.items(), start=1):
    fig.add_trace(go.Scatter(y=s, mode="lines", line=dict(width=0.8), name=sid), row=i, col=1)
fig.update_layout(
    title="M4 Hourly — raw series (sample)",
    height=900,
    showlegend=False,
)
fig.show()

# %% ADF + KPSS stationarity tests

# Interpretation guide:
#   ADF  p < 0.05  → reject unit root  → stationary signal
#   KPSS p < 0.05  → reject stationarity → non-stationary signal
#
#   Both agree stationary  → STATIONARY
#   Both agree non-stat    → NON-STATIONARY → difference
#   Disagree               → AMBIGUOUS → difference to be safe


def stationarity_report(name: str, s: np.ndarray) -> None:
    """Print ADF and KPSS results for a series."""
    adf_stat, adf_p, adf_lags, *_ = adfuller(s, autolag="AIC")
    kpss_stat, kpss_p, kpss_lags, _ = kpss(s, regression="c", nlags="auto")

    adf_verdict = "stationary" if adf_p < 0.05 else "unit root"
    kpss_verdict = "stationary" if kpss_p >= 0.05 else "non-stationary"

    if adf_verdict == "stationary" and kpss_verdict == "stationary":
        conclusion = "STATIONARY"
    elif adf_verdict == "unit root" and kpss_verdict == "non-stationary":
        conclusion = "NON-STATIONARY → difference"
    else:
        conclusion = "AMBIGUOUS → difference to be safe"

    print(
        f"{name:>4}  ADF p={adf_p:.3f} ({adf_verdict:<10})  "
        f"KPSS p={kpss_p:.3f} ({kpss_verdict:<15})  → {conclusion}"
    )


print("─" * 85)
print(f"{'ID':>4}  {'ADF':^30}  {'KPSS':^30}  Conclusion")
print("─" * 85)
for sid, s in series_dict.items():
    stationarity_report(sid, s)
print("─" * 85)

# %% Log-transform check (variance stabilisation)
# Positive correlation between rolling mean and rolling std → multiplicative noise → log-transform.

WINDOW = 24  # one day

fig = make_subplots(rows=len(SAMPLE_IDS), cols=1, subplot_titles=SAMPLE_IDS)
for i, (sid, s) in enumerate(series_dict.items(), start=1):
    roll_mean = np.convolve(s, np.ones(WINDOW) / WINDOW, mode="valid")
    roll_std = np.array([s[j : j + WINDOW].std() for j in range(len(s) - WINDOW + 1)])
    fig.add_trace(
        go.Scatter(
            x=roll_mean.tolist(),
            y=roll_std.tolist(),
            mode="markers",
            marker=dict(size=3, opacity=0.5),
            name=sid,
        ),
        row=i,
        col=1,
    )
    fig.update_xaxes(title_text="rolling mean (24h)", row=i, col=1)
    fig.update_yaxes(title_text="rolling std", row=i, col=1)
fig.update_layout(
    title="Variance–mean relationship (positive slope → log-transform)",
    height=1000,
    showlegend=False,
)
fig.show()

# %% Seasonal differencing (lag 24) — the right operator for hourly data
# Regular diff (y_t - y_{t-1}) removes linear trend.
# Seasonal diff (y_t - y_{t-24}) removes the daily cycle directly.
# For series whose non-stationarity is seasonal rather than trend-based, this is what we want.

M = 24  # daily seasonal period


def seasonal_diff(s: np.ndarray, m: int = M) -> np.ndarray:
    """Return s[m:] - s[:-m]  (one seasonal difference at period m)."""
    return s[m:] - s[:-m]


print("Raw series:")
print("─" * 85)
for sid, s in series_dict.items():
    stationarity_report(sid, s)
print("─" * 85)

print("\nAfter seasonal differencing (lag 24):")
print("─" * 85)
for sid, s in series_dict.items():
    stationarity_report(sid, seasonal_diff(s))
print("─" * 85)

print("\nAfter seasonal diff + regular diff (in case any trend remains):")
print("─" * 85)
for sid, s in series_dict.items():
    stationarity_report(sid, np.diff(seasonal_diff(s)))
print("─" * 85)
# %% ACF / PACF


def plot_acf_pacf(sid: str, m: int = 0, nlags_acf: int = 200, nlags_pacf: int = 48) -> None:
    """Plot raw series, transformed series, ACF, and PACF in a 2×2 grid.

    Parameters
    ----------
    sid : str
        Series ID (must be a key in ``series_dict``).
    m : int
        Seasonal period for differencing. ``0`` → no differencing (use raw series).
    nlags_acf : int
        Number of lags for the ACF plot.
    nlags_pacf : int
        Number of lags for the PACF plot.
    """
    s = series_dict[sid]
    s_transformed = seasonal_diff(s, m) if m > 0 else s
    transform_label = f"seasonal diff (lag {m})" if m > 0 else "raw"

    n = len(s_transformed)
    conf = 1.96 / np.sqrt(n)

    acf_vals: np.ndarray = np.asarray(acf(s_transformed, nlags=nlags_acf, fft=True))
    pacf_vals: np.ndarray = np.asarray(pacf(s_transformed, nlags=nlags_pacf))
    lags_acf = np.arange(len(acf_vals))
    lags_pacf = np.arange(len(pacf_vals))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"{sid} raw",
            f"{sid} {transform_label}",
            f"ACF — {sid} {transform_label} (lags 0–{nlags_acf})",
            f"PACF — {sid} {transform_label} (lags 0–{nlags_pacf})",
        ],
    )

    fig.add_trace(go.Scatter(y=s.tolist(), mode="lines", line=dict(width=0.7), name="raw"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=s_transformed.tolist(), mode="lines", line=dict(width=0.7),
                             name=transform_label), row=1, col=2)

    for lag, val in zip(lags_acf, acf_vals):
        fig.add_shape(type="line", x0=lag, x1=lag, y0=0, y1=float(val),
                      line=dict(color="steelblue", width=1.2), row=2, col=1)
    fig.add_trace(go.Scatter(x=lags_acf.tolist(), y=acf_vals.tolist(), mode="markers",
                             marker=dict(size=3, color="steelblue"), showlegend=False), row=2, col=1)

    for lag, val in zip(lags_pacf, pacf_vals):
        fig.add_shape(type="line", x0=lag, x1=lag, y0=0, y1=float(val),
                      line=dict(color="steelblue", width=1.2), row=2, col=2)
    fig.add_trace(go.Scatter(x=lags_pacf.tolist(), y=pacf_vals.tolist(), mode="markers",
                             marker=dict(size=3, color="steelblue"), showlegend=False), row=2, col=2)

    for xref, yref in (("x3 domain", "y3"), ("x4 domain", "y4")):
        for sign in (1, -1):
            fig.add_shape(type="line", x0=0, x1=1, y0=sign * conf, y1=sign * conf,
                          xref=xref, yref=yref,
                          line=dict(color="crimson", dash="dash", width=1))

    fig.update_layout(
        title=f"{sid} — {transform_label} | spikes expected at lag 24 (daily) and 168 (weekly)",
        height=700,
        showlegend=False,
    )
    fig.show()


# %% Raw H1 — seasonality clearly visible in ACF
plot_acf_pacf("H1", m=0)

# %% H1 after seasonal differencing — daily cycle removed; check what remains
plot_acf_pacf("H1", m=24)

# %%

"""M4 competition dataset loader.

Each frequency comes as a pair of wide-format CSV files: one for training and
one for the forecast horizon (test).  Rows are individual time series; columns
are time steps.  Series of different lengths are right-padded with nulls.

Reference: Makridakis, Spiliotis & Assimakopoulos (2020),
    "The M4 Competition: 100,000 time series and 61 forecasting methods."
"""
# %%
from enum import StrEnum
from pathlib import Path

import inflection
import polars as pl

from beyond_backprop.constants import DATA_PATH

_M4_DIR = DATA_PATH / "m4"


class Freq(StrEnum):
    """Temporal frequency of an M4 time series."""

    YEARLY = "Yearly"
    QUARTERLY = "Quarterly"
    MONTHLY = "Monthly"
    WEEKLY = "Weekly"
    DAILY = "Daily"
    HOURLY = "Hourly"


class Split(StrEnum):
    """Dataset split."""

    TRAIN = "train"
    TEST = "test"


def load_data(freq: Freq, split: Split) -> pl.DataFrame:
    """Load one frequency/split pair from the M4 dataset.

    Returns a wide-format DataFrame where each row is a single time series.
    Shorter series are right-padded with nulls.

    Parameters
    ----------
    freq : Freq
        Temporal frequency (e.g. ``Freq.DAILY``).
    split : Split
        Dataset split (``Split.TRAIN`` or ``Split.TEST``).

    Returns
    -------
    pl.DataFrame
        Columns: ``m4id`` (series identifier) followed by ``t_1``, ``t_2``, …,
        ``t_N`` (Float32 time-step values).
    """
    path = _M4_DIR / f"{freq}-{split}.csv"
    df = pl.read_csv(path, infer_schema_length=0, null_values=[""])

    n_value_cols = df.width - 1
    rename_map = {"V1": "m4id"} | {f"V{i + 2}": f"t_{i + 1}" for i in range(n_value_cols)}
    df = df.rename(rename_map)

    value_cols = [f"t_{i + 1}" for i in range(n_value_cols)]
    return df.with_columns(pl.col(c).cast(pl.Float32) for c in value_cols)


def load_info() -> pl.DataFrame:
    """Load M4 series metadata.

    Parameters
    ----------
    None

    Returns
    -------
    pl.DataFrame
        Columns (snake_case): ``m4id``, ``category``, ``frequency``,
        ``horizon``, ``sp``, ``starting_date``.
    """
    df = pl.read_csv(_M4_DIR / "m4_info.csv")
    return df.rename({col: inflection.underscore(col) for col in df.columns})

# %%
if __name__ == "__main__":
    df = load_data(Freq.WEEKLY, Split.TRAIN)
    print(df.head())
    info = load_info()
    print(info.head())
# %%

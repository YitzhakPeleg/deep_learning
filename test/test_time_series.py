import pointblank as pb
import polars as pl
import pytest

from time_series.load_data import Freq, Split, load_data, load_info
from time_series.validate import validate_data, validate_info


def test_daily_train_shape() -> None:
    """Daily train split has 4 227 series."""
    df = load_data(Freq.DAILY, Split.TRAIN)
    assert df.shape[0] == 4_227
    assert df["m4id"][0] == "D1"


def test_daily_test_columns() -> None:
    """Daily test split has m4id + 14 forecast-step columns in snake_case."""
    df = load_data(Freq.DAILY, Split.TEST)
    assert df.columns[0] == "m4id"
    assert df.columns[1] == "t_1"
    assert df.shape[1] == 15  # m4id + 14 forecast steps


def test_value_columns_float32() -> None:
    """Value columns are cast to Float32."""
    df = load_data(Freq.DAILY, Split.TEST)
    assert df["t_1"].dtype == pl.Float32


def test_info_columns() -> None:
    """load_info returns exactly the expected snake_case columns."""
    info = load_info()
    assert info.columns == ["m4id", "category", "frequency", "horizon", "sp", "starting_date"]
    assert info.shape[0] == 100_000


def test_info_daily_horizon() -> None:
    """All Daily series have a forecast horizon of 14."""
    info = load_info()
    horizons = info.filter(pl.col("m4id").str.starts_with("D"))["horizon"].unique().to_list()
    assert horizons == [14]


@pytest.mark.parametrize("freq", list(Freq))
def test_all_frequencies_load(freq: Freq) -> None:
    """Every frequency loads without error for both splits."""
    for split in Split:
        df = load_data(freq, split)
        assert df.shape[0] > 0
        assert df.columns[0] == "m4id"


# --- validation ---


def test_validate_data_passes() -> None:
    """Daily train validation has zero failures across all steps."""
    v = validate_data(Freq.DAILY, Split.TRAIN)
    assert isinstance(v, pb.Validate)
    assert v.all_passed()


def test_validate_info_passes() -> None:
    """Info validation has zero failures across all steps."""
    v = validate_info()
    assert isinstance(v, pb.Validate)
    assert v.all_passed()


@pytest.mark.parametrize("freq", list(Freq))
def test_validate_all_frequencies(freq: Freq) -> None:
    """Validation passes for every frequency (train split)."""
    v = validate_data(freq, Split.TRAIN)
    assert v.all_passed()

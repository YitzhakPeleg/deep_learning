"""Data-quality validation for the M4 dataset using pointblank."""

from types import MappingProxyType

import pointblank as pb

from time_series.load_data import Freq, Split, load_data, load_info

_ID_PREFIX: MappingProxyType[Freq, str] = MappingProxyType(
    {
        Freq.YEARLY: "Y",
        Freq.QUARTERLY: "Q",
        Freq.MONTHLY: "M",
        Freq.WEEKLY: "W",
        Freq.DAILY: "D",
        Freq.HOURLY: "H",
    }
)

_CATEGORIES: list[str] = [
    "Demographic",
    "Finance",
    "Industry",
    "Macro",
    "Micro",
    "Other",
]
_SP_VALUES: list[str] = ["Daily", "Hourly", "Monthly", "Quarterly", "Weekly", "Yearly"]
_FREQUENCY_VALUES: list[int] = [1, 4, 12, 24]


def validate_data(freq: Freq, split: Split) -> pb.Validate:
    """Validate the structure and content of one M4 frequency/split file.

    Parameters
    ----------
    freq : Freq
        Temporal frequency (e.g. ``Freq.DAILY``).
    split : Split
        Dataset split (``Split.TRAIN`` or ``Split.TEST``).

    Returns
    -------
    pb.Validate
        Interrogated validation object; inspect with ``.get_sundered_data()``
        or render with ``.get_tabular_report()``.
    """
    df = load_data(freq, split)
    prefix = _ID_PREFIX[freq]

    return (
        pb.Validate(data=df, label=f"M4 {freq} {split}")
        .col_vals_not_null(columns="m4id")
        .rows_distinct(columns_subset=["m4id"])
        .col_vals_regex(columns="m4id", pattern=rf"^{prefix}\d+$")
        .col_vals_not_null(columns="t_1")
        .interrogate()
    )


def validate_info() -> pb.Validate:
    """Validate the M4 series metadata file.

    Returns
    -------
    pb.Validate
        Interrogated validation object.
    """
    df = load_info()

    return (
        pb.Validate(data=df, label="M4 info")
        .col_vals_not_null(columns=["m4id", "category", "horizon"])
        .rows_distinct(columns_subset=["m4id"])
        .col_vals_gt(columns="horizon", value=0)
        .col_vals_in_set(columns="category", set=_CATEGORIES)
        .col_vals_in_set(columns="sp", set=_SP_VALUES)
        .col_vals_in_set(columns="frequency", set=_FREQUENCY_VALUES)
        .row_count_match(count=100_000)
        .interrogate()
    )


if __name__ == "__main__":
    for freq in Freq:
        for split in Split:
            validate_data(freq, split).get_tabular_report().show()

    validate_info().get_tabular_report().show()

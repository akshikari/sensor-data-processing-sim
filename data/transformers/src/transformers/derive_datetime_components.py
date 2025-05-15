import logging
from typing import Any, Callable
import pandas as pd

logger = logging.getLogger(__name__)

# class DatetimeParts(Enum):
possible_components = set(
    ["year", "month", "day", "hour", "minute", "second", "microsecond", "nanosecond"]
)


def derive_datetime_components(
    df: pd.DataFrame,
    timestamp_column_id: str,
    datetime_parts: list[str],
) -> pd.DataFrame:
    """
    Derives specified datetime parts from the specified datetime column in a Pandas DataFrame

    :param df: The Pandas DataFrame from which to derive the date info from.
    :param timestamp_column_id: The name of the column in the DataFrame that contains the target timestamp.
    :param datetime_parts: The list of components to pull from the timetamps e.g. ["year", "month", "day", "hour"]
    :returns: The original DataFrame modified to include the datetime component two-digit string values as new columns.
    :raise ValueError: If inputs are invalid
    :raises TypeError: If the timestamp column cannot be converted to datetime
    """
    # Validations
    if df.empty:
        logger.info("No data in incoming DataFrame. Returning DataFrame as is.")
        return df
    if timestamp_column_id not in df.columns:
        raise ValueError(
            f"Provided timestamp column {timestamp_column_id} not found in incoming DataFrame."
        )
    if not datetime_parts or (
        datetime_parts
        and any([part for part in datetime_parts if part not in possible_components])
    ):
        raise ValueError(
            f"Invalid datetime component(s) provided. datetime_parts: {datetime_parts}\nSupported parts are: {possible_components}"
        )

    # Lambdas to map to dataframe to extract datetime components
    component_extractors: dict[str, Callable[[pd.Series[Any]], pd.Series[str]]] = {
        "year": lambda ts: ts.dt.year.astype(str),
        "month": lambda ts: ts.dt.month.map("{:02d}".format),
        "day": lambda ts: ts.dt.day.map("{:02d}".format),
        "hour": lambda ts: ts.dt.hour.map("{:02d}".format),
        "minute": lambda ts: ts.dt.minute.map("{:02d}".format),
        "second": lambda ts: ts.dt.second.map("{:02d}".format),
        "microsecond": lambda ts: ts.dt.microsecond.map("{:02d}".format),
        "nanosecond": lambda ts: ts.dt.nanosecond.map("{:02d}".format),
    }

    # Extract components
    df_transformed = df.copy()
    for part in datetime_parts:
        if part in component_extractors:
            try:
                df_transformed[part] = component_extractors[part](
                    df_transformed[timestamp_column_id]
                )
            except Exception as err:
                raise RuntimeError(
                    f"Failed to derive datetime part {part} due to the following error:\n{err}"
                )

    return df_transformed

import pandas as pd

def _handle_dataframe_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the input DataFrame and handle common errors.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.
    ValueError
        If the DataFrame is None or empty.
    """
    if df is None:
        raise ValueError("The DataFrame cannot be None.")

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The DataFrame cannot be empty.")

    return df
    
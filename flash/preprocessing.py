from typing import Literal
import pandas as pd

def extract_features(
        df: pd.DataFrame,
        var_type: Literal['num', 'cat', 'other', 'all'],
        ignore_cols: list[str] | None = None,
        unique_value_threshold: int = 12
        ) -> list[str] | tuple[list[str], list[str], list[str]]:
        """Extracts features based on their type.

        Parameters
        ----------
        df : pd.DataFrame
            A Pandas DataFrame.
        var_type : {'num', 'cat', 'other', 'all'}
            The type of the feature to extract.
        ignore_cols : list[str], default=None
            Features to exclude from extraction.
        unique_value_threshold : int, default=12
            The threshold below which numerical features are considered categorical.
            If a numerical feature has fewer unique values than this threshold,
            it will be treated as categorical.

        Returns
        -------
        list[str] or tuple[list[str], list[str], list[str]]
            Feature names based on the requested type.

        Raises
        ------
        ValueError
            If `var_type` is not in {'num', 'cat', 'other', 'all'}.
        TypeError
            If `ignore_cols` is not a list of strings or None.
        """

        # Validate inputs
        if var_type not in ['num', 'cat', 'other', 'all']:
            raise ValueError("The 'var_type' parameter must be 'num', 'cat', 'other', or 'all'.")
        if ignore_cols and not isinstance(ignore_cols, list):
            raise TypeError("ignore_cols must be a list of strings.")

        # Prepare DataFrame
        df = df.copy()
        if ignore_cols:
            df = df.drop(columns=ignore_cols)

        # Identify feature types
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['bool', 'object', 'category']).columns.tolist()
        cat_cols += [col for col in num_cols if df[col].nunique() <= unique_value_threshold]
        num_cols = [col for col in num_cols if col not in cat_cols]
        other_cols = [col for col in df.columns if col not in set(num_cols + cat_cols)]

        # Return based on `var_type`
        type_map = {
            'num': num_cols,
            'cat': cat_cols,
            'other': other_cols,
            'all': (num_cols, cat_cols, other_cols)
        }
        return type_map[var_type]
        
def calc_nan_values(df: pd.DataFrame, pct: bool = True) -> pd.Series:
    """Filters out features with missing values from the DataFrame and calculates the
    number of missing values or their percentage.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas DataFrame.
    pct : bool, default=True
        Whether to return missing values as a percentage (True) or as absolute counts (False).

    Returns
    -------
    pd.Series
        A Series indexed by feature names with either the count or percentage of missing values.
    """

    # Count the number of missing values in features with missing values
    missing_values = df.isna().sum().loc[lambda x: x > 0]

    # Return percentage or count of missing values
    return (missing_values / df.shape[0] * 100).round(2) if pct else missing_values
  
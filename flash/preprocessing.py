from typing import Literal
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    FunctionTransformer, PowerTransformer, QuantileTransformer, OneHotEncoder, LabelEncoder
)

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

def feature_transform(
    df_num: pd.DataFrame | pd.Series,
    transformers: dict | None = None,
    epsilon: float = 1e-10
    ):
    """Transforms numerical features.

    Parameters
    ----------
    df_num : pd.DataFrame or pd.Series
        A Pandas DataFrame or a Pandas Series containing numerical features. If `df_num`
        is a Pandas Series, it is converted to Pandas DataFrame
    transformers : dict, default=None
        A dictionary containing transformers. If None, the following default transformers
        are applied:
        - Log transformation
        - Square transformation
        - Square Root transformation
        - Reciprocal transformation
        - Quantile transformation
        - Yeo-Johnson transformation.
    epsilon : float, default=1e-10
         A small value added to avoid issues with zero or negative values for certain
         transformations.

    Returns
    -------
    transformed_data : dict[str, pd.DataFrame]
       A dictionary where keys are the names of the transformations and values are the
       corresponding transformed features.
    """

    if isinstance(df_num, pd.Series):
        df_num = pd.DataFrame(df_num, columns=[df_num.name])

    cols = df_num.columns
    if transformers is None:
        transformers = {
            'Log': FunctionTransformer(func=lambda X: np.log(X + epsilon), validate=False),
            'Square': FunctionTransformer(func=np.square, validate=False),
            'Square Root': FunctionTransformer(func=lambda X: np.sqrt(X + epsilon), validate=False),
            'Reciprocal': FunctionTransformer(func=lambda X: np.reciprocal(X + epsilon), validate=False),
            'Quantile': QuantileTransformer(n_quantiles=df_num.shape[0], output_distribution='normal'),
            'Yeo-Johnson': PowerTransformer(standardize=False)
        }

    transformed_data = {}

    # Apply each transformer and store the result in the dictionary
    for name, transformer in transformers.items():
        transformed_df = pd.DataFrame(transformer.fit_transform(df_num), columns=cols)
        transformed_data[name] = transformed_df

    return transformed_data

def basic_imputer(
    x: pd.Series,
    var_type: Literal['num', 'cat'],
    method: Literal['mean', 'median', 'mode', 'ffill', 'bfill'] | None = None,
    fallback: Literal['mean', 'median', 'mode', 'ffill', 'bfill'] | None = None
    ) -> pd.Series:
    """Imputes missing values using basic statistical measures.

    Parameters
    ----------
    x : pd.Series
        The Series in which to impute missing values.
    var_type : {'num', 'cat'}
        Variable type of the Series. 'num' for numerical, 'cat' for categorical.
    method : {'mean', 'median', 'mode', 'ffill', 'bfill'}, default=None
        The method to impute missing values. If None, the default method for the given `var_type` is used.
    fallback : {'mean', 'median', 'mode', 'ffill', 'bfill'}, default=None
        The fallback imputation strategy if the `method` fails to impute all missing values. If None, the default fallback for the given `var_type` is used.

    Returns
    -------
    x : pd.Series
        The Series with missing values imputed.

    Raises
    ------
    ValueError
        If `var_type` is not 'num' or 'cat'.
        If an invalid method or fallback is specified.
    """

    def _impute(x, method):
        """Performs the imputation based on the selected method."""
        if method == 'mean':
            x.fillna(x.mean(), inplace=True)
        elif method == 'median':
            x.fillna(x.median(), inplace=True)
        elif method == 'mode':
            x.fillna(x.mode()[0], inplace=True)
        elif method == 'ffill':
            x.ffill(inplace=True)
        elif method == 'bfill':
            x.bfill(inplace=True)
        else:
            raise ValueError(f"Invalid option {method}. It should be either 'mean', 'median', 'mode', 'ffill', or 'bfill'")
        return x

    # Default method and fallback based on var_type. Also validate var_type
    if var_type == 'num':
        method = method or 'mean'
        fallback = fallback or 'mean'
    elif var_type == 'cat':
        method = method or 'mode'
        fallback = fallback or 'mode'
    else:
        raise ValueError(f"Invalid var_type {var_type}. It should be either 'num' or 'cat'.")

    x = x.copy()  # Avoid modifying the original series
    x = _impute(x, method)  # Primary imputation

    # Fallback imputation if there are still missing values
    if x.isna().any():
        x = _impute(x, fallback)

    return x

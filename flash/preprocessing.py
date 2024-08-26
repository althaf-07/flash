from typing import List, Optional, Union, Literal, Tuple
import pandas as pd
from .utils import _handle_dataframe_errors

def extract_features(
    df: pd.DataFrame,
    feature_type: Literal['num', 'cat', 'others', 'all'],
    ignore_cols: Optional[Union[str, List[str]]] = None,
    unique_value_threshold: Optional[int] = 12
) -> Union[List[str], Tuple[List[str]]]:
    # Handle DataFrame errors
    df = _handle_dataframe_errors(df)

    # Ensure ignore_cols is a list
    if ignore_cols:
        if isinstance(ignore_cols, str):
            ignore_cols = [ignore_cols]
        elif not isinstance(ignore_cols, list):
            raise TypeError("ignore_cols must be a string or a list of strings.")
        df = df.drop(columns=ignore_cols, errors='ignore')

    # Select numerical features
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    numerical_features = [col for col in numerical_features if df[col].nunique() > unique_value_threshold]

    # Select categorical features
    categorical_features = df.select_dtypes(include=['bool', 'object', 'category']).columns.tolist()
    categorical_features += [col for col in df.columns if df[col].nunique() <= unique_value_threshold and col not in numerical_features]

    if feature_type == 'num':
        return numerical_features
    elif feature_type == 'cat':
        return categorical_features
    elif feature_type == 'others':
        num_cat_features = set(numerical_features + categorical_features)
        other_features = [col for col in df.columns if col not in num_cat_features]
        return other_features
    elif feature_type == 'all':
        other_features = [col for col in df.columns if col not in set(numerical_features + categorical_features)]
        return numerical_features, categorical_features, other_features
    else:
        raise ValueError("The 'feature_type' parameter must be 'num', 'cat', 'others', or 'all'.")
        
def find_outliers(
    df: pd.DataFrame,  
    features_with_outliers: List[str]
) -> pd.DataFrame:
    df = _handle_dataframe_errors(df[features_with_outliers])
    outlier_df = pd.DataFrame()
    for feature in features_with_outliers:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[feature][(df[feature] < lower_bound) | (df[feature] > upper_bound)]

        # Add outliers to the DataFrame if any exist
        if not outliers.empty:
            outlier_df[feature] = outliers.sort_values().reset_index(drop=True)
    return outlier_df
    
def calc_na_values(
    df: pd.DataFrame, 
    features: List[str], 
    pct: bool = True
) -> pd.Series:
    df = _handle_dataframe_errors(df[features])
    
    # Count of missing values in features
    missing_value_count = df.isna().sum()

    # Filter out features with no missing values
    missing_value_count = missing_value_count[missing_value_count > 0]

    # Store features with missing values
    features_with_missing_values = missing_value_count.index.to_list()

    if pct:
        # Percentage of missing values in features
        missing_value_pct = round(missing_value_count / df.shape[0] * 100, 2)
        return pd.Series(missing_value_pct, index=features_with_missing_values)
    else:
        return pd.Series(missing_value_count, index=features_with_missing_values)
  
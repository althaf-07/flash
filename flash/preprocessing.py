from typing import List, Optional, Union, Literal, Tuple
import pandas as pd

def extract_features(
    df: pd.DataFrame,
    feature_type: Literal['num', 'cat', 'others', 'all'],
    ignore_cols: Optional[Union[str, List[str]]] = None,
    unique_value_threshold: Optional[int] = 12
) -> Union[List[str], Tuple[List[str]]]:
    # Validate the feature_type input
    if feature_type not in {'num', 'cat', 'others', 'all'}:
        raise ValueError("The 'feature_type' parameter must be 'num', 'cat', 'others', or 'all'.")

    # Ensure ignore_cols is a list and drop the columns from DataFrame
    if ignore_cols:
        if isinstance(ignore_cols, str):
            ignore_cols = [ignore_cols]
        elif not isinstance(ignore_cols, list):
            raise TypeError("ignore_cols must be a string or a list of strings.")
        df = df.drop(columns=ignore_cols, errors='ignore')

    # Select numerical features
    numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    
    # Select categorical features
    categorical_features = df.select_dtypes(include=['bool', 'object', 'category']).columns.tolist()
    categorical_features += [col for col in numerical_features if df[col].nunique() <= unique_value_threshold]
    
    # Filter out real numerical features (i.e., those with more unique values than the threshold)
    numerical_features = [col for col in numerical_features if col not in categorical_features]

    # Return based on feature_type
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
        
def calc_na_values(
    df: pd.DataFrame, 
    features: List[str], 
    pct: bool = True
) -> pd.Series:
    
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
  
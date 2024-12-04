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

def advanced_numerical_imputer(
    df: pd.DataFrame,
    num_cols_with_nan: list[str],
    imputer: KNNImputer | IterativeImputer,
    cat_cols: list[str] | None = None,
    cat_cols_mode_imputation: bool = False,
    return_full_df: bool = False
    ) -> pd.DataFrame:
    """Imputes missing values of numerical features using advanced imputation methods.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing both numerical and categorical features.
        Features in `num_cols_with_nan` and `cat_cols` must be present in the DataFrame.
    num_cols_with_nan_with_nan : list[str]
        List of numerical features in the DataFrame that has missing values.
    cat_cols : list[str], default=None
        List of categorical features in the DataFrame.
    cat_cols_mode_imputation : bool, default=False
        Whether to impute missing values using mode in cat_cols or not. If False, this
        will not impute the missing values, but OneHotEncoder will make a separate column
        for missing values. If True, this will impute missing values using mode.
    imputer : KNNImputer or IterativeImputer
        KNNImputer or IterativeImputer instance to impute missing values.
    return_full_df : bool, default=False
        If True, this will return the entire imputed DataFrame. If False, this will only
        return the DataFrame with num_cols_with_nan imputed. 

    Returns
    -------
    df_imputed : pd.DataFrame
        A DataFrame with missing values in the numerical columns imputed.
        Only the numerical features are returned after imputation.

    Raises
    ------
    ValueError
        If columns from num_cols_with_nan or cat_cols doesn't exist in the Pandas DataFrame.
    """

    df = df.copy() # Copying to avoid modifying the original DataFrame

    # Input validation
    if not isinstance(cat_cols, list):
        cat_cols = []

    missing_cols = [col for col in num_cols_with_nan + cat_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the DataFrame: {', '.join(missing_cols)}")

    if cat_cols:
        # Impute categorical columns using mode
        if cat_cols_mode_imputation:
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

        # One-Hot Encode categorical features
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = ohe.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())

        # Concatenating encoded categorical features with the rest of the dataset
        df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

    # Impute the data
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    if return_full_df:
        return df_imputed
    else:
        return df_imputed[num_cols_with_nan]

def advanced_categorical_imputer(
    df: pd.DataFrame,
    cat_col_with_nan: str,
    clf_model: ClassifierMixin,
    cat_cols: list[str] | None = None,
    cat_cols_mode_imputation: bool = False,
    ) -> pd.Series:
    """Imputes missing values of categorical features using classifier models.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing categorical features. `cat_cols_with_nan` feature and
        features in `cat_cols` must be present in the DataFrame. Also, missing values of
        numerical features must be imputed before imputing categorial features.
    cat_col_with_nan : str
        The name of the categorical feature that you want to impute.
    clf_model : ClassifierMixin
        Classifier model class (e.g., `LogisticRegression`, `RandomForestClassifier`)
        to use for imputation.
    cat_cols : list[str], default=None
        List of categorical features in the DataFrame.
    cat_cols_mode_imputation : bool, default=False
        Whether to impute missing values using mode in cat_cols or not. If False, this
        will not impute the missing values, but OneHotEncoder will make a separate column
        for missing values. If True, this will impute missing values using mode.

    Returns
    -------
    pd.Series
        The imputed categorical feature
    """

    df = df.copy() # Copying to avoid modifying the original DataFrame 

    # Input validation
    if not isinstance(cat_cols, list):
        cat_cols = []

    missing_cols = [col for col in [cat_col_with_nan] + cat_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the DataFrame: {', '.join(missing_cols)}")

    # Remove cat_col_with_nan from cat_cols
    cat_cols = [col for col in cat_cols if col != cat_col_with_nan]

    # Splitting features and target
    X = df.drop(cat_col_with_nan, axis=1)
    y = df[cat_col_with_nan]

    if cat_cols:
        # Impute categorical columns using mode
        if cat_cols_mode_imputation:
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])

        # One-Hot Encode categorical features (excluding the column to impute)
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = ohe.fit_transform(X[cat_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())
        X = pd.concat([X.drop(columns=cat_cols), encoded_df], axis=1)

    # Splitting into train and test sets based on missing values in the target
    y_notna = y.notna()
    X_train, X_test = X[y_notna], X[~y_notna]
    y_train, y_test = y[y_notna], y[~y_notna]

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    # Label encoding the target feature
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Model training
    clf_model.fit(X_train, y_train_encoded)

    # Predicting on test data
    y_pred = clf_model.predict(X_test)

    # Inverse-transforming the predicted values
    y_pred_inverse = le.inverse_transform(y_pred)

    # Imputing the missing target values with the predicted ones
    df.loc[y_test.index, cat_col_with_nan] = y_pred_inverse

    # Return the target feature with imputed values
    return df[cat_col_with_nan]
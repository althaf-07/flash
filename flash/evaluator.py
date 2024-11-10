from typing import Optional, Dict, List, Literal
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

def basic_imputer(
        series: pd.Series,
        var_type: Literal['num', 'cat'],
        method: Literal['mean', 'median', 'mode', 'ffill', 'bfill'] = None,
        fallback: Literal['mean', 'median', 'mode', 'ffill', 'bfill'] = None
    ) -> pd.Series:
    def _impute(series, method):
        if method == 'mean':
            series.fillna(series.mean(), inplace=True)
        elif method == 'median':
            series.fillna(series.median(), inplace=True)
        elif method == 'mode':
            series.fillna(series.mode()[0], inplace=True)
        elif method == 'ffill':
            series.ffill(inplace=True)
        elif method == 'bfill':
            series.bfill(inplace=True)
        return series

    # Determine default methods based on variable type
    if var_type == 'num':
        method = method or 'mean'
        fallback = fallback or 'mean'
    elif var_type == 'cat':
        method = method or 'mode'
        fallback = fallback or 'mode'

    series = series.copy()  # Avoid modifying the original
    series = _impute(series, method)  # Primary imputation

    if series.isna().any():  # Check for remaining NaNs
        series = _impute(series, fallback)  # Fallback imputation

    return series

def eval_basic_imputer(
        series: pd.Series,y: pd.Series,
        var_type: Literal['num', 'cat'], models: Dict,
        methods: Optional[List[Literal['mean', 'median', 'mode', 'ffill', 'bfill']]] = None,
        ohe: bool = True, scale: bool = False
    ) -> pd.DataFrame:

    # Default methods if not provided
    if methods is None:
        if var_type == 'num':
            methods = ['mean', 'median', 'ffill', 'bfill']
        elif var_type == 'cat':
            methods = ['mode', 'ffill', 'bfill']

    results_dict = {method: [] for method in methods}
    results_dict['Model'] = list(models.keys())

    for method in methods:
        X_imputed = basic_imputer(series, var_type=var_type, method=method)
        X_imputed = pd.DataFrame(X_imputed, columns=[series.name])

        # Apply one-hot encoding or scaling based on variable type
        if var_type == 'cat' and ohe:
            # One-Hot Encoding for categorical features
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            encoded_data = ohe.fit_transform(X_imputed)
            encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())
            X_imputed = encoded_df
        elif var_type == 'num' and scale:
            X_imputed = pd.DataFrame(StandardScaler().fit_transform(X_imputed), columns=[series.name])

        # Evaluate models
        for model in models.values():
            cv_score = cross_val_score(model, X_imputed, y, cv=5, scoring='accuracy')
            results_dict[method].append(cv_score.mean() * 100)

    results = pd.DataFrame(results_dict).set_index('Model')
    results.loc['Mean Accuracy'] = results.mean()

    return results

def advanced_numerical_imputer(
        df: pd.DataFrame,
        cat_cols: List[str],
        target: str,
        method: Literal['knn', 'iterative'] = 'knn'
    ) -> pd.Series:
    """
    Performs numerical imputation on the target column using either KNN or Iterative Imputer.
    Args:
        df (pd.DataFrame): The input DataFrame.
        cat_cols (List[str]): List of categorical columns to be encoded.
        target (str): The target column for imputation.
        method (str): Imputation method ('knn' or 'iterative').
    Returns:
        pd.Series: Imputed values for the target column.
    """
    # Copying to avoid modifying the original DataFrame
    df = df.copy()

    # One-Hot Encoding for categorical features
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = ohe.fit_transform(df[cat_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())

    # Concatenating encoded categorical features with the rest of the dataset
    encoded_df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

    # Select the appropriate imputer
    if method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif method == 'iterative':
        imputer = IterativeImputer()
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'knn' or 'iterative'.")

    # Impute the data and convert back to a DataFrame
    df_imputed = pd.DataFrame(imputer.fit_transform(encoded_df), columns=encoded_df.columns)

    # Return the imputed target column
    return df_imputed[target]

def eval_advanced_numerical_imputer(
        df: pd.DataFrame,
        target: str,
        y: pd.Series,
        cat_cols: List[str],
        models: Dict,
        methods: Optional[List[Literal['knn', 'iterative']]] = None,
        scale: bool = True,
        scoring: str = 'accuracy'
    ) -> pd.DataFrame:
    """
    Evaluates different imputation methods and models for a target column with missing values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Target column name for imputation.
        y (pd.Series): True target values for evaluation.
        cat_cols (List[str]): List of categorical columns to be encoded.
        models (Dict): Dictionary of models to evaluate.
        methods (Optional[List[str]]): List of imputation methods to test. Defaults to ['knn', 'iterative'].
        scale (bool): Whether to scale the imputed data. Defaults to True.
        scoring (str): Scoring metric for model evaluation. Defaults to 'accuracy'.

    Returns:
        pd.DataFrame: DataFrame containing mean accuracy scores for each model and method.
    """

    # Set default imputation methods if none provided
    methods = methods or ['knn', 'iterative']

    # Prepare results dictionary
    results_dict = {method: [] for method in methods}
    results_dict['Model'] = list(models.keys())

    for method in methods:
        # Impute the target column using the specified method
        target_imputed = advanced_numerical_imputer(df, cat_cols, target, method=method)
        target_imputed_df = pd.DataFrame(target_imputed, columns=[target])

        # Scale if specified
        if scale:
            scaler = StandardScaler()
            target_imputed_df[target] = scaler.fit_transform(target_imputed_df[[target]])

        # Evaluate each model using cross-validation
        for model in models.values():
            cv_score = cross_val_score(model, target_imputed_df[target].reshape(-1, 1), y, cv=5, scoring=scoring)
            results_dict[method].append(cv_score.mean() * 100)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_dict).set_index('Model')
    results_df.loc['Mean Accuracy'] = results_df.mean()

    return results_df

def advanced_categorical_imputer(
        df: pd.DataFrame,
        cat_cols: List[str],
        target: str,
        clf_model: ClassifierMixin,
        handle_other_cat_cols: Literal['drop_rows', 'drop_cols', 'ohe', 'mode']='ohe'
        ) -> pd.Series:

    # Copying to avoid modifying original data
    df = df.copy()

    # Splitting features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Handling missing categorical values based on the specified strategy
    if handle_other_cat_cols == 'drop_rows':
        X.dropna(subset=cat_cols, inplace=True)
        y = y.loc[X.index]
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
    elif handle_other_cat_cols == 'drop_cols':
        cols_with_missing_values = X.columns[X.isna().any()].tolist()
        X.drop(columns=cols_with_missing_values, inplace=True)
        cat_cols = [col for col in cat_cols if col not in cols_with_missing_values]
    elif handle_other_cat_cols == 'mode':
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # One-Hot Encoding for categorical features
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = ohe.fit_transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())

    # Concatenating encoded categorical features with the rest of the dataset
    X = pd.concat([X.drop(columns=cat_cols), encoded_df], axis=1)

    # Splitting into train and test sets
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
    df.loc[y_test.index, target] = y_pred_inverse

    # Return the target feature with imputed values
    return df[target]

def eval_advanced_categorical_imputer(
        df: pd.DataFrame,
        target: str,
        y: pd.Series,
        cat_cols: List[str],
        clf_models: Dict[str, ClassifierMixin],
        models: Dict,
        handle_other_cat_cols: Literal['drop_rows', 'drop_cols', 'ohe', 'mode']='ohe',
        scoring: str = 'accuracy'
    ) -> pd.DataFrame:
    """
    Evaluates different imputation methods and models for a target column with missing values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target (str): Target column name for imputation.
        y (pd.Series): True target values for evaluation.
        cat_cols (List[str]): List of categorical columns to be encoded.
        models (Dict): Dictionary of models to evaluate.
        methods (Optional[List[str]]): List of imputation methods to test. Defaults to ['knn', 'iterative'].
        scale (bool): Whether to scale the imputed data. Defaults to True.
        scoring (str): Scoring metric for model evaluation. Defaults to 'accuracy'.

    Returns:
        pd.DataFrame: DataFrame containing mean accuracy scores for each model and method.
    """

    # Prepare results dictionary
    results_dict = {model_name: [] for model_name in clf_models}
    results_dict['Model'] = list(models.keys())

    for model_name, model_instance in clf_models.items():
        # Impute the target column using the specified method
        target_imputed = advanced_categorical_imputer(df, cat_cols, target, model_instance, handle_other_cat_cols)
        target_imputed_df = pd.DataFrame(target_imputed, columns=[target])

        ohe = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = ohe.fit_transform(target_imputed_df)
        encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())

        target_imputed_df = encoded_df

        # Evaluate each model using cross-validation
        for model in models.values():
            cv_score = cross_val_score(model, target_imputed_df, y, cv=5, scoring=scoring)
            results_dict[model_name].append(cv_score.mean() * 100)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_dict).set_index('Model')
    results_df.loc['Mean Accuracy'] = results_df.mean()

    return results_df
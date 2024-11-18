from typing import Optional, Dict, List, Literal
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, PowerTransformer, QuantileTransformer
)

def basic_imputer(
        x: pd.Series,
        var_type: Literal['num', 'cat'],
        method: Optional[Literal['mean', 'median', 'mode', 'ffill', 'bfill']] = None,
        fallback: Optional[Literal['mean', 'median', 'mode', 'ffill', 'bfill']] = None
    ) -> pd.Series:
    """Imputes missing values using basic statistical measures.

    Parameters
    ----------
    x : pd.Series
        The Series in which to impute missing values.
    var_type : {'num', 'cat'}
        Variable type of the Series. 'num' for numerical, 'cat' for categorical.
    method : {'mean', 'median', 'mode', 'ffill', 'bfill'}, optional
        The method to impute missing values. If None, the default method for the given `var_type` is used.
    fallback : {'mean', 'median', 'mode', 'ffill', 'bfill'}, optional
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
        num_cols: List[str],
        cat_cols: List[str],
        handle_cat_cols: Literal['ohe', 'mode', 'drop_rows', 'drop_cols'] = 'ohe',
        method: Literal['knn', 'iterative'] = 'knn'
    ) -> pd.DataFrame:
    """Imputes missing values of numerical features using advanced imputation methods.

    This function imputes missing values in numerical columns using advanced imputation methods, 
    either K-Nearest Neighbors ('knn') or Iterative Imputation ('iterative'). 
    It also handles categorical columns by one-hot encoding them before imputing numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing both numerical and categorical features. Features in `num_cols` and `cat_cols` 
        must be present in the DataFrame.
    num_cols : List[str]
        List of numerical features in the DataFrame.
    cat_cols : List[str]
        List of categorical features in the DataFrame.
    handle_cat_cols : {'ohe', 'mode', 'drop_rows', 'drop_cols'}, default='ohe'
        The strategy to handle missing values in categorical columns:
        - 'ohe': One-hot encoding the categorical features.
        - 'mode': Impute missing values with the most frequent value (mode).
        - 'drop_rows': Drop rows with missing values in categorical columns.
        - 'drop_cols': Drop columns that have missing values in categorical columns.
        From what i have experimented with these values, using 'ohe' or 'mode' provide the best results. 
    method : {'knn', 'iterative'}, default='knn'
        The method to impute missing values. If not specified, 'knn' is used.

    Returns
    -------
    df_imputed : pd.DataFrame
        A DataFrame with missing values in the numerical columns imputed. Only the numerical features 
        are returned after imputation.

    Raises
    ------
    ValueError
        If an invalid method or handle_cat_cols strategy is specified.
    """

    # Check if the columns exist in the DataFrame
    missing_cols = [col for col in num_cols + cat_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the DataFrame: {', '.join(missing_cols)}")

    # Copying to avoid modifying the original DataFrame
    df = df.copy()

    # Handle categorical columns based on the specified strategy
    if handle_cat_cols == 'mode':
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    elif handle_cat_cols == 'drop_rows':
        df.dropna(subset=cat_cols, inplace=True)
        df.reset_index(drop=True, inplace=True)
    elif handle_cat_cols == 'drop_cols':
        df.dropna(subset=cat_cols, axis=1, inplace=True)
        cat_cols_in_df =  df.columns.to_list()
        cat_cols = [col for col in cat_cols if col in cat_cols_in_df]
    elif handle_cat_cols != 'ohe':
        raise ValueError(f"Invalid handle_cat_cols strategy: {handle_cat_cols}. Choose from 'ohe', 'mode', 'drop_rows', 'drop_cols'.")

    # One-Hot Encoding categorical features if still features are in cat_cols
    if cat_cols:
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        encoded_data = ohe.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=ohe.get_feature_names_out())

        # Concatenating encoded categorical features with the rest of the dataset
        df_encoded = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

    # Select the appropriate imputer
    if method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
    elif method == 'iterative':
        imputer = IterativeImputer()
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'knn' or 'iterative'.")

    # Impute the data
    df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df_encoded.columns)

    # Return only the numerical features from the imputed DataFrame
    return df_imputed[num_cols]

def advanced_categorical_imputer(
        df: pd.DataFrame,
        cat_cols: List[str],
        target: str,
        clf_model: ClassifierMixin,
        handle_other_cat_cols: Literal['ohe', 'mode', 'drop_rows', 'drop_cols'] = 'ohe'
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
            df[col] = df[col].fillna(df[col].mode()[0])

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

def imputer_evaluator(
        X_imputed: Dict[str, pd.DataFrame],
        y: pd.Series,
        models: Dict[str, BaseEstimator],
        scale: bool = False,
        ohe: bool = False,
        scoring: str = 'accuracy'
    ) -> pd.DataFrame:
    """Evaluates missing value imputation strategies.

    Parameters
    ----------
    X_imputed : Dict[str, pd.DataFrame]
        A dictionary containing custom names for Dataframes and DataFrames.
    y : pd.Series
        The target feature containing labels. This should be label encoded.
    models : Dict[str, BaseEstimator]
        The dictionary of models to evaluate the imputation strategies.
    scale : bool, default=False
        To control whether to scale the values using `StandardScaler` from sklearn or not.
    ohe : bool, default=False
        To control whether to One-Hot encode the values using `OneHotEncoder` from sklearn or not.
    scoring : See https://scikit-learn.org/1.5/modules/model_evaluation.html
        The scoring metric on which to evaluate the models.

    Returns
    -------
    results_df : pd.DataFrame
        A DataFrame with evaluation results of DataFrames in `X_imputed`.

    Raises
    ------
    """

    # Prepare a dictionary to hold results
    results_dict = {X_name: [] for X_name in X_imputed.keys()}
    results_dict['Model'] = list(models.keys())

    for X_name, X_df in X_imputed.items():
        # Encode if specified
        if ohe:
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            # OneHotEncoder expects a 2D array, so we reshape X_df before fitting
            X_df = ohe.fit_transform(X_df.values.reshape(-1, 1))

        # Scale if specified
        if scale:
            ss = StandardScaler()
            # StandardScaler also expects a 2D array
            X_df = ss.fit_transform(X_df.values.reshape(-1, 1))

        # Evaluate each model using cross-validation
        for model in models.values():
            cv_score = cross_val_score(model, X_df.values.reshape(-1, 1), y, cv=5, scoring=scoring)
            results_dict[X_name].append(cv_score.mean() * 100)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_dict).set_index('Model')
    results_df.loc['Mean Accuracy'] = results_df.mean()

    return results_df

def evaluate_models(X, y, numerical_feature = None, categorical_features = None, models = None):
    # Create a copy of the original DataFrame to avoid modifying it
    X = X.copy()

    if isinstance(categorical_features, list):
        # One-hot encode categorical features and drop original categorical columns
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # Encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Initialize a dictionary to store results for each transformation
    results_dict = {}

    # Feature transformation
    transformed_data = fz.feature_transform_viz(X, [numerical_feature], result='data')

    # Iterate over transformations returned by the feature transform function
    for transform_name, transformed_values in transformed_data.items():
        # Initialize an empty dictionary to store accuracies for this transformation
        accuracies = {}

        # Apply the transformation
        X[numerical_feature] = transformed_values[numerical_feature]

        # Evaluate each model with cross-validation
        for model_name, model in models.items():
            cv_scores_after = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            accuracies[model_name] = cv_scores_after.mean() * 100

        # Add accuracies for this transformation to the results_dict
        if not results_dict:
            # Initialize the dict with model names and transformation-specific accuracy columns
            results_dict['Model'] = list(accuracies.keys())
            results_dict[transform_name] = list(accuracies.values())
        else:
            # Append the new accuracy column to the existing results
            results_dict[transform_name] = list(accuracies.values())

    # Convert the results_dict to a DataFrame
    results_X = pd.DataFrame(results_dict)

    # Calculate mean accuracy for each transformation column and create a DataFrame for it
    mean_accuracies = results_X.iloc[:, 1:].mean()  # Skip 'Model' column
    mean_row = pd.DataFrame([['Mean Accuracy'] + mean_accuracies.tolist()], columns=results_X.columns)

    # Concatenate the mean_row DataFrame to results_X
    results_X = pd.concat([results_X, mean_row], ignore_index=True)

    return results_X

# Log Transformation
def log_transform(X: pd.Series) -> pd.Series:
    """
    Apply log transformation to skewed features.
    Values <= 0 are adjusted before applying log transformation to avoid errors.
    """
    X_log = X.apply(lambda x: np.log(x + 1) if x > 0 else 0)
    return X_log

# Yeo-Johnson Transformation
def yeo_johnson_transform(X: pd.Series) -> pd.Series:
    """
    Apply Yeo-Johnson transformation to features.
    It works on both positive and negative values.
    """
    X = X.copy()
    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    X_transformed = transformer.fit_transform(X.values.reshape(-1, 1))
    return pd.Series(X_transformed.flatten(), name=X.name)

# Standardization (Z-Score)
def standardize_transform(X: pd.Series) -> pd.Series:
    """
    Standardize the data (zero mean, unit variance).
    """
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X.values.reshape(-1, 1))
    return pd.Series(X_standardized.flatten(), name=X.name)

# Min-Max Scaling
def min_max_scale(X: pd.Series) -> pd.Series:
    """
    Apply Min-Max Scaling (scales the data to the range [0, 1]).
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X.values.reshape(-1, 1))
    return pd.Series(X_scaled.flatten(), name=X.name)

# Square Root Transformation
def sqrt_transform(X: pd.Series) -> pd.Series:
    """
    Apply square root transformation to features.
    This is typically used to reduce skewness, especially for data with a strong positive skew.
    """
    X_sqrt = X.apply(lambda x: np.sqrt(x) if x >= 0 else 0)
    return X_sqrt

# Square Transformation
def square_transform(X: pd.Series) -> pd.Series:
    """
    Apply square transformation to features.
    This can be useful for modeling non-linear relationships.
    """
    X_squared = X.apply(lambda x: x**2)
    return X_squared

# Reciprocal Transformation
def reciprocal_transform(X: pd.Series) -> pd.Series:
    """
    Apply reciprocal transformation to features.
    This is useful for reducing the impact of large values.
    """
    X_reciprocal = X.apply(lambda x: 1 / (x + 1) if x != 0 else 0)
    return X_reciprocal

# Quantile Transformation
def quantile_transform(X: pd.Series) -> pd.Series:
    """
    Apply quantile transformation to features.
    This transforms the data to have a uniform or Gaussian distribution.
    """
    transformer = QuantileTransformer(output_distribution='uniform', n_quantiles=X.shape[0])
    X_quantile = transformer.fit_transform(X.values.reshape(-1, 1))
    return pd.Series(X_quantile.flatten(), name=X.name)

# Handle Outliers using IQR method
def handle_outliers(series: pd.Series, method: Literal['iqr', 'robust_model'] = None) -> pd.Series:
    """
    Handle outliers using the IQR method.
    """
    series_copy = series.copy()

    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        # Calculate lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        series_copy = series.apply(
            lambda x: lower_bound if x < lower_bound else (
                upper_bound if x > upper_bound else x
                )
            )
    elif method == 'robust_model':
        # Placeholder for robust models handling (e.g., RandomForest) that naturally handle outliers
        pass

    return series_copy

# Main function to evaluate different outlier methods and transformations
def eval_outlier_method(
        series: pd.Series,
        y: pd.Series,
        methods: List[Literal['iqr', 'robust_model', 'log', 'yeo-johnson', 'standardize', 'min-max',
                             'sqrt', 'square', 'reciprocal', 'quantile']],

        models: Dict[str, BaseEstimator],
        scoring: str = 'accuracy'
        ) -> pd.DataFrame:
    """
    Evaluate different outlier handling methods and transformations for feature data.
    """
    # Initialize the results dictionary
    results_dict = {method: [] for method in methods}
    results_dict['Model'] = list(models.keys())

    # Evaluate for each method
    for method in methods:
        if method not in ['log', 'yeo-johnson', 'standardize', 'min-max', 'sqrt', 'square', 'reciprocal', 'quantile']:
            # Apply the outlier handling (IQR or robust model)
            adjusted_series = handle_outliers(series, method=method)
        else:
            # Apply transformations based on the method
            if method == 'log':
                adjusted_series = log_transform(series)
            elif method == 'yeo-johnson':
                adjusted_series = yeo_johnson_transform(series)
            elif method == 'standardize':
                adjusted_series = standardize_transform(series)
            elif method == 'min-max':
                adjusted_series = min_max_scale(series)
            elif method == 'sqrt':
                adjusted_series = sqrt_transform(series)
            elif method == 'square':
                adjusted_series = square_transform(series)
            elif method == 'reciprocal':
                adjusted_series = reciprocal_transform(series)
            elif method == 'quantile':
                adjusted_series = quantile_transform(series)

            # After transformation, handle outliers using the IQR method
            adjusted_series = handle_outliers(adjusted_series, method='iqr')

        # Evaluate each model using cross-validation
        for model_name, model_instance in models.items():
            # Perform cross-validation and collect the mean score
            cv_score = cross_val_score(model_instance, adjusted_series.values.reshape(-1, 1), y, cv=5, scoring=scoring)
            results_dict[method].append(cv_score.mean() * 100)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_dict).set_index('Model')
    # Add mean accuracy
    results_df.loc['Mean Accuracy'] = results_df.mean()

    return results_df
    
def evaluate_models(X, y, feature, categorical_features, models):
    # Create a copy to avoid modifying the original DataFrame
    X = X.copy()

    # Encode the target variable
    y = LabelEncoder().fit_transform(y)

    def evaluate(X, categorical_features):
        # One-hot encode categorical features
        X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

        accuracies = {}
        for model_name, model in models.items():
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            accuracies[model_name] = cv_scores.mean() * 100
        return accuracies

    # Evaluate before removing the feature
    if feature in X.columns:
        X_before = X.drop(columns=[feature])
        categorical_features = [f for f in categorical_features if f != feature]
        accuracies_before = evaluate(X_before, categorical_features)
    else:
        accuracies_before = evaluate(X, categorical_features)

    # Evaluate after including the feature
    accuracies_after = evaluate(X, categorical_features + [feature] if feature else categorical_features)

    # Create results DataFrame
    results = pd.DataFrame({
        'Model': models.keys(),
        'Before': accuracies_before.values(),
        'After': accuracies_after.values()
    })

    # Add mean row
    results.loc[len(results)] = ['Mean Accuracy', results['Before'].mean(), results['After'].mean()]
    results_df = pd.DataFrame(results).set_index('Model')
    results_df.loc['Mean Accuracy'] = results_df.mean()

    return results_df
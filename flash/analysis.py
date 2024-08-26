import math
from typing import List, Optional, Literal, Dict, Tuple
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, QuantileTransformer
from .utils import _handle_dataframe_errors

def stats_moments(
    df: pd.DataFrame, 
    numerical_features: List[str],
    round_: Optional[int] = 2 
):
    df = _handle_dataframe_errors(df[numerical_features])
    moments_df = pd.DataFrame(
        [
            {
                'Mean': round(float(df[col].mean()), round_),
                'Standard deviation': round(float(df[col].std()), round_),
                'Skewness': round(float(df[col].skew()), round_),
                'Kurtosis': round(float(df[col].kurtosis()), round_)
            }
            for feature in numerical_features
        ],
        index=numerical_features
    )
    return moments_df
    
def hist_box_viz(
    df: pd.DataFrame, 
    numerical_features: List[str], 
    figsize: Optional[Tuple[int, int]] = None, 
    title: Optional[str] = None, 
    hist_xlabel: Optional[str] = None, 
    hist_ylabel: Optional[str] = None, 
    box_xlabel: Optional[str] = None, 
    box_ylabel: Optional[str] = None
):
    """
    Plots histograms and boxplots for the specified numerical features.
    """
    # Handle potential DataFrame errors
    df = _handle_dataframe_errors(df)
    
    # Calculate figure size if not provided
    n_features = len(numerical_features)
    figsize = figsize or (12.5, n_features * 3 + 1)
    
    # Create subplots: one column for histograms, one for boxplots
    fig, axs = plt.subplots(n_features, 2, figsize=figsize)

    for i, feature in enumerate(numerical_features):
        # Plot histogram with KDE
        sns.histplot(df[feature], kde=True, ax=axs[i, 0])
        axs[i, 0].set(title=f'Histogram of {feature}', xlabel=hist_xlabel or '', ylabel=hist_ylabel or '')
        axs[i, 0].grid(True)

        # Plot boxplot
        sns.boxplot(x=df[feature], ax=axs[i, 1])
        axs[i, 1].set(title=f'Boxplot of {feature}', xlabel=box_xlabel or '', ylabel=box_ylabel or '')
        axs[i, 1].grid(True)

    plt.tight_layout()
    plt.show()

def nan_value_viz(
    df: pd.DataFrame, 
    figsize: Optional[Tuple[int, int]] = None, 
    cmap: str = 'Blues', 
    xticks_rotation: Optional[Union[int, float]] = None
):
    """
    Plots a heatmap of missing values in the DataFrame.
    """
    # Handle potential DataFrame errors
    df = _handle_dataframe_errors(df)
    
    # Calculate figure size if not provided
    figsize = figsize or (df.shape[1] / 4 * 5, 4)
    
    # Plot heatmap for missing values
    sns.heatmap(df.isna(), figsize=figsize, cbar=False, cmap=cmap, yticklabels=False)
    
    if xticks_rotation is not None:  
        plt.xticks(rotation=xticks_rotation)
        
    plt.show()

def count_viz(
    df: pd.DataFrame, 
    categorical_features: List[str],
    n_cols: Optional[int] = 3,
    figsize: Optional[Tuple[int, int]] = None,
    rotate_x_labels: Optional[List[str]] = None, 
    rotation: Optional[int] = 45
):
    """
    Plots countplots for categorical features in the DataFrame.
    """
    # Handle potential DataFrame errors
    df = _handle_dataframe_errors(df[categorical_features])

    # Validate n_cols
    if not isinstance(n_cols, int) or n_cols <= 0:
        raise ValueError("n_cols must be a positive integer.")

    # Ensure rotate_x_labels is a list or None
    if rotate_x_labels is not None and not isinstance(rotate_x_labels, list):
        raise TypeError("rotate_x_labels must be a list of feature names or None.")

    # Validate rotation value
    if rotation is not None and (not isinstance(rotation, (int, float)) or rotation < 0):
        raise ValueError("rotation must be a non-negative integer or float, or None.")

    n_features = len(categorical_features)

    # Calculate number of rows needed for subplots
    n_rows = math.ceil(n_features / n_cols)

    # Calculate figure size if not provided
    figsize = figsize or (n_cols * 4 + 1, n_rows * 3)
    
    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()  # Flatten the array for easy iteration

    for i, feature in enumerate(categorical_features):
        sns.countplot(x=df[feature], ax=axs[i])
        axs[i].set_title(feature)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')

        if rotate_x_labels and feature in rotate_x_labels:
            axs[i].tick_params(axis='x', rotation=rotation)

    # Turn off any unused subplots
    for j in range(n_features, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

def pair_viz(
    df: pd.DataFrame, 
    numerical_features: List[str], 
    kind: Optional[Literal['scatter', 'kde', 'hist', 'reg']] = 'scatter', 
    diag_kind: Optional[Literal['auto', 'hist', 'kde', None]] = 'kde', 
    plot_kws: Optional[Dict[str, any]] = None,
    diag_kws: Optional[Dict[str, any]] = None,
    grid_kws: Optional[Dict[str, any]] = None,
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Plots a pairplot for the specified numerical features.
    """
    
    # Handle dataframe errors
    df = _handle_dataframe_errors(df)
    
    n_features = len(numerical_features)

    # Calculate figure size if not provided
    figsize = figsize or (12.5, n_features + 3)

    if kind == 'reg':
        plot_kws = plot_kws or {'line_kws': {'color': 'red'}}

    # Use default arguments for dictionaries
    plot_kws = plot_kws or {}
    diag_kws = diag_kws or {}
    grid_kws = grid_kws or {}

    height = figsize[1] / n_features
    aspect = figsize[0] / figsize[1]

    # Create the pairplot
    g = sns.pairplot(df[numerical_features], kind=kind, diag_kind=diag_kind, 
                     plot_kws=plot_kws, diag_kws=diag_kws, grid_kws=grid_kws, 
                     height=height, aspect=aspect)

    g.map_upper(_hide_current_axis)

    plt.show()
    
def corr_heatmap_viz(
    df: pd.DataFrame,
    numerical_features: List[str],
    method: Literal['pearson', 'spearman'] = 'pearson',
    cmap: Optional[str] = None,
    mask: Optional[Literal['upper', 'lower', None]] = 'upper',  
    title: Optional[str] = None, 
    ax: Optional[plt.Axes] = None
):
    # Handle dataframe errors
    df = _handle_dataframe_erros(df[numerical_features])

    # Validate method
    if method not in ['pearson', 'spearman']:
        raise ValueError("Method must be either 'pearson' or 'spearman'.")
    corr = df.corr(method=method)

    # Create mask if specified
    if mask is not None:
        if mask == 'upper':
             mask_array = np.triu(np.ones_like(corr, dtype=bool))
        elif mask == 'lower':
            mask_array = np.tril(np.ones_like(corr, dtype=bool))
        else:
            raise ValueError("Mask must be either 'upper', 'lower', or None")
    else:
        mask_array = None
        
    # Set default colormap if none is provided
    colors = ["#FF0000", "#FFFF00", "#00FF00"]
    cmap = cmap or mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plot heatmap
    ax = ax or plt.gca()
    sns.heatmap(corr, mask=mask_array, annot=True, cmap=cmap, ax=ax, cbar=False)

    # Set title
    ax.set_title(title or f'{method.capitalize()} Correlation Heatmap')
    
    plt.show()
    
def crosstab_heatmap_viz(
    df: pd.DataFrame, 
    categorical_features: List[str],
    reference_feature: Optional[str] = None,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    annot: Optional[bool] = True,
    cbar: Optional[bool] = False
):
    # Handle dataframe errors
    df = _handle_dataframe_errors(df)

    # Set default colormap if none is provided
    colors = ["#FF0000", "#FFFF00", "#00FF00"]
    cmap = cmap or mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)  

    n_features = len(categorical_features)
    n_plots = n_features * (n_features - 1) // 2 if reference_feature is None else n_features

    # Automatically adjust figure size if not provided
    figsize = figsize or (12.5, n_plots * 5)
    fig, axs = plt.subplots(n_plots, 2, figsize=figsize)
    axs = axs.reshape(-1, 2)  # Flatten the array of subplots

    if reference_feature:
        for i, feature in enumerate(categorical_features):
            table_index = pd.crosstab(df[feature], df[reference_feature], normalize='index') * 100
            table_column = pd.crosstab(df[feature], df[reference_feature], normalize='columns') * 100
            title_index = f"{feature} vs {reference_feature} (Index Normalized)"
            title_column = f"{feature} vs {reference_feature} (Column Normalized)"
            
            _plot(axs[i][0], table_index, title_index)
            _plot(axs[i][1], table_column, title_column)
    else:
        plot_index = 0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                table_index = pd.crosstab(df[categorical_features[i]], df[categorical_features[j]], normalize='index') * 100
                table_column = pd.crosstab(df[categorical_features[i]], df[categorical_features[j]], normalize='columns') * 100
                title_index = f"{categorical_features[i]} vs {categorical_features[j]} (Index Normalized)"
                title_column = f"{categorical_features[i]} vs {categorical_features[j]} (Column Normalized)"
                
                _plot(axs[plot_index][0], table_index, title_index)
                _plot(axs[plot_index][1], table_column, title_column)
                
                plot_index += 1

    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    def _plot(ax, table, title):
        sns.heatmap(table, annot=annot, cmap=cmap, cbar=cbar, fmt='0.2f',
                    xticklabels=True, yticklabels=True, ax=ax)
        ax.set_title(title)
    
def violin_point_viz(
    df: pd.DataFrame,
    numerical_features: Union[List[str], str],
    categorical_features: Union[str, List[str]], 
    figsize: Optional[Tuple[int, int]] = None,
    mean_color: Optional[str] = 'blue', 
    median_color: Optional[str] = 'red'
):
    # Check if inputs are valid
    if isinstance(numerical_features, list) and isinstance(categorical_features, str):
        if not all(col in df.columns for col in numerical_features):
            raise ValueError("One or more numerical features are not in the DataFrame.")
        if categorical_features not in df.columns:
            raise ValueError("Categorical feature is not in the DataFrame.")
        
        n_features = len(numerical_features)
        figsize = figsize or (12.5, n_features * 4)
        fig, axs = plt.subplots(n_features, 2, figsize=figsize, constrained_layout=True)
        
        for i, num in enumerate(numerical_features):
            _plot(num, categorical_features, axs[i])
    
    elif isinstance(categorical_features, list) and isinstance(numerical_features, str):
        if not all(col in df.columns for col in categorical_features):
            raise ValueError("One or more categorical features are not in the DataFrame.")
        if numerical_features not in df.columns:
            raise ValueError("Numerical feature is not in the DataFrame.")
        
        n_features = len(categorical_features)
        figsize = figsize or (12.5, n_features * 4)
        fig, axs = plt.subplots(n_features, 2, figsize=figsize, constrained_layout=True)
        
        for i, cat in enumerate(categorical_features):
            _plot(numerical_features, cat, axs[i])
    else:
        raise TypeError("If numerical_features is a list, categorical_features must be a string. And vice versa.")
    
    plt.show()
    
    def _plot(num, cat, ax):
        sns.violinplot(x=cat, y=num, hue=cat, data=df, ax=ax[0])
        ax[0].set_title(f'Violinplot of {num} by {cat}')
        
        sns.pointplot(x=cat, y=num, errorbar=None, color=mean_color, data=df, ax=ax[1], 
                      label='Mean')
        sns.pointplot(x=cat, y=num, errorbar=None, color=median_color, 
                      estimator='median', data=df, ax=ax[1], label='Median')
        ax[1].set_title(f'Pointplot of {num} by {cat}')
        
def feature_transform_viz(
        df: pd.DataFrame, 
        numerical_features: List[str], 
        result: Optional[Literal['data', 'hist', 'qq']] = None, 
        figsize: Optional[Tuple[int, int]] = (12, 8)
) -> Optional[Dict[str, pd.DataFrame]]:
    df = _handle_dataframe_errors(df[numerical_features])
    epsilon = 1e-10

    transformers = {
        'Log': FunctionTransformer(func=lambda X: np.log(X + epsilon), validate=False),
        'Square Root': FunctionTransformer(func=lambda X: np.sqrt(X + epsilon), validate=False),
        'Square': FunctionTransformer(func=np.square, validate=False),
        'Reciprocal': FunctionTransformer(func=lambda X: np.reciprocal(X + epsilon), validate=False),
        'Yeo-Johnson': PowerTransformer(standardize=False),
        'Quantile': QuantileTransformer(n_quantiles=df.shape[0], output_distribution='normal')
    }
    
    transformed_data = {
        name: pd.DataFrame(transformer.fit_transform(df[numerical_features]), columns=numerical_features)
        for name, transformer in transformers.items()
    }

    if result == 'data':
        return transformed_data

    def _plot_histograms():
        fig, axs = plt.subplots(len(numerical_features), len(transformers) + 1, figsize=figsize)
        for i, feature in enumerate(numerical_features):
            sns.histplot(df[feature], kde=True, ax=axs[i, 0])
            axs[i, 0].set_title(f'Original {feature}')

            for j, (name, transformed_df) in enumerate(transformed_data.items()):
                sns.histplot(transformed_df[feature], kde=True, ax=axs[i, j + 1])
                axs[i, j + 1].set_title(f'{name} {feature}')
        _finalize_plot(axs)

    def _plot_qq_plots():
        fig, axs = plt.subplots(len(numerical_features), len(transformers) + 1, figsize=figsize)
        for i, feature in enumerate(numerical_features):
            stats.probplot(df[feature], dist="norm", plot=axs[i, 0])
            axs[i, 0].set_title(f'Original {feature}')
            axs[i, 0].get_lines()[1].set_color('red')  # Reference line

            for j, (name, transformed_df) in enumerate(transformed_data.items()):
                stats.probplot(transformed_df[feature], dist="norm", plot=axs[i, j + 1])
                axs[i, j + 1].set_title(f'{name} {feature}')
                axs[i, j + 1].get_lines()[1].set_color('red')  # Reference line
        _finalize_plot(axs)

    def _finalize_plot(axs):
        for ax in axs.flatten():
            ax.set_xlabel('')
            ax.set_ylabel('')
        plt.tight_layout()
        plt.show()

    if result == 'hist':
        _plot_histograms()
    elif result == 'qq':
        _plot_qq_plots()
    else:
        raise ValueError(f"Invalid result type '{result}'. Choose from 'data', 'hist', or 'qq'.")
        
    return None
    
def _hide_current_axis():
    plt.gca().set_visible(False)

import math
from typing import List, Union, Optional, Literal, Dict, Tuple
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
                'Mean': round(float(df[feature].mean()), round_),
                'Standard deviation': round(float(df[feature].std()), round_),
                'Skewness': round(float(df[feature].skew()), round_),
                'Kurtosis': round(float(df[feature].kurtosis()), round_)
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
    figsize = figsize or (13, n_features * 3 + 1)
    
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
    plt.figure(figsize=figsize)
    sns.heatmap(df.isna(), cbar=False, cmap=cmap, yticklabels=False)
    
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
    mask: Optional[Literal['upper', 'lower', None]] = 'upper',
    plot_kws: Optional[Dict[str, any]] = None,
    diag_kws: Optional[Dict[str, any]] = None,
    grid_kws: Optional[Dict[str, any]] = None,
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Plots a pairplot for the specified numerical features.
    """
    
    # Handle dataframe errors
    df = _handle_dataframe_errors(df[numerical_features])
    
    n_features = len(numerical_features)

    # Calculate figure size if not provided
    figsize = figsize or (13, n_features + 3)

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
    
    # Apply mask
    _apply_mask_pairplot(g, mask)
    
    plt.show()

def corr_heatmap_viz(
    df: pd.DataFrame,
    numerical_features: List[str],
    figsize: Optional[Tuple[int, int]] = (13, 5),
    annot: bool = True,
    title: Optional[str] = None,
    cmap: Optional[str] = None,
    cbar: bool = True,
    mask: Literal['upper', 'lower', None] = 'upper',
    mask_main_diagonal: Optional[bool] = True
):
    """
    Plots a correlation heatmap for the specified numerical features.
    """

    # Handle dataframe errors
    df = _handle_dataframe_errors(df[numerical_features])

    # Set default colormap if none is provided
    if cmap is None:
        colors = ["#FF0000", "#FFFF00", "#00FF00"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Create subplots
    if cbar:
        fig, axs = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [1, 1, 0.015]})
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize)

    for i, method in enumerate(['pearson', 'spearman']):
        # Calculate correlation table
        corr = df.corr(method=method)

        # Apply mask
        mask_array = _apply_mask_heatmap(corr, mask, mask_main_diagonal)

        # Plot heatmap
        sns.heatmap(corr, mask=mask_array, annot=annot, cmap=cmap, ax=axs[i], cbar=False)

        # Set title
        axs[i].set_title(title or f'{method.capitalize()} Correlation Heatmap')

    if cbar:
        fig.colorbar(axs[0].collections[0], cax=axs[-1])

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()
    
def crosstab_heatmap_viz(
    df: pd.DataFrame, 
    categorical_features: List[str],
    reference_feature: Optional[str] = None,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    annot: bool = True,
    cbar: bool = False
):
    def _plot(ax, table, title):
        sns.heatmap(table, annot=annot, cmap=cmap, cbar=cbar, fmt='0.2f',
                    xticklabels=True, yticklabels=True, ax=ax)
        ax.set_title(title)

    # Handle dataframe errors
    df = _handle_dataframe_errors(df)

    # Set default colormap if none is provided
    colors = ["#FF0000", "#FFFF00", "#00FF00"]
    cmap = cmap or mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)  

    if reference_feature:
        for feature in categorical_features:
            figsize = figsize or (12, 3)
            fig, axs = plt.subplots(1, 2, figsize=figsize)

            table_index = pd.crosstab(df[feature], df[reference_feature], normalize='index') * 100
            table_column = pd.crosstab(df[feature], df[reference_feature], normalize='columns') * 100
            title_index = f"{feature} vs {reference_feature} (Index Normalized)"
            title_column = f"{feature} vs {reference_feature} (Column Normalized)"
            
            _plot(axs[0], table_index, title_index)
            _plot(axs[1], table_column, title_column)

            # Adjust layout
            plt.tight_layout()
            plt.show()
            print("-" * 141)
    else:
        for i, feature_i in enumerate(categorical_features):
            for j, feature_j in enumerate(categorical_features[i+1:], start=i+1):
                figsize = figsize or (12, 3)
                fig, axs = plt.subplots(1, 2, figsize=figsize)

                table_index = pd.crosstab(df[feature_i], df[feature_j], normalize='index') * 100
                table_column = pd.crosstab(df[feature_i], df[feature_j], normalize='columns') * 100
                title_index = f"{feature_i} vs {feature_j} (Index Normalized)"
                title_column = f"{feature_i} vs {feature_j} (Column Normalized)"
                
                _plot(axs[0], table_index, title_index)
                _plot(axs[1], table_column, title_column)

                # Adjust layout
                plt.tight_layout()
                plt.show()
                print("-" * 141)

def violin_point_viz(
    df: pd.DataFrame,
    numerical_features: Union[List[str], str],
    categorical_features: Union[str, List[str]], 
    figsize: Optional[Tuple[int, int]] = None,
    grid: bool = True,
    mean_color: str = 'blue', 
    median_color: str = 'red'
):
    def _plot(num, cat, ax):
        sns.violinplot(data=df, x=cat, y=num, hue=cat, ax=ax[0], legend=False)
        ax[0].set_title(f'Violinplot of {num} by {cat}')
        
        sns.pointplot(data=df, x=cat, y=num, errorbar=None, color=mean_color, ax=ax[1], 
                      label='Mean')
        sns.pointplot(x=cat, y=num, errorbar=None, color=median_color, 
                      estimator='median', data=df, ax=ax[1], label='Median')
        ax[1].set_title(f'Pointplot of {num} by {cat}')

    # Check if inputs are valid
    if isinstance(numerical_features, list) and isinstance(categorical_features, str):
        n_features = len(numerical_features)
        figsize = figsize or (13, n_features * 4)
        fig, axs = plt.subplots(n_features, 2, figsize=figsize)
        
        for i, num in enumerate(numerical_features):
            _plot(num, categorical_features, axs[i])
    
    elif isinstance(categorical_features, list) and isinstance(numerical_features, str):
        n_features = len(categorical_features)
        figsize = figsize or (13, n_features * 4)
        fig, axs = plt.subplots(n_features, 2, figsize=figsize)
        
        for i, cat in enumerate(categorical_features):
            _plot(numerical_features, cat, axs[i])
    else:
        raise TypeError("If numerical_features is a list, categorical_features must be a string. And vice versa.")
    
    if grid:
        for i in range(n_features):
            axs[i, 0].grid(True)
            axs[i, 1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()
        
def feature_transform_viz(
        df: pd.DataFrame, 
        numerical_features: List[str], 
        result: Literal['data', 'hist', 'qq'] = 'hist', 
        figsize: Optional[Tuple[int, int]] = None
) -> Optional[Dict[str, pd.DataFrame]]:
    if result not in ['data', 'hist', 'qq']:
        raise ValueError(f"Invalid result value: {result}. Choose from 'data', 'hist', or 'qq'.") 

    epsilon = 1e-10
    df = _handle_dataframe_errors(df[numerical_features])

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

    def _plot_histograms(axs: np.ndarray):
        for i, feature in enumerate(numerical_features):
            sns.histplot(df[feature], kde=True, ax=axs[i, 0])
            axs[i, 0].set_title(f'Original {feature}')

            for j, (name, transformed_df) in enumerate(transformed_data.items()):
                sns.histplot(transformed_df[feature], kde=True, ax=axs[i, j + 1])
                axs[i, j + 1].set_title(f'{name} {feature}')
        _finalize_plot(axs)

    def _plot_qq_plots(axs: np.ndarray):
        for i, feature in enumerate(numerical_features):
            stats.probplot(df[feature], dist="norm", plot=axs[i, 0])
            axs[i, 0].set_title(f'Original {feature}')
            axs[i, 0].get_lines()[1].set_color('red')  # Reference line

            for j, (name, transformed_df) in enumerate(transformed_data.items()):
                stats.probplot(transformed_df[feature], dist="norm", plot=axs[i, j + 1])
                axs[i, j + 1].set_title(f'{name} {feature}')
                axs[i, j + 1].get_lines()[1].set_color('red')  # Reference line
        _finalize_plot(axs)

    def _finalize_plot(axs: np.ndarray):
        for ax in axs.flatten():
            ax.set_xlabel('')
            ax.set_ylabel('')
        plt.tight_layout()
        plt.show()

    # Select plotting function based on 'result' parameter
    if result == 'data':
        return transformed_data
    else:
        figsize = figsize or (26, len(numerical_features) * 3 + 1)
        fig, axs = plt.subplots(len(numerical_features), len(transformers) + 1, figsize=figsize)
        
        if result == 'hist':
            _plot_histograms(axs)
        elif result == 'qq':
            _plot_qq_plots(axs)
    
def _apply_mask_pairplot(pairplot, mask):
    # Apply the mask to the pairplot
    if mask == 'upper':
        for i in range(len(pairplot.axes)):
            for j in range(i + 1, len(pairplot.axes)):
                pairplot.axes[i, j].set_visible(False)
    elif mask == 'lower':
        for i in range(len(pairplot.axes)):
            for j in range(i):
                pairplot.axes[i, j].set_visible(False)
    elif mask is not None:
        raise ValueError("Invalid mask option. Choose from 'upper', 'lower', or None.")

def _apply_mask_heatmap(corr, mask, mask_main_diagonal):
    # Apply the mask for heatmap
    if mask is None:
        return None
    elif mask == 'upper':
        mask_array = np.triu(np.ones_like(corr, dtype=bool))
    elif mask == 'lower':
        mask_array = np.tril(np.ones_like(corr, dtype=bool))
    else:
        raise ValueError("Invalid mask option. Choose from 'upper', 'lower', or None.")

    if mask_main_diagonal not in [True, False]:
        raise ValueError("mask_main_diagonal must be either True or False")

    np.fill_diagonal(mask_array, not mask_main_diagonal)
    return mask_array

import math
from typing import Literal
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def stats_moments(df_num: pd.DataFrame, round_: int = 2) -> pd.DataFrame:
    """Calculates statistical moments for numerical features.

    Parameters
    ----------
    df_num : pd.DataFrame
        A DataFrame containing numerical features in which to calculate statistical moments.
    round_ : int, default=2
        Rounds the moments' values to the nearest integer.

    Returns
    -------
    moments_df : pd.DataFrame
        A DataFrame containing statistical moments of numerical features. Statistical moments in columns
        and numerical features in rows.
    """

    moments_dict = {
        'mean': df_num.mean().round(round_),
        'std': df_num.std().round(round_),
        'skewness': df_num.skew().round(round_),
        'kurtosis': df_num.kurtosis().round(round_)
        }

    moments_df = pd.DataFrame(moments_dict)
    return moments_df

def hist_box_viz(
    df_num: pd.DataFrame,
    figsize: tuple[int, int] | None = None,
    hist_xlabel: str | None = None,
    hist_ylabel: str | None = None,
    box_xlabel: str | None = None,
    box_ylabel: str | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
    """Plots histograms (with KDE) and boxplots for the specified numerical features.

    Usage (Run this in one single cell for better performance):
        fig, axs = hist_box_viz(df)
        axs[1, 1].grid(False)
        plt.show()

    Parameters
    ----------
    df_num : pd.DataFrame
        A DataFrame containing numerical features in which to plot Histograms and Boxplots.
    figsize : tuple[int, int], default=None
        Figure size of the plot. If None, `figsize` will be automatically 
        calculated using number of features.
    hist_xlabel : str, default=None
        X-axis label for histograms.
    hist_ylabel : str, default=None
        Y-axis label for histograms.
    box_xlabel : str, default=None
        X-axis label for boxplots.
    box_ylabel : str, default=None
        Y-axis label for boxplots.

    Returns
    -------
    fig, axs : tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes objects.
    """

    num_cols = df_num.columns.tolist()
    n_cols = len(num_cols)

    # Calculate figure size if not provided
    figsize = figsize or (13, n_cols * 3 + 1)

    # Create subplots: one column for histograms, one for boxplots
    fig, axs = plt.subplots(n_cols, 2, figsize=figsize)

    for i, feature in enumerate(num_cols):
        # Plot histogram with KDE
        sns.histplot(df_num[feature], kde=True, ax=axs[i, 0])
        axs[i, 0].set(title=f'Histogram of {feature}', xlabel=hist_xlabel, ylabel=hist_ylabel)
        axs[i, 0].grid(True)

        # Plot boxplot
        sns.boxplot(x=df_num[feature], ax=axs[i, 1])
        axs[i, 1].set(title=f'Boxplot of {feature}', xlabel=box_xlabel, ylabel=box_ylabel)
        axs[i, 1].grid(True)

    plt.tight_layout()
    # Prevent the plot being displayed after returning
    plt.close()
    return fig, axs

def nan_value_viz(
    df: pd.DataFrame, 
    figsize: tuple[int, int] | None = None, 
    cmap: str = 'Blues', 
    x_label_rotation: int | float | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
    """Plots a heatmap of missing values in the DataFrame.
    We can use this visualization to identify whether the missing values are 
    missing at random (MAR), missing completely at random (MCAR), 
    or missing not at random (MNAR)

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas DataFrame
    figsize : tuple[int, int], default=None
        Figure size of the plot. If None, `figsize` will be automatically 
        calculated based on the number of features in the DataFrame.
    cmap : str, default='Blues'
        Colour map for the missing values. 
        Read https://matplotlib.org/stable/gallery/color/colormap_reference.html and
        https://matplotlib.org/stable/users/explain/colors/colormaps.html for more info.
    x_label_rotation : int or float, default=None
        This should be a numerical value. With this, we can control the rotation of X-axis
        labels, thus enhancing visibility.

    Returns
    -------
    fig, axs : tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes objects.
    """

    # Calculate figure size if not provided
    figsize = figsize or (df.shape[1] / 4 * 5, 4)
    
    # Create figure and subplot to plot the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.isna(), cbar=False, cmap=cmap, yticklabels=False)
    
    if x_label_rotation is not None:
        ax.tick_params('x', labelrotation=x_label_rotation)
    
    # Prevent the plot being displayed after returning
    plt.close()
    return fig, ax

def count_viz(
    df_cat: pd.DataFrame,
    n_cols: int = 3,
    figsize: tuple[int, int] | None = None,
    x_label_rotation: dict[str, int | float] | None = None
    ) -> tuple[plt.Figure, plt.Axes]:
    """Plots countplots for categorical features in the DataFrame.

    Parameters
    ----------
    df_cat : pd.DataFrame
        A DataFrame containing categorical features in which to plot countplots.
    n_cols : int, default=3
        The number of plots you want in a single row.
    figsize : tuple[int, int], default=None
        Figure size for the entire figure. If None, `figsize` is automatically caluculated
        based on the number of features in DataFrame and `n_cols`.
    x_label_rotation : dict[str, int], default=None
        This dictionary should contain the features' x labels you want to rotate and the
        rotation value. If None, there will be no rotation for any of the x labels.

    Returns
    -------
    fig, axs : tuple[plt.Figure, plt.Axes]
        The matplotlib figure and axes objects.
    """

    cat_features = df_cat.columns.tolist()
    n_cat_features = len(cat_features)

    # Calculate number of rows needed for subplots
    n_rows = math.ceil(n_cat_features / n_cols)

    # Calculate figure size if not provided
    figsize = figsize or (n_cols*4 + 1, n_rows*3)

    # Create subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten() # Flatten the array for easy iteration

    for i, feature in enumerate(cat_features):
        sns.countplot(x=df_cat[feature], ax=axs[i])
        axs[i].set_title(feature)
        axs[i].set_xlabel('')
        axs[i].set_ylabel('')

    # Turn off any unused subplots
    for j in range(n_cat_features, len(axs)):
        axs[j].axis('off')

    if isinstance(x_label_rotation, dict):
        # Create a mapping of titles of axes
        title_of_ax = {ax.get_title(): ax for ax in axs}

        # Apply the label rotation only to matching titles
        for feature, rotation in x_label_rotation.items():
            if feature in title_of_ax:
                title_of_ax[feature].tick_params('x', labelrotation=rotation)

    plt.tight_layout()
    # Prevent the plot being displayed after returning
    plt.close()
    return fig, axs

def pair_viz(
    df_num: pd.DataFrame,
    kind: Literal['scatter', 'kde', 'hist', 'reg'] = 'scatter',
    diag_kind: Literal['auto', 'hist', 'kde'] | None = 'kde',
    mask: Literal['upper', 'lower'] | None = 'upper',
    figsize: tuple[int, int] | None = None,
    plot_kws: dict[str, any] | None = None,
    diag_kws: dict[str, any] | None = None,
    grid_kws: dict[str, any] | None = None
    ) -> sns.PairGrid:
    """Plots a pairplot for numerical features.

    To understand the parameters better, you can refer to Seaborn pairplot
    documentation -> (https://seaborn.pydata.org/generated/seaborn.pairplot.html).

    Parameters
    ----------
    df_num : pd.DataFrame
        A Pandas DataFrame containing numerical features.
    kind : {'scatter', 'kde', 'hist', 'reg'}, default='scatter'
        The type of the pairplot to plot.
    diag_kind : {'auto', 'hist', 'kde'}, default='kde'
        Kind of plot for the diagonal subplots. If 'auto', choose based on whether or 
        not hue is used.
    mask : {'upper', 'lower'}, default='upper'
        Mask out the upper or lower triangle of the plot grid.
    figsize : tuple[int, int], default=None
        The figure size of the pairplot.
    {plot, diag, grid}_kws : dicts, default=None
        Dictionaries of keyword arguments. `plot_kws` are passed to the bivariate plotting function,
        `diag_kws` are passed to the univariate plotting function, and `grid_kws` are passed to the
        PairGrid constructor.

    Returns
    -------
    grid : sns.PairGrid
        The Seaborn PairGrid object with the pairplot.

    Raises
    ------
    ValueError
        If `kind` not in {'scatter', 'kde', 'hist', 'reg'}
        If `diag_kind` not in {'auto', 'hist', 'kde'} or not None
        If `mask` not in {'upper', 'lower'} or not None
    """

    # Validate inputs
    if kind not in ['scatter', 'kde', 'hist', 'reg']:
        raise ValueError("The 'kind' parameter must be 'scatter', 'kde', 'hist', or 'reg'.")
    if diag_kind not in ['auto', 'hist', 'kde'] and diag_kind is not None:
        raise ValueError("The 'diag_kind' parameter must be 'auto', 'hist', 'kde', or None.")
    if mask not in ['upper', 'lower'] and mask is not None:
        raise ValueError("The 'mask' parameter must be 'upper', 'lower', or None.")

    # Calculate figure size if not provided
    n_cols = df_num.shape[1]
    figsize = figsize or (13, n_cols + 3)
    height = figsize[1] / n_cols
    aspect = figsize[0] / figsize[1]

    # Set default plot arguments
    plot_kws = plot_kws or {}
    diag_kws = diag_kws or {}
    grid_kws = grid_kws or {}
    if kind == 'reg' and not plot_kws:
        plot_kws = {'line_kws': {'color': 'red'}}
    
    # Create the pairplot
    grid = sns.pairplot(
        df_num,
        kind=kind,
        diag_kind=diag_kind,
        plot_kws=plot_kws,
        diag_kws=diag_kws,
        grid_kws=grid_kws,
        height=height,
        aspect=aspect
        )

    # Apply mask
    if mask == 'upper':
        for i in range(len(grid.axes)):
            for j in range(i+1, len(grid.axes)):
                grid.axes[i, j].set_visible(False)
    elif mask == 'lower':
        for i in range(len(grid.axes)):
            for j in range(i):
                grid.axes[i, j].set_visible(False)

    plt.tight_layout()
    plt.close() # Prevent the plot being displayed after returning
    return grid

def corr_heatmap_viz(
    df_num: pd.DataFrame,
    method: Literal['pearson', 'kendall', 'spearman'] = 'pearson',
    mask: Literal['upper', 'lower'] | None = 'upper',
    figsize: tuple[int, int] | None = None,
    heatmap_kws: dict[str, any] | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
    """Plots a correlation heatmap for the specified numerical features.

    df_num : pd.DataFrame
        A Pandas DataFrame containing only numerical features.
    method : {'person', 'kendall', 'spearman'}, default='pearson'
        Method of correlation.
    mask : {'upper', 'lower'}, default='upper'
        The way you want to mask the duplicate correlations.
    figsize : tuple[int, int], default=None
        The figure size for the plot.
    heatmap_kws : dict[str, any], default=None
        Keyword arguments for the heatmap plot.
        See https://seaborn.pydata.org/generated/seaborn.heatmap.html for more info.

    Raises
    ------
    ValueError
        If mask is not either of 'upper', 'lower', or None.

    Returns
    -------
    fig, ax: tuple[plt.Figure, plt.Axes]
        The figure object and the axis for the heatmap.
    """

    # Validate inputs
    if mask not in ['upper', 'lower', None]:
        raise ValueError("Invalid mask option. Choose from 'upper', 'lower', or None.")
    # Set default values for heatmap arguments if not provided
    if heatmap_kws is None:
        colors = ["#FF0000", "#FFFF00", "#00FF00"]
        heatmap_kws = {
            "annot": True,
            "cmap": mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors),
            "cbar": True,
            "xticklabels": True,
            "yticklabels": True,
            "fmt": '0.2f'
        }

    # Calculate correlation table
    corr = df_num.corr(method=method)

    # Apply mask
    mask_array = None
    if mask:
        if mask == 'upper':
            mask_array = np.triu(np.ones_like(corr, dtype=bool))
        else:
            mask_array = np.tril(np.ones_like(corr, dtype=bool))
        np.fill_diagonal(mask_array, False)

    # Default figsize
    if figsize is None:
        aspect_ratio = 1.4
        width = max(5, min(20, df_num.shape[1] + 2)) # Cap width between 5 and 20
        height = width / aspect_ratio
        figsize = (width, height)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask_array, ax=ax, **heatmap_kws)
    ax.set_title(f'{method.capitalize()} Correlation Heatmap')
    plt.tight_layout()
    plt.close() # Prevent the plot being displayed after returning
    return fig, ax
    
def crosstab_heatmap_viz(
    df_cat: pd.DataFrame,
    normalize: Literal['index', 'columns', 'both'] = 'index',
    ref_cols: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    heatmap_kws: dict[str, any] | None = None
    ) -> None:
    """Plots heatmap of crosstab for categorical features.

    Parameters
    ----------
    df_cat : pd.DataFrame
        A Pandas DataFrame containing categorical features.
    normalize : {'index', 'columns', 'both'}, default='index'
        The way you want to normalize the crosstab for categorical features.
    ref_col : list[str], default=None
        The columns to compare the other categorical features to. Defaults to all columns.
    figsize : tuple[int, int], default=None
        The figure size for the plots. If None, it will be calculated based on the number
        of unique values in features of the DataFrame.
    heatmap_kws : dict[str, any] , default=None
        Keyword arguments for the heatmap. If None, defaults are used. See Seaborn docs
        for more options: https://seaborn.pydata.org/generated/seaborn.heatmap.html.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If normalize is not either of these {'index', 'columns', 'both'}.
    """

    # Validate inputs
    if normalize not in ['index', 'columns', 'both']:
        raise ValueError("The parameter normalize must be either 'index', 'columns', or 'both'")

    # Set default values for heatmap arguments if not provided
    if heatmap_kws is None:
        colors = ["#FF0000", "#FFFF00", "#00FF00"]
        heatmap_kws = {
            "annot": True,
            "cmap": mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors),
            "cbar": False,
            "xticklabels": True,
            "yticklabels": True,
            "fmt": '0.2f'
        }

    # Get the list of categorical columns and the reference columns
    cat_cols = df_cat.columns.tolist()
    ref_cols = ref_cols or cat_cols

    # The number of unique values for each column for dynamic figure sizing
    n_unique = {col: df_cat[col].nunique() for col in set(cat_cols + ref_cols)}

    # Function to generate and plot the crosstab heatmap
    def _plot(normalize):
        if normalize == 'both':
            fig_width = max(n_unique[columns_col] * 2, 13)
            fig_height = min(n_unique[index_col]*2 - 1, 26)
            figsize = figsize or (fig_width, fig_height)
            fig, axs = plt.subplots(1, 2, figsize=figsize)

            # Normalize by index and by columns
            index_table = pd.crosstab(df_cat[index_col], df_cat[columns_col], normalize='index') * 100
            columns_table = pd.crosstab(df_cat[index_col], df_cat[columns_col], normalize='columns') * 100

            sns.heatmap(index_table, ax=axs[0], **heatmap_kws)
            axs[0].set_title(f"{index_col} vs {columns_col} (Index Normalized)")
            sns.heatmap(columns_table, ax=axs[1], **heatmap_kws)
            axs[1].set_title(f"{index_col} vs {columns_col} (Column Normalized)")
        else:
            table = pd.crosstab(df_cat[index_col], df_cat[columns_col], normalize=normalize) * 100

            fig_width = n_unique[columns_col] * 2
            fig_height = n_unique[index_col]*2 - 1
            figsize = figsize or (fig_width, fig_height)
            plt.figure(figsize=(fig_width, fig_height))

            sns.heatmap(table, **heatmap_kws)
            plt.title(f"{index_col} vs {columns_col}")

        # Adjust layout
        plt.tight_layout()
        plt.show()

    # Plot the heatmap of crosstabs for each pair of categorical features
    used_index_cols = []
    for index_col in cat_cols:
        for columns_col in ref_cols:
            if index_col == columns_col or columns_col in used_index_cols:
                continue

            _plot(normalize)
            print("-" * 150)

        # Track the columns we have already used for plotting
        used_index_cols.append(index_col)

def num_cat_viz(
    df: pd.DataFrame,
    num_cols: list[str] | str,
    cat_cols: str | list[str],
    kind: Literal['box', 'point', 'kde'] = 'box',
    col_wrap: int = 2,
    figsize: tuple[int, int] | None = None,
    grid: bool = True,
    estimators: list[Literal['mean', 'median', 'mode', 'std', 'min', 'max']] = ['mean', 'median'],
    class_imbalance: bool = True
    ) -> tuple[plt.Figure, plt.Axes]:
    """Plots box, point, and kde plots for numerical-categorical analysis.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas DataFrame.
    num_cols : list or string of numerical column(s) in 'df' 
        This must be a list of numerical columns if 'cat_cols' is a string. If 'cat_cols'
        is a list of categorical columns, this must me a string.
    cat_cols : list or string of categorical columns(s) in 'df'
        This must be a list of categorical columns if 'num_cols' is a string. If 'num_cols'
        is a list of numerical columns, this must be a string.
    kind : {'box', 'point', 'kde'}, default=True
        Type of the plot you want to plot.
    col_wrap : int, default=2
        This parameter controls how many plots should be in a single row.
    grid : bool, default=True
        This parameter controls whether grid should be applied in plots. For box plots, it
        will only apply y-axis grids. For point and kde plot it will apply grid to both axes.
    estimators: list of {'mean', 'median', 'mode', 'std', 'min', 'max'}, default=['mean', 'median']
        Estimators for the point plot.
    class_imbalance : bool, default=True
        This parameter controls whether we should consider class imbalance in categorical
        features when plotting kde plots. If True, it will consider class imbalance and plot
        accordingly. If False, it will not consider the the class imbalancement.

    Returns
    -------
    fig, axs : tuple[plt.Figure, plt.Axes]
        The figure and axes object containing the plots.

    Raises
    ------
    ValueError
        If kind is not either of 'box', 'point', or 'kde'.
    TypeError
        If 'num_cols' is a list while 'cat_cols' is not a string. Or, 'cat_cols' is a list
        while 'num_cols' is not a string.
    """

    def _setup_fig_axes(list_type: list, figsize):
        n_cols = len(list_type)
        n_rows = int(np.ceil(n_cols / col_wrap))
        figsize = figsize or (col_wrap * 6, n_rows * 4)
        fig, axs = plt.subplots(n_rows, col_wrap, figsize=figsize)
        if n_rows > 1:
            axs = axs.flatten()

        # Turn off unused subplots
        for ax in axs[n_cols:]:
            fig.delaxes(ax)
        axs = axs[:n_cols] # Update the `axs` variable to exclude the deleted axes
        return fig, axs

    def _plot_box():
        if isinstance(num_cols, list) and isinstance(cat_cols, str):
            fig, axs = _setup_fig_axes(num_cols, figsize)
            for i, col in enumerate(num_cols):
                sns.boxplot(data=df, x=cat_cols, y=col, hue=cat_cols, legend=False, ax=axs[i])
                axs[i].set_title(f"{col} vs {cat_cols}")
        elif isinstance(num_cols, str) and isinstance(cat_cols, list):
            fig, axs = _setup_fig_axes(cat_cols, figsize)
            for i, col in enumerate(cat_cols):
                sns.boxplot(data=df, x=col, y=num_cols, hue=col, legend=False, ax=axs[i])
                axs[i].set_title(f"{col} vs {num_cols}")

        # Remove x and y labels and apply grid lines for y-axis
        for ax in axs:
            ax.set_xlabel('')
            ax.set_ylabel('')
            if grid:
                ax.yaxis.grid(True)

        plt.tight_layout()
        plt.close()
        return fig, axs

    def _plot_point():
        if isinstance(num_cols, list) and isinstance(cat_cols, str):
            fig, axs = _setup_fig_axes(num_cols, figsize)
            for i, col in enumerate(num_cols):
                for estimator in estimators:
                    sns.pointplot(data=df, x=cat_cols, y=col, errorbar=None,
                                label=estimator, estimator=estimator, ax=axs[i])
                axs[i].set_title(f"{col} vs {cat_cols}")
        elif isinstance(num_cols, str) and isinstance(cat_cols, list):
            fig, axs = _setup_fig_axes(cat_cols, figsize)
            for i, col in enumerate(cat_cols):
                for estimator in estimators:
                    sns.pointplot(data=df, x=col, y=num_cols, errorbar=None,
                                label=estimator, estimator=estimator, ax=axs[i])
                axs[i].set_title(f"{col} vs {num_cols}")
            
        # Remove x and y labels and apply grid lines
        for ax in axs:
            ax.set_xlabel('')
            ax.set_ylabel('')
            if grid:
                ax.grid(True)

        plt.tight_layout()
        plt.close()
        return fig, axs

    def _plot_kde():
        if isinstance(num_cols, list) and isinstance(cat_cols, str):
            fig, axs = _setup_fig_axes(num_cols, figsize)
            for i, col in enumerate(num_cols):
                if class_imbalance:
                    for value in df[cat_cols].unique():
                        sns.kdeplot(data=df[col][df[cat_cols] == value], label=value, ax=axs[i])
                        axs[i].legend()
                else:
                    sns.kdeplot(data=df, x=col, hue=cat_cols, ax=axs[i])
                    legend = axs[i].get_legend() # Get the current axis legend
                    legend.set_title('') # Remove the legend title
                axs[i].set_title(f"{col} vs {cat_cols}")
        elif isinstance(num_cols, str) and isinstance(cat_cols, list):
            fig, axs = _setup_fig_axes(cat_cols, figsize)
            for i, col in enumerate(cat_cols):
                if class_imbalance:
                    for value in df[col].unique():
                        sns.kdeplot(data=df[num_cols][df[col] == value], label=value, ax=axs[i])
                        axs[i].legend()
                else:
                    sns.kdeplot(data=df, x=col, hue=cat_cols, ax=axs[i])
                    legend = axs[i].get_legend() # Get the current axis legend
                    legend.set_title('') # Remove the legend title
                axs[i].set_title(f"{col} vs {num_cols}")
            
        # Apply grid lines
        if grid:
            for ax in axs:
                ax.grid(True)

        plt.tight_layout()
        plt.close()
        return fig, axs

    if not (
        (isinstance(num_cols, list) and isinstance(cat_cols, str)) or
        (isinstance(num_cols, str) and isinstance(cat_cols, list))
        ):
        raise TypeError(
            "Expected 'num_cols' to be a list and 'cat_cols' to be a string, "
            "or 'num_cols' to be a string and 'cat_cols' to be a list."
            )

    if kind == 'box':
        fig, axs = _plot_box()
    elif kind == 'point':
        fig, axs = _plot_point()
    elif kind == 'kde':
        fig, axs = _plot_kde()
    else:
        raise ValueError("The 'kind' parameter must be either of 'box', 'point', or 'kde'")

    return fig, axs

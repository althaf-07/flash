import math
from typing import Literal
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, QuantileTransformer

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
    figsize : tuple[int, int], optional, default=None
        Figure size of the plot. If None, `figsize` will be automatically 
        calculated using number of features.
    hist_xlabel : str | None, optional, default=None
        X-axis label for histograms.
    hist_ylabel : str | None, optional, default=None
        Y-axis label for histograms.
    box_xlabel : str | None, optional, default=None
        X-axis label for boxplots.
    box_ylabel : str | None, optional, default=None
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
    figsize : Tuple[int, int] | None, optional, default=None
        Figure size of the plot. If None, `figsize` will be automatically 
        calculated based on the number of features in the DataFrame.
    cmap : str, default='Blues'
        Colour map for the missing values. 
        Read https://matplotlib.org/stable/gallery/color/colormap_reference.html and
        https://matplotlib.org/stable/users/explain/colors/colormaps.html for more info.
    x_label_rotation : int | float | None, optional, default=None
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
    figsize : tuple[int, int] | None, optional
        Figure size for the entire figure. If None, `figsize` is automatically caluculated
        based on the number of features in DataFrame and `n_cols`.
    x_label_rotation : dict[str, int | float] | None, optional, default=None
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
        diag_kind : {'auto', 'hist', 'kde'} | None, optional, default='kde'
            Kind of plot for the diagonal subplots. If 'auto', choose based on whether or 
            not hue is used.
        mask : {'upper', 'lower'} | None, optional, default='upper'
            Mask out the upper or lower triangle of the plot grid.
        figsize : tuple[int, int] | None, optional, default=None
            The figure size of the pairplot.
        plot_kws : dict[str, any] | None, optional, default=None
            Arguments passed to the bivariate plotting function.
        diag_kws : dict[str, any] | None, optional, default=None
            Arguments passed to the univariate plotting function.
        grid_kws : dict[str, any] | None, optional, default=None
            Arguments passed to the PairGrid constructor.

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
        # Prevent the plot being displayed after returning
        plt.close()
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
        mask : {'upper', 'lower'} | None, optional, default='upper'
            The way you want to mask the duplicate correlations.
        figsize : tuple[int, int] | None, optional, default=None
            The figure size for the plot.
        heatmap_kws : dict[str, any] | None, optional, default=None
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
    ref_col : list[str] | None, optional, default=None
        The columns to compare the other categorical features to. Defaults to all columns.
    figsize : tuple[int, int] | None, optional, default=None
        The figure size for the plots. If None, it will be calculated based on the number
        of unique values in features of the DataFrame.
    heatmap_kws : dict[str, any] | None, optional, default=None
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
            fig, axs = plt.subplots(1, 2, figsize=(fig_width, fig_height))

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
    numerical_features: Union[List[str], str],
    categorical_features: Union[str, List[str]],
    kind: Literal['box', 'kde', 'point'] = 'box',
    col_wrap: int = 2,
    figsize: Optional[Tuple[int, int]] = None,
    grid: bool = True
):
    def _setup_fig_axes(n_features, col_wrap, figsize):
        """Set up figure and axes."""
        n_rows = int(np.ceil(n_features / col_wrap))
        figsize = figsize or (col_wrap * 6, n_rows * 4)
        fig, axs = plt.subplots(n_rows, col_wrap, figsize=figsize)
        axs = axs.flatten() if n_rows > 1 else [axs]
        # Turn off unused subplots
        for ax in axs[n_features:]:
            fig.delaxes(ax)
        return fig, axs

    def _apply_grid(ax):
        """Apply grid lines to the plots."""
        if grid:
            if kind == 'box':
                for a in ax:
                    a.yaxis.grid(True)
            else:
                for a in ax:
                    a.grid(True)

    def _plot_helper(plot_func, y_var, hue_var, title_template):
        """Helper function to plot based on the kind of plot."""
        if isinstance(y_var, list) and isinstance(hue_var, str):
            fig, ax = _setup_fig_axes(len(y_var), col_wrap, figsize)
            for i, feature in enumerate(y_var):
                plot_func(df, feature, hue_var, ax[i])
                ax[i].set_title(title_template.format(feature, hue_var))
                ax[i].set_ylabel('')

        elif isinstance(hue_var, list) and isinstance(y_var, str):
            fig, ax = _setup_fig_axes(len(hue_var), col_wrap, figsize)
            for i, feature in enumerate(hue_var):
                plot_func(df, y_var, feature, ax[i])
                ax[i].set_title(title_template.format(y_var, feature))
                ax[i].set_ylabel('')
        else:
            raise ValueError(
                "Expected 'numerical_features' to be a list and 'categorical_features' to be a string, or "
                "'numerical_features' to be a string and 'categorical_features' to be a list."
            )
            
        _apply_grid(ax)
        plt.tight_layout()
        plt.show()

    def _boxplot_func(df, y, hue, ax):
        """Plot boxplot."""
        sns.boxplot(data=df, y=y, hue=hue, ax=ax)

    def _kdeplot_func(df, y, hue, ax):
        """Plot KDE plot."""
        for value in df[hue].unique():
            sns.kdeplot(df[y][df[hue] == value], label=value, ax=ax, warn_singular=False)
        ax.set_yticks([])
        ax.legend()

    def _pointplot_func(df, y, hue, ax):
        """Plot point plot."""
        sns.pointplot(data=df, x=hue, y=y, errorbar=None, label='Mean', ax=ax)
        sns.pointplot(data=df, x=hue, y=y, errorbar=None, label='Median', ax=ax, estimator='median')

    def _plot_box():
        """Call boxplot plotting function."""
        _plot_helper(_boxplot_func, numerical_features, categorical_features, '{} by {}')

    def _plot_kde():
        """Call KDE plot function."""
        _plot_helper(_kdeplot_func, numerical_features, categorical_features, '{} by {}')

    def _plot_point():
        """Call point plot function."""
        _plot_helper(_pointplot_func, numerical_features, categorical_features, '{} by {}')

    if kind == 'box':
        _plot_box()
    elif kind == 'kde':
        _plot_kde()
    elif kind == 'point':
        _plot_point()
    else:
        raise ValueError("Invalid 'kind' parameter. Choose from 'box', 'kde', or 'point'.")
        
def feature_transform_viz(
        df: pd.DataFrame, 
        numerical_features: List[str], 
        result: Literal['data', 'hist', 'qq'] = 'hist', 
        figsize: Optional[Tuple[int, int]] = None
) -> Optional[Dict[str, pd.DataFrame]]:
    if result not in ['data', 'hist', 'qq']:
        raise ValueError(f"Invalid result value: {result}. Choose from 'data', 'hist', or 'qq'.") 

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

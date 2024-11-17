# Importing key functions from modules to make them accessible at the package level
from .analysis import (
    stats_moments, hist_box_viz, nan_value_viz, count_viz, pair_viz, corr_heatmap_viz, 
    crosstab_heatmap_viz, num_cat_viz, feature_transform_viz
    )
from .preprocessing import (
    extract_features, calc_na_values
    )
    
from .evaluator import (
    basic_imputer, eval_basic_imputer, advanced_numerical_imputer, eval_advanced_numerical_imputer, advanced_categorical_imputer,
    eval_advanced_categorical_imputer
    )

# Metadata
__version__ = '0.1.0'
__author__ = 'Althaf Muhammad'
__email__ = 'flashlib0308@gmail.com'
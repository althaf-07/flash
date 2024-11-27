import os
import pandas as pd

def export(df: pd.DataFrame, file_path: str, force_overwrite: bool = False) -> None:
    """Exports a DataFrame to a CSV file with overwrite control.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to export.
    file_path : str
        Path where the DataFrame will be saved as a CSV file.
    force_overwrite : bool, default=False
        Whether to overwrite the file if it exists, by default False.

    Raises
    ------
    TypeError
        If `df` is not a pandas DataFrame.
    ValueError
        If `file_path` already exists and `force_overwrite` is False.
    """

    if os.path.exists(file_path) and not force_overwrite:
        raise FileExistsError(f"File {file_path} already exists. Set force_overwrite=True to overwrite it.")
    
    df.to_csv(file_path, index=False)
    print(f"Data exported to {file_path}")
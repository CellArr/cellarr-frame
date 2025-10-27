from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tiledb
from cellarr_array import SparseCellArray

from .base import CellArrFrame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class SparseCellArrFrame(CellArrFrame):
    """Handler for sparse dataframes using a 2D sparse TileDB array."""

    def __init__(self, uri: str, config_or_context: Optional[tiledb.Config] = None):
        super().__init__(uri, config_or_context)
        self._array = SparseCellArray(uri=self.uri, attr="value", config_or_context=self._ctx, return_sparse=False)

    def write_dataframe(self, df: pd.DataFrame, **kwargs) -> None:
        """Write a sparse pandas DataFrame to a 2D sparse TileDB array.

        The DataFrame is converted to a coordinate format (row_idx, col_idx, value).

        Args:
            df:
                The sparse pandas DataFrame to write.

            **kwargs:
                Additional arguments for the write operation.
        """
        # Convert the sparse DataFrame to a COO-like format
        sdf = df.stack(future_stack=True).dropna()
        if sdf.empty:
            return

        coords = sdf.index.to_frame()
        rows = coords.iloc[:, 0].to_numpy()
        cols = coords.iloc[:, 1].to_numpy()
        values = sdf.to_numpy(dtype=str)

        with self._array.open_array(mode="w") as A:
            A[rows, cols] = values

    def read_dataframe(
        self,
        subset: Optional[Union[slice, int, str]] = None,
        columns: Optional[List[str]] = None,
        query: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Read a sparse pandas DataFrame from a 2D sparse TileDB array.

        Args:
            subset:
                A slice or index to select rows.

            columns:
                A list of column names to read.

            query:
                A TileDB query condition string.

            **kwargs:
                Additional arguments for the read operation.

        Returns:
            The reconstructed sparse pandas DataFrame.
        """
        if query:
            data = self._array[query]
        elif subset is not None:
            data = self._array[subset]
        else:
            data = self._array[:]

        if not data or not data["value"].size:
            return pd.DataFrame()

        rows = data["rows"]
        cols = data["cols"]
        values = data["value"]

        # Reconstruct the sparse DataFrame
        s = pd.Series(values, index=[rows, cols])
        df = s.unstack()

        try:
            df.index = pd.to_numeric(df.index)
        except (ValueError, TypeError):
            pass

        try:
            df.columns = pd.to_numeric(df.columns)
        except (ValueError, TypeError):
            pass

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        if columns:
            df = df[columns]

        return df

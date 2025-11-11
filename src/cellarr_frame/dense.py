import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import CellArrayFrame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class DenseCellArrayFrame(CellArrayFrame):
    """Handler for dense dataframes using TileDB's native dataframe support."""

    def write_dataframe(self, df: pd.DataFrame, **kwargs) -> None:
        """Write a dense pandas DataFrame to a 1D TileDB array.

        This assumes the array was created using tiledb.from_pandas or
        the helper function. It appends the dataframe starting at row 0.

        Args:
            df:
                The pandas DataFrame to write.

            **kwargs:
                Additional arguments.
        """
        self.append_dataframe(df, row_offset=0)

    def read_dataframe(
        self,
        columns: Optional[List[str]] = None,
        query: Optional[str] = None,
        subset: Optional[Union[slice, int, str]] = None,
        primary_key_column_name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Read a pandas DataFrame from the TileDB array.

        Args:
            columns:
                A list of column names to read.

            query:
                A TileDB query condition string.

            subset:
                A slice or index to select rows.

            primary_key_column_name:
                Name of the primary key column.

            **kwargs:
                Additional arguments for the read operation.

        Returns:
            The pandas DataFrame.
        """
        dim_name = self.dim_names[0]
        result = None

        with self.open_array(mode="r") as A:
            if query:
                if primary_key_column_name is None:
                    pk_candidates = [d for d in self.dim_names if d in self.attr_names or d == "__tiledb_rows"]
                    if len(pk_candidates) == 1:
                        primary_key_column_name = pk_candidates[0]
                    elif "__tiledb_rows" in self.dim_names:
                        primary_key_column_name = "__tiledb_rows"
                    else:
                        raise ValueError("'primary_key_column_name' must be provided for queries on dense frames.")

                all_columns = columns.copy() if columns else [A.attr(i).name for i in range(A.nattr)]
                if primary_key_column_name not in all_columns and primary_key_column_name in self.attr_names:
                    all_columns.append(primary_key_column_name)

                q = A.query(cond=query, attrs=all_columns, **kwargs)
                data = q.df[:]

                if primary_key_column_name in data.columns:
                    mask = A.attr(primary_key_column_name).fill
                    if isinstance(mask, bytes):
                        mask = mask.decode("ascii")
                    filtered_df = data[data[primary_key_column_name] != mask]
                else:
                    filtered_df = data

                result = filtered_df
                if columns:
                    result = result[columns]

            elif subset is not None:
                adjusted_subset = subset
                if isinstance(subset, slice) and subset.stop is not None:
                    if subset.start is None or subset.stop > subset.start:
                        if subset.stop > 0:
                            adjusted_subset = slice(subset.start, subset.stop - 1, subset.step)

                result = A.df[adjusted_subset]
                if columns:
                    result = result[columns]

            else:
                result = A.df[:]
                if columns:
                    result = result[columns]

        if dim_name in result.columns:
            user_requested_dim = columns is not None and dim_name in columns
            dim_is_also_attr = dim_name in self.attr_names

            if not user_requested_dim and not dim_is_also_attr:
                result = result.drop(columns=[dim_name], errors="ignore")

        # Replace null characters with NaN
        re_null = re.compile(pattern="\x00")
        result = result.replace(regex=re_null, value=np.nan)

        return result

    def get_shape(self) -> tuple:
        """Get the shape (number of rows) of the dense dataframe array."""
        with self.open_array(mode="r") as A:
            non_empty = A.nonempty_domain()
            if non_empty is None or self.ndim == 0:
                return (0,)

            if self.ndim == 1:
                return (non_empty[0][1] + 1,)

            return tuple(ned[1] + 1 for ned in non_empty)

    def append_dataframe(self, df: pd.DataFrame, row_offset: Optional[int] = None) -> None:
        """Append a pandas DataFrame to the dense TileDB array.

        Args:
            df:
                The pandas DataFrame to write.

            row_offset:
                Row offset to write the rows to.
        """
        if row_offset is None:
            row_offset = self.get_shape()[0]

        write_data = {col: df[col].to_numpy() for col in df.columns}

        with self.open_array(mode="w") as A:
            end_row = row_offset + len(df)
            A[row_offset:end_row] = write_data

    def __getitem__(self, key):
        if isinstance(key, str):  # Column selection
            return self.read_dataframe(columns=[key])
        if isinstance(key, list):  # Column selection
            return self.read_dataframe(columns=key)
        if isinstance(key, (slice, int)):  # Row selection
            return self.read_dataframe(subset=key)
        if isinstance(key, tuple):  # Row and column selection
            rows, cols = key
            cols_list = cols if isinstance(cols, list) else [cols]
            return self.read_dataframe(subset=rows, columns=cols_list)

        raise TypeError(f"Unsupported key type for slicing: {type(key)}")

    @property
    def shape(self) -> tuple:
        """Get the shape (rows, columns) of the dataframe."""
        with self.open_array(mode="r") as A:
            non_empty = A.nonempty_domain()
            num_cols = len(self.columns)
            if non_empty is None or self.ndim == 0:
                return (0, num_cols)

            if self.ndim == 1:
                return (non_empty[0][1] + 1, num_cols)

            return (non_empty[0][1] + 1, num_cols)

    @property
    def columns(self) -> pd.Index:
        """Get the column names (attributes) of the dataframe."""
        return pd.Index(self.attr_names)

    @property
    def index(self) -> pd.Index:
        """Get the row index of the dataframe."""
        return pd.RangeIndex(start=0, stop=self.shape[0], step=1)

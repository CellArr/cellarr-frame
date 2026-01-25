from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
import tiledb
from cellarr_array import SparseCellArray

from .base import CellArrayFrame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class SparseCellArrayFrame(CellArrayFrame):
    """Handler for sparse dataframes using a 2D sparse TileDB array.

    This class wraps a `cellarr_array.SparseCellArray` instance, assuming
    it's a 2D sparse array with string/object data.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        tiledb_array_obj: Optional[tiledb.Array] = None,
        mode: Optional[Literal["r", "w", "d", "m"]] = None,
        config_or_context: Optional[Union[tiledb.Config, tiledb.Ctx]] = None,
    ):
        """Initialize the object.

        Args:
            uri:
                URI to the array.

            tiledb_array_obj:
                Optional, an already opened ``tiledb.Array`` instance.

            mode:
                Default open mode.

            config_or_context:
                Optional config or context object.
        """
        super().__init__(uri=uri, tiledb_array_obj=tiledb_array_obj, mode=mode, config_or_context=config_or_context)
        self._array = SparseCellArray(
            uri=self.uri,
            tiledb_array_obj=tiledb_array_obj,
            attr="value",
            mode=mode,
            config_or_context=self._ctx,
            return_sparse=False,
        )

    def write_dataframe(self, df: pd.DataFrame, **kwargs) -> None:
        """Write a sparse pandas DataFrame to a 2D sparse TileDB array.

        The DataFrame is converted to a coordinate format (row_idx, col_idx, value).

        Args:
            df:
                The sparse pandas DataFrame to write.

            **kwargs:
                Additional arguments for the write operation.
        """
        if df is None:
            return

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
        """Read a pandas DataFrame from the TileDB array.

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
            The pandas DataFrame.
        """
        if columns is not None:
            final_columns = pd.Index(columns)
        else:
            final_columns = self.columns

        final_index = None
        row_dim_dtype = self._array.dim_dtypes[0]

        if subset is None:
            if query is None:
                final_index = self.index
        elif isinstance(subset, slice):
            if np.issubdtype(row_dim_dtype, np.integer):
                start = subset.start if subset.start is not None else 0

                if subset.stop is not None:
                    final_index = pd.RangeIndex(
                        start=start, stop=subset.stop, step=subset.step if subset.step is not None else 1
                    )
                else:
                    all_rows = self.index
                    final_index = all_rows[all_rows.slice_indexer(start, None, subset.step)]
            else:
                all_rows = self.index
                final_index = all_rows[subset]
                subset = final_index.tolist()
        elif isinstance(subset, (int, str)):
            final_index = pd.Index([subset])
        elif isinstance(subset, list):
            final_index = pd.Index(subset)

        if query:
            slice_key = query
        elif subset is not None and columns is not None:
            slice_key = (subset, columns)
        elif subset is not None:
            slice_key = (subset, slice(None))
        elif columns is not None:
            slice_key = (slice(None), columns)
        else:
            slice_key = (slice(None), slice(None))

        data = self._array[slice_key]

        if not data or not data["value"].size:
            return pd.DataFrame(index=final_index, columns=final_columns)

        rows = data[self._array.dim_names[0]]
        cols = data[self._array.dim_names[1]]
        values = data["value"]

        if len(rows) > 0 and isinstance(rows[0], bytes):
            rows = [r.decode() for r in rows]
        if len(cols) > 0 and isinstance(cols[0], bytes):
            cols = [c.decode() for c in cols]

        s = pd.Series(values, index=[rows, cols])
        s.index.names = self._array.dim_names

        df = s.unstack()
        df_index_to_use = final_index if final_index is not None else df.index
        df = df.reindex(index=df_index_to_use, columns=final_columns)

        try:
            df.index = pd.to_numeric(df.index)
        except (ValueError, TypeError):
            pass

        try:
            df.columns = pd.to_numeric(df.columns)
        except (ValueError, TypeError):
            pass

        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        if columns:
            df = df[columns]

        df.index.name = None
        df.columns.name = None

        return df

    def get_shape(self) -> tuple:
        """Get the shape based on the non-empty domain for sparse arrays."""
        with self._array.open_array(mode="r") as A:
            non_empty = A.nonempty_domain()
            if non_empty is None:
                return (0, 0)

            rows_ned, cols_ned = non_empty

            if np.issubdtype(self._array.dim_dtypes[0], np.str_):
                n_rows = len(A.unique_dim_values(self._array.dim_names[0]))
            else:
                n_rows = rows_ned[1] + 1 if isinstance(rows_ned[1], (int, np.integer)) else 0

            if np.issubdtype(self._array.dim_dtypes[1], np.str_):
                n_cols = len(A.unique_dim_values(self._array.dim_names[1]))
            else:
                n_cols = cols_ned[1] + 1 if isinstance(cols_ned[1], (int, np.integer)) else 0

            return (n_rows, n_cols)

    def append_dataframe(self, df: pd.DataFrame, row_offset: Optional[int] = None) -> None:
        """Append data points from a pandas DataFrame to the sparse TileDB array.

        If row_offset is provided, adjusts the row indices of the appended data.
        Assumes integer row dimensions for offset calculation.

        Args:
            df:
                The pandas DataFrame to write.

            row_offset:
                Row offset to write the rows to.
        """
        if df.empty:
            return

        if row_offset is None:
            row_dim_type = self._array.dim_dtypes[0]
            if np.issubdtype(row_dim_type, np.integer):
                current_shape = self.get_shape()

                if current_shape[0] is not None and current_shape[0] > 0:
                    row_offset = current_shape[0]
                else:
                    row_offset = 0
            else:
                row_offset = 0

        sdf = df.stack(future_stack=True).dropna()
        if sdf.empty:
            return

        coords = sdf.index.to_frame()
        rows = coords.iloc[:, 0].to_numpy()
        cols = coords.iloc[:, 1].to_numpy()
        values = sdf.to_numpy(dtype=str)

        rows_adjusted = rows
        if row_offset != 0:
            row_dim_type = self._array.dim_dtypes[0]
            if np.issubdtype(row_dim_type, np.integer):
                rows_adjusted = rows + row_offset
            else:
                print(f"Warning: Row offset {row_offset} ignored for non-integer row dimension.")

        with self._array.open_array(mode="w") as A:
            A[rows_adjusted, cols] = values

    def __getitem__(self, key):
        """Optimized slicing for the DataFrame."""
        if isinstance(key, str):  # Column selection, e.g., df['col_A']
            return self.read_dataframe(columns=[key])

        if isinstance(key, list):  # Column selection, e.g., df[['col_A', 'col_B']]
            return self.read_dataframe(columns=key)

        if isinstance(key, (slice, int)):  # Row selection, e.g., df[0:10] or df[3]
            return self.read_dataframe(subset=key)

        if isinstance(key, tuple):  # Row and column selection, e.g., df[0:10, ['col_A']]
            rows, cols = key
            cols_list = cols if isinstance(cols, list) else [cols]

            # Support positional indexing for columns
            if cols_list and all(isinstance(c, int) for c in cols_list):
                all_cols = self.columns
                try:
                    cols_list = [all_cols[i] for i in cols_list]
                except IndexError:
                    raise IndexError("Column index out of bounds")

            return self.read_dataframe(subset=rows, columns=cols_list)

        raise TypeError(f"Unsupported key type for slicing: {type(key)}")

    @property
    def shape(self) -> tuple:
        """Get the shape (unique rows, unique columns) of the dataframe."""
        return self.get_shape()

    @property
    def columns(self) -> pd.Index:
        """Get the column names (unique values from 2nd dim) of the dataframe."""
        with self._array.open_array("r") as A:
            cols = A.unique_dim_values(self._array.dim_names[1])
            decoded_cols = [c.decode() if isinstance(c, bytes) else c for c in cols]

            try:
                return pd.Index(pd.to_numeric(decoded_cols))
            except (ValueError, TypeError):
                return pd.Index(decoded_cols)

    @property
    def index(self) -> pd.Index:
        """Get the row index (unique values from 1st dim) of the dataframe."""
        with self._array.open_array("r") as A:
            rows = A.unique_dim_values(self._array.dim_names[0])
            decoded_rows = [r.decode() if isinstance(r, bytes) else r for r in rows]

            try:
                return pd.Index(pd.to_numeric(decoded_rows))
            except (ValueError, TypeError):
                return pd.Index(decoded_rows)

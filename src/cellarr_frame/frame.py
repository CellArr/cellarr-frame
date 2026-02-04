from typing import Any, List, Optional

import pandas as pd
import tiledb

from .base import CellArrayBaseFrame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CellArrayFrame(CellArrayBaseFrame):
    """Implementation for TileDB DataFrames."""

    def _read_slice(self, rows: Any, cols: Optional[List[str]]) -> pd.DataFrame:
        """Read data using direct slicing.

        Args:
            rows:
                Slice object, specific index (int/str), or list of indices.

            cols:
                List of column names to retrieve.
        """
        if isinstance(rows, slice):
            start = rows.start
            stop = rows.stop
            step = rows.step

            if stop is not None and isinstance(stop, int):
                if stop == 0 and start is None:
                    stop = -1
                else:
                    stop -= 1

            rows = slice(start, stop, step)

        with self.open_array(mode="r") as array:
            attrs = cols if cols is not None else self.column_names
            query = array.query(attrs=attrs)

            return query.df[rows]

    def _read_query(self, condition: str, columns: Optional[List[str]]) -> pd.DataFrame:
        """Read data using a string query condition.

        Args:
            condition:
                TileDB query string (e.g. "val > 5.0").

            columns:
                List of column names to retrieve.
        """
        with self.open_array(mode="r") as array:
            attrs = columns if columns is not None else self.column_names
            return array.query(cond=condition, attrs=attrs).df[:]

    def write_batch(self, data: pd.DataFrame, append: bool = True, **kwargs) -> None:
        """Write a batch of data to the frame.

        Args:
            data:
                Pandas DataFrame to write.

            append:
                If True, appends to existing array. If False, might overwrite/schema_only
                depending on lower-level tiledb.from_pandas behavior, but mostly used for appending.
        """
        mode = "append" if append else "ingest"
        tiledb.from_pandas(uri=self.uri, dataframe=data, mode=mode, ctx=self._ctx, **kwargs)

    @classmethod
    def create(
        cls, uri: str, data: pd.DataFrame, index_dims: Optional[List[str]] = None, full_domain: bool = True, **kwargs
    ):
        """Helper to create a new CellFrame from a dataframe.

        Args:
            uri:
                Path to create array.

            data:
                Initial dataframe (can be empty schema if used with mode='schema_only').

            index_dims:
                Columns to use as dimensions (indices).

            full_domain:
                Whether to allow the domain to extend to the full range of the dtype (default True).
        """
        tiledb.from_pandas(uri, data, index_dims=index_dims, full_domain=full_domain, **kwargs)
        return cls(uri=uri)

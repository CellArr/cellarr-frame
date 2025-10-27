import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tiledb

from .base import CellArrFrame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class DenseCellArrFrame(CellArrFrame):
    """Handler for dense dataframes using TileDB's native dataframe support."""

    def write_dataframe(self, df: pd.DataFrame, **kwargs) -> None:
        """Write a dense pandas DataFrame to a 1D TileDB array.

        Args:
            df:
                The pandas DataFrame to write.

            **kwargs:
                Additional arguments passed to tiledb.from_pandas.
        """
        tiledb.from_pandas(self.uri, df, ctx=self._ctx, **kwargs)

    def read_dataframe(
        self,
        columns: Optional[List[str]] = None,
        query: Optional[str] = None,
        subset: Optional[Union[slice, int, str]] = None,
        primary_key_column_name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Read a dense pandas DataFrame from a 1D TileDB array.

        Args:
            subset:
                A slice or index to select rows.

            columns:
                A list of column names to read.

            query:
                A TileDB query condition string.

            **kwargs:
                Additional arguments passed to tiledb.open_dataframe.

        Returns:
            The pandas DataFrame.
        """
        with tiledb.open(self.uri, "r", ctx=self._ctx) as A:
            if query:
                if primary_key_column_name is None:
                    raise ValueError("'primary_key_column_name' must be provided for queries.")

                all_columns = columns.copy() if columns else [A.attr(i).name for i in range(A.nattr)]
                if primary_key_column_name not in all_columns:
                    all_columns.append(primary_key_column_name)

                q = A.query(cond=query, attrs=all_columns, **kwargs)
                data = q.df[:]

                # Filter out fill values
                mask = A.attr(primary_key_column_name).fill
                if isinstance(mask, bytes):
                    mask = mask.decode("ascii")

                filtered_df = data[data[primary_key_column_name] != mask]

                if columns:
                    result = filtered_df[columns]
                else:
                    result = filtered_df.drop(columns=primary_key_column_name, errors="ignore")

            elif subset is not None:
                result = A.df[subset]
                if columns:
                    result = result[columns]
            else:
                result = A.df[:]
                if columns:
                    result = result[columns]

        # Replace null characters with NaN
        re_null = re.compile(pattern="\x00")
        result = result.replace(regex=re_null, value=np.nan)

        return result

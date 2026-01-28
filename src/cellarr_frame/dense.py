import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tiledb

from .base import CellArrayFrame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class DenseCellArrayFrame(CellArrayFrame):
    """Handler for dense dataframes using TileDB's native dataframe support."""

    @classmethod
    def from_dataframe(cls, uri: str, df: pd.DataFrame, **kwargs) -> "DenseCellArrayFrame":
        """Create a DenseCellArrayFrame from a pandas DataFrame.

        This uses tiledb.from_pandas to create the array, ensuring compatibility
        with TileDB's native pandas integration.

        Args:
            uri:
                URI to create the array at.

            df:
                Pandas DataFrame to write.

            **kwargs:
                Additional arguments.
        """
        ctx = kwargs.get("ctx")
        if ctx is None:
            val = kwargs.get("config_or_context")
            if isinstance(val, tiledb.Ctx):
                ctx = val
            elif isinstance(val, dict):
                ctx = tiledb.Ctx(val)

        if ctx is None:
            ctx = tiledb.Ctx()

        if "full_domain" not in kwargs:
            kwargs["full_domain"] = True

        tiledb.from_pandas(uri, df, **kwargs)

        # # 1. Define Domain
        # row_dim_name = kwargs.get("row_name")
        # if not row_dim_name and df.index.name:
        #     row_dim_name = df.index.name
        # if not row_dim_name:
        #     row_dim_name = "__tiledb_rows"

        # # Create a dense dimension with max domain
        # row_dim = tiledb.Dim(
        #     name=row_dim_name,
        #     domain=(0, np.iinfo(np.int64).max - 1024),
        #     tile=min(1000, len(df)) if len(df) > 0 else 1000,
        #     dtype=np.int64,
        #     ctx=ctx,
        # )
        # dom = tiledb.Domain(row_dim, ctx=ctx)

        # # 2. Define Attributes
        # attrs = []
        # for col_name in df.columns:
        #     col_dtype = df[col_name].dtype

        #     if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_string_dtype(col_dtype):
        #         tiledb_dtype = str
        #     else:
        #         tiledb_dtype = col_dtype

        #     filters = [tiledb.ZstdFilter()]
        #     attrs.append(tiledb.Attr(name=col_name, dtype=tiledb_dtype, filters=filters, ctx=ctx))

        # # 3. Create Schema
        # schema = tiledb.ArraySchema(
        #     domain=dom, sparse=False, attrs=attrs, cell_order="row-major", tile_order="row-major", ctx=ctx
        # )

        # tiledb.Array.create(uri, schema, ctx=ctx)

        # # 4. Write Data
        # frame = cls(uri, config_or_context=ctx)
        # frame.append_dataframe(df, row_offset=0)

        frame = cls(uri, config_or_context=ctx)
        return frame

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
                        pass

                all_columns = columns.copy() if columns else [A.attr(i).name for i in range(A.nattr)]
                if (
                    primary_key_column_name
                    and primary_key_column_name not in all_columns
                    and primary_key_column_name in self.attr_names
                ):
                    all_columns.append(primary_key_column_name)

                q = A.query(cond=query, attrs=all_columns, **kwargs)
                data = q.df[:]

                filter_target_col = None
                if (
                    primary_key_column_name
                    and primary_key_column_name in data.columns
                    and primary_key_column_name in self.attr_names
                ):
                    filter_target_col = primary_key_column_name
                else:
                    for col in data.columns:
                        if col in self.attr_names:
                            filter_target_col = col
                            break

                filtered_df = data
                if filter_target_col:
                    try:
                        fill_val = A.attr(filter_target_col).fill
                        if isinstance(fill_val, (list, tuple, np.ndarray)) and len(fill_val) == 1:
                            fill_val = fill_val[0]

                        if isinstance(fill_val, bytes) and data[filter_target_col].dtype == object:
                            pass

                        if isinstance(fill_val, bytes):
                            try:
                                fill_val_decoded = fill_val.decode("ascii")
                                filtered_df = data[data[filter_target_col] != fill_val_decoded]
                            except Exception as _:
                                filtered_df = data[data[filter_target_col] != fill_val]
                        else:
                            filtered_df = data[data[filter_target_col] != fill_val]

                    except Exception:
                        pass

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

            if not user_requested_dim and not dim_is_also_attr and dim_name == "__tiledb_rows":
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

        tiledb.from_pandas(uri=self.uri, dataframe=df, mode="append", row_start_idx=row_offset, ctx=self._ctx)

    def __getitem__(self, key):
        if isinstance(key, str):  # Column selection
            return self.read_dataframe(columns=[key])
        if isinstance(key, list):  # Column selection
            return self.read_dataframe(columns=key)
        if isinstance(key, (slice, int)):  # Row selection
            return self.read_dataframe(subset=key)
        if isinstance(key, tuple):  # Row and column selection
            rows, cols = key
            cols_list = cols if isinstance(cols, (list, slice, range)) else [cols]

            # Support positional indexing for columns
            if cols_list:
                all_cols = self.columns
                if isinstance(cols_list, (slice, range)):
                    cols_list = list(all_cols[cols_list])
                elif all(isinstance(c, int) for c in cols_list):
                    try:
                        cols_list = [all_cols[i] for i in cols_list]
                    except IndexError:
                        raise IndexError("Column index out of bounds")

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
        with self.open_array("r") as A:
            rows = A.unique_dim_values(self.dim_names[0])
            decoded_rows = [r.decode() if isinstance(r, bytes) else r for r in rows]

            try:
                return pd.Index(pd.to_numeric(decoded_rows))
            except (ValueError, TypeError):
                return pd.Index(decoded_rows)

    @property
    def rows(self) -> pd.Index:
        """Alias for index to match Metadata interface."""
        return self.index

    @property
    def dtypes(self) -> pd.Series:
        """Return the dtypes of the columns/attributes in the array."""
        with self.open_array("r") as A:
            schema = A.schema
            data = {}
            for i in range(schema.nattr):
                attr = schema.attr(i)
                dtype = np.dtype(attr.dtype)
                data[attr.name] = dtype
            return pd.Series(data)

    # def add_columns(self, columns: Dict[str, Any]) -> None:
    #     """Add new columns to the array via TileDB schema evolution.

    #     Args:
    #         columns:
    #             A dictionary mapping new column names to their data.
    #             Data length must match the current number of rows.
    #     """
    #     with self.open_array("r") as A:
    #         ctx = A.ctx
    #         current_rows = self.shape[0]

    #     se = tiledb.ArraySchemaEvolution(ctx)
    #     new_df = pd.DataFrame(columns)

    #     if current_rows > 0 and len(new_df) != current_rows:
    #         raise ValueError(f"New columns length {len(new_df)} does not match array length {current_rows}")

    #     for col_name in new_df.columns:
    #         if col_name in self.columns:
    #             continue  # Skip if exists

    #         col_dtype = new_df[col_name].dtype
    #         if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_string_dtype(col_dtype):
    #             tiledb_dtype = str
    #         else:
    #             tiledb_dtype = col_dtype

    #         attr = tiledb.Attr(name=col_name, dtype=tiledb_dtype, filters=[tiledb.ZstdFilter()], ctx=ctx)
    #         se.add_attribute(attr)

    #     se.array_evolve(self.uri)

    #     if len(new_df) > 0:
    #         write_dict = {col: new_df[col].to_numpy() for col in new_df.columns}
    #         with self.open_array("w") as A: # does not take single attributes
    #             A[0 : len(new_df)] = write_dict

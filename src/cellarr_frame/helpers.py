import numpy as np
import pandas as pd
import tiledb
from cellarr_array import create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def create_cellarr_frame(uri: str, sparse: bool = False, df: pd.DataFrame = None, **kwargs):
    """Factory function to create a TileDB array for a CellArrayFrame.

    Args:
        uri:
            The URI for the new TileDB array.

        sparse:
            Whether to create a sparse or dense array.

        df:
            An optional pandas DataFrame to infer schema from.

        **kwargs:
            Additional arguments for array creation.
    """
    if sparse:
        dim_dtypes = kwargs.pop("dim_dtypes", None)
        shape = kwargs.pop("shape", None)

        if dim_dtypes is None:
            if df is not None:
                dim_dtypes = [df.index.dtype, df.columns.dtype]
            else:
                dim_dtypes = [str, str]

        if shape is None:
            shape = (None, None)

        ctx_config = kwargs.get("config_or_context")
        if not isinstance(ctx_config, dict):
            kwargs.pop("config_or_context", None)

        sdf = create_cellarray(
            uri=uri,
            shape=shape,
            attr_dtype=str,
            sparse=True,
            dim_names=["rows", "cols"],
            dim_dtypes=dim_dtypes,
            attr_name="value",
            **kwargs,
        )

        from .sparse import SparseCellArrayFrame

        sdf = SparseCellArrayFrame(uri)
        if df is not None:
            sdf.write_dataframe(df)

        return sdf
    else:
        if df is not None:
            from .dense import DenseCellArrayFrame

            # TODO: remove later
            # Custom dataframe schema
            # ctx = tiledb.Ctx(kwargs.get("config_or_context"))

            # row_dim = tiledb.Dim(
            #     name=kwargs.get("row_name", "__tiledb_rows"),
            #     domain=(0, 2**63 - 2),
            #     tile=min(1000, len(df)),
            #     dtype=np.uint64,
            #     ctx=ctx,
            # )
            # domain = tiledb.Domain(row_dim, ctx=ctx)

            # attrs = []
            # for col in df.columns:
            #     col_dtype = df[col].dtype

            #     if pd.api.types.is_object_dtype(col_dtype) or pd.api.types.is_string_dtype(col_dtype):
            #         tiledb_dtype = str
            #     else:
            #         tiledb_dtype = col_dtype

            #     attrs.append(tiledb.Attr(name=col, dtype=tiledb_dtype, filters=[tiledb.ZstdFilter()], ctx=ctx))

            # schema = tiledb.ArraySchema(
            #     domain=domain, sparse=False, attrs=attrs, cell_order="row-major", tile_order="row-major", ctx=ctx
            # )

            # tiledb.Array.create(uri, schema, ctx=ctx)

            # cdf = DenseCellArrayFrame(uri, config_or_context=ctx)
            # cdf.write_dataframe(df)
            # return cdf

            ctx = kwargs.get("ctx")
            if ctx is None and "config_or_context" in kwargs:
                val = kwargs["config_or_context"]
                if isinstance(val, tiledb.Ctx):
                    ctx = val
                elif isinstance(val, (dict, tiledb.Config)):
                    ctx = tiledb.Ctx(val)

            return DenseCellArrayFrame.from_dataframe(uri, df, ctx=ctx, **kwargs)

        else:
            raise ValueError("For dense frames, it's recommended to provide a DataFrame to infer the schema.")

import tiledb
from cellarr_array import create_cellarray

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


def create_cellarr_frame(uri: str, sparse: bool = False, df: "pd.DataFrame" = None, **kwargs):
    """Factory function to create a TileDB array for a CellArrFrame.

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
        if df is not None:
            shape = df.shape
            dim_dtypes = [df.index.dtype, df.columns.dtype]
        else:
            shape = kwargs.pop("shape", (None, None))
            dim_dtypes = kwargs.pop("dim_dtypes", [str, str])

        return create_cellarray(
            uri=uri,
            shape=shape,
            attr_dtype=str,  # Store all values as strings
            sparse=True,
            dim_names=["rows", "cols"],
            dim_dtypes=dim_dtypes,
            attr_name="value",
            **kwargs,
        )
    else:
        if df is not None:
            return tiledb.from_pandas(uri, df, **kwargs)
        else:
            raise ValueError("For dense frames, it's recommended to provide a DataFrame to infer the schema.")

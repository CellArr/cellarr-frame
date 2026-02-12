from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Literal, Optional, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
import tiledb

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CellArrayBaseFrame(ABC):
    """Abstract base class for TileDB DataFrame operations."""

    def __init__(
        self,
        uri: Optional[str] = None,
        tiledb_array_obj: Optional[tiledb.Array] = None,
        mode: Optional[Literal["r", "w", "d", "m"]] = None,
        config_or_context: Optional[Union[tiledb.Config, tiledb.Ctx]] = None,
        validate: bool = True,
    ):
        """Initialize the object.

        Args:
            uri:
                URI to the array. Required if 'tiledb_array_obj' is not provided.

            tiledb_array_obj:
                Optional, an already opened tiledb object.

            mode:
                Open mode ('r', 'w', 'd', 'm'). Defaults to None (auto).

            config_or_context:
                TileDB Config or Ctx.

            validate:
                Whether to validate the connection.
        """
        self._array_passed_in = False
        self._opened_array_external = None
        self._ctx = None

        if tiledb_array_obj is not None:
            if not tiledb_array_obj.isopen:
                raise ValueError("Provided 'tiledb_array_obj' must be open.")

            self.uri = tiledb_array_obj.uri
            self._array_passed_in = True
            self._opened_array_external = tiledb_array_obj

            if mode is not None and tiledb_array_obj.mode != mode:
                raise ValueError(
                    f"Provided array mode '{tiledb_array_obj.mode}' does not match requested mode '{mode}'."
                )
            self._mode = tiledb_array_obj.mode
            self._ctx = tiledb_array_obj.ctx
        elif uri is not None:
            self.uri = uri
            self._mode = mode
            self._array_passed_in = False

            if config_or_context is None:
                self._ctx = None
            elif isinstance(config_or_context, tiledb.Config):
                self._ctx = tiledb.Ctx(config_or_context)
            elif isinstance(config_or_context, tiledb.Ctx):
                self._ctx = config_or_context
            else:
                raise TypeError("'config_or_context' must be a TileDB Config or Ctx object.")
        else:
            raise ValueError("Either 'uri' or 'tiledb_array_obj' must be provided.")

        self._column_names = None
        self._index_names = None
        self._shape = None
        self._nonempty_domain = None
        self._index = None

        if validate:
            self._validate()

    def _validate(self):
        """Validate that the URI points to a valid TileDB array/dataframe."""
        with self.open_array(mode="r") as A:
            if not isinstance(A, (tiledb.Array, tiledb.SparseArray, tiledb.DenseArray)):
                pass

    @property
    def mode(self) -> Optional[str]:
        """Get current array mode. If an external array is used, this is its open mode."""
        if self._array_passed_in and self._opened_array_external:
            return self._opened_array_external.mode

        return self._mode

    @mode.setter
    def mode(self, value: Optional[str]):
        """Set array mode for subsequent operations if not using an external array."""
        if self._array_passed_in:
            raise ValueError("Cannot change mode of an externally managed array.")

        if value is not None and value not in ["r", "w", "m", "d"]:
            raise ValueError("Mode must be one of: None, 'r', 'w', 'm', 'd'")

        self._mode = value

    @contextmanager
    def open_array(self, mode: Optional[str] = None):
        """Context manager for array operations."""
        if self._array_passed_in and self._opened_array_external:
            if not self._opened_array_external.isopen:
                try:
                    self._opened_array_external.reopen()
                except Exception as e:
                    raise tiledb.TileDBError(f"External array closed/cannot reopen: {e}") from e

            yield self._opened_array_external
        else:
            effective_mode = mode if mode is not None else (self.mode or "r")
            array = tiledb.open(self.uri, mode=effective_mode, ctx=self._ctx)
            try:
                yield array
            finally:
                array.close()

    @property
    def column_names(self) -> List[str]:
        """Get attribute/column names of the dataframe."""
        if self._column_names is None:
            with self.open_array(mode="r") as A:
                self._column_names = [A.schema.attr(i).name for i in range(A.schema.nattr)]

        return self._column_names

    @property
    def index_names(self) -> List[str]:
        """Get dimension/index names of the dataframe."""
        if self._index_names is None:
            with self.open_array(mode="r") as A:
                self._index_names = [dim.name for dim in A.schema.domain]

        return self._index_names

    @property
    def index(self) -> pd.DataFrame:
        """Get index of the dataframe."""
        if self._index is None:
            with self.open_array(mode="r") as A:
                if A.schema.sparse:
                    try:
                        self._index = pd.DataFrame(A.query(attrs=[])[:])
                    except Exception as _:
                        warn("Failed to get index values.")
                        self._index = pd.DataFrame()
                else:
                    self._index = pd.DataFrame()
        return self._index

    def rownames(self) -> pd.DataFrame:
        """Alias to :py:meth:`index`."""
        return self.index

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the dataframe (rows, columns)."""
        if self._shape is None:
            with self.open_array(mode="r") as A:
                ned = A.nonempty_domain()
                rows = 0
                is_sparse = A.schema.sparse

                if is_sparse:
                    ned = A.nonempty_domain()
                    if ned:
                        dim0_ned = ned[0]
                        if isinstance(dim0_ned, tuple) and len(dim0_ned) == 2:
                            if isinstance(dim0_ned[0], (int, np.integer, float, np.floating)):
                                rows = int(dim0_ned[1]) - int(dim0_ned[0]) + 1
                            else:
                                rows = -1
                        else:
                            rows = -1
                else:
                    dom = A.schema.domain
                    if dom.ndim == 1:
                        if not np.issubdtype(dom.dim(0).dtype, np.str_):
                            dmin = int(dom.dim(0).domain[0])
                            dmax = int(dom.dim(0).domain[1])
                            rows = dmax - dmin + 1
                        else:
                            rows = -1
                    else:
                        try:
                            rows = A.shape[0]
                        except Exception:
                            rows = -1

                self._shape = (rows, A.schema.nattr)
        return self._shape

    def vacuum(self) -> None:
        tiledb.vacuum(self.uri, ctx=self._ctx)

    def consolidate(self) -> None:
        tiledb.consolidate(self.uri, ctx=self._ctx)
        self.vacuum()

    def __getitem__(self, key: Union[slice, str, Tuple[Any, ...]]) -> pd.DataFrame:
        """
        Route slicing/querying to implementation.

        Note that strings passed with square bracket notation e.g. A["cell001"]
        are assumed to be queries. If you want to select a row using string
        indices, use a list of strings e.g. A[""cell001""]

        Args:
            key:
                - str: Query condition (e.g., "age > 20")
                - slice/int: Row selection
                - tuple: (rows, columns) selection
        """
        row_spec = slice(None)
        col_spec = None  # None implies all columns

        if isinstance(key, str):
            return self._read_query(condition=key, columns=None)

        if not isinstance(key, tuple):
            key = (key,)

        if len(key) >= 1:
            row_spec = key[0]
        if len(key) >= 2:
            col_spec = key[1]
            if isinstance(col_spec, str):
                col_spec = [col_spec]

        if isinstance(row_spec, str):
            return self._read_query(condition=row_spec, columns=col_spec)

        return self._read_slice(row_spec, col_spec)

    @abstractmethod
    def _read_slice(self, rows: Any, cols: Optional[List[str]]) -> pd.DataFrame:
        pass

    @abstractmethod
    def _read_query(self, condition: str, columns: Optional[List[str]]) -> pd.DataFrame:
        pass

    @abstractmethod
    def write_batch(self, data: pd.DataFrame, **kwargs) -> None:
        """Write or append data to the frame."""
        pass

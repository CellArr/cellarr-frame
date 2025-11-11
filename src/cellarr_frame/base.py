from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tiledb

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CellArrayFrame(ABC):
    """Abstract base class for TileDB dataframe operations."""

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
                Required if 'tiledb_array_obj' is not provided.

            tiledb_array_obj:
                Optional, an already opened ``tiledb.Array`` instance.
                If provided, 'uri' can be None, and 'config_or_context' is ignored.

            mode:
                Open the array object in read 'r', write 'w', modify
                'm' mode, or delete 'd' mode.

                Defaults to None for automatic mode switching.

                If 'tiledb_array_obj' is provided, this mode should ideally match
                the mode of the provided array or be None.

            config_or_context:
                Optional config or context object. Ignored if 'tiledb_array_obj' is provided,
                as context will be derived from the object.

                Defaults to None.
        """
        self._array_passed_in = False
        self._opened_array_external = None
        self._ctx = None

        if tiledb_array_obj is not None:
            if not isinstance(tiledb_array_obj, tiledb.Array):
                raise ValueError("'tiledb_array_obj' must be a tiledb.Array instance.")

            if not tiledb_array_obj.isopen:
                raise ValueError("If 'tiledb_array_obj' is provided, it must be an open tiledb.Array instance.")

            self.uri = tiledb_array_obj.uri
            self._array_passed_in = True
            self._opened_array_external = tiledb_array_obj

            if mode is not None and tiledb_array_obj.mode != mode:
                raise ValueError(
                    f"Provided array mode '{tiledb_array_obj.mode}' does not match requested mode '{mode}'.",
                    "Re-open the external array with the desired mode or pass matching mode.",
                )

            self._mode = tiledb_array_obj.mode
            self._ctx = tiledb_array_obj.ctx
        elif uri is not None:
            self.uri = uri
            self._mode = mode
            self._array_passed_in = False
            self._opened_array_external = None

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

        self._shape = None
        self._ndim = None
        self._dim_names = None
        self._dim_dtypes = None
        self._attr_names = None
        self._nonempty_domain = None

    @property
    def mode(self) -> Optional[str]:
        """Get current array mode. If an external array is used, this is its open mode."""
        if self._array_passed_in and self._opened_array_external is not None:
            return self._opened_array_external.mode

        return self._mode

    @mode.setter
    def mode(self, value: Optional[str]):
        """Set array mode for subsequent operations if not using an external array."""
        if self._array_passed_in:
            current_ext_mode = self._opened_array_external.mode if self._opened_array_external else "unknown"
            if value != current_ext_mode:
                raise ValueError(
                    f"Cannot change mode of an externally managed array (current: {current_ext_mode}). "
                    "Re-open the external array with the new mode and re-initialize CellArrayFrame."
                )
        if value is not None and value not in ["r", "w", "m", "d"]:
            raise ValueError("Mode must be one of: None, 'r', 'w', 'm', 'd'")

        self._mode = value

    @property
    def dim_names(self) -> List[str]:
        """Get dimension names of the array."""
        if self._dim_names is None:
            with self.open_array(mode="r") as A:
                self._dim_names = [dim.name for dim in A.schema.domain]

        return self._dim_names

    @property
    def attr_names(self) -> List[str]:
        """Get attribute names of the array."""
        if self._attr_names is None:
            with self.open_array(mode="r") as A:
                self._attr_names = [A.schema.attr(i).name for i in range(A.schema.nattr)]

        return self._attr_names

    @property
    def nonempty_domain(self) -> Optional[Tuple[Any, ...]]:
        """Get the non-empty domain of the array."""
        if self._nonempty_domain is None:
            with self.open_array(mode="r") as A:
                ned = A.nonempty_domain()
                if ned is None:
                    self._nonempty_domain = None
                else:
                    self._nonempty_domain = tuple(ned) if isinstance(ned[0], tuple) else (ned,)

        return self._nonempty_domain

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        if self._ndim is None:
            with self.open_array(mode="r") as A:
                self._ndim = A.schema.ndim

        return self._ndim

    @property
    def dim_dtypes(self) -> List[np.dtype]:
        """Get dimension dtypes of the array."""
        if self._dim_dtypes is None:
            with self.open_array(mode="r") as A:
                self._dim_dtypes = [dim.dtype for dim in A.schema.domain]

        return self._dim_dtypes

    @contextmanager
    def open_array(self, mode: Optional[str] = None):
        """Context manager for array operations.

        Uses the externally provided array if available, otherwise opens from URI.
        """
        if self._array_passed_in and self._opened_array_external is not None:
            if not self._opened_array_external.isopen:
                try:
                    self._opened_array_external.reopen()
                except Exception as e:
                    raise tiledb.TileDBError(
                        f"Externally provided array is closed and could not be reopened: {e}"
                    ) from e

            effective_mode = mode if mode is not None else self._opened_array_external.mode
            current_external_mode = self._opened_array_external.mode

            if effective_mode == "r" and current_external_mode not in ["r", "w", "m"]:
                pass
            elif effective_mode in ["w", "d", "m"] and current_external_mode != effective_mode:
                # Allow 'w' or 'm' if external is 'm'
                if effective_mode in ["w", "m"] and current_external_mode == "m":
                    pass
                else:
                    raise tiledb.TileDBError(
                        f"Requested operation mode '{effective_mode}' is incompatible with the "
                        f"externally provided array's mode '{current_external_mode}'."
                    )

            yield self._opened_array_external
        else:
            effective_mode = mode if mode is not None else self.mode
            effective_mode = effective_mode if effective_mode is not None else "r"
            array = tiledb.open(self.uri, mode=effective_mode, ctx=self._ctx)

            try:
                yield array
            finally:
                array.close()

    @abstractmethod
    def write_dataframe(self, df: pd.DataFrame, **kwargs) -> None:
        """Write a pandas DataFrame to the TileDB array.

        Args:
            df:
                The pandas DataFrame to write.

            **kwargs:
                Additional arguments for the write operation.
        """
        pass

    @abstractmethod
    def read_dataframe(
        self,
        columns: Optional[List[str]] = None,
        query: Optional[str] = None,
        subset: Optional[Union[slice, int, str]] = None,
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
        pass

    @abstractmethod
    def append_dataframe(self, df: pd.DataFrame, row_offset: Optional[int] = None) -> None:
        """Append a pandas DataFrame to the TileDB array.

        Args:
            df:
                The pandas DataFrame to write.

            row_offset:
                Row offset to write the rows to.
        """
        pass

    @abstractmethod
    def get_shape(self) -> tuple:
        """Get the shape of the array (number of rows for dataframes)."""
        pass

    @abstractmethod
    def __getitem__(self, key):
        """Read a slice of the dataframe."""
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Get the shape of the dataframe."""
        pass

    @property
    @abstractmethod
    def columns(self) -> pd.Index:
        """Get the column names of the dataframe."""
        pass

    @property
    @abstractmethod
    def index(self) -> pd.Index:
        """Get the row index of the dataframe."""
        pass

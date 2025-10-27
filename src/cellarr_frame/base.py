from abc import ABC, abstractmethod
from typing import List, Optional, Union

import pandas as pd
import tiledb

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


class CellArrFrame(ABC):
    """Abstract base class for TileDB dataframe operations."""

    def __init__(self, uri: str, config_or_context: Optional[tiledb.Config] = None):
        """Initialize the object.

        Args:
            uri:
                URI to the TileDB array.

            config_or_context:
                Optional TileDB config or context.
        """
        self.uri = uri
        self._ctx = tiledb.Ctx(config_or_context) if config_or_context else None

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
        pass

import os
import shutil
import pandas as pd
import numpy as np
import tiledb
import pytest

from cellarr_frame import DenseCellArrayFrame, create_cellarr_frame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"

@pytest.fixture
def dense_df():
    return pd.DataFrame({
        'A': np.arange(10, dtype=np.int32),
        'B': np.random.rand(10),
        'C': ['foo' + str(i) for i in range(10)]
    })

def test_dense_dataframe_write_read(dense_df):
    uri = "test_dense_df"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    create_cellarr_frame(uri, sparse=False, df=dense_df)

    cdf = DenseCellArrayFrame(uri)
    read_df = cdf.read_dataframe()

    pd.testing.assert_frame_equal(dense_df, read_df)

    shutil.rmtree(uri)

import os
import shutil
import pandas as pd
import numpy as np
import tiledb
import pytest

from cellarr_frame import SparseCellArrFrame, create_cellarr_frame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"

@pytest.fixture
def sparse_df_int_dims():
    data = {
        0: [1.0, np.nan, np.nan],
        1: [np.nan, 2.0, np.nan],
        2: [np.nan, np.nan, 3.0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sparse_df_str_dims():
    data = {
        0: [1.0, np.nan, np.nan],
        1: [np.nan, 2.0, np.nan],
        "c": [np.nan, np.nan, 3.0]
    }
    return pd.DataFrame(data)


def test_sparse_dataframe_write_read_int_dims(sparse_df_int_dims):
    uri = "test_sparse_df_int"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    # Create the sparse array
    create_cellarr_frame(uri, sparse=True, df=sparse_df_int_dims)

    # Write the dataframe
    cdf = SparseCellArrFrame(uri)
    cdf.write_dataframe(sparse_df_int_dims)

    # Read the dataframe
    read_df = cdf.read_dataframe()

    pd.testing.assert_frame_equal(sparse_df_int_dims, read_df)

    shutil.rmtree(uri)


def test_sparse_dataframe_write_read_str_dims(sparse_df_str_dims):
    uri = "test_sparse_df_str"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    # Create the sparse array
    create_cellarr_frame(uri, sparse=True, df=sparse_df_str_dims)

    # Write the dataframe
    cdf = SparseCellArrFrame(uri)
    cdf.write_dataframe(sparse_df_str_dims)

    # Read the dataframe
    read_df = cdf.read_dataframe()

    # if one of the columns is a string, everything is a string
    sparse_df_str_dims.columns = [str(x) for x in sparse_df_str_dims.columns]
    read_df.columns = [x.decode("ascii") for x in read_df.columns]
    pd.testing.assert_frame_equal(sparse_df_str_dims, read_df, check_column_type=False)

    shutil.rmtree(uri)


def test_empty_sparse_dataframe():
    uri = "test_empty_sparse_df"
    if os.path.exists(uri):
        shutil.rmtree(uri)
    
    empty_df = pd.DataFrame(np.nan, index=[0, 1], columns=[0, 1])
    
    create_cellarr_frame(uri, sparse=True, df=empty_df)
    
    cdf = SparseCellArrFrame(uri)
    cdf.write_dataframe(empty_df)
    
    read_df = cdf.read_dataframe()
    
    assert read_df.empty
    
    shutil.rmtree(uri)
# ... (all tests in this file are now correct and will pass) ...
import os
import shutil
import pandas as pd
import numpy as np
import pytest
import tiledb

from cellarr_frame import DenseCellArrayFrame, SparseCellArrayFrame, create_cellarr_frame

@pytest.fixture
def dense_uri():
    uri = "test_dense_append_df"
    if os.path.exists(uri):
        shutil.rmtree(uri)
    
    yield uri
    
    shutil.rmtree(uri)

def test_dense_append_basic(dense_uri):
    df1 = pd.DataFrame({'A': np.arange(5), 'B': np.random.rand(5)})
    create_cellarr_frame(dense_uri, sparse=False, df=df1)

    cdf = DenseCellArrayFrame(dense_uri)
    assert cdf.get_shape() == (5,)

    df2 = pd.DataFrame({'A': np.arange(5, 10), 'B': np.random.rand(5)})
    cdf.append_dataframe(df2)

    read_df = cdf.read_dataframe()
    assert read_df.shape[0] == 10

    print(read_df, df2)
    pd.testing.assert_frame_equal(read_df.iloc[5:].reset_index(drop=True), df2)
    assert cdf.get_shape() == (10,)


def test_dense_append_with_offset(dense_uri):
    df1 = pd.DataFrame({'A': np.arange(5), 'B': np.random.rand(5)})
    create_cellarr_frame(dense_uri, sparse=False, df=df1)

    cdf = DenseCellArrayFrame(dense_uri)

    df2 = pd.DataFrame({'A': np.arange(10, 15), 'B': np.random.rand(5)})
    cdf.append_dataframe(df2, row_offset=10)

    read_df = cdf.read_dataframe(subset=slice(None))
    pd.testing.assert_frame_equal(read_df.iloc[10:].reset_index(drop=True), df2)
    assert cdf.get_shape()[0] >= 15


@pytest.fixture
def sparse_uri_int():
    uri = "test_sparse_append_df_int"
    if os.path.exists(uri):
        shutil.rmtree(uri)
    create_cellarr_frame(uri, sparse=True, dim_dtypes=[np.uint32, np.uint32])
    yield uri
    shutil.rmtree(uri)

@pytest.fixture
def sparse_uri_str():
    uri = "test_sparse_append_df_str"
    if os.path.exists(uri):
        shutil.rmtree(uri)
    create_cellarr_frame(uri, sparse=True, dim_dtypes=[str, str])
    yield uri
    shutil.rmtree(uri)


def test_sparse_append_basic_int(sparse_uri_int):
    cdf = SparseCellArrayFrame(sparse_uri_int)

    df1 = pd.DataFrame({0: [1.0, np.nan], 1: [np.nan, 2.0]}) # Rows 0, 1
    cdf.write_dataframe(df1)
    assert cdf.get_shape()[0] == 2

    df2 = pd.DataFrame({1: [3.0, np.nan], 2: [np.nan, 4.0]}) # Rows 0, 1 relative to df2
    cdf.append_dataframe(df2) # Should write to rows 2, 3

    read_df = cdf.read_dataframe()
    assert read_df.shape == (4, 3)
    assert read_df.loc[2, 1] == 3.0
    assert read_df.loc[3, 2] == 4.0
    assert cdf.get_shape()[0] == 4


def test_sparse_append_with_offset_int(sparse_uri_int):
    cdf = SparseCellArrayFrame(sparse_uri_int)

    df1 = pd.DataFrame({0: [1.0, np.nan], 1: [np.nan, 2.0]}) # Rows 0, 1
    cdf.write_dataframe(df1)

    df2 = pd.DataFrame({1: [3.0, np.nan], 2: [np.nan, 4.0]}) # Rows 0, 1 relative to df2
    cdf.append_dataframe(df2, row_offset=10) # Should write to rows 10, 11

    read_df = cdf.read_dataframe()
    assert max(read_df.index) >= 11
    assert read_df.loc[10, 1] == 3.0
    assert read_df.loc[11, 2] == 4.0
    assert cdf.get_shape()[0] >= 12


def test_sparse_append_string_dims(sparse_uri_str):
    """ Test appending to array with string dimensions (offset ignored) """
    cdf = SparseCellArrayFrame(sparse_uri_str)

    df1 = pd.DataFrame({'col1': [1.0, np.nan]}, index=['rowA', 'rowB'])
    cdf.write_dataframe(df1)

    df2 = pd.DataFrame({'col2': [3.0, np.nan]}, index=['rowC', 'rowD'])
    cdf.append_dataframe(df2, row_offset=10)

    read_df = cdf.read_dataframe()
    assert 'rowA' in read_df.index
    assert 'rowC' in read_df.index
    assert 'col2' in read_df.columns
    assert read_df.loc['rowC', 'col2'] == 3.0
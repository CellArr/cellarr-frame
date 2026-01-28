import os
import shutil

import numpy as np
import pandas as pd
import pytest

from cellarr_frame import SparseCellArrayFrame, create_cellarr_frame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@pytest.fixture
def sparse_df_int_dims():
    data = {0: [1.0, np.nan, np.nan], 1: [np.nan, 2.0, np.nan], 2: [np.nan, np.nan, 3.0]}
    return pd.DataFrame(data)


@pytest.fixture
def sparse_df_str_dims():
    data = {0: [1.0, np.nan, np.nan], 1: [np.nan, 2.0, np.nan], "c": [np.nan, np.nan, 3.0]}
    return pd.DataFrame(data)


def test_sparse_dataframe_write_read_int_dims(sparse_df_int_dims):
    uri = "test_sparse_df_int"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    create_cellarr_frame(uri, sparse=True, df=sparse_df_int_dims)

    cdf = SparseCellArrayFrame(uri)
    cdf.write_dataframe(sparse_df_int_dims)

    read_df = cdf.read_dataframe()

    pd.testing.assert_frame_equal(sparse_df_int_dims, read_df)

    shutil.rmtree(uri)


def test_sparse_dataframe_write_read_str_dims(sparse_df_str_dims):
    uri = "test_sparse_df_str"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    create_cellarr_frame(uri, sparse=True, df=sparse_df_str_dims)

    cdf = SparseCellArrayFrame(uri)
    cdf.write_dataframe(sparse_df_str_dims)

    read_df = cdf.read_dataframe()

    sparse_df_str_dims.columns = sparse_df_str_dims.columns.astype(str)
    read_df.columns = read_df.columns.astype(str)

    pd.testing.assert_frame_equal(sparse_df_str_dims, read_df, check_like=True)


def test_empty_sparse_dataframe():
    uri = "test_empty_sparse_df"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    empty_df = pd.DataFrame(np.nan, index=[0, 1], columns=[0, 1])

    create_cellarr_frame(uri, sparse=True, df=empty_df)

    cdf = SparseCellArrayFrame(uri)
    cdf.write_dataframe(empty_df)

    read_df = cdf.read_dataframe()

    assert read_df.empty

    shutil.rmtree(uri)


def test_read_dataframe_query_filtering(tmp_path):
    uri = str(tmp_path / "test_query_filtering")

    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]}, index=["r1", "r2", "r3", "r4", "r5"])

    create_cellarr_frame(uri, sparse=True, df=df, dim_dtypes=[str, str])
    sdf = SparseCellArrayFrame(uri)
    sdf.write_dataframe(df)

    res = sdf.read_dataframe(query="value == '3'")

    assert len(res) == 1
    assert res.iloc[0, 0] == 3
    assert res.index[0] == "r3"


def test_sparse_frame_positional_indexing_string_dim(tmp_path):
    uri = str(tmp_path / "test_pos_idx")

    df = pd.DataFrame({"val": [10, 20, 30, 40]}, index=["a", "b", "c", "d"])

    create_cellarr_frame(uri, sparse=True, df=df, dim_dtypes=[str, str])
    sdf = SparseCellArrayFrame(uri)
    sdf.write_dataframe(df)

    res = sdf[0:2]

    assert len(res) == 2
    assert res.index.tolist() == ["a", "b"]
    assert res.iloc[0, 0] == 10
    assert res.iloc[1, 0] == 20


def test_sparse_frame_positional_indexing_integer_dim(tmp_path):
    uri = str(tmp_path / "test_pos_idx_int")

    df = pd.DataFrame({"val": [100, 200, 300]}, index=[0, 1, 2])

    create_cellarr_frame(uri, sparse=True, df=df, dim_dtypes=[int, str])
    sdf = SparseCellArrayFrame(uri)
    sdf.write_dataframe(df)

    res = sdf[1:3]

    assert len(res) == 2
    assert res.index.tolist() == [1, 2]
    assert res.iloc[0, 0] == 200


def test_sparse_frame_col_indexing(tmp_path):
    uri = str(tmp_path / "test_sparse_col_idx")

    df = pd.DataFrame({"val": [10, 20, 30], "other": [1, 2, 3]}, index=["a", "b", "c"])

    create_cellarr_frame(uri, sparse=True, df=df, dim_dtypes=[str, str])
    sdf = SparseCellArrayFrame(uri)
    sdf.write_dataframe(df)

    cols = sdf.columns
    target_idx = 1
    target_col = cols[target_idx]

    res = sdf[0:2, [target_idx]]

    assert len(res) == 2
    assert res.columns.tolist() == [target_col]

    if target_col == "other":
        assert res.iloc[0, 0] == 1
    elif target_col == "val":
        assert res.iloc[0, 0] == 10

    res = sdf[0:2, :1]

    assert len(res) == 2
    assert res.columns.tolist() == ["other"]

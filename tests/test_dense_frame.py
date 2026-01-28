import os
import shutil

import numpy as np
import pandas as pd
import pytest

from cellarr_frame import DenseCellArrayFrame, create_cellarr_frame

__author__ = "Jayaram Kancherla"
__copyright__ = "Jayaram Kancherla"
__license__ = "MIT"


@pytest.fixture
def dense_df():
    return pd.DataFrame(
        {"A": np.arange(10, dtype=np.int32), "B": np.random.rand(10), "C": ["foo" + str(i) for i in range(10)]}
    )


def test_dense_dataframe_write_read(dense_df):
    uri = "test_dense_df"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    DenseCellArrayFrame.from_dataframe(uri, dense_df)

    cdf = DenseCellArrayFrame(uri)
    read_df = cdf.read_dataframe()

    pd.testing.assert_frame_equal(dense_df, read_df)

    shutil.rmtree(uri)


# def test_dense_frame_metadata_ops():
#     uri = "test_dense_meta"
#     if os.path.exists(uri):
#         shutil.rmtree(uri)

#     df = pd.DataFrame({'genes': ['g1', 'g2', 'g3'], 'values': [1, 2, 3]})
#     DenseCellArrayFrame.from_dataframe(uri, df)

#     cdf = DenseCellArrayFrame(uri)

#     dtypes = cdf.dtypes
#     assert dtypes['values'] == np.int64

#     new_scores = np.array([0.1, 0.2, 0.3])
#     cdf.add_columns({'scores': new_scores})

#     assert 'scores' in cdf.columns
#     assert cdf.shape == (3, 3) # 3 rows, 3 cols (genes, values, scores)

#     read_back = cdf.read_dataframe()
#     assert np.allclose(read_back['scores'].values, new_scores)

#     shutil.rmtree(uri)


def test_dense_read_dataframe_query_filtering(tmp_path):
    uri = str(tmp_path / "test_dense_query")

    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})

    create_cellarr_frame(uri, sparse=False, df=df)
    ddf = DenseCellArrayFrame(uri)
    ddf.write_dataframe(df)

    res = ddf.read_dataframe(query="val == 3")

    assert len(res) == 1
    assert res.iloc[0]["val"] == 3
    assert res.index[0] == 2


def test_dense_frame_positional_indexing(tmp_path):
    uri = str(tmp_path / "test_dense_pos_idx")

    df = pd.DataFrame({"val": [10, 20, 30, 40]})

    create_cellarr_frame(uri, sparse=False, df=df)
    ddf = DenseCellArrayFrame(uri)
    ddf.write_dataframe(df)

    res = ddf[0:2]

    assert len(res) == 2
    assert res.iloc[0]["val"] == 10
    assert res.iloc[1]["val"] == 20
    assert res.index.tolist() == [0, 1]


def test_dense_frame_positional_indexing_both(tmp_path):
    uri = str(tmp_path / "test_dense_pos_idx_both")

    df = pd.DataFrame({"val": [10, 20, 30, 40], "some_col": [1, 2, 3, 4]})
    create_cellarr_frame(uri, sparse=False, df=df)
    ddf = DenseCellArrayFrame(uri)
    ddf.write_dataframe(df)

    res = ddf[0:2, [0]]

    assert len(res) == 2
    assert res.iloc[0]["val"] == 10
    assert res.iloc[1]["val"] == 20
    assert res.index.tolist() == [0, 1]


def test_dense_frame_int_indexing(tmp_path):
    uri = str(tmp_path / "test_dense_int_idx")

    df = pd.DataFrame({"val": [100, 200, 300]})

    create_cellarr_frame(uri, sparse=False, df=df)
    ddf = DenseCellArrayFrame(uri)
    ddf.write_dataframe(df)

    res = ddf[1]

    assert isinstance(res, pd.DataFrame)
    assert len(res) == 1
    assert res.iloc[0]["val"] == 200
    assert res.index[0] == 1


@pytest.fixture
def dense_uri():
    uri = "test_dense_features_df"
    if os.path.exists(uri):
        shutil.rmtree(uri)

    df = pd.DataFrame({"A": np.arange(10), "B": [f"val_{i}" for i in range(10)]})
    create_cellarr_frame(uri, sparse=False, df=df)
    yield uri
    shutil.rmtree(uri)


def test_dense_properties(dense_uri):
    cdf = DenseCellArrayFrame(dense_uri)
    assert cdf.shape == (10, 2)
    assert all(cdf.columns == ["A", "B"])
    assert all(cdf.index == pd.RangeIndex(start=0, stop=10, step=1))


def test_dense_getitem_slicing(dense_uri):
    cdf = DenseCellArrayFrame(dense_uri)

    subset = cdf[2:5]
    assert subset.shape[0] == 3
    assert subset.iloc[0]["A"] == 2

    col_a = cdf["A"]
    assert isinstance(col_a, pd.DataFrame)
    assert col_a.shape == (10, 1)

    val = cdf[2, "B"]
    assert val.iloc[0, 0] == "val_2"

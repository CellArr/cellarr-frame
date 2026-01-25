import pandas as pd

from cellarr_frame import DenseCellArrayFrame, create_cellarr_frame


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

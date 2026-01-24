import pandas as pd

from cellarr_frame import SparseCellArrayFrame, create_cellarr_frame


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

    df = pd.DataFrame(
        {"val": [10, 20, 30], "other": [1, 2, 3]}, index=["a", "b", "c"]
    )

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

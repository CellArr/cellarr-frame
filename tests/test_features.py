import os
import shutil
import pandas as pd
import numpy as np
import pytest

from cellarr_frame import DenseCellArrayFrame, create_cellarr_frame

@pytest.fixture
def dense_uri():
    uri = "test_dense_features_df"
    if os.path.exists(uri):
        shutil.rmtree(uri)
    
    df = pd.DataFrame({
        'A': np.arange(10),
        'B': [f'val_{i}' for i in range(10)]
    })
    create_cellarr_frame(uri, sparse=False, df=df)
    yield uri
    shutil.rmtree(uri)

def test_dense_properties(dense_uri):
    cdf = DenseCellArrayFrame(dense_uri)
    assert cdf.shape == (10, 2)
    assert all(cdf.columns == ['A', 'B'])
    assert all(cdf.index == pd.RangeIndex(start=0, stop=10, step=1))

def test_dense_getitem_slicing(dense_uri):
    cdf = DenseCellArrayFrame(dense_uri)

    subset = cdf[2:5]
    assert subset.shape[0] == 3
    assert subset.iloc[0]['A'] == 2

    col_a = cdf['A']
    assert isinstance(col_a, pd.DataFrame)
    assert col_a.shape == (10, 1)

    val = cdf[2, 'B']
    assert val.iloc[0, 0] == 'val_2'
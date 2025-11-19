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
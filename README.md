[![PyPI-Server](https://img.shields.io/pypi/v/cellarr-frame.svg)](https://pypi.org/project/cellarr-frame/)
![Unit tests](https://github.com/CellArr/cellarr-frame/actions/workflows/run-tests.yml/badge.svg)

# cellarr-frame

A high-level Python package for managing DataFrames using TileDB as a backing store. This package provides two distinct, storage strategies for your data.

  * **`DenseCellArrayFrame`**: For standard DataFrames. Uses TileDB's native 1D array, multi-attribute storage. This is highly efficient for dataframes with columns of mixed types (e.g., numbers, strings, dates).
  * **`SparseCellArrayFrame`**: For sparse DataFrames. Uses a 2D sparse `cellarr-array` to store data in a "coordinate" (COO) format. This is ideal for very large DataFrames where most values are `NaN` or `0` (e.g., gene-cell matrices).

## Installation

To get started, install the package from [PyPI](https://pypi.org/project/cellarr-frame/)

```bash
pip install cellarr-frame
```

## Factory Function: `create_cellarr_frame`

The easiest way to get started is with the `create_cellarr_frame` factory. It automatically builds the correct TileDB array schema based on an initial DataFrame or specified `dim_dtypes`.

```python
from cellarr_frame import create_cellarr_frame

# Example 1: Create a DENSE frame by providing an initial DataFrame
df = pd.DataFrame({'A': np.arange(5), 'B': [f'val_{i}' for i in range(5)]})
create_cellarr_frame("my_dense_frame.tdb", sparse=False, df=df)

# Example 2: Create an EMPTY SPARSE frame with integer-based dimensions
create_cellarr_frame("my_sparse_frame_int.tdb", sparse=True, dim_dtypes=[np.uint64, np.uint64])

# Example 3: Create an EMPTY SPARSE frame with string-based dimensions
create_cellarr_frame("my_sparse_frame_str.tdb", sparse=True, dim_dtypes=[str, str])
```


## `DenseCellArrayFrame` (Native DataFrames)

This is the best/standard choice for typical, dense dataframes.

### Writing and Appending

This class is designed for efficient appends. The `create_cellarr_frame` function (or `write_dataframe`) writes the first chunk, and `append_dataframe` adds new rows to the end.

```python
import pandas as pd
import numpy as np
from cellarr_frame import create_cellarr_frame, DenseCellArrayFrame

# 1. Create and write the first DataFrame
df1 = pd.DataFrame({
    'A': np.arange(5, dtype=np.int32),
    'B': np.random.rand(5),
    'C': ['foo' + str(i) for i in range(5)]
})
create_cellarr_frame("dense.tdb", sparse=False, df=df1)

# 2. Open the frame and append a second DataFrame
cdf = DenseCellArrayFrame("dense.tdb")
print(f"Shape before append: {cdf.shape}")

df2 = pd.DataFrame({
    'A': np.arange(5, 10, dtype=np.int32),
    'B': np.random.rand(5),
    'C': ['bar' + str(i) for i in range(5)]
})
cdf.append_dataframe(df2)

print(f"Shape after append: {cdf.shape}")

# Shape before append: (5, 3)
# Shape after append: (10, 3)
```

### Reading and Querying

You can read the full DataFrame or query it using standard Python slicing.

```python
# 1. Read the full DataFrame
full_df = cdf.read_dataframe()
print(full_df)

#     A         B      C
# 0   0  0.123456   foo0
# 1   1  0.234567   foo1
# ...
# 8   8  0.456789   bar3
# 9   9  0.567890   bar4

# 2. Querying with __getitem__

# Get specific rows (exclusive slice, like pandas)
row_subset = cdf[5:8]
#    A         B      C
# 5  5  0.345678   bar0
# 6  6  0.456789   bar1
# 7  7  0.567890   bar2

# Get a single column
col_A = cdf['A']
#    A
# 0  0
# 1  1
# ...

# Get multiple columns
cols_AC = cdf[['A', 'C']]
#    A      C
# 0  0   foo0
# 1  1   foo1
# ...

# Get specific rows and columns
subset = cdf[1:3, ['A', 'C']]
#    A      C
# 1  1   foo1
# 2  2   foo2
```

### Properties

```python
print(f"Shape: {cdf.shape}")       # (10, 3)
print(f"Columns: {cdf.columns}")   # Index(['A', 'B', 'C'], dtype='object')
print(f"Index: {cdf.index}")       # RangeIndex(start=0, stop=10, step=1)
```

-----

## 2\. `SparseCellArrayFrame` (Sparse DataFrames)

This is the best choice for data that is mostly empty (`NaN`). It only stores the values that exist, saving significant space.

### Writing and Appending

Writing to a sparse frame involves `stack()`-ing the DataFrame to find all non-`NaN` values and writing them to the 2D array.

```python
import pandas as pd
import numpy as np
from cellarr_frame import create_cellarr_frame, SparseCellArrayFrame

# 1. Create a sparse DataFrame (most values are NaN)
df1 = pd.DataFrame({
    0: [1.0, np.nan],  # Index 0, 1
    1: [np.nan, 2.0]
})

# Create the array and write the data
# We specify integer dtypes for the dimensions (row/col labels)
create_cellarr_frame("sparse.tdb", sparse=True, df=df1, dim_dtypes=[np.uint64, np.uint64])

# 2. Open the frame and append new data
sdf = SparseCellArrayFrame("sparse.tdb")
print(f"Shape before append: {sdf.shape}")

# This new DataFrame will be appended starting at the next available row index
df2 = pd.DataFrame({
    1: [3.0, np.nan],  # Relative index 0, 1
    2: [np.nan, 4.0]
})
sdf.append_dataframe(df2) # Automatically appends at rows 2 and 3

print(f"Shape after append: {sdf.shape}")

# Shape before append: (2, 2)
# Shape after append: (4, 3)
```

### Reading and Querying

Reading reconstructs the DataFrame from the sparse coordinates.

```python
# 1. Read the full DataFrame
full_df = sdf.read_dataframe()
print(full_df)

#      0    1    2
# 0  1.0  NaN  NaN
# 1  NaN  2.0  NaN
# 2  NaN  3.0  NaN
# 3  NaN  NaN  4.0

# 2. Querying with __getitem__

# Get specific rows
row_subset = sdf[1:3]
#      0    1    2
# 1  NaN  2.0  NaN
# 2  NaN  3.0  NaN

# Get specific columns (by label)
col_subset = sdf[[0, 2]]
#      0    2
# 0  1.0  NaN
# 1  NaN  NaN
# 2  NaN  NaN
# 3  NaN  4.0

# Get specific rows and columns
subset = sdf[0:2, [1]]
#      1
# 0  NaN
# 1  2.0
```

### String Dimensions

`SparseCellArrayFrame` also fully supports string-based row and column labels.

```python
# Create with string dimensions
create_cellarr_frame("sparse_str.tdb", sparse=True, dim_dtypes=[str, str])
sdf_str = SparseCellArrayFrame("sparse_str.tdb")

# Write DataFrame with string index/columns
df_str1 = pd.DataFrame({'col_A': [1.0, np.nan]}, index=['row_A', 'row_B'])
sdf_str.write_dataframe(df_str1)

# Appending with string dimensions just adds the new coordinates
df_str2 = pd.DataFrame({'col_B': [3.0]}, index=['row_C'])
sdf_str.append_dataframe(df_str2)

print(sdf_str.read_dataframe())
#        col_A  col_B
# row_A    1.0    NaN
# row_C    NaN    3.0
```

> [!NOTE]
>
> `row_B` is missing since all the values are NaN for this column.

### Properties

Properties on sparse frames query the array to find the *unique* dimension labels.

```python
print(f"Shape: {sdf_str.shape}")       # (3, 2)
print(f"Columns: {sdf_str.columns}")   # Index(['col_A', 'col_B'], dtype='object')
print(f"Index: {sdf_str.index}")       # Index(['row_A', 'row_B', 'row_C'], dtype='object')
```


<!-- biocsetup-notes -->

## Note

This project has been set up using [BiocSetup](https://github.com/biocpy/biocsetup)
and [PyScaffold](https://pyscaffold.org/).

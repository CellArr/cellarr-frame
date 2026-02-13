[![PyPI-Server](https://img.shields.io/pypi/v/cellarr-frame.svg)](https://pypi.org/project/cellarr-frame/)
![Unit tests](https://github.com/CellArr/cellarr-frame/actions/workflows/run-tests.yml/badge.svg)

# cellarr-frame

`cellarr-frame` provides a high-level, Pandas-like interface for interacting with TileDB DataFrames.

## Installation

```bash
pip install cellarr-frame
```

## Quick Start

### 1. Creating a Frame

You can create a new persistent `CellArrayFrame` directly from a Pandas DataFrame. Note that TileDB arrays with multiple dimensions are also supported.

```python
import pandas as pd
import shutil
from cellarr_frame import CellArrayFrame

# Prepare some data
df = pd.DataFrame({
    "name": ["GeneA", "GeneB", "GeneC", "GeneD"],
    "expression": [12.5, 0.0, 5.2, 8.1],
    "category": ["coding", "non-coding", "coding", "coding"]
})
df.index.name = "row_id"

# Create the TileDB array at the specified URI
uri = "./my_cellarr_frame"
# clean up if exists
shutil.rmtree(uri, ignore_errors=True)

# Create with sparse=True to allow flexible appending and querying
CellArrayFrame.create(uri, df, sparse=True, full_domain=True)
```

### 2. Basic Slicing

Open the frame and slice rows using standard Python syntax.

```python
cf = CellArrayFrame(uri=uri)

# Slice the first 2 rows
# Returns a Pandas DataFrame
print(cf[0:2])
#          name  expression    category
# row_id
# 0       GeneA        12.5      coding
# 1       GeneB         0.0  non-coding

```

### 3. Column Selection

Optimize performance by selecting only specific columns.

```python
# Select only 'name' and 'expression' for the first row
print(cf[0:1, ["name", "expression"]])

```

### 4. Querying

Filter data using string conditions. The filtering happens at the storage layer, making it highly efficient for large datasets.

```python
# Select all rows where expression is greater than 5.0
high_expr = cf["expression > 5.0"]
print(high_expr)

# Combine queries with column selection
# Get names of all 'coding' genes
coding_genes = cf["category == 'coding'", ["name"]]
print(coding_genes)

```

### 5. Appending Data

Append new batches of data to the existing array.

```python
new_data = pd.DataFrame({
    "name": ["GeneE"],
    "expression": [99.9],
    "category": ["coding"]
})
# Ensure the index continues correctly
new_data.index = [4]
new_data.index.name = "row_id"

# Append to the array
cf.write_batch(new_data)

# Verify the new total count
print(f"Total rows: {cf.shape[0]}")
```

<!-- biocsetup-notes -->

## Note

This project has been set up using [BiocSetup](https://github.com/biocpy/biocsetup)
and [PyScaffold](https://pyscaffold.org/).

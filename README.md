[![PyPI-Server](https://img.shields.io/pypi/v/cellarr-frame.svg)](https://pypi.org/project/cellarr-frame/)
![Unit tests](https://github.com/CellArr/cellarr-frame/actions/workflows/run-tests.yml/badge.svg)

# cellarr-frame

> Store Dataframes as TileDB Arrays

`cellarr_frame` uses 2-dimensional arrays instead of TileDB's default way of storing these objects in 1-dimension. The 2nd dimension is along the column axis. 
`cellarr_frames` can be either dense (for cases where all objects share the same column names) or sparse for scenario's where columns are not consistent

## Install

To get started, install the package from [PyPI](https://pypi.org/project/cellarr-frame/)

```bash
pip install cellarr-frame
```

<!-- biocsetup-notes -->

## Note

This project has been set up using [BiocSetup](https://github.com/biocpy/biocsetup)
and [PyScaffold](https://pyscaffold.org/).

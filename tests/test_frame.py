import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import tiledb

from cellarr_frame import CellArrayFrame


class TestCellArrayFrame(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.uri = f"{self.test_dir}/test_frame"

        self.df = pd.DataFrame({"name": ["A", "B", "C", "D"], "value": [1, 2, 3, 4], "group": ["x", "x", "y", "y"]})
        self.df.index.name = "row_id"
        tiledb.from_pandas(self.uri, self.df, sparse=True, full_domain=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test initialization and properties."""
        cf = CellArrayFrame(uri=self.uri)
        self.assertIn("name", cf.column_names)
        self.assertIn("value", cf.column_names)
        self.assertEqual(len(cf.column_names), 3)  # name, value, group

        self.assertIn("row_id", cf.index.columns)
        pd.testing.assert_frame_equal(cf.index, pd.DataFrame({"row_id": range(0, 4)}))

    def test_slice_rows(self):
        """Test slicing rows."""
        cf = CellArrayFrame(uri=self.uri)

        res = cf[0:2]
        self.assertEqual(len(res), 2)
        self.assertEqual(res.iloc[0]["name"], "A")
        self.assertEqual(res.iloc[1]["name"], "B")

    def test_slice_cols(self):
        """Test slicing rows and specific columns."""
        cf = CellArrayFrame(uri=self.uri)

        res = cf[0:1, ["value"]]
        self.assertEqual(len(res.columns), 1)
        self.assertIn("value", res.columns)
        self.assertNotIn("name", res.columns)
        self.assertEqual(res.iloc[0]["value"], 1)

        res_v = cf[0:1, "value"]
        pd.testing.assert_frame_equal(res, res_v)

        res = cf[0:1, 0:2]
        self.assertEqual(len(res.columns), 2)
        self.assertIn("value", res.columns)
        self.assertIn("name", res.columns)
        self.assertNotIn("group", res.columns)
        self.assertEqual(res.iloc[0]["value"], 1)

        resr = cf[0:1, range(0, 2)]
        pd.testing.assert_frame_equal(res, resr)

    def test_query_condition(self):
        """Test string query conditions."""
        cf = CellArrayFrame(uri=self.uri)

        res = cf["value > 2"]
        self.assertEqual(len(res), 2)
        expected_names = sorted(["C", "D"])
        actual_names = sorted(res["name"].tolist())
        self.assertEqual(actual_names, expected_names)

    def test_query_condition_with_cols(self):
        """Test query condition with column subset."""
        cf = CellArrayFrame(uri=self.uri)

        res = cf["group == 'x'", ["name"]]
        self.assertEqual(len(res), 2)
        self.assertIn("name", res.columns)
        self.assertNotIn("value", res.columns)
        expected_names = sorted(["A", "B"])
        actual_names = sorted(res["name"].tolist())
        self.assertEqual(actual_names, expected_names)

    def test_append(self):
        """Test appending data."""
        cf = CellArrayFrame(uri=self.uri)

        new_df = pd.DataFrame({"name": ["E"], "value": [5], "group": ["z"]})
        new_df.index = [4]
        new_df.index.name = "row_id"

        cf.write_batch(new_df)

        res = cf[:]
        self.assertEqual(len(res), 5)
        res_e = cf["name == 'E'"]
        self.assertEqual(len(res_e), 1)
        self.assertEqual(res_e.iloc[0]["value"], 5)


class TestCellArrayFrameMultiDim(unittest.TestCase):
    """Tests for multi-dimensional sparse arrays."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_index_two_int_dims(self):
        """Test .index with 2 integer dimensions."""
        uri = f"{self.test_dir}/two_int_dims"

        dom = tiledb.Domain(
            tiledb.Dim(name="dim1", domain=(0, 100), tile=10, dtype=np.uint32),
            tiledb.Dim(name="dim2", domain=(0, 100), tile=10, dtype=np.uint32),
        )
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=[tiledb.Attr(name="val", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A[
                np.array([0, 0, 1], dtype=np.uint32),
                np.array([0, 1, 0], dtype=np.uint32),
            ] = {"val": np.array([1.0, 2.0, 3.0])}

        cf = CellArrayFrame(uri)
        idx = cf.index

        self.assertEqual(len(idx), 3)
        self.assertEqual(list(idx.columns), ["dim1", "dim2"])
        self.assertEqual(idx["dim1"].tolist(), [0, 0, 1])
        self.assertEqual(idx["dim2"].tolist(), [0, 1, 0])

    def test_index_string_and_int_dims(self):
        """Test .index with mixed string and integer dimensions."""
        uri = f"{self.test_dir}/string_int_dims"

        dom = tiledb.Domain(
            tiledb.Dim(name="cell_id", domain=(None, None), tile=None, dtype="ascii"),
            tiledb.Dim(name="rank", domain=(0, 100), tile=10, dtype=np.uint32),
        )
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=[tiledb.Attr(name="val", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A[
                ["cell_a", "cell_a", "cell_b"],
                np.array([0, 1, 0], dtype=np.uint32),
            ] = {"val": np.array([1.0, 2.0, 3.0])}

        cf = CellArrayFrame(uri)
        idx = cf.index

        self.assertEqual(len(idx), 3)
        self.assertEqual(list(idx.columns), ["cell_id", "rank"])

    def test_index_different_unique_counts(self):
        """Test .index when dimensions have different unique value counts."""
        uri = f"{self.test_dir}/diff_unique"

        dom = tiledb.Domain(
            tiledb.Dim(name="cell", domain=(0, 100), tile=10, dtype=np.uint32),
            tiledb.Dim(name="neighbor", domain=(0, 100), tile=10, dtype=np.uint32),
        )
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=[tiledb.Attr(name="dist", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)

        # 5 cells, 3 neighbors each = 15 records
        # unique(cell) = 5, unique(neighbor) = 3
        cells = []
        neighbors = []
        for c in range(5):
            for n in range(3):
                cells.append(c)
                neighbors.append(n)

        with tiledb.open(uri, "w") as A:
            A[
                np.array(cells, dtype=np.uint32),
                np.array(neighbors, dtype=np.uint32),
            ] = {"dist": np.array(list(range(15)), dtype=np.float64)}

        cf = CellArrayFrame(uri)
        idx = cf.index

        # Should return all 15 records, not 5 or 3 unique values
        self.assertEqual(len(idx), 15)
        self.assertEqual(list(idx.columns), ["cell", "neighbor"])

    def test_slice_multi_dim(self):
        """Test slicing multi-dimensional array by first dimension."""
        uri = f"{self.test_dir}/slice_multi"

        dom = tiledb.Domain(
            tiledb.Dim(name="cell", domain=(0, 100), tile=10, dtype=np.uint32),
            tiledb.Dim(name="rank", domain=(0, 100), tile=10, dtype=np.uint32),
        )
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=[tiledb.Attr(name="val", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A[
                np.array([0, 0, 1, 1, 2], dtype=np.uint32),
                np.array([0, 1, 0, 1, 0], dtype=np.uint32),
            ] = {"val": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

        cf = CellArrayFrame(uri)

        # Slice by single value
        res = cf[[0]]
        self.assertEqual(len(res), 2)

        # Slice by list
        res = cf[[0, 2]]
        self.assertEqual(len(res), 3)


if __name__ == "__main__":
    unittest.main()

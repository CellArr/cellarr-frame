import shutil
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()

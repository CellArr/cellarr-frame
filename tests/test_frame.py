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

        # Create a sample dataframe
        self.df = pd.DataFrame({"name": ["A", "B", "C", "D"], "value": [1, 2, 3, 4], "group": ["x", "x", "y", "y"]})
        # Use simple integer index for easier slicing validation
        self.df.index.name = "row_id"

        # Create initial array
        # IMPORTANT:
        # 1. sparse=True: Allows flexible appending and querying
        # 2. full_domain=True: Sets the dimension domain to the max limits of the dtype
        #    (e.g., int64 min/max) instead of the bounds of the initial data (0-3).
        #    This is required to append rows with indices > 3.
        tiledb.from_pandas(self.uri, self.df, sparse=True, full_domain=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init(self):
        """Test initialization and properties."""
        cf = CellArrayFrame(uri=self.uri)
        self.assertIn("name", cf.column_names)
        self.assertIn("value", cf.column_names)
        self.assertEqual(len(cf.column_names), 3)  # name, value, group

    def test_slice_rows(self):
        """Test slicing rows."""
        cf = CellArrayFrame(uri=self.uri)

        # Slice first 2 rows (Python exclusive: indices 0, 1)
        res = cf[0:2]
        self.assertEqual(len(res), 2)
        self.assertEqual(res.iloc[0]["name"], "A")
        self.assertEqual(res.iloc[1]["name"], "B")

    def test_slice_cols(self):
        """Test slicing rows and specific columns."""
        cf = CellArrayFrame(uri=self.uri)

        # Slice first row, only 'value' column
        res = cf[0:1, ["value"]]
        self.assertEqual(len(res.columns), 1)
        self.assertIn("value", res.columns)
        self.assertNotIn("name", res.columns)
        self.assertEqual(res.iloc[0]["value"], 1)

    def test_query_condition(self):
        """Test string query conditions."""
        cf = CellArrayFrame(uri=self.uri)

        # Query value > 2 (Expect C and D)
        res = cf["value > 2"]
        self.assertEqual(len(res), 2)
        expected_names = sorted(["C", "D"])
        actual_names = sorted(res["name"].tolist())
        self.assertEqual(actual_names, expected_names)

    def test_query_condition_with_cols(self):
        """Test query condition with column subset."""
        cf = CellArrayFrame(uri=self.uri)

        # Query group == 'x', select 'name' (Expect A and B)
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
        # Ensure index continues
        new_df.index = [4]
        new_df.index.name = "row_id"

        cf.write_batch(new_df)

        # Verify count
        res = cf[:]
        self.assertEqual(len(res), 5)
        # Check the appended row
        res_e = cf["name == 'E'"]
        self.assertEqual(len(res_e), 1)
        self.assertEqual(res_e.iloc[0]["value"], 5)


if __name__ == "__main__":
    unittest.main()

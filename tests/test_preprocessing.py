###
## cluster_maker
## Unit tests for preprocessing module
###

import unittest
import pandas as pd
import numpy as np
from cluster_maker.preprocessing import select_features, standardise_features

class TestPreprocessing(unittest.TestCase):
    
    def test_select_features_non_numeric(self):
        """
        Test that select_features raises a TypeError when a selected column 
        contains non-numeric data (e.g., strings). This is critical to prevent 
        mathematical operations from failing silently or confusingly later in the pipeline.
        """
        # Create a DataFrame with a string column
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['cat', 'dog', 'mouse']
        })
        
        # Expect a TypeError when trying to select the string column 'B'
        with self.assertRaises(TypeError):
            select_features(df, ['A', 'B'])

    def test_select_features_missing_columns(self):
        """
        Test that select_features raises a KeyError when requested columns 
        are not present in the input DataFrame. This detects common user errors 
        like typos in column names immediately.
        """
        df = pd.DataFrame({
            'col1': [10, 20],
            'col2': [30, 40]
        })
        
        # Ask for 'col3', which doesn't exist
        with self.assertRaises(KeyError):
            select_features(df, ['col1', 'col3'])

    def test_standardise_features_correctness(self):
        """
        Test that standardise_features correctly scales data to have zero mean 
        and unit variance. This verifies the mathematical correctness of the 
        preprocessing step, which is essential for distance-based clustering.
        """
        # Create a simple numpy array
        # Using a range ensures non-zero variance
        data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        
        scaled = standardise_features(data)
        
        # Check mean is approximately 0
        self.assertAlmostEqual(np.mean(scaled), 0.0, places=5)
        
        # Check standard deviation is approximately 1
        self.assertAlmostEqual(np.std(scaled), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
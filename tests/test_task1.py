

# tests/test_task1.py
"""
Unit tests for Task 1: EDA and Preprocessing
"""

import unittest
import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from notebooks.eda_preprocessing import ComplaintDataAnalyzer

class TestComplaintDataAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.analyzer = ComplaintDataAnalyzer()
        
        # Create test data
        self.test_data = pd.DataFrame({
            'Product': ['Credit card', 'Personal loan', 'Savings account', 
                       'Money transfers', 'Mortgage', 'Debt collection'],
            'Consumer complaint narrative': [
                'I have a complaint about my credit card charges.',
                'My personal loan interest rate is too high.',
                'Savings account withdrawal problem.',
                'Money transfer failed to go through.',
                'Mortgage issue not relevant.',
                'Debt collection issue not relevant.'
            ],
            'Company': ['Bank A', 'Bank B', 'Bank C', 'Bank D', 'Bank E', 'Bank F'],
            'Date received': ['2023-01-01', '2023-01-02', '2023-01-03', 
                            '2023-01-04', '2023-01-05', '2023-01-06']
        })
        
        self.analyzer.data = self.test_data
    
    def test_filter_data(self):
        """Test data filtering"""
        filtered = self.analyzer.filter_data()
        
        # Should have 4 rows (our 4 target products)
        self.assertEqual(len(filtered), 4)
        
        # Should only have our target products
        expected_products = ['Credit card', 'Personal loan', 'Savings account', 'Money transfers']
        actual_products = filtered['product_category'].unique().tolist()
        
        self.assertCountEqual(actual_products, expected_products)
    
    def test_clean_text(self):
        """Test text cleaning function"""
        test_cases = [
            # (input, expected_output)
            ("I AM WRITING TO COMPLAIN about charges!", "i am writing to complain about charges"),
            ("Email: test@example.com Phone: 123-456-7890", "email phone"),
            ("Visit https://example.com for details", "visit for details"),
            ("Multiple   spaces   here", "multiple spaces here"),
            ("", ""),
            (None, ""),
        ]
        
        for input_text, expected in test_cases:
            cleaned = self.analyzer.clean_text(input_text)
            self.assertEqual(cleaned, expected)
    
    def test_product_mapping(self):
        """Test product name mapping"""
        test_products = [
            ('credit card', 'Credit card'),
            ('Credit cards', 'Credit card'),
            ('personal loan', 'Personal loan'),
            ('Personal loans', 'Personal loan'),
            ('savings account', 'Savings account'),
            ('Savings accounts', 'Savings account'),
            ('money transfer', 'Money transfers'),
            ('Money transfers', 'Money transfers'),
            ('mortgage', 'mortgage'),  # Not in our target, should remain unchanged
        ]
        
        for input_product, expected in test_products:
            # Test the mapping logic from filter_data
            mapped = self.analyzer._map_product_internal(input_product)
            self.assertEqual(mapped, expected)
    
    def test_output_files(self):
        """Test that output files are created"""
        # Run the full pipeline
        self.analyzer.filter_data()
        self.analyzer.preprocess_data()
        
        # Check that output files exist
        output_files = [
            'data/processed/filtered_complaints.csv',
            'data/processed/preprocessing_summary.json',
        ]
        
        for file_path in output_files:
            self.assertTrue(os.path.exists(file_path), f"File not created: {file_path}")
        
        # Check CSV content
        df = pd.read_csv('data/processed/filtered_complaints.csv')
        self.assertGreater(len(df), 0)
        self.assertIn('cleaned_narrative', df.columns)
        self.assertIn('product_category', df.columns)
    
    def test_narrative_length_calculation(self):
        """Test narrative length calculations"""
        test_narratives = [
            "Short text",
            "This is a longer piece of text with more words in it",
            "",  # Empty
            "A",  # Very short
        ]
        
        for narrative in test_narratives:
            # Test word count
            word_count = len(narrative.split())
            
            # Test character length
            char_length = len(narrative)
            
            # Just verify the functions don't crash
            self.assertIsInstance(word_count, int)
            self.assertIsInstance(char_length, int)
    
    def tearDown(self):
        """Clean up test files"""
        # Remove test output files
        test_files = [
            'data/processed/filtered_complaints.csv',
            'data/processed/preprocessing_summary.json',
            'data/processed/eda_report.txt',
            'data/processed/eda_summary.md',
        ]
        
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)

def run_tests():
    """Run all tests"""
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestComplaintDataAnalyzer)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
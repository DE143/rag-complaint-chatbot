# src/task1_eda_preprocessing.py
#!/usr/bin/env python3
"""
Task 1: Exploratory Data Analysis and Data Preprocessing
Main script for running the complete EDA and preprocessing pipeline.
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from notebooks.eda_preprocessing import ComplaintDataAnalyzer, create_sample_data

def main():
    parser = argparse.ArgumentParser(description='Run EDA and preprocessing for complaint data')
    parser.add_argument('--data_path', type=str, help='Path to input complaint data file')
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size for testing')
    parser.add_argument('--create_sample', action='store_true', help='Create sample data if no data file exists')
    parser.add_argument('--skip_eda', action='store_true', help='Skip EDA and only run preprocessing')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ComplaintDataAnalyzer(data_path=args.data_path)
    
    try:
        # Check if data file exists
        if args.data_path and not os.path.exists(args.data_path):
            if args.create_sample:
                print(f"Data file not found at {args.data_path}. Creating sample data...")
                create_sample_data()
                analyzer.data_path = 'data/raw/sample_complaints.csv'
            else:
                print(f"Error: Data file not found at {args.data_path}")
                print("Use --create_sample flag to create sample data, or specify correct --data_path")
                return 1
        
        # Load data
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        data = analyzer.load_data(sample_size=args.sample_size)
        
        if not args.skip_eda:
            # Perform EDA
            print("\n" + "="*60)
            print("EXPLORATORY DATA ANALYSIS")
            print("="*60)
            analyzer.basic_eda()
        
        # Filter data
        print("\n" + "="*60)
        print("FILTERING DATA")
        print("="*60)
        filtered_data = analyzer.filter_data()
        
        # Preprocess data
        print("\n" + "="*60)
        print("PREPROCESSING DATA")
        print("="*60)
        cleaned_data = analyzer.preprocess_data()
        
        # Generate report
        print("\n" + "="*60)
        print("GENERATING REPORTS")
        print("="*60)
        analyzer.generate_eda_report()
        
        # Final summary
        print("\n" + "="*60)
        print("TASK 1 COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nSummary:")
        print(f"  Original complaints: {len(data):,}")
        print(f"  After filtering: {len(filtered_data):,}")
        print(f"  After cleaning: {len(cleaned_data):,}")
        print(f"  Retention rate: {(len(cleaned_data)/len(data)*100):.1f}%")
        
        print(f"\nOutput files:")
        print(f"  1. Cleaned data: data/processed/filtered_complaints.csv")
        print(f"  2. Preprocessing summary: data/processed/preprocessing_summary.json")
        print(f"  3. EDA report: data/processed/eda_report.txt")
        print(f"  4. Markdown summary: data/processed/eda_summary.md")
        print(f"\nVisualizations: notebooks/figures/")
        
        return 0
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
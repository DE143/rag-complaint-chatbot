# Task 1: Exploratory Data Analysis and Data Preprocessing

## Overview

This task involves analyzing and preparing the consumer complaint data for the RAG Complaint Chatbot. The goal is to understand the data structure, clean the text narratives, and prepare the data for embedding and retrieval.

## Files Structure

```
rag-complaint-chatbot/
├── data/
│ ├── raw/ # Original complaint data
│ └── processed/ # Processed data (output of Task 1)
├── notebooks/
│ ├── eda_preprocessing.py # Main EDA and preprocessing class
│ ├── eda_notebook.ipynb # Jupyter notebook for interactive EDA
│ └── figures/ # Generated visualizations
├── src/
│ └── task1_eda_preprocessing.py # Command-line script for Task 1
└── setup_environment.py # Environment setup script
```

## How to Run

### Option 1: Using the Command Line Script

1. First, set up the environment:
```
python setup_environment.py
```
2. Run the EDA and preprocessing pipeline:
# With your own data file
```
python src/task1_eda_preprocessing.py --data_path data/raw/complaints.csv
```

# Create and use sample data
```
python src/task1_eda_preprocessing.py --create_sample
```
# Run with sample size for testing
```
python src/task1_eda_preprocessing.py --create_sample --sample_size 10000
```
Option 2: Using the Jupyter Notebook

    1. Launch Jupyter:
 ```
    jupyter notebook notebooks/eda_notebook.ipynb
 ```

     Run the cells sequentially to perform EDA and preprocessing interactively.

Output Files

After successful execution, the following files will be generated:

    data/processed/filtered_complaints.csv - The cleaned and filtered dataset containing only:

        Target products (Credit card, Personal loan, Savings account, Money transfers)

        Non-empty complaint narratives

        Cleaned text (lowercase, special characters removed, etc.)

    data/processed/preprocessing_summary.json - JSON file with preprocessing statistics

    data/processed/eda_report.txt - Detailed EDA report

    data/processed/eda_summary.md - Markdown summary for the final report

    notebooks/figures/ - Directory containing all generated visualizations

Key Steps Performed
1. Data Loading

    Load the CFPB complaint dataset

    Handle different file formats (CSV, JSON, Parquet)

    Optional sampling for testing

2. Exploratory Data Analysis

    Analyze data structure and types

    Check for missing values

    Analyze product distribution

    Analyze narrative length distribution

    Temporal analysis (if date information available)

    Company analysis

3. Data Filtering

    Filter to include only target product categories

    Remove complaints without narratives

    Remove duplicate complaints

    Standardize product names

4. Text Preprocessing

    Convert to lowercase

    Expand contractions

    Remove URLs, email addresses, phone numbers

    Remove special characters

    Remove common boilerplate text

    Remove very short narratives after cleaning

5. Analysis and Reporting

    Generate comprehensive statistics

    Create visualizations

    Generate summary reports

Data Quality Insights

The preprocessing pipeline ensures:

    All narratives are non-empty and meaningful

    Text is cleaned for better embedding quality

    Product categories are standardized

    Data is ready for chunking and embedding in Task 2

Next Steps

After completing Task 1, proceed to:

    Task 2: Text chunking, embedding, and vector store indexing

    Task 3: Building the RAG core logic and evaluation

    Task 4: Creating an interactive chat interface
Summary

I've provided you with a complete implementation for Task 1 that includes:
1. Main Components:

    notebooks/eda_preprocessing.py - Core EDA and preprocessing class

    notebooks/eda_notebook.ipynb - Interactive Jupyter notebook

    src/task1_eda_preprocessing.py - Command-line script

    tests/test_task1.py - Unit tests

2. Key Features:

    Comprehensive EDA: Product distribution, narrative length analysis, temporal trends

    Data Filtering: Filters to target products, removes empty narratives

    Text Cleaning: Lowercasing, special character removal, boilerplate removal

    Visualizations: Multiple plots for better understanding

    Report Generation: Detailed reports in multiple formats

3. How to Use:

# Setup environment
```
python setup_environment.py
```
# Run Task 1 with sample data
```
python src/task1_eda_preprocessing.py --create_sample
```
# Run with your own data
```
python src/task1_eda_preprocessing.py --data_path /path/to/your/complaints.csv
```
# Run tests
```
python tests/test_task1.py
```


4. Expected Output:

The pipeline will generate:

    Cleaned dataset with 4 product categories

    Multiple visualization files

    Comprehensive reports

    Preprocessing summary

This implementation provides a solid foundation for Task 1 and prepares the data for subsequent tasks in the RAG pipeline. The code is modular, well-documented, and follows best practices for data preprocessing and EDA.
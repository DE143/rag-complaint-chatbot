# setup_environment.py
"""
Setup script for the RAG Complaint Chatbot project environment.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing requirements...")
    
    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install requirements
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    else:
        print(f"Error: {requirements_file} not found")
        return False
    
    # Additional packages that might be needed
    additional_packages = [
        "ipykernel",  # For Jupyter notebooks
        "notebook",   # Jupyter notebook
        "jupyterlab", # Jupyter Lab
        "faker",      # For generating sample data
    ]
    
    print("\nInstalling additional packages...")
    for package in additional_packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"  ✓ {package}")
        except:
            print(f"  ✗ Failed to install {package}")
    
    print("\nAll packages installed successfully!")
    return True

def check_environment():
    """Check if all required packages are installed"""
    print("Checking environment...")
    
    required_packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "scikit-learn",
        "transformers",
        "sentence-transformers",
        "langchain",
        "chromadb",
        "faiss-cpu",
        "gradio",
        "streamlit",
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ✗ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\nAll required packages are installed!")
        return True

def setup_directories():
    """Create necessary directories"""
    print("\nSetting up project directories...")
    
    directories = [
        'data/raw',
        'data/processed',
        'vector_store',
        'notebooks/figures',
        'src',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  Created: {directory}/")
    
    # Create empty __init__.py files
    for dir in ['notebooks', 'src', 'tests']:
        init_file = os.path.join(dir, '__init__.py')
        with open(init_file, 'w') as f:
            f.write('')
        print(f"  Created: {init_file}")
    
    print("\nDirectory structure created successfully!")
    return True

def main():
    """Main setup function"""
    print("="*60)
    print("RAG COMPLAINT CHATBOT - ENVIRONMENT SETUP")
    print("="*60)
    
    try:
        # Setup directories
        setup_directories()
        
        # Install requirements
        install_success = install_requirements()
        
        if install_success:
            # Check environment
            env_ok = check_environment()
            
            if env_ok:
                print("\n" + "="*60)
                print("SETUP COMPLETED SUCCESSFULLY!")
                print("="*60)
                print("\nNext steps:")
                print("1. Place your complaint data in data/raw/")
                print("2. Run: python src/task1_eda_preprocessing.py")
                print("3. Or use the notebook: notebooks/eda_notebook.ipynb")
                return 0
            else:
                print("\nEnvironment check failed. Please install missing packages.")
                return 1
        else:
            print("\nFailed to install requirements.")
            return 1
            
    except Exception as e:
        print(f"\nError during setup: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
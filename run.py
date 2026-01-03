# run.py
"""
Main script to run the complete project
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='RAG Complaint Chatbot')
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4],
                       help='Run specific task (1-4)')
    parser.add_argument('--interface', type=str, choices=['gradio', 'streamlit'],
                       default='gradio', help='Web interface to use')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation only')
    
    args = parser.parse_args()
    
    if args.task == 1:
        print("Running Task 1: EDA and Preprocessing...")
        from src.task1_eda_preprocessing import main as task1_main
        task1_main()
        
    elif args.task == 2:
        print("Running Task 2: Chunking and Embedding...")
        from src.task2_chunking_embedding import main as task2_main
        task2_main()
        
    elif args.task == 3:
        print("Running Task 3: RAG Pipeline and Evaluation...")
        from src.task3_rag_pipeline import main as task3_main
        sys.argv = [sys.argv[0]]
        if args.eval:
            sys.argv.append('--eval_only')
        task3_main()
        
    elif args.task == 4:
        print(f"Running Task 4: {args.interface.title()} Interface...")
        if args.interface == 'gradio':
            from app import main as gradio_main
            gradio_main()
        else:
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app_streamlit.py"])
    
    else:
        print("RAG Complaint Chatbot - Complete Project")
        print("="*50)
        print("\nAvailable tasks:")
        print("  1. EDA and Data Preprocessing")
        print("  2. Text Chunking and Embedding")
        print("  3. RAG Pipeline and Evaluation")
        print("  4. Interactive Chat Interface")
        print("\nUsage:")
        print("  python run.py --task 1  # Run Task 1")
        print("  python run.py --task 4 --interface gradio  # Launch Gradio interface")
        print("\nFor detailed instructions, see README.md")

if __name__ == "__main__":
    main()
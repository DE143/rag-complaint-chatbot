# src/task3_rag_pipeline.py
"""
Task 3: Building the RAG Core Logic and Evaluation
"""

import os
import sys
import json
import warnings
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# Vector store and embeddings
import chromadb
from chromadb.config import Settings
import faiss
from sentence_transformers import SentenceTransformer

# LLM and LangChain
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')

class RAGSystem:
    """
    RAG System for complaint analysis
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 llm_model_name: str = "microsoft/DialoGPT-medium",
                 use_prebuilt: bool = True,
                 vector_store_type: str = "chroma"):
        """
        Initialize RAG system
        
        Args:
            embedding_model_name: Name of embedding model
            llm_model_name: Name of LLM model
            use_prebuilt: Use pre-built vector store
            vector_store_type: Type of vector store ('chroma' or 'faiss')
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.use_prebuilt = use_prebuilt
        self.vector_store_type = vector_store_type
        
        # Initialize components
        self.embedding_model = None
        self.llm = None
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
        # Paths
        self.prebuilt_dir = Path("data/prebuilt_vector_store")
        self.vector_store_dir = Path("vector_store")
        self.evaluation_dir = Path("evaluation_results")
        self.evaluation_dir.mkdir(exist_ok=True)
        
        print(f"Initialized RAG System with:")
        print(f"  Embedding model: {embedding_model_name}")
        print(f"  LLM model: {llm_model_name}")
        print(f"  Vector store: {vector_store_type}")
        print(f"  Use pre-built: {use_prebuilt}")
    
    def load_prebuilt_vector_store(self):
        """
        Load pre-built vector store from provided files
        """
        print("\n" + "="*60)
        print("LOADING PRE-BUILT VECTOR STORE")
        print("="*60)
        
        # Check if pre-built data exists
        parquet_path = self.prebuilt_dir / "complaint_embeddings.parquet"
        if not parquet_path.exists():
            # Try alternative locations
            alt_paths = [
                "data/prebuilt_vector_store/complaint_embeddings.parquet",
                "complaint_embeddings.parquet",
                "data/processed/complaint_embeddings.parquet"
            ]
            
            for path in alt_paths:
                if os.path.exists(path):
                    parquet_path = Path(path)
                    break
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Pre-built embeddings not found at: {parquet_path}")
        
        # Load Parquet data
        print(f"Loading pre-built embeddings from: {parquet_path}")
        self.prebuilt_data = pd.read_parquet(parquet_path)
        print(f"Loaded {len(self.prebuilt_data)} chunks from pre-built data")
        
        # Display sample
        print(f"\nSample of pre-built data:")
        print(f"Columns: {list(self.prebuilt_data.columns)}")
        print(f"First row:")
        print(self.prebuilt_data.iloc[0].to_dict())
        
        # Initialize embedding model
        print(f"\nInitializing embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Create vector store from pre-built data
        if self.vector_store_type.lower() == "chroma":
            self._create_chromadb_from_prebuilt()
        else:
            self._create_faiss_from_prebuilt()
        
        print("Pre-built vector store loaded successfully!")
    
    def _create_chromadb_from_prebuilt(self):
        """Create ChromaDB vector store from pre-built data"""
        print("Creating ChromaDB vector store from pre-built data...")
        
        # Initialize ChromaDB
        chroma_dir = self.vector_store_dir / "chroma_db_prebuilt"
        chroma_dir.mkdir(exist_ok=True)
        
        client = chromadb.PersistentClient(path=str(chroma_dir))
        
        # Create collection
        collection_name = "prebuilt_complaints"
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        collection = client.create_collection(
            name=collection_name,
            metadata={"description": "Pre-built complaint embeddings", "model": self.embedding_model_name}
        )
        
        # Prepare documents and metadata
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        print(f"Preparing {len(self.prebuilt_data)} documents for ChromaDB...")
        for idx, row in tqdm(self.prebuilt_data.iterrows(), total=len(self.prebuilt_data), desc="Processing chunks"):
            chunk_id = f"chunk_{idx:08d}"
            chunk_text = row.get('chunk_text', '')
            
            if not chunk_text or not isinstance(chunk_text, str):
                continue
            
            # Create metadata
            metadata = {
                'complaint_id': str(row.get('complaint_id', '')),
                'product_category': str(row.get('product_category', 'Unknown')),
                'product': str(row.get('product', 'Unknown')),
                'issue': str(row.get('issue', 'Unknown')),
                'sub_issue': str(row.get('sub_issue', 'Unknown')),
                'company': str(row.get('company', 'Unknown')),
                'state': str(row.get('state', 'Unknown')),
                'date_received': str(row.get('date_received', 'Unknown')),
                'chunk_index': int(row.get('chunk_index', 0)),
                'total_chunks': int(row.get('total_chunks', 1))
            }
            
            # Generate embedding if not present
            if 'embedding' in row and isinstance(row['embedding'], list):
                embedding = row['embedding']
            else:
                embedding = self.embedding_model.encode([chunk_text])[0].tolist()
            
            documents.append(chunk_text)
            metadatas.append(metadata)
            ids.append(chunk_id)
            embeddings.append(embedding)
        
        # Add to collection in batches
        batch_size = 1000
        print(f"Adding documents to ChromaDB in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(documents), batch_size), 
                     desc="Adding to ChromaDB",
                     total=len(documents) // batch_size + 1):
            batch_end = min(i + batch_size, len(documents))
            
            collection.add(
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end]
            )
        
        self.chroma_collection = collection
        print(f"ChromaDB collection created with {collection.count()} documents")
    
    def _create_faiss_from_prebuilt(self):
        """Create FAISS index from pre-built data"""
        print("Creating FAISS index from pre-built data...")
        
        # Prepare embeddings and documents
        embeddings = []
        documents = []
        metadatas = []
        
        print(f"Processing {len(self.prebuilt_data)} chunks...")
        for idx, row in tqdm(self.prebuilt_data.iterrows(), total=len(self.prebuilt_data), desc="Processing chunks"):
            chunk_text = row.get('chunk_text', '')
            
            if not chunk_text or not isinstance(chunk_text, str):
                continue
            
            # Generate embedding
            if 'embedding' in row and isinstance(row['embedding'], list):
                embedding = np.array(row['embedding'], dtype=np.float32)
            else:
                embedding = self.embedding_model.encode([chunk_text])[0].astype(np.float32)
            
            # Create document
            metadata = {
                'complaint_id': str(row.get('complaint_id', '')),
                'product_category': str(row.get('product_category', 'Unknown')),
                'product': str(row.get('product', 'Unknown')),
                'issue': str(row.get('issue', 'Unknown')),
                'sub_issue': str(row.get('sub_issue', 'Unknown')),
                'company': str(row.get('company', 'Unknown')),
                'state': str(row.get('state', 'Unknown')),
                'date_received': str(row.get('date_received', 'Unknown')),
                'chunk_index': int(row.get('chunk_index', 0)),
                'total_chunks': int(row.get('total_chunks', 1))
            }
            
            documents.append(Document(
                page_content=chunk_text,
                metadata=metadata
            ))
            embeddings.append(embedding)
            metadatas.append(metadata)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        # Create LangChain FAISS store
        self.embedding_model_langchain = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
        
        self.faiss_index = index
        self.faiss_documents = documents
        self.faiss_metadatas = metadatas
        
        print(f"FAISS index created with {index.ntotal} vectors")
    
    def initialize_llm(self):
        """
        Initialize the language model
        """
        print("\n" + "="*60)
        print("INITIALIZING LANGUAGE MODEL")
        print("="*60)
        
        try:
            # Try to use a smaller, faster model first
            print(f"Loading LLM: {self.llm_model_name}")
            
            # Use text generation pipeline
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(self.llm_model_name)
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            # Create LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            print("LLM initialized successfully!")
            
        except Exception as e:
            print(f"Failed to load {self.llm_model_name}: {e}")
            print("Falling back to smaller model...")
            
            # Fallback to a smaller model
            try:
                fallback_model = "gpt2"
                print(f"Trying fallback model: {fallback_model}")
                
                tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                model = AutoModelForCausalLM.from_pretrained(fallback_model)
                
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_length=256,
                    temperature=0.7
                )
                
                self.llm = HuggingFacePipeline(pipeline=pipe)
                print(f"Successfully loaded fallback model: {fallback_model}")
                
            except Exception as e2:
                print(f"Failed to load fallback model: {e2}")
                print("Using mock LLM for demonstration...")
                self.llm = self._create_mock_llm()
    
    def _create_mock_llm(self):
        """Create a mock LLM for testing"""
        from langchain.llms.base import BaseLLM
        from langchain.schema import Generation, LLMResult
        
        class MockLLM(BaseLLM):
            def _generate(self, prompts, stop=None, run_manager=None, **kwargs):
                responses = []
                for prompt in prompts:
                    # Simple rule-based response
                    if "credit card" in prompt.lower():
                        response = "Based on customer complaints, the main issues with credit cards are: unauthorized charges, billing disputes, and poor customer service. Customers report fraudulent transactions and difficulties in resolving billing errors."
                    elif "loan" in prompt.lower():
                        response = "Personal loan complaints primarily focus on: high interest rates, hidden fees, and payment processing issues. Customers express frustration with unclear terms and unexpected charges."
                    elif "savings" in prompt.lower():
                        response = "Savings account complaints include: withdrawal problems, missing deposits, and account access issues. Customers report difficulties in accessing their funds and delays in transaction processing."
                    elif "money transfer" in prompt.lower():
                        response = "Money transfer issues involve: failed transactions, delayed transfers, and high fees. Customers complain about funds not reaching recipients and poor communication from service providers."
                    else:
                        response = "Based on the complaint data, customers are experiencing various issues across financial products. The most common themes include billing problems, poor customer service, and technical issues with account access."
                    
                    responses.append(Generation(text=response))
                
                return LLMResult(generations=[responses])
            
            @property
            def _llm_type(self):
                return "mock"
        
        return MockLLM()
    
    def create_prompt_template(self):
        """
        Create prompt template for the RAG system
        """
        print("\nCreating prompt template...")
        
        template = """You are a helpful financial analyst assistant for CrediTrust Financial. 
Your task is to analyze customer complaints and provide insightful answers based on the retrieved complaint excerpts.

CONTEXT INFORMATION:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Analyze the provided context carefully
2. Identify key themes, issues, and patterns in the complaints
3. Provide a concise, evidence-based answer
4. If the context doesn't contain relevant information, state: "I don't have enough information from the complaint data to answer this question."
5. Focus on actionable insights that could help improve products/services
6. Mention specific product categories when relevant

ANSWER:
"""
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        print("Prompt template created successfully!")
        return self.prompt
    
    def setup_retrieval_qa(self, k: int = 5):
        """
        Set up the Retrieval QA chain
        
        Args:
            k: Number of documents to retrieve
        """
        print("\n" + "="*60)
        print("SETTING UP RETRIEVAL QA CHAIN")
        print("="*60)
        
        # Initialize LLM if not already done
        if self.llm is None:
            self.initialize_llm()
        
        # Create prompt template
        if not hasattr(self, 'prompt'):
            self.create_prompt_template()
        
        # Set up retriever based on vector store type
        if self.vector_store_type.lower() == "chroma":
            self._setup_chromadb_retriever(k)
        else:
            self._setup_faiss_retriever(k)
        
        # Create QA chain
        print("Creating QA chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
        
        print("Retrieval QA chain setup completed!")
    
    def _setup_chromadb_retriever(self, k: int):
        """Set up ChromaDB retriever"""
        if not hasattr(self, 'chroma_collection'):
            raise ValueError("ChromaDB collection not initialized")
        
        print(f"Setting up ChromaDB retriever with k={k}...")
        
        # Create LangChain Chroma retriever
        embedding_function = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
        
        # Create Chroma vector store
        chroma = Chroma(
            collection_name=self.chroma_collection.name,
            embedding_function=embedding_function,
            persist_directory=str(self.vector_store_dir / "chroma_db_prebuilt"),
            client_settings=Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.vector_store_dir / "chroma_db_prebuilt")
            )
        )
        
        self.retriever = chroma.as_retriever(
            search_kwargs={"k": k}
        )
    
    def _setup_faiss_retriever(self, k: int):
        """Set up FAISS retriever"""
        if not hasattr(self, 'faiss_index'):
            raise ValueError("FAISS index not initialized")
        
        print(f"Setting up FAISS retriever with k={k}...")
        
        # Create FAISS vector store
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
        
        # Create document store
        faiss_store = FAISS.from_documents(
            documents=self.faiss_documents,
            embedding=embeddings
        )
        
        # Replace the index with our pre-built one
        faiss_store.index = self.faiss_index
        
        self.retriever = faiss_store.as_retriever(
            search_kwargs={"k": k}
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized. Call setup_retrieval_qa() first.")
        
        print(f"\nQuery: {question}")
        print("-" * 50)
        
        # Get response
        start_time = datetime.now()
        response = self.qa_chain({"query": question})
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Extract answer and sources
        answer = response.get("result", "No answer generated")
        source_docs = response.get("source_documents", [])
        
        # Format sources
        sources = []
        for i, doc in enumerate(source_docs):
            source_info = {
                "source_id": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "product": doc.metadata.get("product_category", "Unknown"),
                "relevance_score": None  # Can be calculated from distance if available
            }
            sources.append(source_info)
        
        result = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
            "response_time": f"{elapsed:.2f}s"
        }
        
        # Display results
        print(f"Answer: {answer}")
        print(f"\nSources ({len(sources)}):")
        for source in sources[:3]:  # Show top 3 sources
            print(f"  [{source['source_id']}] {source['product']}: {source['content']}")
        
        return result
    
    def evaluate_questions(self, questions: List[Dict[str, str]]) -> pd.DataFrame:
        """
        Evaluate the RAG system with a set of questions
        
        Args:
            questions: List of question dictionaries with 'question' and 'category' keys
            
        Returns:
            DataFrame with evaluation results
        """
        print("\n" + "="*60)
        print("EVALUATING RAG SYSTEM")
        print("="*60)
        
        results = []
        
        for i, q_info in enumerate(tqdm(questions, desc="Evaluating questions")):
            question = q_info["question"]
            category = q_info.get("category", "General")
            
            print(f"\n[{i+1}/{len(questions)}] Question: {question}")
            
            try:
                # Get response
                response = self.query(question)
                
                # Manual evaluation (in real scenario, this would be automated or human-evaluated)
                # For now, we'll use simple heuristics for demonstration
                evaluation = self._evaluate_response(response, question)
                
                result = {
                    "question_id": i + 1,
                    "question": question,
                    "category": category,
                    "answer": response["answer"],
                    "num_sources": response["num_sources"],
                    "response_time": response["response_time"],
                    **evaluation
                }
                
                results.append(result)
                
                # Print evaluation
                print(f"  Evaluation: Quality={result['quality_score']}/5, Relevance={result['relevance_score']}/5")
                
            except Exception as e:
                print(f"  Error processing question: {e}")
                results.append({
                    "question_id": i + 1,
                    "question": question,
                    "category": category,
                    "answer": f"Error: {str(e)}",
                    "num_sources": 0,
                    "response_time": "0.00s",
                    "quality_score": 1,
                    "relevance_score": 1,
                    "has_answer": False,
                    "notes": str(e)
                })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.evaluation_dir / f"evaluation_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        
        # Generate summary statistics
        self._generate_evaluation_summary(results_df, results_path)
        
        return results_df
    
    def _evaluate_response(self, response: Dict[str, Any], question: str) -> Dict[str, Any]:
        """
        Evaluate a single response
        
        Args:
            response: Response dictionary
            question: Original question
            
        Returns:
            Evaluation metrics
        """
        answer = response["answer"]
        sources = response["sources"]
        
        # Initialize scores
        quality_score = 3  # Default medium score
        relevance_score = 3
        has_answer = True
        
        # Check if answer indicates no information
        no_info_phrases = [
            "don't have enough information",
            "no information",
            "cannot answer",
            "not in the context"
        ]
        
        if any(phrase in answer.lower() for phrase in no_info_phrases):
            has_answer = False
            quality_score = 1
            relevance_score = 1
        
        # Check answer length and quality
        answer_length = len(answer.split())
        
        if answer_length < 20:
            quality_score = max(1, quality_score - 1)
        
        # Check for specific keywords from question
        question_keywords = set(question.lower().split())
        answer_keywords = set(answer.lower().split())
        common_keywords = question_keywords.intersection(answer_keywords)
        
        if len(common_keywords) > 0:
            relevance_score = min(5, relevance_score + 1)
        
        # Check if sources were retrieved
        if len(sources) == 0:
            relevance_score = max(1, relevance_score - 1)
        
        # Check answer structure
        if "based on" in answer.lower() or "according to" in answer.lower():
            quality_score = min(5, quality_score + 1)
        
        # Check for actionable insights
        actionable_terms = ["recommend", "suggest", "improve", "issue", "problem", "solution"]
        if any(term in answer.lower() for term in actionable_terms):
            quality_score = min(5, quality_score + 1)
        
        return {
            "quality_score": quality_score,
            "relevance_score": relevance_score,
            "has_answer": has_answer,
            "answer_length": answer_length,
            "notes": "Automated evaluation - manual review recommended"
        }
    
    def _generate_evaluation_summary(self, results_df: pd.DataFrame, results_path: Path):
        """Generate evaluation summary"""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        # Calculate statistics
        total_questions = len(results_df)
        avg_quality = results_df["quality_score"].mean()
        avg_relevance = results_df["relevance_score"].mean()
        avg_response_time = pd.to_numeric(
            results_df["response_time"].str.replace("s", ""),
            errors='coerce'
        ).mean()
        has_answer_rate = results_df["has_answer"].mean() * 100
        
        print(f"\nTotal Questions Evaluated: {total_questions}")
        print(f"Average Quality Score: {avg_quality:.2f}/5")
        print(f"Average Relevance Score: {avg_relevance:.2f}/5")
        print(f"Average Response Time: {avg_response_time:.2f}s")
        print(f"Questions Answered: {has_answer_rate:.1f}%")
        
        # By category
        if "category" in results_df.columns:
            print(f"\nPerformance by Category:")
            category_stats = results_df.groupby("category").agg({
                "quality_score": "mean",
                "relevance_score": "mean",
                "has_answer": "mean"
            }).round(2)
            
            for category, row in category_stats.iterrows():
                print(f"  {category}:")
                print(f"    Quality: {row['quality_score']}/5")
                print(f"    Relevance: {row['relevance_score']}/5")
                print(f"    Answer Rate: {row['has_answer']*100:.1f}%")
        
        # Save summary
        summary = {
            "evaluation_date": datetime.now().isoformat(),
            "total_questions": int(total_questions),
            "average_quality_score": float(avg_quality),
            "average_relevance_score": float(avg_relevance),
            "average_response_time": float(avg_response_time),
            "answer_rate_percentage": float(has_answer_rate),
            "results_file": str(results_path)
        }
        
        summary_path = self.evaluation_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDetailed results saved to: {results_path}")
        print(f"Summary saved to: {summary_path}")
        
        # Create markdown report
        self._create_markdown_report(results_df, summary)
    
    def _create_markdown_report(self, results_df: pd.DataFrame, summary: Dict[str, Any]):
        """Create markdown evaluation report"""
        report_path = self.evaluation_dir / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# RAG System Evaluation Report\n\n")
            f.write(f"**Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Questions**: {summary['total_questions']}\n")
            f.write(f"- **Average Quality Score**: {summary['average_quality_score']:.2f}/5\n")
            f.write(f"- **Average Relevance Score**: {summary['average_relevance_score']:.2f}/5\n")
            f.write(f"- **Average Response Time**: {summary['average_response_time']:.2f}s\n")
            f.write(f"- **Answer Rate**: {summary['answer_rate_percentage']:.1f}%\n\n")
            
            f.write("## Detailed Results\n\n")
            f.write("| Question ID | Question | Category | Quality | Relevance | Answer Length | Has Answer? |\n")
            f.write("|-------------|----------|----------|---------|-----------|---------------|-------------|\n")
            
            for _, row in results_df.iterrows():
                has_answer = "✓" if row["has_answer"] else "✗"
                f.write(f"| {row['question_id']} | {row['question'][:50]}... | {row['category']} | ")
                f.write(f"{row['quality_score']} | {row['relevance_score']} | {row['answer_length']} | {has_answer} |\n")
            
            f.write("\n## Sample Questions and Answers\n\n")
            
            # Show top and bottom performing questions
            top_questions = results_df.nlargest(3, "quality_score")
            bottom_questions = results_df.nsmallest(3, "quality_score")
            
            f.write("### Top Performing Questions\n\n")
            for _, row in top_questions.iterrows():
                f.write(f"**Question {row['question_id']}**: {row['question']}\n\n")
                f.write(f"**Answer**: {row['answer'][:200]}...\n\n")
                f.write(f"*Quality: {row['quality_score']}/5, Relevance: {row['relevance_score']}/5*\n\n")
                f.write("---\n\n")
            
            f.write("### Areas for Improvement\n\n")
            for _, row in bottom_questions.iterrows():
                f.write(f"**Question {row['question_id']}**: {row['question']}\n\n")
                f.write(f"**Answer**: {row['answer'][:200]}...\n\n")
                f.write(f"*Quality: {row['quality_score']}/5, Relevance: {row['relevance_score']}/5*\n\n")
                f.write("---\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Improve Retrieval**: Consider tuning the embedding model or chunking strategy\n")
            f.write("2. **Enhance Prompt Engineering**: Refine the prompt template for better answers\n")
            f.write("3. **Increase Training Data**: Add more diverse complaint examples\n")
            f.write("4. **Implement Hybrid Search**: Combine semantic search with keyword matching\n")
        
        print(f"Markdown report saved to: {report_path}")

def get_sample_questions() -> List[Dict[str, str]]:
    """
    Get sample questions for evaluation
    
    Returns:
        List of question dictionaries
    """
    questions = [
        # Credit Card Questions
        {"question": "What are the main complaints about credit cards?", "category": "Credit Card"},
        {"question": "Why are customers unhappy with credit card services?", "category": "Credit Card"},
        {"question": "What billing issues do customers report with credit cards?", "category": "Credit Card"},
        
        # Personal Loan Questions
        {"question": "What problems do customers face with personal loans?", "category": "Personal Loan"},
        {"question": "What are the common complaints about loan interest rates?", "category": "Personal Loan"},
        {"question": "How do customers describe their loan payment experiences?", "category": "Personal Loan"},
        
        # Savings Account Questions
        {"question": "What issues do customers report with savings accounts?", "category": "Savings Account"},
        {"question": "Why are people complaining about their bank accounts?", "category": "Savings Account"},
        {"question": "What withdrawal problems do customers encounter?", "category": "Savings Account"},
        
        # Money Transfer Questions
        {"question": "What are the complaints about money transfer services?", "category": "Money Transfer"},
        {"question": "Why do customers complain about wire transfers?", "category": "Money Transfer"},
        {"question": "What problems occur with international money transfers?", "category": "Money Transfer"},
        
        # Cross-product Questions
        {"question": "What are the most common customer service complaints across all products?", "category": "General"},
        {"question": "How do complaint patterns differ between credit cards and personal loans?", "category": "Comparative"},
        {"question": "What are the top 3 issues reported by customers in the last year?", "category": "General"},
        
        # Specific Scenario Questions
        {"question": "What do customers say about unauthorized transactions?", "category": "Fraud"},
        {"question": "How do customers describe their experiences with hidden fees?", "category": "Fees"},
        {"question": "What problems do customers report with online banking access?", "category": "Technical"},
        
        # Action-oriented Questions
        {"question": "What should CrediTrust improve based on customer complaints?", "category": "Recommendations"},
        {"question": "Which product has the most complaints and why?", "category": "Analytical"},
        {"question": "What are customers saying about response times to their complaints?", "category": "Service"}
    ]
    
    return questions

def main():
    """Main function for Task 3"""
    parser = argparse.ArgumentParser(description='Build and evaluate RAG system')
    parser.add_argument('--use_prebuilt', action='store_true', default=True,
                       help='Use pre-built vector store')
    parser.add_argument('--vector_store', type=str, default='chroma',
                       choices=['chroma', 'faiss'],
                       help='Vector store type')
    parser.add_argument('--embedding_model', type=str, default='all-MiniLM-L6-v2',
                       help='Embedding model name')
    parser.add_argument('--llm_model', type=str, default='microsoft/DialoGPT-medium',
                       help='LLM model name')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of documents to retrieve')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only run evaluation (skip setup)')
    parser.add_argument('--questions_file', type=str,
                       help='JSON file with custom questions')
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag_system = RAGSystem(
        embedding_model_name=args.embedding_model,
        llm_model_name=args.llm_model,
        use_prebuilt=args.use_prebuilt,
        vector_store_type=args.vector_store
    )
    
    try:
        if not args.eval_only:
            # Load vector store
            if args.use_prebuilt:
                rag_system.load_prebuilt_vector_store()
            else:
                # In a real scenario, you would build from scratch here
                print("Building vector store from scratch...")
                # This would call methods from Task 2
                pass
            
            # Setup RAG pipeline
            rag_system.setup_retrieval_qa(k=args.k)
            
            # Test with sample questions
            print("\n" + "="*60)
            print("TESTING RAG SYSTEM")
            print("="*60)
            
            test_questions = [
                "What are the main complaints about credit cards?",
                "What problems do customers face with money transfers?"
            ]
            
            for question in test_questions:
                rag_system.query(question)
                print("\n" + "-"*60 + "\n")
        
        # Load questions for evaluation
        if args.questions_file:
            with open(args.questions_file, 'r') as f:
                questions = json.load(f)
        else:
            questions = get_sample_questions()
        
        # Run evaluation
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE EVALUATION")
        print("="*60)
        
        results_df = rag_system.evaluate_questions(questions)
        
        print("\n" + "="*60)
        print("TASK 3 COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Display top questions
        print("\nTop Performing Questions:")
        top_results = results_df.nlargest(3, "quality_score")
        for _, row in top_results.iterrows():
            print(f"\nQuestion {row['question_id']}: {row['question']}")
            print(f"Answer: {row['answer'][:100]}...")
            print(f"Quality: {row['quality_score']}/5")
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
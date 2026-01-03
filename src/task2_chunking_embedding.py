# src/task2_chunking_embedding.py
"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
import hashlib
from tqdm import tqdm
import argparse
from typing import List, Dict, Any, Optional, Tuple

# Vector store libraries
import faiss
import chromadb
from chromadb.config import Settings

# Embedding and text processing
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch

warnings.filterwarnings('ignore')

class VectorStoreBuilder:
    """
    Class for building vector stores from complaint data
    """
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 sample_size: Optional[int] = 15000,
                 random_state: int = 42):
        """
        Initialize the VectorStoreBuilder
        
        Args:
            embedding_model_name: Name of the embedding model
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            sample_size: Number of complaints to sample (None for all)
            random_state: Random seed for reproducibility
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sample_size = sample_size
        self.random_state = random_state
        
        # Initialize components
        self.embedding_model = None
        self.text_splitter = None
        self.data = None
        self.chunks = []
        self.metadata = []
        
        # Output directories
        self.output_dir = Path("vector_store")
        self.output_dir.mkdir(exist_ok=True)
        
        # Chunks directory
        self.chunks_dir = Path("data/chunks")
        self.chunks_dir.mkdir(exist_ok=True)
        
        print(f"Initialized VectorStoreBuilder with:")
        print(f"  Model: {embedding_model_name}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Chunk overlap: {chunk_overlap}")
        print(f"  Sample size: {sample_size}")
    
    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and prepare the complaint data
        
        Args:
            data_path: Path to the cleaned complaint data
            
        Returns:
            DataFrame with complaint data
        """
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Use default path if not provided
        if data_path is None:
            data_path = "data/processed/filtered_complaints.csv"
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data
        self.data = pd.read_csv(data_path)
        print(f"Loaded {len(self.data)} complaints from {data_path}")
        
        # Check required columns
        required_columns = ['cleaned_narrative', 'product_category']
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Add complaint ID if not present
        if 'complaint_id' not in self.data.columns:
            self.data['complaint_id'] = self.data.index.astype(str)
        
        # Display data info
        print(f"\nData shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        print(f"\nProduct distribution:")
        print(self.data['product_category'].value_counts())
        
        return self.data
    
    def create_stratified_sample(self) -> pd.DataFrame:
        """
        Create a stratified sample of complaints
        
        Returns:
            Sampled DataFrame
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.sample_size is None or self.sample_size >= len(self.data):
            print(f"Using all {len(self.data)} complaints")
            return self.data
        
        print(f"\nCreating stratified sample of {self.sample_size} complaints...")
        
        # Calculate sample size per product category
        product_counts = self.data['product_category'].value_counts()
        total = len(self.data)
        
        # Calculate proportions
        proportions = {product: count / total for product, count in product_counts.items()}
        
        # Calculate sample sizes per category
        sample_sizes = {}
        for product, prop in proportions.items():
            sample_sizes[product] = max(1, int(self.sample_size * prop))
        
        # Adjust if total doesn't match desired sample size
        total_sample = sum(sample_sizes.values())
        if total_sample != self.sample_size:
            # Adjust the largest category
            largest_product = max(sample_sizes, key=sample_sizes.get)
            adjustment = self.sample_size - total_sample
            sample_sizes[largest_product] += adjustment
        
        print(f"Sample sizes per product:")
        for product, size in sample_sizes.items():
            print(f"  {product}: {size}")
        
        # Perform stratified sampling
        sampled_dfs = []
        for product, size in sample_sizes.items():
            product_data = self.data[self.data['product_category'] == product]
            if len(product_data) > size:
                sampled = product_data.sample(n=size, random_state=self.random_state)
            else:
                sampled = product_data
            sampled_dfs.append(sampled)
        
        sampled_data = pd.concat(sampled_dfs, ignore_index=True)
        
        print(f"\nSampled data shape: {sampled_data.shape}")
        print(f"Sample product distribution:")
        print(sampled_data['product_category'].value_counts())
        
        return sampled_data
    
    def initialize_components(self):
        """
        Initialize embedding model and text splitter
        """
        print("\n" + "="*60)
        print("INITIALIZING COMPONENTS")
        print("="*60)
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Get embedding dimension
        sample_embedding = self.embedding_model.encode(["sample text"])
        self.embedding_dim = sample_embedding.shape[1]
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize text splitter
        print(f"Initializing text splitter with chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        print("Components initialized successfully!")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into chunks
        
        Args:
            text: Text to chunk
            metadata: Metadata for the text
            
        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        if not text or not isinstance(text, str):
            return []
        
        # Split text
        chunks = self.text_splitter.split_text(text)
        
        # Create metadata for each chunk
        chunk_metadata_list = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_text': chunk,  # Store the actual chunk text
                'chunk_length': len(chunk),
                'chunk_hash': hashlib.md5(chunk.encode()).hexdigest()[:8]
            })
            chunk_metadata_list.append((chunk, chunk_metadata))
        
        return chunk_metadata_list
    
    def process_complaints(self, data: pd.DataFrame):
        """
        Process all complaints: chunk text and prepare for embedding
        
        Args:
            data: DataFrame with complaint data
        """
        print("\n" + "="*60)
        print("CHUNKING COMPLAINT NARRATIVES")
        print("="*60)
        
        self.chunks = []
        self.metadata = []
        
        total_chunks = 0
        complaints_without_chunks = 0
        
        # Process each complaint
        for idx, row in tqdm(data.iterrows(), total=len(data), desc="Chunking complaints"):
            complaint_id = str(row.get('complaint_id', idx))
            text = str(row.get('cleaned_narrative', ''))
            
            # Skip empty text
            if not text or len(text.strip()) < 10:
                complaints_without_chunks += 1
                continue
            
            # Create base metadata
            base_metadata = {
                'complaint_id': complaint_id,
                'product_category': row.get('product_category', 'Unknown'),
                'product': row.get('product', row.get('Product', 'Unknown')),
                'issue': row.get('issue', 'Unknown'),
                'sub_issue': row.get('sub_issue', 'Unknown'),
                'company': row.get('company', row.get('Company', 'Unknown')),
                'state': row.get('state', 'Unknown'),
                'date_received': row.get('date_received', 'Unknown'),
                'original_text': text[:200] + "..." if len(text) > 200 else text
            }
            
            # Chunk the text
            chunk_data = self.chunk_text(text, base_metadata)
            
            if chunk_data:
                for chunk_text, chunk_metadata in chunk_data:
                    self.chunks.append(chunk_text)
                    self.metadata.append(chunk_metadata)
                total_chunks += len(chunk_data)
            else:
                complaints_without_chunks += 1
        
        print(f"\nChunking completed!")
        print(f"  Total complaints processed: {len(data)}")
        print(f"  Total chunks created: {total_chunks}")
        print(f"  Average chunks per complaint: {total_chunks / len(data):.2f}")
        print(f"  Complaints without chunks: {complaints_without_chunks}")
        
        if total_chunks == 0:
            raise ValueError("No chunks were created. Check your data and text splitter configuration.")
        
        # Save chunks to file for inspection
        self._save_chunks_for_inspection()
        
        return self.chunks, self.metadata
    
    def _save_chunks_for_inspection(self):
        """Save sample chunks for inspection"""
        sample_size = min(100, len(self.chunks))
        sample_data = []
        
        for i in range(sample_size):
            sample_data.append({
                'chunk_id': i,
                'chunk_text': self.chunks[i][:200] + "..." if len(self.chunks[i]) > 200 else self.chunks[i],
                'chunk_length': len(self.chunks[i]),
                **self.metadata[i]
            })
        
        sample_df = pd.DataFrame(sample_data)
        sample_path = self.chunks_dir / "sample_chunks.csv"
        sample_df.to_csv(sample_path, index=False)
        print(f"Saved sample chunks to: {sample_path}")
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Generate embeddings for all chunks
        
        Returns:
            NumPy array of embeddings
        """
        print("\n" + "="*60)
        print("GENERATING EMBEDDINGS")
        print("="*60)
        
        if not self.chunks:
            raise ValueError("No chunks available. Call process_complaints() first.")
        
        print(f"Generating embeddings for {len(self.chunks)} chunks...")
        
        # Generate embeddings in batches
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(self.chunks), batch_size), 
                     desc="Generating embeddings",
                     total=len(self.chunks) // batch_size + 1):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, 
                                                          show_progress_bar=False,
                                                          convert_to_numpy=True)
            embeddings.append(batch_embeddings)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(embeddings)
        
        print(f"\nEmbeddings generated!")
        print(f"  Embedding shape: {all_embeddings.shape}")
        print(f"  Memory usage: {all_embeddings.nbytes / 1024 / 1024:.2f} MB")
        
        # Save embeddings for later use
        embeddings_path = self.output_dir / "embeddings.npy"
        np.save(embeddings_path, all_embeddings)
        print(f"Saved embeddings to: {embeddings_path}")
        
        return all_embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS vector index
        
        Args:
            embeddings: NumPy array of embeddings
        """
        print("\n" + "="*60)
        print("BUILDING FAISS INDEX")
        print("="*60)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        
        # Use Inner Product (cosine similarity) index
        # FAISS doesn't have built-in cosine similarity, so we normalize vectors
        print("Normalizing embeddings for cosine similarity...")
        faiss.normalize_L2(embeddings)
        
        # Create index
        print(f"Creating FAISS index with dimension {dimension}...")
        index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        
        # Add vectors to index
        print(f"Adding {len(embeddings)} vectors to index...")
        index.add(embeddings)
        
        # Save index
        index_path = self.output_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_path))
        
        # Save metadata
        metadata_path = self.output_dir / "faiss_metadata.json"
        metadata_to_save = []
        for meta in self.metadata:
            # Remove large text fields for metadata file
            meta_copy = meta.copy()
            meta_copy.pop('chunk_text', None)
            meta_copy.pop('original_text', None)
            metadata_to_save.append(meta_copy)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_to_save, f, indent=2)
        
        print(f"\nFAISS index built successfully!")
        print(f"  Index size: {index.ntotal} vectors")
        print(f"  Index saved to: {index_path}")
        print(f"  Metadata saved to: {metadata_path}")
        
        # Test the index
        self._test_faiss_index(index, embeddings[:5])
        
        return index
    
    def build_chromadb_store(self, embeddings: np.ndarray):
        """
        Build ChromaDB vector store
        
        Args:
            embeddings: NumPy array of embeddings
        """
        print("\n" + "="*60)
        print("BUILDING CHROMADB VECTOR STORE")
        print("="*60)
        
        # Initialize ChromaDB client
        chroma_dir = self.output_dir / "chroma_db"
        chroma_dir.mkdir(exist_ok=True)
        
        print(f"Initializing ChromaDB at: {chroma_dir}")
        client = chromadb.PersistentClient(path=str(chroma_dir))
        
        # Create or get collection
        collection_name = "complaint_chunks"
        try:
            collection = client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except:
            collection = client.create_collection(
                name=collection_name,
                metadata={"description": "Customer complaint chunks", "model": self.embedding_model_name}
            )
            print(f"Created new collection: {collection_name}")
        
        # Prepare documents and IDs
        documents = []
        metadatas = []
        ids = []
        
        print(f"Preparing {len(self.chunks)} documents for ChromaDB...")
        for i, (chunk, meta) in enumerate(tqdm(zip(self.chunks, self.metadata), 
                                               total=len(self.chunks),
                                               desc="Preparing documents")):
            # Create unique ID
            chunk_id = f"chunk_{i:08d}"
            
            # Prepare metadata
            chroma_meta = meta.copy()
            chroma_meta.pop('chunk_text', None)  # Don't duplicate
            
            documents.append(chunk)
            metadatas.append(chroma_meta)
            ids.append(chunk_id)
        
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
                embeddings=embeddings[i:batch_end].tolist()  # ChromaDB expects list of lists
            )
        
        # Get collection info
        collection_count = collection.count()
        print(f"\nChromaDB collection built successfully!")
        print(f"  Collection: {collection_name}")
        print(f"  Total documents: {collection_count}")
        print(f"  Data directory: {chroma_dir}")
        
        # Test the collection
        self._test_chromadb_collection(collection)
        
        return collection
    
    def _test_faiss_index(self, index, test_embeddings: np.ndarray):
        """Test FAISS index with sample queries"""
        print("\nTesting FAISS index...")
        
        # Normalize test embeddings
        faiss.normalize_L2(test_embeddings)
        
        # Search for similar vectors
        k = 3
        distances, indices = index.search(test_embeddings, k)
        
        print(f"Test search results (top {k} matches):")
        for i in range(min(3, len(test_embeddings))):
            print(f"\nQuery {i + 1}:")
            for j in range(k):
                idx = indices[i][j]
                dist = distances[i][j]
                if idx != -1:  # Valid index
                    product = self.metadata[idx].get('product_category', 'Unknown')
                    print(f"  Match {j + 1}: Index={idx}, Score={dist:.4f}, Product={product}")
    
    def _test_chromadb_collection(self, collection):
        """Test ChromaDB collection with sample queries"""
        print("\nTesting ChromaDB collection...")
        
        # Sample queries
        test_queries = [
            "credit card fraud",
            "loan payment issue",
            "bank account problem"
        ]
        
        # Generate embeddings for test queries
        query_embeddings = self.embedding_model.encode(test_queries)
        
        for i, (query, embedding) in enumerate(zip(test_queries, query_embeddings)):
            results = collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=2,
                include=["documents", "metadatas", "distances"]
            )
            
            print(f"\nQuery: '{query}'")
            if results['documents'] and results['documents'][0]:
                for j, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    product = metadata.get('product_category', 'Unknown')
                    print(f"  Match {j + 1}: Distance={distance:.4f}, Product={product}")
                    print(f"     Preview: {doc[:100]}...")
            else:
                print("  No results found")
    
    def save_parquet_output(self):
        """
        Save embeddings and metadata in Parquet format
        """
        print("\n" + "="*60)
        print("SAVING PARQUET OUTPUT")
        print("="*60)
        
        # Create DataFrame with embeddings and metadata
        print("Creating Parquet dataset...")
        
        data_list = []
        for i, (chunk, meta) in enumerate(zip(self.chunks, self.metadata)):
            # Create hash for embedding reference
            chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:16]
            
            record = {
                'chunk_id': f"chunk_{i:08d}",
                'chunk_text': chunk,
                'chunk_hash': chunk_hash,
                **meta
            }
            data_list.append(record)
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        
        # Save to Parquet
        parquet_path = self.output_dir / "complaint_embeddings.parquet"
        df.to_parquet(parquet_path, index=False)
        
        print(f"Saved Parquet file to: {parquet_path}")
        print(f"  Total records: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    
    def run_pipeline(self, data_path: Optional[str] = None, use_chromadb: bool = True):
        """
        Run the complete vector store building pipeline
        
        Args:
            data_path: Path to complaint data
            use_chromadb: Whether to build ChromaDB store (otherwise FAISS)
        """
        print("\n" + "="*60)
        print("VECTOR STORE BUILDING PIPELINE")
        print("="*60)
        
        # 1. Load data
        self.load_data(data_path)
        
        # 2. Create sample (if specified)
        if self.sample_size:
            data_to_process = self.create_stratified_sample()
        else:
            data_to_process = self.data
        
        # 3. Initialize components
        self.initialize_components()
        
        # 4. Process complaints (chunking)
        self.process_complaints(data_to_process)
        
        # 5. Generate embeddings
        embeddings = self.generate_embeddings()
        
        # 6. Build vector store
        if use_chromadb:
            self.build_chromadb_store(embeddings)
        else:
            self.build_faiss_index(embeddings)
        
        # 7. Save Parquet output
        self.save_parquet_output()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"  Total complaints processed: {len(data_to_process)}")
        print(f"  Total chunks created: {len(self.chunks)}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Vector store: {'ChromaDB' if use_chromadb else 'FAISS'}")
        print(f"\nOutput files saved to: {self.output_dir}/")

def main():
    """Main function for Task 2"""
    parser = argparse.ArgumentParser(description='Build vector store from complaint data')
    parser.add_argument('--data_path', type=str, 
                       default='data/processed/filtered_complaints.csv',
                       help='Path to cleaned complaint data')
    parser.add_argument('--sample_size', type=int, default=15000,
                       help='Number of complaints to sample (use 0 for all)')
    parser.add_argument('--chunk_size', type=int, default=500,
                       help='Size of text chunks')
    parser.add_argument('--chunk_overlap', type=int, default=50,
                       help='Overlap between chunks')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='Embedding model name')
    parser.add_argument('--use_faiss', action='store_true',
                       help='Use FAISS instead of ChromaDB')
    parser.add_argument('--no_sample', action='store_true',
                       help='Use all data (ignore sample_size)')
    
    args = parser.parse_args()
    
    # Adjust sample size
    if args.no_sample:
        sample_size = None
    else:
        sample_size = args.sample_size if args.sample_size > 0 else None
    
    # Initialize builder
    builder = VectorStoreBuilder(
        embedding_model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sample_size=sample_size
    )
    
    # Run pipeline
    try:
        builder.run_pipeline(
            data_path=args.data_path,
            use_chromadb=not args.use_faiss
        )
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
# app.py
"""
Task 4: Interactive Chat Interface for RAG Complaint Chatbot
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import gradio as gr
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from task3_rag_pipeline import RAGSystem

class ComplaintChatbot:
    """
    Interactive chatbot for complaint analysis
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the chatbot
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.rag_system = None
        self.chat_history = []
        self.initialize_rag_system()
        
        # Create output directories
        self.chat_logs_dir = Path("chat_logs")
        self.chat_logs_dir.mkdir(exist_ok=True)
        
        print("Complaint Chatbot initialized!")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "microsoft/DialoGPT-medium",
            "vector_store_type": "chroma",
            "retrieval_k": 5,
            "use_prebuilt": True,
            "max_chat_history": 10,
            "enable_streaming": False,
            "show_sources": True,
            "show_confidence": True
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def initialize_rag_system(self):
        """Initialize the RAG system"""
        print("Initializing RAG system...")
        
        self.rag_system = RAGSystem(
            embedding_model_name=self.config["embedding_model"],
            llm_model_name=self.config["llm_model"],
            use_prebuilt=self.config["use_prebuilt"],
            vector_store_type=self.config["vector_store_type"]
        )
        
        # Load vector store
        self.rag_system.load_prebuilt_vector_store()
        
        # Setup QA chain
        self.rag_system.setup_retrieval_qa(k=self.config["retrieval_k"])
        
        print("RAG system initialized successfully!")
    
    def process_query(self, 
                     query: str, 
                     chat_history: List[List[str]],
                     show_sources: bool = True) -> tuple:
        """
        Process a user query
        
        Args:
            query: User question
            chat_history: List of previous conversations
            show_sources: Whether to show source documents
            
        Returns:
            Tuple of (answer, chat_history, sources_html)
        """
        if not query or query.strip() == "":
            return "", chat_history, ""
        
        print(f"Processing query: {query}")
        
        try:
            # Get response from RAG system
            start_time = time.time()
            response = self.rag_system.qa_chain({"query": query})
            response_time = time.time() - start_time
            
            # Extract answer and sources
            answer = response.get("result", "I couldn't generate an answer.")
            source_docs = response.get("source_documents", [])
            
            # Format answer with confidence if enabled
            if self.config["show_confidence"] and len(source_docs) > 0:
                confidence = min(95, 70 + (len(source_docs) * 5))
                answer = f"**Answer** (confidence: {confidence}%):\n\n{answer}"
            else:
                answer = f"**Answer**:\n\n{answer}"
            
            # Add response time
            answer += f"\n\n*Generated in {response_time:.2f} seconds*"
            
            # Format sources
            sources_html = ""
            if show_sources and source_docs:
                sources_html = self._format_sources(source_docs)
            
            # Update chat history
            chat_history.append([query, answer])
            
            # Keep only recent history
            if len(chat_history) > self.config["max_chat_history"]:
                chat_history = chat_history[-self.config["max_chat_history"]:]
            
            # Log the conversation
            self._log_conversation(query, answer, source_docs, response_time)
            
            return answer, chat_history, sources_html
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            chat_history.append([query, error_msg])
            return error_msg, chat_history, ""
    
    def _format_sources(self, source_docs: List) -> str:
        """Format source documents as HTML"""
        sources_html = "<div class='sources-container'>"
        sources_html += "<h3>📚 Sources Used:</h3>"
        
        for i, doc in enumerate(source_docs[:5]):  # Show max 5 sources
            metadata = doc.metadata
            content = doc.page_content
            
            # Truncate content for display
            if len(content) > 300:
                content = content[:300] + "..."
            
            source_html = f"""
            <div class='source-card'>
                <div class='source-header'>
                    <span class='source-number'>#{i+1}</span>
                    <span class='source-product'>{metadata.get('product_category', 'Unknown')}</span>
                    <span class='source-company'>{metadata.get('company', 'Unknown')}</span>
                </div>
                <div class='source-content'>
                    {content}
                </div>
                <div class='source-meta'>
                    <span>Issue: {metadata.get('issue', 'Unknown')}</span>
                    <span>Date: {metadata.get('date_received', 'Unknown')}</span>
                </div>
            </div>
            """
            sources_html += source_html
        
        sources_html += "</div>"
        return sources_html
    
    def _log_conversation(self, query: str, answer: str, sources: List, response_time: float):
        """Log conversation to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "answer": answer[:500],  # Truncate for logging
            "num_sources": len(sources),
            "response_time": response_time,
            "sources": [
                {
                    "product": doc.metadata.get("product_category", "Unknown"),
                    "company": doc.metadata.get("company", "Unknown"),
                    "preview": doc.page_content[:100]
                }
                for doc in sources[:3]
            ]
        }
        
        # Save to daily log file
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.chat_logs_dir / f"chat_log_{date_str}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def get_suggested_questions(self) -> List[str]:
        """Get suggested questions for users"""
        return [
            "What are the main complaints about credit cards?",
            "What problems do customers face with personal loans?",
            "How do savings account complaints differ from checking accounts?",
            "What are the common issues with money transfers?",
            "Which product has the most customer complaints?",
            "What do customers say about hidden fees?",
            "How are customers affected by unauthorized transactions?",
            "What improvements do customers suggest for online banking?"
        ]
    
    def clear_chat(self) -> tuple:
        """Clear chat history"""
        self.chat_history = []
        return "", [], ""
    
    def create_interface(self):
        """Create Gradio interface"""
        print("Creating Gradio interface...")
        
        # Custom CSS for styling
        css = """
        .gradio-container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .chatbot {
            min-height: 500px;
            max-height: 600px;
            overflow-y: auto;
        }
        .sources-container {
            margin-top: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
        }
        .source-card {
            background: white;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        .source-header {
            display: flex;
            gap: 10px;
            margin-bottom: 5px;
            font-weight: bold;
        }
        .source-number {
            background: #4CAF50;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
        }
        .source-product {
            color: #2196F3;
        }
        .source-company {
            color: #FF9800;
        }
        .source-content {
            color: #666;
            font-size: 0.9em;
            margin: 5px 0;
        }
        .source-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.8em;
            color: #888;
        }
        .suggested-questions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 10px 0;
        }
        .suggested-btn {
            padding: 5px 15px;
            background: #e3f2fd;
            border: 1px solid #2196F3;
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.9em;
        }
        .suggested-btn:hover {
            background: #bbdefb;
        }
        """
        
        # Suggested questions
        suggested_questions = self.get_suggested_questions()
        
        # Create interface
        with gr.Blocks(css=css, theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🏦 CrediTrust Complaint Analysis Chatbot
            
            **Ask questions about customer complaints** across:
            - 💳 Credit Cards
            - 📝 Personal Loans
            - 💰 Savings Accounts
            - 🔄 Money Transfers
            
            *The chatbot analyzes real complaint data to provide insights and patterns.*
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        elem_classes=["chatbot"],
                        height=500
                    )
                    
                    # Query input
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask about customer complaints...",
                            scale=4
                        )
                        submit_btn = gr.Button("Submit", variant="primary", scale=1)
                    
                    # Suggested questions
                    gr.Markdown("### 💡 Suggested Questions:")
                    with gr.Row(elem_classes=["suggested-questions"]):
                        for i, question in enumerate(suggested_questions[:6]):
                            btn = gr.Button(
                                question,
                                size="sm",
                                elem_classes=["suggested-btn"]
                            )
                            btn.click(
                                lambda q=question: q,
                                outputs=[query_input]
                            )
                    
                    # Control buttons
                    with gr.Row():
                        clear_btn = gr.Button("🧹 Clear Chat", variant="secondary")
                        show_sources = gr.Checkbox(
                            label="Show Sources",
                            value=self.config["show_sources"]
                        )
                
                with gr.Column(scale=1):
                    # Sources display
                    sources_output = gr.HTML(
                        label="Retrieved Sources",
                        value="<div style='padding: 20px; text-align: center; color: #666;'>Sources will appear here...</div>"
                    )
                    
                    # Stats and info
                    with gr.Accordion("📊 Chat Info", open=False):
                        gr.Markdown(f"""
                        **Configuration:**
                        - Embedding Model: {self.config['embedding_model']}
                        - LLM: {self.config['llm_model']}
                        - Vector Store: {self.config['vector_store_type'].upper()}
                        - Retrieval: Top {self.config['retrieval_k']} documents
                        
                        **Tips:**
                        1. Be specific with your questions
                        2. Ask about particular products or issues
                        3. Use comparative questions (e.g., "compare credit card and loan complaints")
                        4. Questions about patterns and trends work well
                        """)
            
            # Event handlers
            submit_btn.click(
                fn=self.process_query,
                inputs=[query_input, chatbot, show_sources],
                outputs=[query_input, chatbot, sources_output]
            ).then(
                lambda: "",  # Clear input
                outputs=[query_input]
            )
            
            # Enter key submission
            query_input.submit(
                fn=self.process_query,
                inputs=[query_input, chatbot, show_sources],
                outputs=[query_input, chatbot, sources_output]
            ).then(
                lambda: "",
                outputs=[query_input]
            )
            
            # Clear chat
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[query_input, chatbot, sources_output]
            )
            
            # Additional suggested question buttons
            if len(suggested_questions) > 6:
                gr.Markdown("### More Questions:")
                with gr.Row(elem_classes=["suggested-questions"]):
                    for question in suggested_questions[6:]:
                        btn = gr.Button(
                            question,
                            size="sm",
                            elem_classes=["suggested-btn"]
                        )
                        btn.click(
                            lambda q=question: q,
                            outputs=[query_input]
                        )
        
        return interface

def main():
    """Main function to run the chatbot"""
    print("="*60)
    print("CREDITRUST COMPLAINT ANALYSIS CHATBOT")
    print("="*60)
    
    # Initialize chatbot
    chatbot = ComplaintChatbot()
    
    # Create interface
    interface = chatbot.create_interface()
    
    # Launch
    print("\nLaunching chatbot interface...")
    print("Open your browser and navigate to the local URL shown below")
    print("Press Ctrl+C to stop the server\n")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nChatbot server stopped.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
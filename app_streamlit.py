# app_streamlit.py
"""
Alternative Streamlit interface for the Complaint Chatbot
"""

import streamlit as st
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from task3_rag_pipeline import RAGSystem

class StreamlitChatbot:
    """Streamlit version of the complaint chatbot"""
    
    def __init__(self):
        self.initialize_session_state()
        self.initialize_rag_system()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = None
        if 'show_sources' not in st.session_state:
            st.session_state.show_sources = True
    
    def initialize_rag_system(self):
        """Initialize RAG system"""
        if st.session_state.rag_system is None:
            with st.spinner("Loading AI model and complaint data..."):
                st.session_state.rag_system = RAGSystem(
                    embedding_model_name="all-MiniLM-L6-v2",
                    llm_model_name="microsoft/DialoGPT-medium",
                    use_prebuilt=True,
                    vector_store_type="chroma"
                )
                
                # Load vector store
                st.session_state.rag_system.load_prebuilt_vector_store()
                
                # Setup QA chain
                st.session_state.rag_system.setup_retrieval_qa(k=5)
    
    def get_suggested_questions(self):
        """Get suggested questions"""
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
    
    def process_query(self, query):
        """Process a user query"""
        if not query or query.strip() == "":
            return None, None
        
        try:
            # Get response
            response = st.session_state.rag_system.qa_chain({"query": query})
            
            answer = response.get("result", "I couldn't generate an answer.")
            source_docs = response.get("source_documents", [])
            
            # Format answer
            formatted_answer = f"**Answer**:\n\n{answer}"
            
            # Format sources
            sources_html = ""
            if st.session_state.show_sources and source_docs:
                sources_html = self.format_sources(source_docs)
            
            return formatted_answer, sources_html
            
        except Exception as e:
            return f"Error: {str(e)}", ""
    
    def format_sources(self, source_docs):
        """Format source documents"""
        sources_html = "<div style='margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 10px;'>"
        sources_html += "<h4>📚 Sources Used:</h4>"
        
        for i, doc in enumerate(source_docs[:3]):  # Show max 3 sources
            metadata = doc.metadata
            content = doc.page_content
            
            if len(content) > 200:
                content = content[:200] + "..."
            
            source_html = f"""
            <div style='background: white; padding: 10px; margin: 10px 0; border-radius: 5px; border: 1px solid #dee2e6;'>
                <div style='display: flex; gap: 10px; margin-bottom: 5px; font-weight: bold;'>
                    <span style='background: #28a745; color: white; padding: 2px 8px; border-radius: 3px;'>#{i+1}</span>
                    <span style='color: #007bff;'>{metadata.get('product_category', 'Unknown')}</span>
                    <span style='color: #fd7e14;'>{metadata.get('company', 'Unknown')}</span>
                </div>
                <div style='color: #6c757d; font-size: 0.9em; margin: 5px 0;'>
                    {content}
                </div>
                <div style='display: flex; justify-content: space-between; font-size: 0.8em; color: #adb5bd;'>
                    <span>Issue: {metadata.get('issue', 'Unknown')}</span>
                    <span>Date: {metadata.get('date_received', 'Unknown')}</span>
                </div>
            </div>
            """
            sources_html += source_html
        
        sources_html += "</div>"
        return sources_html
    
    def clear_chat(self):
        """Clear chat history"""
        st.session_state.chat_history = []
        st.rerun()
    
    def render(self):
        """Render the Streamlit interface"""
        # Page configuration
        st.set_page_config(
            page_title="CrediTrust Complaint Chatbot",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 4px solid #2196F3;
        }
        .assistant-message {
            background-color: #f5f5f5;
            border-left: 4px solid #4CAF50;
        }
        .suggested-question {
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            border-radius: 20px;
            background-color: #e3f2fd;
            border: 1px solid #2196F3;
            cursor: pointer;
            display: inline-block;
        }
        .suggested-question:hover {
            background-color: #bbdefb;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.title("🏦 CrediTrust")
            st.markdown("### Complaint Analysis Chatbot")
            
            st.markdown("---")
            st.markdown("### 📊 Configuration")
            
            # Toggle sources
            st.session_state.show_sources = st.checkbox(
                "Show Sources",
                value=st.session_state.show_sources,
                help="Display the source complaint excerpts used for answers"
            )
            
            # Clear chat button
            if st.button("🧹 Clear Chat", use_container_width=True):
                self.clear_chat()
            
            st.markdown("---")
            st.markdown("### 💡 Suggested Questions")
            
            # Suggested questions
            suggested_questions = self.get_suggested_questions()
            for question in suggested_questions:
                if st.button(question, use_container_width=True):
                    st.session_state.last_question = question
            
            st.markdown("---")
            st.markdown("### ℹ️ About")
            st.markdown("""
            This chatbot analyzes customer complaints across:
            - **Credit Cards** 💳
            - **Personal Loans** 📝
            - **Savings Accounts** 💰
            - **Money Transfers** 🔄
            
            *Powered by RAG (Retrieval-Augmented Generation)*
            """)
        
        # Main content
        st.title("🏦 CrediTrust Complaint Analysis Chatbot")
        st.markdown("Ask questions about customer complaints to uncover patterns and insights.")
        
        # Chat container
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, (query, answer) in enumerate(st.session_state.chat_history):
                # User message
                with st.chat_message("user"):
                    st.markdown(f"**You:** {query}")
                
                # Assistant message
                with st.chat_message("assistant"):
                    st.markdown(answer)
        
        # Query input
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.chat_input("Ask about customer complaints...")
            
            # Check if there's a question from sidebar button
            if 'last_question' in st.session_state:
                query = st.session_state.last_question
                del st.session_state.last_question
        
        with col2:
            if st.button("Submit", use_container_width=True):
                pass  # Handled by chat_input
        
        # Process query
        if query:
            # Add user query to history and display
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(f"**You:** {query}")
            
            # Get response
            with st.spinner("Analyzing complaints..."):
                answer, sources_html = self.process_query(query)
            
            # Add to history
            st.session_state.chat_history.append((query, answer))
            
            # Display answer
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    # Display sources if enabled
                    if st.session_state.show_sources and sources_html:
                        st.markdown(sources_html, unsafe_allow_html=True)
            
            # Rerun to update display
            st.rerun()

def main():
    """Main function"""
    chatbot = StreamlitChatbot()
    chatbot.render()

if __name__ == "__main__":
    main()
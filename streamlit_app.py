import streamlit as st
from rag_core import run_rag

# Minimal background and text color via CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: midnightblue;
        color: white;
    }
    .stApp h1 {
        font-size: 40px !important; 
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Website QA using RAG", layout="centered")
st.title("🧠 Website Answer Seeking using RAG")

url = st.text_input("Enter Website URL", value="https://example.com")
question = st.text_input("Ask a Question", value="What is this website about?")

if st.button("Run RAG"):
    if not url or not question:
        st.warning("Both fields are required.")
    else:
        with st.spinner("Crawling website and generating answer..."):
            try:
                answer = run_rag(url, question)
                st.success("Answer generated:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")

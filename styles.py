import streamlit as st

def apply_styles():
    """
    Apply custom CSS styles to the Streamlit app
    """
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .stMetric {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        
        .stTable {
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        
        .stDownloadButton {
            background-color: #0031f3;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            margin-top: 1rem;
        }
        
        .stDownloadButton:hover {
            background-color: #0026c9;
        }
        
        h1 {
            color: #1e1e1e;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        h2 {
            color: #2e2e2e;
            font-weight: 600;
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

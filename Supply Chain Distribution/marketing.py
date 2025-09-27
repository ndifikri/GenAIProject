import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]

# --- Configure Streamlit Page ---
st.set_page_config(
    page_title="AI Marketing Assistant",
    page_icon="🤖"
)
st.title("AI Marketing Assistant")

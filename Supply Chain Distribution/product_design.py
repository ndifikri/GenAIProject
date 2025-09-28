import streamlit as st

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO

import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.5-flash'
    )

client = genai.Client()

def GenerateImage(prompt):
    prompt = (prompt)
    response = client.models.generate_content(
        model="gemini-2.5-flash-image-preview",
        contents=[prompt],
    )
    return response

        
def GeneratePrompt(desc_requested):
    fix_prompt = f'''You are an Master Prompter. Your task is to create a compact and solid prompt to visualize product based on given information.
    You can add some details that not mentioned in information to build a fascinating visual.
    Return only the final prompt.
    
    Product Information
    {desc_requested}'''
    image_prompt = llm.invoke(fix_prompt)
    return image_prompt.content

# --- Configure Streamlit Page ---
st.set_page_config(
    page_title="AI Product Design",
    page_icon="📍"
)
st.title("Product Design Assistant")

st.header("Assumptions")
st.markdown(f'''
-  Use Form or Free-text method for describe product that you want to design.
-  While click "submit", agent will help for prompt refinment and generate product image based on prompt.
''')
        
tab1, tab2 = st.tabs(["Design Form", "Free-Text Prompt"])

with tab1:
    title = st.text_input("Product Title", "Magical Eye Cream")
    category = st.text_input("Product Category", "Face Care - Daily Moisturizer")
    description = st.text_input("Product Description", "The Problem: Irritated and over-exfoliated skin due to aggressive skincare routines. Target: Skincare enthusiasts aged 20-35 looking for fast skin recovery. Core Benefits: 1. Repairs and strengthens the skin barrier in 7 days. 2. Ultra-soothing, non-sticky balm-to-gel texture. 3. Contains a high concentration of 3x Ceramides and Oat Extract.")
    visual = st.text_input("Product Visual Explanation", "Minimalist packaging (milky white with grey font accents). Desired texture is a soft, whipped butter.")

    desc_requested = f'''-  Product Title: {title}
-  Product Category: {category}
-  Product Description: {description}
-  Product Visual Explanation: {visual}'''
    
    if st.button("Submit Form"):
        prompt = GeneratePrompt(desc_requested)
        st.header("Prompt Refinment")
        st.markdown(prompt)

        st.header("Image Result")
        response = GenerateImage(prompt)
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                st.image(part.inline_data.data)

with tab2:
    p = '''-  Product Title: Magical Eye Cream
-  Product Category: Face Care - Daily Moisturizer
-  Product Description: The Problem: Irritated and over-exfoliated skin due to aggressive skincare routines. Target: Skincare enthusiasts aged 20-35 looking for fast skin recovery. Core Benefits: 1. Repairs and strengthens the skin barrier in 7 days. 2. Ultra-soothing, non-sticky balm-to-gel texture. 3. Contains a high concentration of 3x Ceramides and Oat Extract.
-  Product Visual Explanation: Minimalist packaging (milky white with grey font accents). Desired texture is a soft, whipped butter.'''
    txt = st.text_area("Describe the Product",p)
    if st.button("Submit Description"):
        prompt = GeneratePrompt(txt)
        st.header("Prompt Refinment")
        st.markdown(prompt)

        st.header("Image Result")
        response = GenerateImage(prompt)
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                st.image(part.inline_data.data)
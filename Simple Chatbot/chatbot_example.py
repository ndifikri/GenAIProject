import os
import streamlit as st
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# # Load API_KEY from local environment (kalau di local laptop)
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load API_KEY from streamlit cloud secret (apabila mau deploy di streamlit cloud)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

if OPENAI_API_KEY:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key = OPENAI_API_KEY
        )

def chat_ai(question, history):
    system_message = '''Kamu adalah AI Assistant yang ahli dalam bidang Data Science dan Artificial Intelligence.
    Tugas utama kamu adalah menjawab pertanyaan user terkait dengan Data Science, AI, machine learning, deep learning, dan bidang terkait lainnya.
    Jangan jawab pertanyaan diluar bidang tersebut. Selalu bersikap ramah dan seru dalam berinteraksi dengan User.
    Gunakan emoji apabila diperlukan.'''

    final_prompt = f'''**System Message**: {system_message}

    **History Chat**: {history}

    **Question**: {question}'''

    response = llm.invoke(final_prompt)
    return response

# Judul Streamlit
st.title("CHATBOT SEDERHANA")

# Mendefinisikan tempat untuk chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mendefinisikan role dan content pada setiap percakapan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan saya tentang Data Science dan AI"):
    messages_history = st.session_state.get("messages", [])[-20:] #hanya ambil 10 percakapan sebelumnya
    history = "\n\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Mendefinisikan role Human untuk chat yang ditulis oleh user
    with st.chat_message("Human"):
        st.markdown(prompt)
    # Disini mencatat pertanyaan user di chat history
    st.session_state.messages.append({"role": "Human", "content": prompt})

    # Mendefinisikan jawaban dari chatbot
    with st.chat_message("AI"):
        response = chat_ai(prompt, history)
        answer = response.content
        st.markdown(answer)
        # Disini mencatat jawaban AI di chat history
        st.session_state.messages.append({"role": "AI", "content": answer})
    
    with st.expander("History Chat"):
        st.markdown(history)
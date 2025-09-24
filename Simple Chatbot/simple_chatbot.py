import streamlit as st
from langchain_openai import ChatOpenAI

# Definisikan cara mengambil OpenAI API Key
if st.secrets == True:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = st.text_input("Enter your API Key:", type="password")

if OPENAI_API_KEY:
    # Definisikan LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY
    )

# Fungsi untuk memproses pesan pengguna dan menghasilkan respons
def get_chatbot_response(user_input, history):
    prompt_chatbot = f"""Kamu adalah AI Assistant yang ramah dan ceria. Jawab pertanyaan dengan menggunakan bahasa Indonesia yang natural dengan bahasa sehari-hari.
    Gunakan riwayat percakapan untuk menambah informasi konteks percakapan dan gunakan emoji apabila diperlukan dalam jawaban.
    
    Berikut ini riwayat percakapan:
    {history}
    
    User : {user_input}"""
    response = llm.invoke(prompt_chatbot)
    return response

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Chatbot Sederhana",
    page_icon="🤖"
)

st.title("🤖 Chatbot Sederhana")
st.write("Silakan mulai percakapan di bawah.")

# Inisialisasi riwayat obrolan di session state Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

history = st.session_state.messages[-10:]

# Tambahkan expander untuk menampilkan riwayat obrolan
with st.expander("Lihat Riwayat Obrolan"):
    # Tampilkan pesan dari riwayat obrolan pada setiap refresh aplikasi di dalam expander
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Tampilkan pesan dari riwayat obrolan pada setiap refresh aplikasi
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Tangani input dari pengguna
if prompt := st.chat_input("Apa yang ingin Anda katakan?"):
    # Tambahkan pesan pengguna ke riwayat obrolan
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Tampilkan pesan pengguna di antarmuka
    with st.chat_message("user"):
        st.markdown(prompt)

    # Dapatkan respons dari chatbot
    with st.chat_message("assistant"):
        response = get_chatbot_response(prompt, history)
        answer = response.content
        st.markdown(answer)

    # Tambahkan respons chatbot ke riwayat obrolan
    st.session_state.messages.append({"role": "assistant", "content": answer})
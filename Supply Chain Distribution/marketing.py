import streamlit as st
import os

from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
os.environ["QDRANT_API_KEY"] = os.environ.get("QDRANT_API_KEY") or st.secrets["QDRANT_API_KEY"]
os.environ["QDRANT_URL"] = os.environ.get("QDRANT_URL") or st.secrets["QDRANT_URL"]
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or st.secrets["GOOGLE_AOPENAI_API_KEYPI_KEY"]

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="cosmetics",
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)
llm = ChatGoogleGenerativeAI(
        model = 'gemini-2.5-flash'
    )

@tool
def SearchonGoogle(promptsearch):
    """Tool for get information from internet and search something with Google search."""
    client = genai.Client()

    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=promptsearch,
        config=config,
    )

    return response.text

@tool
def check_relevant_products(query: str):
  """Tool for retrieving informations about cosmetics product and It's information in database."""
  documents = qdrant.similarity_search(query, k=5)
  return documents

def get_chatbot_response(prompt, history):
    fix_prompt = f'''User: {prompt}
    History Chat: {history}'''

    agent = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[SearchonGoogle, check_relevant_products],
        prompt=SystemMessage(content="You are a Marketing Expert.Your task is optimizing marketing campaigns for a New Product Development (NPD) initiative, marketing decision-making, recommend actions, and other topics that related to marketing. "),
    )
    return agent.invoke({"messages": fix_prompt})

# --- Configure Streamlit Page ---
st.set_page_config(
    page_title="AI Marketing Assistant",
    page_icon="🤖"
)
st.title("AI Marketing Assistant")

st.header("Assumptions")
st.markdown(f'''
-  Chatbot's Persona: A powerful assistant for optimizing marketing campaigns for a New Product Development (NPD) initiative, marketing decision-making, recommend actions, and other topics that related to marketing.
-  Agents Architecture: Supervisor with Multi-agent as Tools
-  Integrated Capabilities:
      1. Adaptive for tools usage.
      2. Searching informations via Google Search.
      3. Integrated with custom dataset in vector database.
''')


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
        answer = response["messages"][-1].content
        st.markdown(answer)

    # Tambahkan respons chatbot ke riwayat obrolan
    st.session_state.messages.append({"role": "assistant", "content": answer})
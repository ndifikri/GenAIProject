import streamlit as st
import base64, time
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.callbacks import get_openai_callback
from langchain_core.messages import HumanMessage

def chat(req_content, uploaded_cv):
    file_content = uploaded_cv.read()
    file_name = uploaded_cv.name
    encoded_base64 = base64.b64encode(file_content).decode('utf-8')


    fix_prompt = f'''You are an Expert Recruiters. Your task is review candidate data wether It's match for job requirements or not with given candidate data provided.
Always answer in Indonesia language.
Here is job requirements:
{req_content}
'''

    prompt_text = HumanMessage(
        content=[
            {
                "type": "text",
                "text": fix_prompt
                },
            {
                "type": "file",
                "file": {
                    "filename": file_name,
                    "file_data": f"data:application/pdf;base64,{encoded_base64}"
                    }
                },
        ]
    )

    class ResponseFormatter(BaseModel):
        score: int = Field(description="Give score from 0 to 100 for how much this candidate suits for AI Engineer role")
        reason: str = Field(description="Give the reason about match or not the candidate with needed role")
        desc: str = Field(description="Describe the candidate's skills and capability for needed role")

    model_with_structure = llm.with_structured_output(ResponseFormatter)

    with get_openai_callback() as cb:
        structured_response = model_with_structure.invoke([prompt_text])
        completion_tokens = cb.completion_tokens
        prompt_tokens = cb.prompt_tokens
        score = structured_response.score
        reason = structured_response.reason
        desc = structured_response.desc
        price = 17_000 * (prompt_tokens*0.15 + completion_tokens*0.6)/1_000_000

    response = {
        "score" : score,
        "reason" : reason,
        "desc" : desc,
        "completion_tokens" : completion_tokens,
        "prompt_tokens" : prompt_tokens,
        "price_idr" : price
    }
    return response

def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Judul aplikasi llm
st.title("CV Assesment")

api_key = OPENAI_API_KEY
if api_key:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=api_key
    )

# Uploaded requirement
uploaded_req = st.file_uploader(
    "**Drop Job Requirements Here**", type="txt", accept_multiple_files=False
)

if uploaded_req:
    req_content = uploaded_req.read().decode("utf-8")
    with st.expander("Job Requirements Detail"):
        st.markdown(req_content)
    
    uploaded_cvs = st.file_uploader(
        "**Upload PDF CV**", type="pdf", accept_multiple_files=True
    )

    if uploaded_cvs:
        if st.button("Analyze"):
            st.write("Candidate Analysis Results:")

            result_list = []
            for uploaded_cv in uploaded_cvs:
                st.subheader(f"📘 {uploaded_cv.name}")
                result = chat(req_content, uploaded_cv)
                result['filename'] = uploaded_cv.name
                result_list.append(result)
                st.write(result)
import os
import streamlit as st
from langChain_backend import get_openai_response
import uuid

openai_api_key = st.secrets["OPENAI_API_KEY"]    
os.environ["OPENAI_API_KEY"] = openai_api_key

def format_user_input(user_input):
    return f"**You:** {user_input}"

def format_response(output_parsed):
    # output_parsed is of type Response, which has a list of Paragraphs
    formatted_paragraphs = []
    for para in output_parsed.paragraphs:
        # Format references as clickable URLs if present
        refs = ""
        if para.references:
            refs_list = [
                f'<a href="{ref}" target="_blank">{ref}</a>' for ref in para.references
            ]
            refs = "<br>".join(refs_list)
            refs = f"<br><small>References:<br>{refs}</small>"
        formatted_paragraphs.append(f"{para.content}{refs}")
    return "<br><br>".join(formatted_paragraphs)

def on_user_input_change():
    if(st.session_state.user_input.strip() == ""):
        return
    
    st.session_state.chat_history_raw.append({"role": "user", "content": st.session_state.user_input})
    st.session_state.chat_history_display.append(st.session_state.user_input)

    response = get_openai_response(
        st.session_state.chat_history_raw,
        parameters=st.session_state.input
    )

    st.session_state.chat_history_raw.append({"role": "assistant", "content": response["output_text"]})    
    st.session_state.chat_history_display.append(format_response(response["output_parsed"]))
    st.session_state.user_input = ""

def on_interview_start():
    st.session_state.chat_history_raw = []
    st.session_state.chat_history_display = []
    st.session_state.in_conversation = True
    st.session_state.user_input = ""
    st.session_state.input = {
        # "temperature": st.session_state.temperature,
        # "position": st.session_state.position,
        # "industry": st.session_state.industry,
        # "number_of_questions": st.session_state.number_of_questions,
        # "model": st.session_state.model,
        # "top_p": st.session_state.top_p,
        "thread_id": str(uuid.uuid4())
    }

def on_interview_end():
    st.session_state.in_conversation = False
    st.session_state.user_input = ""
    st.session_state.input = None

if "chat_history_raw" not in st.session_state:
    st.session_state.chat_history_raw = []

if "chat_history_display" not in st.session_state:
    st.session_state.chat_history_display = []  

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

if "in_conversation" not in st.session_state:
    st.session_state.in_conversation = False

if "input" not in st.session_state:
    on_interview_start()

st.title("Pelno mokesčio įstatymo pokalbių asistentas")

# if st.session_state.in_conversation == False:
#     st.markdown(f"**Model parameters**")
#     st.slider('Temperature', min_value=0.0, max_value=2.0, value=0.7, step=0.1, key="temperature")
#     st.slider("top_p", min_value=0.0, max_value=1.0, value=1.0, step=0.1, key="top_p")
#     st.selectbox("Select model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-3.5-turbo"], key="model")
#     st.markdown(f"**Interview parameters**")
#     st.slider('Number of questions', min_value=1, max_value=10, value=3, step=1, key="number_of_questions")
#     st.text_input("What position are you interviewing for?", key="position")
#     st.text_input("What industry/company is the job in?", key="industry")
#     st.button("Start Interview", on_click=on_interview_start)
# else:    
# st.button("Start new interview", on_click=on_interview_end)
st.markdown(f"**Asistentas:** Sveiki, kaip galiu jums padėti?")

for msg in st.session_state.chat_history_display:
    st.markdown(msg.replace("\n", "<br>"), unsafe_allow_html=True)

st.text_area("Type your message...", key="user_input", on_change=on_user_input_change)

st.markdown(
    """
    <script>
    document.querySelector('div.st-key-user_input input').focus();
    </script>
    """,
    unsafe_allow_html=True
)

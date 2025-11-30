# streamlit: title = "Pokalbių asistentas"
import os
import re
import streamlit as st
import streamlit.components.v1 as components
from ESeimasAgent import ESeimasAgent
eseimas_agent = ESeimasAgent(db_name="pm_chroma_db", law_name="Pelno mokesčio įstatymas")
import uuid
import logging

# Set up logging
logging.basicConfig(
    filename="app.log",
    level=logging.ERROR,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8"
)
import traceback
from datetime import datetime

openai_api_key = st.secrets["OPENAI_API_KEY"]    
os.environ["OPENAI_API_KEY"] = openai_api_key

def format_user_input(user_input):
    return f"<b>Jūs:</b> {user_input}"

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
            refs = f"<br><small>Šaltiniai:<br>{refs}</small>"
        formatted_paragraphs.append(f"{para.content}{refs}")

    formatted_paragraphs[0] = f"<b>Asistentas:</b> {formatted_paragraphs[0]}"

    return "<br><br>".join(formatted_paragraphs)

def on_user_input_change():
    try:
        # --- Rate limiting: max 5 requests per 60 seconds per session ---
        now = datetime.now()
        window_seconds = 60
        max_requests = 5
        if "request_timestamps" not in st.session_state:
            st.session_state.request_timestamps = []
        # Remove timestamps older than window
        st.session_state.request_timestamps = [ts for ts in st.session_state.request_timestamps if (now - ts).total_seconds() < window_seconds]
        if len(st.session_state.request_timestamps) >= max_requests:
            st.warning(f"Per daug užklausų. Bandykite po {window_seconds} sekundžių.")
            return
        st.session_state.request_timestamps.append(now)

        if(st.session_state.user_input.strip() == ""):
            return

        user_input = st.session_state.user_input
        sanitized, error = validate_and_sanitize_input(user_input)

        if error:
            st.warning(error)
            return

        st.session_state.chat_history_raw.append({"role": "user", "content": sanitized})
        st.session_state.chat_history_display.append(format_user_input(st.session_state.user_input))

        with st.spinner("Atsakymas generuojamas..."):
            response = eseimas_agent.get_agent_response(
                st.session_state.chat_history_raw[-1],
                parameters=st.session_state.input
            )
        st.session_state.chat_history_raw.append({"role": "assistant", "content": response["output_text"]})    
        st.session_state.chat_history_display.append(format_response(response["output_parsed"]))
        st.session_state.execution_trace = response["execution_trace"]
        st.session_state.token_usage = response.get("token_usage", {})
        st.session_state.user_input = ""
    except Exception as e:
        st.error("Įvyko klaida apdorojant užklausą.")
        logging.error("Exception in on_user_input_change: %s", traceback.format_exc())

def on_interview_start():
    st.session_state.chat_history_raw = []
    st.session_state.chat_history_display = []
    st.session_state.execution_trace = []
    st.session_state.in_conversation = True
    st.session_state.user_input = ""
    st.session_state.input = {
        "thread_id": str(uuid.uuid4())
    }

def on_interview_end():
    st.session_state.in_conversation = False
    st.session_state.user_input = ""
    st.session_state.input = None

def validate_and_sanitize_input(user_input, max_length=500):
    # Remove leading/trailing whitespace
    cleaned = user_input.strip()
    # Reject empty input
    if not cleaned:
        return None, "Įveskite klausimą."
    # Limit length
    if len(cleaned) > max_length:
        return None, f"Žinutė per ilga (max {max_length} simbolių)."
    # Remove HTML tags
    cleaned = re.sub(r'<.*?>', '', cleaned)
    
    return cleaned, None

def render_UI():
    # --- Token usage and cost display ---
    if "token_usage" in st.session_state and st.session_state.token_usage:
        tu = st.session_state.token_usage
        input_cost_str = "{:.6f}".format(tu.get('input_cost', 0))
        output_cost_str = "{:.6f}".format(tu.get('output_cost', 0))
        total_cost_str = "{:.6f}".format(tu.get('total_cost', 0))

        st.markdown(
            f"""
            <div style="
            background: #eaf3fc;
            border-radius: 8px;
            padding: 0.8em 1.2em;
            margin-bottom: 1em;
            border: 1px solid #b6d4f7;
            color: #174a8b;
            font-size: 0.98rem;
            ">
            <b>Naudotojo žinutė:</b> {tu.get('user_tokens', 0)} tokenų (kaina: {input_cost_str} USD)<br>
            <b>Atsakymas:</b> {tu.get('assistant_tokens', 0)} tokenų (kaina: {output_cost_str} USD)<br>
            <b>Viso:</b> {tu.get('total_tokens', 0)} tokenų | <b>Bendra kaina:</b> {total_cost_str} USD
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Small header at the top ---
    st.markdown('<div style="font-size:1.1rem;font-weight:600;padding:0.5rem 0 0.5rem 1rem;background:#f5f5f5;border-bottom:1px solid #ddd;">Pelno mokesčio įstatymo pokalbių asistentas</div>', unsafe_allow_html=True)

    # --- Scrollable chat area with fixed height, using components.html for reliable JS execution ---
    chat_html = ""
    if not st.session_state.chat_history_display:
        chat_html += "<p><i>Sveiki, aš esu pelno mokesčio įstatymo pokalbių asistentas, kaip galiu jums padėti?</i></p>"
    else:
        for msg in st.session_state.chat_history_display:
            chat_html += f'<div style="margin-bottom:1em;">{msg.replace("\n", "<br>")}</div>'

    components.html(
        f'''
        <style>
        #chat-history {{
            font-family: "Inter", "system-ui", "Segoe UI", Arial, sans-serif;
            font-size: 1rem;
            color: #262730;
        }}
        #chat-history a {{
            color: #2471c8;
            text-decoration: underline;
            word-break: break-all;
            transition: color 0.2s;
        }}
        #chat-history a:hover {{
            color: #174a8b;
        }}
        #chat-history b {{
            color: #262730;
        }}
        </style>
        <div id="chat-history" style="height:350px;overflow-y:auto;padding:1rem 1rem 0.5rem 1rem;background:#fff;border-bottom:1px solid #eee;">
            {chat_html}
        </div>
        <script>
        setTimeout(function() {{
          var chatHistory = document.getElementById('chat-history');
          if (chatHistory) {{ chatHistory.scrollTop = chatHistory.scrollHeight; }}
        }}, 100);
        </script>
        ''',
        height=370
    )

    # --- Input area at the bottom ---
    st.text_area("Type your message...", key="user_input", on_change=on_user_input_change, label_visibility="collapsed")

    # --- Execution trace scrollable area below input ---
    if "execution_trace" not in st.session_state:
        st.session_state.execution_trace = []

    trace_html = ""
    for trace in st.session_state.execution_trace:
        trace_html += f'<div style="margin-bottom:0.5em;">{trace.replace("\n", "<br>")}</div>'

    # --- Conversation export button ---
    import json
    if st.session_state.chat_history_raw:
        chat_json = json.dumps(st.session_state.chat_history_raw, ensure_ascii=False, indent=2)
        st.download_button(
            label="Atsisiųsti pokalbio istoriją (JSON)",
            data=chat_json,
            file_name="pokalbio_istorija.json",
            mime="application/json"
        )

    components.html(
        f'''
        <style>
        #trace-history {{
            font-family: "Inter", "system-ui", "Segoe UI", Arial, sans-serif;
            font-size: 0.75rem;
            color: #444;
            background: #f8f9fa;
        }}
        #trace-history code {{
            font-family: "Fira Mono", "Consolas", "monospace";
            background: #eef2f6;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 0.95em;
        }}
        </style>
        <div id="trace-history" style="height:240px;overflow-y:auto;padding:0.7rem 1rem 0.7rem 1rem;border-top:1px solid #eee;border-bottom:1px solid #eee;">{trace_html}</div>
        <script>
        setTimeout(function() {{
          var traceHistory = document.getElementById('trace-history');
          if (traceHistory) {{ traceHistory.scrollTop = traceHistory.scrollHeight; }}
        }}, 100);
        </script>
        ''',
        height=280
    )

    # --- Auto-scroll chat to bottom ---
    st.markdown(
        """
        <script>
        var chatHistory = document.getElementById('chat-history');
        if (chatHistory) { chatHistory.scrollTop = chatHistory.scrollHeight; }
        </script>
        """,
        unsafe_allow_html=True
    )


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

if "execution_trace" not in st.session_state:
    st.session_state.execution_trace = []

st.set_page_config(page_title="Pokalbių asistentas")

render_UI()

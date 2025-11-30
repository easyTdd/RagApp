import streamlit as st
import os

LOG_FILE = "app.log"

st.set_page_config(page_title="Klaidų žurnalas")
st.title("Klaidų žurnalas (Admin)")

# Simple password protection (for demo purposes)
password = st.text_input("Įveskite slaptažodį:", type="password")
if password != st.secrets.get("ADMIN_PASSWORD", "admin"):
    st.warning("Neteisingas slaptažodis.")
    st.stop()

st.success("Prisijungta kaip administratorius.")

def parse_log_entries(log_text):
    import re
    entries = []
    entry = {}
    lines = log_text.splitlines()
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    for line in lines:
        if date_pattern.match(line):
            if entry:
                entries.append(entry)
            entry = {"date": line[:19], "message": line[20:], "stack": []}
        elif line.startswith("Traceback") or line.startswith("  File"):
            entry.setdefault("stack", []).append(line)
        elif entry:
            if entry.get("stack"):
                entry["stack"].append(line)
            else:
                entry["message"] += "\n" + line
    if entry:
        entries.append(entry)
    return entries

# Read and display log entries
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log_text = f.read()
    entries = parse_log_entries(log_text)
    if entries:
        for entry in entries:
            st.markdown(f"**Data:** `{entry['date']}`")
            st.markdown(f"**Klaidos žinutė:**\n```text\n{entry['message']}\n```")
            if entry.get("stack"):
                st.markdown("**Call Stack:**\n" + "```text\n" + "\n".join(entry["stack"]) + "\n```")
            st.markdown("---")
    else:
        st.info("Klaidų žurnalas tuščias.")
else:
    st.info("Klaidų žurnalo failas nerastas.")

if st.button("Išvalyti"):
    open(LOG_FILE, "w").close()
    st.rerun()

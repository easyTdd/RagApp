# streamlit: title = "Duomenų importavimas"
import streamlit as st
from Store import Store
store = Store("pm_chroma_db")

st.set_page_config(page_title="Duomenų importavimas")
st.title("Žinių bazės atnaujinimas (Admin)")

# Simple password protection (for demo purposes)
password = st.text_input("Įveskite slaptažodį:", type="password")
if password != st.secrets.get("ADMIN_PASSWORD", "admin"):
    st.warning("Neteisingas slaptažodis.")
    st.stop()

st.success("Prisijungta kaip administratorius.")

urls = st.text_area("Įveskite dokumentų URL (po vieną eilutėje):")

if st.button("Atnaujinti žinių bazę"):
    url_list = [u.strip() for u in urls.splitlines() if u.strip()]
    if not url_list:
        st.error("Nėra URL adresų.")
    else:
        with st.spinner("Vyksta atnaujinimas..."):
            try:
                store.prefill(url_list)
                st.success("Žinių bazė sėkmingai atnaujinta!")
            except Exception as e:
                st.error(f"Klaida atnaujinant: {e}")

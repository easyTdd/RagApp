import streamlit as st
import os

# pmReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/AKYcONSsXt"
# gpmReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.171369/eWDIZvPgyS"
# pvmReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.163423/rmSIuMfMqe"
# mbReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.429530/CjeIzIuHJG"
# vsdReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.1327/EVOKGDpzti"
# psdReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.28356/bDHXHxxPqm"
# ckReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.107687/XBdbMIpvQc"

"""
https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/lRPSSghBrM
https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/sEicFwzxgj
https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/AKYcONSsXt
https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/OJIyjmFAua
"""

pmEditions = [
    "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/lRPSSghBrM",
    "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/sEicFwzxgj",
    "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/AKYcONSsXt",
    "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/OJIyjmFAua"
]

openai_api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = openai_api_key


from Store import Store

store = Store("pm_chroma_db")
store.prefill(pmEditions)




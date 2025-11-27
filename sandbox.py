from ESeimasHtmlLoader import ESeimasHtmlLoader
from pprint import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter

pmReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/AKYcONSsXt"
gpmReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.171369/eWDIZvPgyS"
pvmReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.163423/rmSIuMfMqe"
mbReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.429530/CjeIzIuHJG"
vsdReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.1327/EVOKGDpzti"
psdReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.28356/bDHXHxxPqm"
ckReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.107687/XBdbMIpvQc"

pmEditions = [
    "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/lRPSSghBrM",
    "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/sEicFwzxgj",
    "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/AKYcONSsXt",
    "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/OJIyjmFAua"
]

from rag import prefill_rag, get_embedding_model

prefill_rag(pmEditions, "pm_chroma_db")

#print(docs[1].metadata)
# with open("result.txt", "w", encoding="utf-8") as f:
#     for doc in chunks:
#         f.write(doc.page_content)
#         f.write("\n" + "-"*28 + "\n")
#         for key, value in doc.metadata.items():
#             f.write(f"{key}: {value}\n")
#         f.write("\n" + "-"*28 + "\n")


from langchain_community.vectorstores import Chroma
import streamlit as st
import os

openai_api_key = st.secrets["OPENAI_API_KEY"]    
os.environ["OPENAI_API_KEY"] = openai_api_key

# Load your vector store
vector_store = Chroma(
    persist_directory="./pm_chroma_db",
    embedding_function=get_embedding_model()  # Use the same as when writing
)

# Get all documents
result = vector_store.get()
documents = result['documents']
metadatas = result['metadatas']

seen = set()
duplicates = []
for doc, meta in zip(documents, metadatas):
    key = (meta.get("reference"), meta.get("chunk_number"))
    if key in seen:
        duplicates.append(key)
    else:
        seen.add(key)

print(f"Found {len(duplicates)} duplicate chunks.")
# if duplicates:
#     print("Duplicates:", duplicates)
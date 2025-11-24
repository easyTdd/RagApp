from ESeimasHtmlLoader import ESeimasHtmlLoader
from pprint import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = ESeimasHtmlLoader(max_depth=3)

pmReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.157066/AKYcONSsXt"
gpmReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.171369/eWDIZvPgyS"
pvmReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.163423/rmSIuMfMqe"
mbReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.429530/CjeIzIuHJG"
vsdReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.1327/EVOKGDpzti"
psdReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.28356/bDHXHxxPqm"
ckReferalUrl = "https://e-seimas.lrs.lt/portal/legalAct/lt/TAD/TAIS.107687/XBdbMIpvQc"

docs = loader.load(pmReferalUrl)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

print(f"Sukurta dokumentų: {len(chunks)}")

#print(docs[1].metadata)
with open("result.txt", "w", encoding="utf-8") as f:
    for doc in chunks:
        f.write(doc.page_content)
        f.write("\n" + "-"*28 + "\n")
        for key, value in doc.metadata.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "-"*28 + "\n")

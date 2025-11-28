from datetime import datetime
#from openai import OpenAI
import os
from pydantic import BaseModel

from pydantic import BaseModel
from typing import List, Optional

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.tools import tool
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rag import query_rag, retrieve_list_of_changes
import requests
import re
from bs4 import BeautifulSoup, Tag

checkpointer = InMemorySaver()

class Paragraph(BaseModel):
    content: str
    references: Optional[List[str]] = []

class Response(BaseModel):    
    paragraphs: List[Paragraph]

class LawChange(BaseModel):
    url: str
    description: str

class LawChanges(BaseModel):
    changes: Optional[List[LawChange]] = []

prompts = [
  {
      "content": "Tu esi asistentas aiškinantis lietuvos respublikos įstatymus. Atsakyk trumpai ir aiškiai į vartotojo užduodamus klausimus pagal pateiktą informaciją.\n\n"
  }
];

def get_openai_response(
        history,
        parameters):

    model = init_chat_model(
        f"openai:gpt-4.1",
       # temperature=parameters["temperature"],
       # top_p=parameters["top_p"]
    )

    agent = create_agent(
        model=model,
        system_prompt=prompts[-1]["content"],
        response_format=Response,
        checkpointer=checkpointer,
        tools=[retrieve_pm_context, get_current_date, retrieve_law_changes, retrieve_law_text]
    )

    result = agent.invoke(
        {"messages": [history[-1]]},
        config={"configurable": {"thread_id": parameters["thread_id"]}}
    )

    for msg in result["messages"]:
        msg.pretty_print()

#    print(result)

    raw_text = json.dumps(result['structured_response'].model_dump(), indent=2)

    return {
        "output_text": raw_text,
        "output_parsed": result['structured_response']
    }

@tool(response_format="content_and_artifact")
def retrieve_pm_context(query: str, date: str):
    """
    Ieško ir grąžina aktualią informaciją iš RAG duomenų bazės pagal užklausą. 
    RAG duomenų bazėje yra Lietuvos Respublikos Pelno mokesčio įstatymo tekstai ir susijusi informacija.
    query: Užklausos tekstas. Pateikti kuo daugiau konteksto, kad būtų galima tiksliai atsakyti.
    date: Data ISO formatu (YYYY-MM-DD), kuri nurodo, kuriuo laikotarpiu turi būti taikoma informacija.
    """
    
    retrieved_docs = query_rag(
        query, 
        date,
        "pm_chroma_db")  # get more to filter

    print(f"Retrieved {len(retrieved_docs)} documents from RAG.")

    for doc in retrieved_docs:        
        print(json.dumps({"metadata": doc.metadata}, indent=2, ensure_ascii=False))

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

@tool(response_format="content")
def get_current_date() -> str:
    """
    Grąžina dabartinę datą ISO formato eilutėje (YYYY-MM-DD).
    Naudoti šią funkciją, kai reikia žinoti dabartinę datą.
    """
    return datetime.now().date().isoformat()

@tool(response_format="content")
def retrieve_law_changes(date: str):
    """
    Grąžina įstatymo redakcijos, galiojančios nurodytą datą, pakeitimus, kurie įsigaliojo nuo tos redakcijos galiojimo pradžios.
    date: Data ISO formatu (YYYY-MM-DD), iki kurios ieškoma pakeitimų.
    """
    list_of_changes_of_current_edition = retrieve_list_of_changes(date , "pm_chroma_db")

    if not list_of_changes_of_current_edition:
        return "Nerasta jokių įstatymo pakeitimų nurodytai datai galiojančiai įstatymo redakcijai."

    serialized = json.dumps([
        {"text": change["text"], "url": change["url"]}
        for change in list_of_changes_of_current_edition
    ], ensure_ascii=False, indent=2)

    return serialized

@tool(response_format="content")
def retrieve_law_text(url: str):
    """
    Grąžina įstatymo tekstą pagal pateiktą URL.
    url: Įstatymo redakcijos URL.
    """

    match = re.search(r'documentId=([a-fA-F0-9]+)', url)
    if match:
        doc_id = match.group(1)
        url = f"https://www.e-tar.lt/rs/legalact/{doc_id}/"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://e-seimas.lrs.lt/",
        "Connection": "keep-alive",
    }
    resp = requests.get(url, timeout=30, headers=headers)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"

    soup = BeautifulSoup(resp.text, "lxml")

    root = soup.find("div", class_="WordSection1")

    if not root:
        return resp.text
    
    return root.get_text()
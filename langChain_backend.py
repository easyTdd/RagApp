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
from rag import query_rag

checkpointer = InMemorySaver()

class Paragraph(BaseModel):
    content: str
    references: Optional[List[str]] = []

class Response(BaseModel):    
    paragraphs: List[Paragraph]

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
        tools=[retrieve_pm_context, get_current_date]
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

    # Filter by effective_from <= date and (effective_to >= date or effective_to is None)
    # filtered_docs = []
    # for doc in retrieved_docs:
    #     meta = doc.metadata or {}
    #     eff_from = meta.get("effective_from")
    #     eff_to = meta.get("effective_to")
    #     if eff_from and eff_from > date:
    #         continue
    #     if eff_to and eff_to < date:
    #         continue
    #     filtered_docs.append(doc)
    # filtered_docs = filtered_docs[:5]  # limit to 5 after filtering

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
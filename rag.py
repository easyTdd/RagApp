from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import os
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

from ESeimasHtmlLoader import ESeimasHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from datetime import datetime, timedelta

def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-large")

def prefill_rag(urls: List[str], db_name: str) -> None:

    embeddings = get_embedding_model()
    chunks = retrieve_chunks(urls)

    with open("result.txt", "w", encoding="utf-8") as f:
        for doc in chunks:
            f.write(doc.page_content)
            f.write("\n" + "-"*28 + "\n")
            for key, value in doc.metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "-"*28 + "\n")

    persist_directory = f"./{db_name}"
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"ChromaDB stored at {persist_directory}")

def retrieve_chunks(urls: List[str]) -> List[Document]:
    loader = ESeimasHtmlLoader()
    docs = []

    for url in urls:
        print(f"Loading document from {url}...")
        loaded_docs = loader.load(url)
        print(f" - Loaded {len(loaded_docs)} documents.")
        docs.extend(loaded_docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []

    for doc in docs:
        chunks = text_splitter.split_documents([doc])
        total = len(chunks)
        for idx, chunk in enumerate(chunks, 1):
            if not hasattr(chunk, "metadata") or chunk.metadata is None:
                chunk.metadata = {}
            chunk.metadata["chunk_number"] = idx
            chunk.metadata["chunk_total"] = total
            chunk.metadata["is_last_chunk"] = (idx == total)
            if idx != 1:
                chunk.page_content = chunk.metadata["heararchy"] + " > " + chunk.metadata["title"] + "\n" + chunk.page_content
        all_chunks.extend(chunks)
    return all_chunks

def query_rag(query: str, date: str, db_name: str) -> List[Document]:

    date_int = int(date.replace("-", ""))
    filter = {
            "$and": [
                {"effective_from": {"$lte": date_int}},
                {"effective_to": {"$gt": date_int}},
                {"title": {"$ne": "Pakeitimai:"}}
            ]
        }

    embeddings = get_embedding_model()
    persist_directory = f"./{db_name}"

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    result = vector_store.similarity_search_with_relevance_scores(query, k=10, filter=filter)

    top_ids = resolve_top_k_doc_ids(result, k=3)

    top_docs =[]

    for id in top_ids:
        full_doc = resolve_full_document_by_reference(vector_store, id, date_int)
        if full_doc is not None:
            print(f"Resolved full document for reference {full_doc.metadata.get('title')}.")
            top_docs.append(full_doc)

    return top_docs

def resolve_top_k_doc_ids(retrieved_docs: List[tuple[Document, float]], k: int = 3) -> List[str]:
    from collections import defaultdict
    grouped = defaultdict(list)
    for doc, score in retrieved_docs:
        id = doc.metadata.get("id")
        grouped[id].append((doc, score))

    group_stats = []
    for id, items in grouped.items():
        count = len(items)
        min_score = min(score for _, score in items)
        group_stats.append((id, count, min_score, items))

    # Sort by count (desc), then by min_score (asc)
    group_stats.sort(key=lambda x: (-x[1], x[2]))

    top_groups = group_stats[:k]

    # Return only the id of top_groups
    return [id for id, _, _, _ in top_groups]

def resolve_full_document_by_reference(vector_store: Chroma, id: str, date_int: int) -> Optional[Document]:
    
    # Only filter by reference in the query
    where = {
        "$and": [
            {"id": id},
            {"effective_from": {"$lte": date_int}},            
            {"effective_to": {"$gt": date_int}}
        ]
    }

    result = vector_store.get(where=where)

    print(f"Resolving full document for reference {id}, found {len(result['documents'])} chunks.")

    return merge_chunks_to_single_document(result)

def merge_chunks_to_single_document(result: Dict[str, List[Any]]) -> Optional[Document]:
    # result: output from vector_store.get(filter=...)
    documents = result['documents']
    metadatas = result['metadatas']
    if not documents:
        return None

    # Pair and sort by chunk_number
    pairs = sorted(
        zip(documents, metadatas),
        key=lambda x: x[1].get("chunk_number", 0)
    )

    merged_content = []
    for i, (content, meta) in enumerate(pairs):
        lines = content.splitlines()
        if i == 0:
            merged_content.extend(lines)
        else:
            merged_content.extend(lines[1:] if len(lines) > 1 else [])

    content = "\n".join(merged_content)

    meta = pairs[0][1]
    selected_meta = {k: meta.get(k) for k in [
        "id", "url", "reference", "heararchy", "title", "effective_from", "effective_to"
    ]}

    return Document(page_content=content, metadata=selected_meta)

def retrieve_list_of_changes(date: str, db_name: str) -> List[dict]:
    date_int = int(date.replace("-", ""))
    filter = {
        "title": "Pakeitimai:"
    }

    embeddings = get_embedding_model()
    persist_directory = f"./{db_name}"

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    result = vector_store.get(where=filter)

    current_edition_change_docs = get_documents_by_effective_date(date_int, result)

    if not current_edition_change_docs:
        return []
    
    eff_from = current_edition_change_docs[0].metadata.get("effective_from", 0)
    date_of_previous_edition = 0

    if eff_from:
        eff_from_str = str(eff_from)
        eff_from_date = datetime.strptime(eff_from_str, "%Y%m%d")
        previous_day = eff_from_date - timedelta(days=1)
        date_of_previous_edition = int(previous_day.strftime("%Y%m%d"))

    if date_of_previous_edition == 0:
        return []
    
    previous_edition_change_docs = resolve_latest_change_list(
        get_documents_by_effective_date(date_of_previous_edition, result)
    )

    # Example of how to get the highest 'number' from previous_edition_change_docs if needed:
    last_change = sorted(
        previous_edition_change_docs, 
        key=lambda doc: doc["number"], 
        reverse=True)[0]["number"] if previous_edition_change_docs else None
    
    if last_change is None:
        return []

    return [doc for doc in resolve_latest_change_list(current_edition_change_docs) if doc["number"] > last_change]

def get_documents_by_effective_date(date_int: int, result: Dict[str, List[Any]]) -> Optional[List[Document]]:
    
    documents = result['documents']
    metadatas = result['metadatas']
    if not documents:
        return None

    pairs = zip(documents, metadatas)

    filtered_docs = []
    for content, meta in pairs:
        eff_from = meta.get("effective_from")
        eff_to = meta.get("effective_to")
        if eff_from and eff_from > date_int:
            continue
        if eff_to and eff_to < date_int:
            continue
        filtered_docs.append(Document(page_content=content, metadata=meta))

    return filtered_docs

def resolve_latest_change_list(change_docs: Optional[List[Document]]) -> List[dict]:
    
    if not change_docs:
        return []

    sorted_docs = sorted(
        change_docs,
        key=lambda doc: doc.metadata.get("chunk_number", 0),
        reverse=True
    )
    
    last_changes = "\n".join(doc.page_content for doc in sorted_docs[:2])

    pattern = re.compile(
        r'(?P<number>\d+)\.\n'
        r'(?P<text1>.+?)\n'
        r'Nr\. [^\[]+\[href="(?P<url>[^"]+)"\], [^\n]+\n'
        r'(?P<text2>.+?)(?=\n\d+\.|\Z)', re.DOTALL
    )

    results = []
    for match in pattern.finditer(last_changes):
        number = int(match.group('number'))
        url = match.group('url')
        # Combine text1 and text2, remove the [href=...] part from text1
        text1 = re.sub(r'\[href="[^"]+"\]', '', match.group('text1')).strip()
        text2 = match.group('text2').strip()
        text = f"{text1}\n{match.group(0).splitlines()[2]}\n{text2}".strip()
        # Remove any [href=...] from the text
        text = re.sub(r'\[href="[^"]+"\]', '', text)
        results.append({
            "number": number,
            "url": url,
            "text": text
        })

    return results

def resolve_ranges_of_available_editions(db_name: str) -> List[dict]:
    embeddings = get_embedding_model()
    persist_directory = f"./{db_name}"

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    filter = {
        "title": "Pakeitimai:",
        "chunk_number": 1
    }

    result = vector_store.get(where=filter)

    documents = result['documents']
    metadatas = result['metadatas']
    if not documents:
        return []

    pairs = zip(documents, metadatas)

    ranges = []

    for _, meta in pairs:
        eff_from = meta.get("effective_from")
        eff_to = meta.get("effective_to")
        ranges.append({"effective_from": eff_from, "effective_to": eff_to})

    return ranges
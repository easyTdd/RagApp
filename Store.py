from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.types import Metadata

from ESeimasHtmlLoader import ESeimasHtmlLoader
import re
from datetime import datetime, timedelta

class Store:
    def __init__(self, db_name: str):
        self.db_name = db_name
        self.persist_directory = f"./{db_name}"
        self.embeddings = self._get_embedding_model()

    def _get_embedding_model(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(model="text-embedding-3-large")

    def prefill(self, urls: List[str]) -> None:
        chunks = self._retrieve_chunks(urls)
        with open("result.txt", "w", encoding="utf-8") as f:
            for doc in chunks:
                f.write(doc.page_content)
                f.write("\n" + "-"*28 + "\n")
                for key, value in doc.metadata.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n" + "-"*28 + "\n")
        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        vectordb = Chroma(
            collection_name=f"{self.db_name}",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )
        def build_id_from_doc(doc: Document) -> str:
            m = doc.metadata
            return f'{m["reference"]}-{m["chunk_number"]}'
        texts = [d.page_content for d in chunks]
        metadatas: List[Metadata] = [d.metadata for d in chunks]
        ids = [build_id_from_doc(d) for d in chunks]
        collection = vectordb._collection
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"Prefilled RAG database '{self.db_name}' with {len(chunks)} chunks.")

    def query(self, query: str, date: str) -> List[Document]:
        date_int = int(date.replace("-", ""))
        filter = {
                "effective_from": {"$lte": date_int},
                "effective_to": {"$gte": date_int},
                "title": {"$ne": "Pakeitimai:"}
            
        }
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        result = vector_store.similarity_search_with_relevance_scores(query, k=10, filter=filter)
        top_ids = self._resolve_top_k_doc_ids(result, k=3)
        top_docs =[]
        for id in top_ids:
            full_doc = self._resolve_full_document_by_reference(vector_store, id, date_int)
            if full_doc is not None:
                print(f"Resolved full document for reference {full_doc.metadata.get('title')}.")
                top_docs.append(full_doc)
        return top_docs

    def resolve_full_document_by_article_no(self, no: str, date: str) -> Optional[Document]:
        date_int = int(date.replace("-", ""))
        no = re.sub(r'^(\d+\.\d+)\((\d+)\)$', r'\1-\2', no)
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        where = {
            "$and": [
                {"article_no": no},
                {"effective_from": {"$lte": date_int}},            
                {"effective_to": {"$gte": date_int}}
            ]
        }
        result = vector_store.get(where=where)
        print(f"Resolving full document for article no: {no}, found {len(result['documents'])} chunks.")
        if len(result['documents']) == 0:
            return None
        return self._merge_chunks_to_single_document(result)

    def retrieve_list_of_changes(self, date: str) -> List[dict]:
        date_int = int(date.replace("-", ""))
        filter = {"title": "Pakeitimai:"}
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        result = vector_store.get(where=filter)
        current_edition_change_docs = self._get_documents_by_effective_date(date_int, result)
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
        previous_edition_change_docs = self._resolve_latest_change_list(
            self._get_documents_by_effective_date(date_of_previous_edition, result)
        )
        last_change = sorted(
            previous_edition_change_docs, 
            key=lambda doc: doc["number"], 
            reverse=True)[0]["number"] if previous_edition_change_docs else None
        if last_change is None:
            return []
        return [doc for doc in self._resolve_latest_change_list(current_edition_change_docs) if doc["number"] > last_change]

    def resolve_ranges_of_available_editions(self) -> List[dict]:
        vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        filter = {
            "$and": [
                {"title": {"$eq": "Pakeitimai:"}},
                {"chunk_number": {"$eq": 1}}
            ]
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
            eff_from_str = str(eff_from)
            eff_to_str = str(eff_to)
            eff_from_fmt = f"{eff_from_str[:4]}-{eff_from_str[4:6]}-{eff_from_str[6:]}" if eff_from_str and len(eff_from_str) == 8 else None
            eff_to_fmt = f"{eff_to_str[:4]}-{eff_to_str[4:6]}-{eff_to_str[6:]}" if eff_to_str and len(eff_to_str) == 8 else None
            if eff_to == 30000000:
                title = f"Suvestinė redakcija nuo {eff_from_fmt}"
            else:
                title = f"Suvestinė redakcija nuo {eff_from_fmt} iki {eff_to_fmt}"
            ranges.append({
                "title": title,
                "effective_from": eff_from_fmt,
                "effective_to": eff_to_fmt
            })
        return ranges

    # Private methods
    def _retrieve_chunks(self, urls: List[str]) -> List[Document]:
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

    def _resolve_top_k_doc_ids(self, retrieved_docs: List[tuple[Document, float]], k: int = 3) -> List[str]:
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
        group_stats.sort(key=lambda x: (-x[1], x[2]))
        top_groups = group_stats[:k]
        return [id for id, _, _, _ in top_groups]

    def _resolve_full_document_by_reference(self, vector_store: Chroma, id: str, date_int: int) -> Optional[Document]:
        where = {
            "$and": [
                {"id": id},
                {"effective_from": {"$lte": date_int}},            
                {"effective_to": {"$gte": date_int}}
            ]
        }
        result = vector_store.get(where=where)
        print(f"Resolving full document for reference {id}, found {len(result['documents'])} chunks.")
        return self._merge_chunks_to_single_document(result)

    def _merge_chunks_to_single_document(self, result: Dict[str, List[Any]]) -> Optional[Document]:
        documents = result['documents']
        metadatas = result['metadatas']
        if not documents:
            return None
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

    def _get_documents_by_effective_date(self, date_int: int, result: Dict[str, List[Any]]) -> Optional[List[Document]]:
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

    def _resolve_latest_change_list(self, change_docs: Optional[List[Document]]) -> List[dict]:
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
            text1 = re.sub(r'\[href="[^"]+"\]', '', match.group('text1')).strip()
            text2 = match.group('text2').strip()
            text = f"{text1}\n{match.group(0).splitlines()[2]}\n{text2}".strip()
            text = re.sub(r'\[href="[^"]+"\]', '', text)
            results.append({
                "number": number,
                "url": url,
                "text": text
            })
        return results

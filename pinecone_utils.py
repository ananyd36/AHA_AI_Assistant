from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from pypdf import PdfReader
import os
from dotenv import load_dotenv

load_dotenv()

# Shared Pinecone index connection and encoding components reused across indexing and retrieval
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=256)
bm25 = BM25Encoder.default()  # Pre-trained MS-MARCO weights — no corpus fitting needed
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

_context_prompt = ChatPromptTemplate.from_template(
    "Document summary: {summary}\n\nChunk: {chunk}\n\n"
    "In 1-2 sentences, explain this chunk's role in the document. Don't say 'This chunk is about'."
)


def _load_and_split(file_path: str) -> list[Document]:
    """Load a PDF file and split into overlapping text chunks."""
    reader = PdfReader(file_path)
    pages = [Document(page_content=p.extract_text()) for p in reader.pages]
    return text_splitter.split_documents(pages)


def _generate_chunk_context(doc_summary: str, chunk: str) -> str:
    """Ask the LLM to write a 1-2 sentence context for a chunk relative to the full document."""
    return (_context_prompt | llm).invoke({"summary": doc_summary, "chunk": chunk}).content


def index_document_to_pinecone(file_path: str, file_id: int, filename: str) -> bool:
    """
    Index a PDF into Pinecone with contextual chunk augmentation and hybrid (dense + sparse) vectors.

    Each chunk is prepended with an LLM-generated context sentence so both the dense embedding
    and the BM25 sparse vector are computed over context-enriched text.
    """
    try:
        splits = _load_and_split(file_path)

        # Use first + last 3 chunks as a document proxy to avoid token limit on full text
        sample = " ".join([s.page_content for s in (splits[:3] + splits[-3:])])
        doc_summary = llm.invoke(
            f"Summarize the structure and objectives of this document in 3-4 sentences: {sample[:4000]}"
        ).content

        # Build context-augmented texts
        texts, metadatas, ids = [], [], []
        for i, split in enumerate(splits):
            context = _generate_chunk_context(doc_summary, split.page_content)
            augmented = f"Context: {context}\n\nContent: {split.page_content}"
            texts.append(augmented)
            metadatas.append({"file_id": file_id, "source_document": filename, "context_summary": context})
            ids.append(f"{file_id}_{i}")

        # Encode once in batch for efficiency
        sparse_vectors = bm25.encode_documents(texts)
        dense_vectors = embeddings.embed_documents(texts)

        # Skip chunks where BM25 produced no sparse terms (empty pages, headers, page numbers, etc.)
        vectors = [
            {"id": vid, "values": dense, "sparse_values": sparse, "metadata": {**meta, "text": text}}
            for vid, dense, sparse, meta, text in zip(ids, dense_vectors, sparse_vectors, metadatas, texts)
            if sparse.get("indices")
        ]

        # Upsert in batches of 100 to stay within Pinecone request size limits
        for i in range(0, len(vectors), 100):
            pinecone_index.upsert(vectors=vectors[i:i + 100])

        return True
    except Exception as e:
        print(f"Error indexing {filename}: {e}")
        return False


def delete_doc_from_pinecone(file_id: int) -> bool:
    """Delete all Pinecone vectors associated with a given file_id."""
    try:
        pinecone_index.delete(filter={"file_id": {"$eq": file_id}})
        return True
    except Exception as e:
        print(f"Error deleting file_id {file_id}: {e}")
        return False

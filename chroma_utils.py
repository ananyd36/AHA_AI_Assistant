'''
This document containes the functions related to chroma vector store operations, 
including loading and splitting documents, generating context for chunks, 
indexing documents into Chroma, and deleting documents from Chroma.

Here we have leverage contextual retreival techniques to enhance the relevance of retrieved chunks by 
providing context summaries for each chunk.
'''


from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")  

# Initialize text splitter and embedding function
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
embedding_function = OpenAIEmbeddings(openai_api_key=api_key)

# Initialize Chroma vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)


def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PdfReader(file_path)
        documents = []
        for page in loader.pages:
            text = page.extract_text()
            documents.append(Document(page_content=text))
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return text_splitter.split_documents(documents)



# Initialize a cheaper LLM for context generation
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=api_key)

def generate_context_for_chunk(doc_summary: str, chunk_content: str) -> str:
    """Generates a brief context string for a specific chunk."""
    prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant helping organize a curriculum for Edge AI. "
        "Document Summary: {summary}\n\n"
        "Specific Chunk: {chunk}\n\n"
        "Briefly (in 1-2 sentences) explain the context of this chunk within the document "
        "so a retriever can understand its relevance. Do not say 'This chunk is about'—just give context."
    )
    chain = prompt | llm
    response = chain.invoke({"summary": doc_summary, "chunk": chunk_content})
    return response.content

def index_document_to_chroma(file_path: str, file_id: int, filename: str) -> bool:
    try:
        raw_splits = load_and_split_document(file_path)
        
        # 1. Generate a whole-doc summary (using the first few/last few pages or LLM)
        doc_full_text = " ".join([d.page_content for d in raw_splits[:]]) # Getting all context
        doc_summary_prompt = f"Summarize the whole structure and objectives of each part of this document. Keep in mind as this would be later used to identify specific chunk of the document: {doc_full_text[:]}"
        doc_summary = llm.predict(doc_summary_prompt)

        contextualized_docs = []

        # 2. Augment each split
        for split in raw_splits:
            context = generate_context_for_chunk(doc_summary, split.page_content)
            
            # Prepend context to content for better vector search
            enhanced_content = f"Context: {context}\n\nContent: {split.page_content}"
            
            new_doc = Document(
                page_content=enhanced_content,
                metadata={
                    **split.metadata,
                    'file_id': file_id,
                    'source_document': filename,
                    'context_summary': context # Storing it as metadata too for filtering
                }
            )
            contextualized_docs.append(new_doc)

        vectorstore.add_documents(contextualized_docs)
        return True
    except Exception as e:
        print(f"Error indexing: {e}")
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")

        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")

        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False
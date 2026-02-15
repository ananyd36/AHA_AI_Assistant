from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma 
from typing import List
from langchain_core.documents import Document
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



def index_document_to_chroma(file_path: str, file_id: int, filename : str) -> bool:
    try:
        splits = load_and_split_document(file_path)

        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
            split.metadata['source_document'] = os.path.basename(filename)  
            print(f"Prepared document chunk with metadata: {split.metadata}")

        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
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
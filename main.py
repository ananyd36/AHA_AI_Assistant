from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain, classify_query, get_greeting_response
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from pinecone_utils import index_document_to_pinecone, delete_doc_from_pinecone
import os
import uuid
import logging
import shutil

from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(filename='app.log', level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to chrome-extension://<id> before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    """Run a user question through the history-aware Pinecone hybrid RAG chain."""
    session_id = query_input.session_id or str(uuid.uuid4())
    logging.info("Session: %s | Model: %s | Q: %s", session_id, query_input.model, query_input.question)

    route = classify_query(query_input.question)

    if route == "greeting":
        answer = get_greeting_response(query_input.question)
        insert_application_logs(session_id, query_input.question, answer, model=query_input.model)
        return QueryResponse(answer=answer, session_id=session_id, model=query_input.model, sources=[])

    if route == "out_of_scope":
        answer = "I am specialized in the Edge AI curriculum. I don't have enough context to answer that accurately."
        insert_application_logs(session_id, query_input.question, answer, model=query_input.model)
        return QueryResponse(answer=answer, session_id=session_id, model=query_input.model, sources=[])

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(model_name=query_input.model)
    response = rag_chain.invoke({"input": query_input.question, "chat_history": chat_history})

    answer = response.get("answer")
    context_docs = response.get("context", [])
    sources = [context_docs[0].metadata.get("source_document")] if context_docs else []
    insert_application_logs(session_id, query_input.question, answer, model=query_input.model)

    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model, sources=sources)


@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    """Accept a PDF upload, index it into Pinecone, and register it in the document store."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_id = insert_document_record(file.filename)
        success = index_document_to_pinecone(temp_path, file_id, file.filename)

        if success:
            return {"message": f"{file.filename} uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    """Return all documents registered in the document store."""
    return get_all_documents()


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    """Delete a document's vectors from Pinecone and its record from the document store."""
    if not delete_doc_from_pinecone(request.file_id):
        return {"error": f"Failed to delete file_id {request.file_id} from Pinecone."}

    if not delete_document_record(request.file_id):
        return {"error": f"Deleted from Pinecone but failed to remove DB record for file_id {request.file_id}."}

    return {"message": f"Deleted file_id {request.file_id} from Pinecone and document store."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

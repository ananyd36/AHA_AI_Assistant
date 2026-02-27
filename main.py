from urllib import response
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from langchain_utils import get_rag_chain
from db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from chroma_utils import index_document_to_chroma, delete_doc_from_chroma
import os
import uuid
import logging
import shutil


os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

app = FastAPI()


@app.get("/health")
def health_check():
    return {
        "status":"ok"
    }

@app.post("/chat", response_model = QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    question = query_input.question
    logging.info("Session ID: %s, Question: %s, Model : %s", session_id, question, query_input.model)

    chat_history = get_chat_history(session_id)
    logging.info("Chat History: %s", chat_history)
    rag_chain = get_rag_chain(model_name=query_input.model)
    response = rag_chain.invoke({
        "input" : question,
        "chat_history" : chat_history
    })

    logging.info("Answer: %s", response)
    context_documents = response.get('context', [])
    ret_sources = [doc.metadata.get('source_document') for doc in context_documents]
    final_answer = response.get('answer')
    insert_application_logs(session_id, question, final_answer, model=query_input.model)
    return QueryResponse(
        answer = final_answer,
        session_id = session_id,
        model = query_input.model,
        sources = set(ret_sources)
    )


@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf']
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")

    temp_file_path = f"temp_{file.filename}"

    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id, file.filename)

        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)




@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()



@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}





if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
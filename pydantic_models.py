from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    GPT = "gpt-3.5-turbo"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    sources : list = Field(default_factory=list)

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int
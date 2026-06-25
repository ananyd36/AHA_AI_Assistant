from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    GPT35 = "gpt-3.5-turbo"
    GPT4O_MINI = "gpt-4o-mini"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4O_MINI)

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
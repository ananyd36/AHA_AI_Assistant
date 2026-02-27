from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever

from typing import List
from langchain_core.documents import Document
import os
from chroma_utils import vectorstore

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# Add Multi-Query Expansion
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever, 
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
)

# Add reranking with Flashrank
compressor = FlashrankRerank(top_n=3, model="ms-marco-MiniLM-L-12-v2")

advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=multi_query_retriever
)

output_parser = StrOutputParser()


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

system_instruction = (
    "### ROLE\n"
    "You are the 'Edge AI Curriculum Support Specialist'. Your sole purpose is to assist teachers "
    "with the 5 specific modules, Arduino IDE setup, and Edge Impulse ESP-based library integration.\n\n"
    
    "### CONTEXT\n"
    "Teachers are often in a classroom setting with time constraints. They are dealing with "
    "physical ESP microcontrollers and the Edge Impulse web interface. You have access to "
    "the curriculum knowledge base provided in the context below.\n\n"
    
    "### TASK & ReAct METHODOLOGY\n"
    "For every user query, follow this internal process:\n"
    "1. Thought: Analyze the error or question. Is it related to the 5 modules, Arduino setup, or Edge Impulse?\n"
    "2. Reason: Identify the likely failure point (e.g., Driver, Library version, Logic error).\n"
    "3. Action: Search the provided {context} for the specific procedural fix.\n"
    "4. Response: Provide a concise, step-by-step solution for the teacher.\n\n"
    
    "### CONSTRAINTS (STRICT)\n"
    "1. SCOPE: Answer ONLY questions regarding the 5 curriculum modules, Arduino IDE, ESP hardware, and Edge Impulse.\n"
    "2. OUT-OF-SCOPE: If the question is about general Python, unrelated hardware, or topics not in the context, "
    "politely state: 'I am specialized in the Edge AI curriculum. I don't have enough context to answer that accurately.'\n"
    "3. SOURCE: Do not use external knowledge. Only use the provided context below.\n"
    "4. FORMAT: Use bolding for technical terms (e.g., **COM Port**, **Baud Rate**, **edge-impulse-daemon**).\n\n"
    
    "### KNOWLEDGE BASE (CONTEXT):\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

def get_rag_chain(model_name="gpt-3.5-turbo"):
    llm = ChatOpenAI(model=model_name)
    history_aware_retriever = create_history_aware_retriever(llm, advanced_retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain) 
    return rag_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_utils import pinecone_index, embeddings, bm25  # reuse shared instances

# Hybrid retriever: alpha=0.5 blends dense (semantic) and sparse (BM25 lexical) equally.
# top_k=5 returns the most relevant chunks directly — no reranker needed with hybrid search.
advanced_retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25,
    index=pinecone_index,
    top_k=5,
    alpha=0.5,
    text_key="text"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference prior context, "
     "formulate a standalone question that can be understood without the chat history. "
     "Do NOT answer — just reformulate if needed, otherwise return as-is."),
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
    "For every user query, follow this internal process and reason about the final answer using this approach:\n"
    "1. Thought: Analyze the error or question. Is it related to the 5 modules, Arduino setup, or Edge Impulse?\n"
    "2. Reason: Identify the likely failure point (e.g., Driver, Library version, Logic error).\n"
    "3. Action: Search the provided {context} for the specific procedural fix.\n"
    "4. Response: Only provide a concise, step-by-step solution for the teacher.\n\n"

    "### CONSTRAINTS (STRICT)\n"
    "1. SCOPE: Answer ONLY questions regarding the 5 curriculum modules, Arduino IDE, ESP hardware, and Edge Impulse.\n"
    "2. OUT-OF-SCOPE: If the question is outside this scope, respond with: "
    "'I am specialized in the Edge AI curriculum. I don't have enough context to answer that accurately.'\n"
    "3. SOURCE: Only use the provided context. Do not use external knowledge.\n"
    "4. FORMAT: Bold technical terms (e.g., **COM Port**, **Baud Rate**, **edge-impulse-daemon**).\n\n"

    "### KNOWLEDGE BASE (CONTEXT):\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])


_scope_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

_scope_prompt = ChatPromptTemplate.from_template(
    "Classify the message into exactly one of: 'rag', 'greeting', or 'out_of_scope'.\n"
    "- 'rag': question about Arduino, Arduino IDE, ESP microcontrollers, Edge Impulse, embedded systems, or Edge AI curriculum\n"
    "- 'greeting': casual greeting or social message (hi, hello, thanks, how are you, etc.)\n"
    "- 'out_of_scope': anything else\n\n"
    "Message: {question}\n\nCategory:"
)

_greeting_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a friendly assistant for an Edge AI curriculum tool used by teachers. "
     "Respond warmly and briefly to the greeting, and let them know you're here to help with "
     "the curriculum, Arduino IDE, or Edge Impulse questions."),
    ("human", "{question}")
])


def classify_query(question: str) -> str:
    """Classify the query as 'rag', 'greeting', or 'out_of_scope' to route before hitting Pinecone."""
    result = (_scope_prompt | _scope_llm).invoke({"question": question}).content.strip().lower()
    if "greeting" in result:
        return "greeting"
    if "rag" in result:
        return "rag"
    if "out_of_scope" in result or "out of scope" in result:
        return "out_of_scope"
    # Fallback: very short inputs are almost always greetings
    return "greeting" if len(question.split()) <= 4 else "out_of_scope"


def get_greeting_response(question: str) -> str:
    """Return a short friendly reply for greetings without invoking RAG."""
    return (_greeting_prompt | _scope_llm).invoke({"question": question}).content


def get_rag_chain(model_name="gpt-3.5-turbo"):
    """Build a history-aware RAG chain using Pinecone hybrid retrieval."""
    llm = ChatOpenAI(model=model_name)
    history_aware_retriever = create_history_aware_retriever(llm, advanced_retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

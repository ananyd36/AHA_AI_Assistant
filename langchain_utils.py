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
    "You are an Edge AI Curriculum Support Specialist helping teachers deliver hands-on lessons "
    "involving ESP microcontrollers, Arduino IDE, and Edge Impulse. "
    "Teachers are in fast-paced classroom environments and need answers they can act on immediately.\n\n"

    "When answering, reason through the problem internally — identify what the teacher is struggling with "
    "and why — but do NOT show your reasoning in the response. Only output the final answer.\n\n"

    "Your response must:\n"
    "- Start with a one-sentence summary of what the issue is or what needs to be done.\n"
    "- Follow with clear, numbered steps the teacher can execute right now.\n"
    "- Bold all technical terms (e.g., **COM Port**, **Board Manager**, **edge-impulse-daemon**).\n"
    "- Be complete enough that the teacher does not need to look elsewhere, but not longer than necessary.\n\n"

    "Only answer questions about the 5 curriculum modules, Arduino IDE setup, ESP hardware, and Edge Impulse. "
    "Base your answer strictly on the context provided below. "
    "If the question is outside this scope or not covered in the context, respond with: "
    "'I'm specialized in the AHA Edge AI curriculum and don't have enough context to answer that accurately.'\n\n"

    "Context:\n{context}"
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

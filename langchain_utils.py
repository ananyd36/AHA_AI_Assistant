from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import BaseOutputParser
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MultiQueryRetriever
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_utils import pinecone_index, embeddings, bm25  # reuse shared instances


class _LineListOutputParser(BaseOutputParser):
    """Splits newline-separated query variants into a list."""
    def parse(self, text: str):
        return [q.strip() for q in text.strip().split("\n") if q.strip()]


# Base hybrid retriever: top_k=15 per query variant gives enough candidates for dedup
_base_retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25,
    index=pinecone_index,
    top_k=15,
    alpha=0.5,
    text_key="text"
)

# Rewrites the query into 3 curriculum-scoped variants before retrieval
_rewrite_prompt = ChatPromptTemplate.from_template(
    "You are helping a teacher find answers in an Edge AI curriculum covering "
    "Arduino IDE, ESP microcontrollers, and Edge Impulse.\n\n"
    "Rewrite the question below into 3 alternative versions using different vocabulary "
    "or angle (one technical, one procedural, one using error/symptom language). "
    "Output exactly 3 questions, one per line, no numbering.\n\n"
    "Question: {question}"
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

# MultiQueryRetriever: generates 3 query variants, retrieves top_k=15 for each, deduplicates
advanced_retriever = MultiQueryRetriever(
    retriever=_base_retriever,
    llm_chain=_rewrite_prompt | _scope_llm | _LineListOutputParser(),
    parser_key="lines",
)

_scope_prompt = ChatPromptTemplate.from_template(
    "Classify the message below into exactly one of: 'rag', 'greeting', or 'out_of_scope'.\n\n"
    "- 'rag': any question or statement related to the Edge AI curriculum, including:\n"
    "  * Curriculum modules, lessons, or student-facing activities\n"
    "  * Named activities: AHA Adventure Land, AHA Card Game\n"
    "  * Hardware: ESP microcontrollers, Arduino boards\n"
    "  * Software: Arduino IDE, Edge Impulse, edge-impulse-daemon\n"
    "  * Setup, configuration, installation, or troubleshooting\n"
    "  * Embedded systems or edge machine learning concepts\n"
    "- 'greeting': casual social messages (hi, hello, thanks, good morning, how are you, etc.)\n"
    "- 'out_of_scope': anything not related to the curriculum or greetings\n\n"
    "Examples:\n"
    "Message: How do I install the Arduino IDE? → rag\n"
    "Message: What is AHA Adventure Land? → rag\n"
    "Message: How does the AHA Card Game activity work? → rag\n"
    "Message: My ESP32 won't connect → rag\n"
    "Message: Hello! → greeting\n"
    "Message: Thanks for your help → greeting\n"
    "Message: What is the capital of France? → out_of_scope\n\n"
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


def get_rag_chain(model_name="gpt-4o-mini"):
    """Build a history-aware RAG chain using Pinecone hybrid retrieval with multi-query expansion."""
    llm = ChatOpenAI(model=model_name)
    history_aware_retriever = create_history_aware_retriever(llm, advanced_retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

import os
import re
from operator import itemgetter

from dotenv import load_dotenv

load_dotenv()

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# On Windows the Tesseract binary usually isn't on PATH, so point pytesseract at it directly.
_TESSERACT_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.name == "nt" and os.path.exists(_TESSERACT_WIN_PATH):
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_WIN_PATH

# Initialize Vector DB and Embeddings
DB_DIR = "./chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

# Using local HuggingFace embeddings (runs on CPU for free!)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# LLM: OpenRouter (OpenAI-compatible API, proxies many models)
llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    temperature=0.2,
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# Keywords that signal a "global" question needing broad book context rather than
# a narrow local lookup (Bug 2).
GLOBAL_KEYWORDS = [
    "entire book", "whole book", "summarize the book", "summarize this book",
    "summary of the book", "overall", "all chapters", "every chapter",
    "best problem", "throughout the book", "across the book", "as a whole",
    "full book", "the whole book", "main themes", "main ideas", "in general",
    "give me an overview", "overview of the book",
]


def is_global_question(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in GLOBAL_KEYWORDS)


# Bug 1 hardening: "What is on page 4?" style questions.
# The [Source: PDF Viewer Page X] tag we inject makes the page number visible to
# the embedding model, but all-MiniLM-L6-v2 still ranks by *semantic meaning*, not
# exact digits -- "page 4" and "page 34"/"page 40"/"page 41" embed as nearly the
# same vector, so a plain top-k similarity search on a short numeric query like
# "what is on page 4?" frequently drops the actual page-4 chunk out of the top k
# entirely (verified: page 4 was missing from the top 30 hits on a real query).
# The fix: when the question names an explicit page number, skip semantic ranking
# and pull that page's chunks directly via an exact metadata filter instead.
PAGE_QUERY_RE = re.compile(r"\bpage\s*#?\s*(\d+)\b", re.IGNORECASE)


def _extract_page_numbers(question: str) -> list[int]:
    return sorted({int(n) for n in PAGE_QUERY_RE.findall(question)})


def _get_chunks_for_page(vector_db: Chroma, page_number: int) -> list[Document]:
    """Exact-match retrieval by page metadata (0-indexed in storage, 1-indexed
    to the user), bypassing embedding similarity entirely."""
    if page_number < 1:
        return []
    result = vector_db.get(where={"page": page_number - 1}, include=["documents", "metadatas"])
    return [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(result.get("documents", []), result.get("metadatas", []))
    ]


def _ocr_pdf_to_documents(file_path: str) -> list[Document]:
    """
    Bug 4 fix: Manga/Comic pages are just pictures of text, so PyMuPDFLoader's
    text-layer extraction returns nothing. Instead, render every page to an image
    with PyMuPDF and run it through Tesseract OCR to pull out the dialogue.
    """
    documents = []
    pdf = fitz.open(file_path)
    try:
        for page_index in range(len(pdf)):
            page = pdf[page_index]
            # Render at 2x zoom for sharper OCR accuracy on small manga text/SFX.
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            text = pytesseract.image_to_string(image)
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_index},
                )
            )
    finally:
        pdf.close()
    return documents


def process_and_store_document(file_path: str, book_type: str = "coding") -> str:
    """
    Processes the uploaded file based on the book_type and stores it in ChromaDB.
    """
    if book_type == "manga":
        # Bug 4: bypass the standard text loader and OCR the page images instead.
        docs = _ocr_pdf_to_documents(file_path)
    else:
        # 1. Load the PDF with PyMuPDF
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

    # 2. Chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)

    # Bug 1 fix: inject the absolute physical page number directly into the chunk's
    # page_content (not just metadata) so semantic search can "see" exact locations.
    # Applied per-chunk (not per-page) so every chunk carries its own marker even
    # when a single page gets split into multiple chunks.
    for chunk in chunks:
        page_number = chunk.metadata.get("page", 0) + 1
        chunk.page_content = f"[Source: PDF Viewer Page {page_number}]\n{chunk.page_content}"

    # 3. Store in Vector DB, isolated per book_type (coding/novel/manga) -- otherwise
    # every upload lands in the same collection and retrieval mixes chunks from
    # unrelated books, corrupting exact-page lookups (Bug 1) across categories.
    vector_db = Chroma(
        collection_name=book_type,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )
    # A fresh upload replaces whatever was previously stored for this category.
    existing_ids = vector_db.get(include=[])["ids"]
    if existing_ids:
        vector_db.delete(ids=existing_ids)
    vector_db.add_documents(chunks)
    return "Success"


def _format_history(history: list[dict]) -> str:
    """Bug 3 fix: render the conversation so far into text the prompt can use."""
    if not history:
        return "No previous conversation."
    lines = []
    for turn in history:
        role = "User" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def _format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


STANDARD_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful Interactive Study Tutor with two distinct jobs -- decide which one applies before answering:

1. META QUESTIONS ABOUT THE CONVERSATION -- if the Current Question asks about the conversation itself
   (e.g. "what did I just ask?", "what was your previous answer?", "summarize what we've discussed") or uses a
   pronoun/reference that only makes sense by looking at earlier turns (e.g. "it", "that", "the one you mentioned"),
   answer using ONLY the Conversation History below. Quote or paraphrase the actual prior message(s) -- do not just
   repeat the Current Question back as if it were the history.

2. TEXTBOOK QUESTIONS -- otherwise, the user is asking about the book. Answer ONLY using the Context below.
   Each snippet is tagged with its physical page number, e.g. "[Source: PDF Viewer Page X]" -- use that tag to
   answer exact-location questions precisely. If the Context does not contain the answer, say
   "I cannot find the answer to this in the textbook." Do not hallucinate or guess.

Conversation History (earlier turns, oldest first):
{chat_history}

Context from the textbook:
{context}

Current Question: {question}

Answer:"""
)

# Bug 2: Global questions need a different strategy than "stuff 30 chunks into a
# prompt" -- we map over batches of retrieved chunks first, then reduce the notes
# into one final answer. This keeps each LLM call within a safe context size no
# matter how large the retrieved k gets.
GLOBAL_MAP_PROMPT = ChatPromptTemplate.from_template(
    """You are analyzing one section of a larger textbook to help answer a reader's question.

Question: {question}

Section:
{context}

Write concise bullet-point notes capturing anything in this section relevant to the question.
If nothing in this section is relevant, respond with exactly: Nothing relevant."""
)

GLOBAL_REDUCE_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful Interactive Study Tutor. A reader asked a "global" question that
requires understanding of the whole textbook, not just one page.

Conversation so far:
{chat_history}

Below are notes gathered from many sections across the book:
{notes}

Combine these notes into one clear, well-organized final answer to the reader's question.

Question: {question}

Final Answer:"""
)

GLOBAL_MAP_BATCH_SIZE = 25
GLOBAL_MAX_CHUNKS = 300
# Free-tier LLM APIs (e.g. OpenRouter's ":free" models) are rate-limited and
# intermittently return "at capacity" / queue-timeout errors under bursts of
# calls. Cap how many map calls run at once instead of firing them all in
# parallel (which trips the rate limit harder) or fully sequentially (which is
# both slow and still bursts -- one call's retry backoff doesn't stop the next
# call from firing immediately after).
GLOBAL_MAP_CONCURRENCY = 3


def _answer_global_question(user_message: str, chat_history_str: str, vector_db: Chroma) -> str:
    # Dynamically expand k: pull as much of the book as exists, capped so the
    # number of map-reduce LLM calls stays bounded on very large books.
    total_chunks = vector_db._collection.count()
    k = max(1, min(total_chunks, GLOBAL_MAX_CHUNKS))
    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(user_message)

    batches = [
        docs[i:i + GLOBAL_MAP_BATCH_SIZE]
        for i in range(0, len(docs), GLOBAL_MAP_BATCH_SIZE)
    ]

    # Retry each map call with backoff, and if a batch still fails after every
    # retry, drop it (treat as "nothing relevant") instead of failing the
    # entire summary because one of ~10+ calls hit a transient capacity error.
    map_chain = (GLOBAL_MAP_PROMPT | llm | StrOutputParser()).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )
    inputs = [{"question": user_message, "context": _format_docs(batch)} for batch in batches]
    results = map_chain.batch(
        inputs,
        config={"max_concurrency": GLOBAL_MAP_CONCURRENCY},
        return_exceptions=True,
    )
    notes = [
        r for r in results
        if not isinstance(r, Exception) and "nothing relevant" not in r.strip().lower()
    ]

    reduce_chain = (GLOBAL_REDUCE_PROMPT | llm | StrOutputParser()).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )
    return reduce_chain.invoke({
        "chat_history": chat_history_str,
        "notes": "\n\n".join(notes) if notes else "No relevant notes found in the book.",
        "question": user_message,
    })


def query_rag_system(user_message: str, book_type: str = "coding", history: list | None = None) -> str:
    """
    Queries the Vector DB and generates an answer using the LLM.
    """
    if llm is None:
        return "ERROR: You must initialize an LLM in rag_engine.py first!"

    chat_history_str = _format_history(history or [])
    # Query the same isolated collection this book_type was stored under, so
    # switching tabs (coding/novel/manga) never pulls chunks from another book.
    vector_db = Chroma(
        collection_name=book_type,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    if is_global_question(user_message):
        # Bug 2: route global questions through the expanded map-reduce path.
        return _answer_global_question(user_message, chat_history_str, vector_db)

    # Bug 1 hardening: explicit page-number questions ("what is on page 4?") get
    # exact metadata-filtered chunks instead of relying on semantic similarity.
    page_numbers = _extract_page_numbers(user_message)
    if page_numbers:
        page_docs = []
        for page_number in page_numbers:
            page_docs.extend(_get_chunks_for_page(vector_db, page_number))
        if page_docs:
            direct_chain = STANDARD_PROMPT | llm | StrOutputParser()
            return direct_chain.invoke({
                "context": _format_docs(page_docs),
                "question": user_message,
                "chat_history": chat_history_str,
            })
        # No chunks stored for that exact page (e.g. page out of range) -- fall
        # through to semantic search rather than dead-ending with no answer.

    # Standard local-lookup path
    retriever = vector_db.as_retriever(search_kwargs={"k": 30})
    rag_chain = (
        {
            "context": itemgetter("question") | retriever | _format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | STANDARD_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke({"question": user_message, "chat_history": chat_history_str})

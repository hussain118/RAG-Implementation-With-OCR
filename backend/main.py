from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import shutil
import os

# Import the RAG engine functions that the interns will implement
from rag_engine import process_and_store_document, query_rag_system

app = FastAPI(title="Zenvyrolabs Task 3 API")

# Enable CORS so the frontend can hit the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    book_type: str = "coding"
    # Bug 3 fix: accept the full conversation history sent by the frontend
    history: List[ChatMessage] = []

@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...), 
    book_type: str = Form("coding")
):
    """
    Endpoint for uploading a PDF/Image. It saves the file to disk and passes the path 
    to the RAG engine for processing.
    """
    # Bug fix: sanitize the filename (os.path.basename strips any ../ path traversal)
    safe_filename = os.path.basename(file.filename or "upload.pdf")
    file_path = os.path.join(TEMP_DIR, safe_filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the document for RAG, passing the mode!
        process_and_store_document(file_path, book_type)

        return {"message": f"Successfully processed {safe_filename} as a {book_type} document!"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        # Bug fix: the temp upload was never deleted, silently filling the disk over time
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint for handling user questions.
    """
    try:
        # Pass the book_type and conversation history into the RAG engine
        history = [turn.model_dump() for turn in request.history]
        answer = query_rag_system(request.message, request.book_type, history)
        return {"answer": answer}
    except Exception as e:
        return {"error": f"Failed to query system: {str(e)}"}

# Serve the static frontend files
# This allows the API to serve index.html directly from the root URL "/"
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Cloud providers inject a dynamic PORT environment variable. We use 8000 as a fallback.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

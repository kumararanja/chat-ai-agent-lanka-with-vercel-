import replicate
import os
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pdfplumber

# --- 1. Initialize Core Components ---

# Get the Replicate API token from the environment variable
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
if not replicate_api_token:
    raise ValueError("REPLICATE_API_TOKEN environment variable not set. Please set it in your Render dashboard.")

# Create a Replicate client instance using the token
replicate_client = replicate.Client(replicate_api_token)

# Initialize the FastAPI app
app = FastAPI()

# Initialize ChromaDB (in-memory for now, can be persistent)
# This is our vector database to store the PDF content
client = chromadb.Client()
collection = client.create_collection("pdf_knowledge_base", embedding_function=SentenceTransformerEmbeddingFunction())

# --- 2. API Endpoint to Ingest PDF ---
# This endpoint allows you to upload a PDF and store its content in ChromaDB
@app.post("/ingest-pdf/")
async def ingest_pdf(file: UploadFile = File(...)):
    # Read the PDF content
    with pdfplumber.open(file.file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()

    # Simple text chunking (you can make this more sophisticated)
    chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
    
    # Add chunks to ChromaDB
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": file.filename, "chunk_id": i}],
            ids=[f"{file.filename}_{i}"]
        )
    
    return {"message": f"Successfully ingested {len(chunks)} chunks from {file.filename}"}

# --- 3. API Endpoint to Answer Questions ---
# This endpoint takes a question and returns an answer based on the PDF content
class Question(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(question: Question):
    # 1. Initialize the embedding model INSIDE the function
    # This makes the app more resilient to model availability issues
    embedding_model = replicate_client.models.get("sentence-transformers/all-MiniLM-l6-v2")

    # 2. Generate an embedding for the user's question
    # This allows us to find relevant text chunks in our database
    question_embedding = embedding_model.predict(input=question.question)

    # 3. Retrieve the most relevant text chunks from ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5  # Get top 5 most relevant chunks
    )

    # 4. Format the context for the LLM
    # We combine the retrieved chunks into a single block of text
    context = "\n\n".join(results['documents'][0])
    prompt = f"Answer the question based ONLY on the following context:\n\nContext: {context}\n\nQuestion: {question.question}\n\nAnswer:"

    # 5. Initialize and call the LLM (Gemini 2.5 Flash) on Replicate
    llm_model = replicate_client.models.get("google/gemini-2.5-flash")
    output = llm_model.predict(
        input={
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.1  # Lower temperature for more factual answers
        }
    )

    # 6. Combine the streamed output into a single string
    answer = "".join(output)
    
    return {"answer": answer}

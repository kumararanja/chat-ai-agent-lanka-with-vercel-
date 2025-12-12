import replicate
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import pdfplumber

# --- 1. Initialize Components ---
app = FastAPI()
# Use a cheap, effective embedding model on Replicate
embedding_model = replicate.models.get("sentence-transformers/all-MiniLM-l6-v2")
# Initialize ChromaDB (in-memory for now, can be persistent)
client = chromadb.Client()
collection = client.create_collection("pdf_knowledge_base", embedding_function=SentenceTransformerEmbeddingFunction())

# --- 2. API Endpoint to Ingest PDF ---
@app.post("/ingest-pdf/")
async def ingest_pdf(file: UploadFile = File(...)):
    # Read the PDF content
    with pdfplumber.open(file.file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text()

    # Chunk the text (simple example, you can make this smarter)
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
class Question(BaseModel):
    question: str

@app.post("/ask/")
async def ask_question(question: Question):
    # 1. Generate embedding for the user's question
    question_embedding = embedding_model.predict(input=question.question)

    # 2. Retrieve relevant chunks from ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=5  # Get top 5 most relevant chunks
    )

    # 3. Format the context for the LLM
    context = "\n\n".join(results['documents'][0])
    prompt = f"Answer the question based ONLY on the following context:\n\nContext: {context}\n\nQuestion: {question.question}\n\nAnswer:"

    # 4. Call the LLM on Replicate to generate the answer
    # Using a cost-effective model like Llama 3.1 8B
    llm_model = replicate.models.get("meta/llama-3.1-8b-instruct")
    output = llm_model.predict(
        input={
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.1  # Lower temp for more factual answers
        }
    )

    # 5. Combine the streamed output into a single string
    answer = "".join(output)
    
    return {"answer": answer}

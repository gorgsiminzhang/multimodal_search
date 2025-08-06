from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from modules.pdf_utils import Extractor
from modules.chunk_utils import Chunker
from modules.embedding_generator import EmbeddingGenerator
from modules.weaviate_store import WeaviateStore  # Make sure this is the path to your WeaviateStore
import uvicorn
from typing import Optional

app = FastAPI()

# Enable CORS for frontend communication (e.g., Gradio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
extractor = Extractor()
chunker = Chunker(window_size=3, stride=1)
embedding_generator = EmbeddingGenerator()
vector_store = WeaviateStore(collection_name="ChunkEmbedding")

# ✅ Use dict to persist state across reloads
app.state.embedding_ready = False

@app.post("/process")
async def process_pdf_or_query(
    file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None)
):
    global embedding_ready

    # --- Mode 1: PDF Upload ---
    if file:
        try:
            pdf_bytes = await file.read()
            result = extractor.extract(pdf_bytes, filename=file.filename)
            if "error" in result:
                return JSONResponse(content=result, status_code=500)

            markdown_text = result["markdown"]
            sentences = chunker.split_into_sentences(markdown_text)
            chunks = chunker.create_chunks(markdown_text)
            embeddings = embedding_generator.generate_embeddings(chunks)
            vector_store.clear_collection()
            vector_store.store_embeddings(chunks, embeddings, source=file.filename)
            embedding_ready = True

            return JSONResponse(content={
                "mode": "upload",
                "markdown": markdown_text,
                "chunks": chunks
            })
        except Exception as e:
            return JSONResponse(content={"error": f"❌ PDF processing error: {str(e)}"}, status_code=500)

    # --- Mode 2: Query ---
    elif query:
        if not embedding_ready:
            return JSONResponse(content={"error": "⏳ Please upload and process a PDF first."}, status_code=400)
        try:
            query_embedding = embedding_generator.generate_embeddings([query])[0]
            results = vector_store.collection.query.near_vector(query_embedding, limit=3)
            output = [f"Result {i+1}. {obj.properties['text']}" for i, obj in enumerate(results.objects)]
            return JSONResponse(content={"mode": "query", "results": output})
        except Exception as e:
            return JSONResponse(content={"error": f"❌ Query error: {str(e)}"}, status_code=500)

    # --- Invalid Input ---
    return JSONResponse(content={"error": "❌ You must provide either a file or a query."}, status_code=400)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

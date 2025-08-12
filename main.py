from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from modules.pdf_utils import Extractor
from modules.chunk_utils import Chunker
from modules.embedding_generator import EmbeddingGenerator
from modules.weaviate_store import WeaviateStore
import uvicorn
from typing import Optional
import traceback
import asyncio
import torch

app = FastAPI()

# Enable CORS for frontend communication (e.g., Gradio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend origin if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components — pick your OCR backend: "mineru" | "easyocr" | "off"
extractor = Extractor(
    use_gpu=True,
    device_id=0,           # choose which GPU; or None to not pin
    dpi=160,               # lower = faster raster if you keep any pre-render
    prefer_text=True,
    enable_tables=False,
    enable_formula=False,
    keep_pdfium_convert=False,
    ocr_backend="mineru",  # <— UPDATED: use "easyocr","mineru" or "off" if you want
    ocr_langs=("en",),     # used when ocr_backend="easyocr"
    ocr_dpi=160,           # used when ocr_backend="easyocr"
)

chunker = Chunker(window_size=3, stride=1)
embedding_generator = EmbeddingGenerator()
vector_store = WeaviateStore(collection_name="ChunkEmbedding")

# ✅ Persist state across reloads
app.state.embedding_ready = False

@app.get("/gpu")
def gpu_status():
    ok = torch.cuda.is_available()
    return {
        "gpu": bool(ok),
        "device_count": torch.cuda.device_count() if ok else 0,
        "device_name": (torch.cuda.get_device_name(0) if ok else None)
    }

@app.post("/process")
async def process_pdf_or_query(
    request: Request,
    file: Optional[UploadFile] = File(None),
    query: Optional[str] = Form(None)
):
    # --- Mode 1: PDF Upload ---
    if file:
        try:
            pdf_bytes = await file.read()

            # run heavy work off the event loop
            result = await asyncio.to_thread(extractor.extract, pdf_bytes, file.filename)
            if "error" in result:
                return JSONResponse(content=result, status_code=500)

            markdown_text = result.get("markdown", "")
            ocr_backend = result.get("ocr_backend", "mineru")

            # If OCR is OFF and there was no text layer, don't try to chunk/embed
            if ocr_backend == "off" and ("No extractable text layer" in markdown_text or not markdown_text.strip()):
                return JSONResponse(content={
                    "mode": "upload",
                    "markdown": markdown_text or "No extractable text layer (OCR is OFF).",
                    "chunks": [],
                    "ocr_backend": ocr_backend
                })

            # Proceed to chunk + embed
            chunks = chunker.create_chunks(markdown_text)
            embeddings = embedding_generator.generate_embeddings(chunks)
            vector_store.store_embeddings(chunks, embeddings, source=file.filename)

            app.state.embedding_ready = True

            return JSONResponse(content={
                "mode": "upload",
                "markdown": markdown_text,
                "chunks": chunks,
                "ocr_backend": ocr_backend
            })

        except Exception as e:
            traceback.print_exc()
            return JSONResponse(content={"error": f"❌ PDF processing error: {str(e)}"}, status_code=500)

    # --- Mode 2: Query ---
    elif query:
        if not app.state.embedding_ready:
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

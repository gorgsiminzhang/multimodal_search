from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from modules.pdf_utils import Extractor
from modules.chunk_utils import Chunker
from modules.embedding_generator import EmbeddingGenerator
import uvicorn

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

@app.post("/extract-pdf-text")
async def extract_pdf_text(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()

        #Step1: Extract PDF
        result = extractor.extract(pdf_bytes, filename=file.filename)
        if "error" in result:
            return JSONResponse(content=result, status_code=500)
        markdown_text = result["markdown"]
        sentences = chunker.split_into_sentences(markdown_text)
        print("The length of sentences is", len(sentences))
        
        #Step2: Split sentence-based window sliding Chunks
        chunks = chunker.create_chunks(markdown_text)
        print("The Chunk number is",len(chunks))
        
        #Step3: Generate Embeddings
        embeddings = embedding_generator.generate_embeddings(chunks)
        # ✅ Print first 3 chunks with embeddings
        for i, (chunk, emb) in enumerate(zip(chunks[:3], embeddings[:3])):
            print(f"\n🔹 Chunk {i+1}:\n{chunk}")
            print(f"🔸 Embedding {i+1} (first 5 dims): {emb[:5]}... (dim = {len(emb)})")
            
        
            
        return JSONResponse(content={
            "markdown": markdown_text,
            "chunks": chunks,
            "embeddings": embeddings
        })

    except Exception as e:
        return JSONResponse(content={"error": f"❌ Extraction error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

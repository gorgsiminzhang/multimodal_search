# 📄 Multimodal Search: PDF Embeddings with Weaviate and Gradio

This project enables you to **upload PDFs**, extract their content, generate **chunked embeddings**, store them in **Weaviate**, and perform **semantic search** over the stored content — all through an intuitive Gradio interface.

---

## 🚀 Features

- ✅ Upload any PDF file
- ✅ Extract clean Markdown text (minerU)
- ✅ Chunk text using a sliding window (sentence-based) 
- ✅ Generate OpenAI (or custom) embeddings for each chunk
- ✅ Store and retrieve embeddings using Weaviate vector DB
- ✅ Perform semantic search with natural language queries

---

## 🖼️ Architecture Overview

PDF ──▶ Extractor ──▶ Chunker ──▶ Embeddings ──▶ Weaviate
▲ │
└──── Query <────┘

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multimodal_search.git
cd multimodal_search
```

### 2. Create a Virtual Environment and Install Dependencies
```bash
conda create -n multimodal_search_env python=3.10
conda activate multimodal_search_env
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
1) Setup openAI embedding api key
2) Create Weaviate Cluster, get REST Endpoint URL
3) Create Weaviate Admin api 

Add all three keys and url above in .env file

### 4. Run the Backend Server
```bash
python main.py
```

### 5. Launch the Gradio UI
```bash
python frontend.py
```
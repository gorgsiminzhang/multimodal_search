import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/process"

def upload_pdf_and_show_chunks(file):
    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f, "application/pdf")}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                if result.get("mode") == "upload":
                    markdown = result.get("markdown", "")
                    chunks = result.get("chunks", [])
                    chunk_preview = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks[:5])])
                    return markdown.strip(), chunk_preview.strip(), gr.update(visible=True)
            return f"❌ Error: {response.status_code}", "", gr.update(visible=False)
    except Exception as e:
        return f"❌ Exception: {e}", "", gr.update(visible=False)


def query_chunks(query):
    try:
        response = requests.post(API_URL, data={"query": query})
        if response.status_code == 200:
            result = response.json()
            return "\n\n".join(result["results"])
        else:
            return response.json().get("error", f"❌ Error: {response.status_code}")
    except Exception as e:
        return f"❌ Exception: {e}"


with gr.Blocks() as demo:
    gr.Markdown("## 📄 PDF to Embeddings + 🔍 Query")

    with gr.Row():
        file_input = gr.File(label="Upload your PDF")

    submit_btn = gr.Button("Process PDF")

    with gr.Accordion("📝 Markdown Output", open=True):
        markdown_output = gr.Textbox(lines=20, interactive=False)

    with gr.Accordion("🧩 Chunk Preview (First 5)", open=False):
        chunk_output = gr.Textbox(lines=15, interactive=False)

    # Query UI
    with gr.Group(visible=False) as query_block:
        gr.Markdown("## 🔍 Query Stored Embeddings")
        query_input = gr.Textbox(placeholder="Enter your search query", label="Query")
        query_submit_btn = gr.Button("Search")
        query_output = gr.Textbox(lines=4, interactive=False, label="Result")

    # Events
    submit_btn.click(
        fn=upload_pdf_and_show_chunks,
        inputs=[file_input],
        outputs=[markdown_output, chunk_output, query_block]
    )

    query_submit_btn.click(
        fn=query_chunks,
        inputs=[query_input],
        outputs=[query_output]
    )

demo.launch()

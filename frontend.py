import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/extract-pdf-text"

def upload_pdf_and_show_chunks(file):
    try:
        with open(file.name, "rb") as f:
            files = {"file": (file.name, f, "application/pdf")}
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                markdown = result.get("markdown", "")
                chunks = result.get("chunks", [])
                return markdown, "\n\n".join(chunks)
            else:
                return f"❌ Error: {response.status_code}", ""
    except Exception as e:
        return f"❌ Exception: {e}", ""

iface = gr.Interface(
    fn=upload_pdf_and_show_chunks,
    inputs=gr.File(label="Upload PDF"),
    outputs=[
        gr.Textbox(label="Markdown Output", lines=20),
        gr.Textbox(label="Chunks", lines=20)
    ],
    title="PDF Processor with Chunking"
)

if __name__ == "__main__":
    iface.launch()

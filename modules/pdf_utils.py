from typing import Literal, Optional, Tuple
import os, tempfile
from pathlib import Path
from functools import lru_cache
from io import BytesIO

# MinerU
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
from mineru.backend.pipeline.pipeline_analyze import doc_analyze
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
from mineru.utils.enum_class import MakeMode
from mineru.data.data_reader_writer import FileBasedDataWriter

# GPU env

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("ORT_CUDA_DEVICE_ID", "0")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "2")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    import torch
except Exception:
    torch = None

# Fallback OCR deps
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import easyocr
except Exception:
    easyocr = None

import re
import numpy as np


@lru_cache(maxsize=4)
def _cached_easyocr_reader(langs: Tuple[str, ...], use_gpu: bool):
    if easyocr is None:
        raise RuntimeError("easyocr is not installed. pip install easyocr")
    return easyocr.Reader(list(langs), gpu=use_gpu)


class Extractor:
    def __init__(
        self,
        use_gpu: bool = True,
        device_id: Optional[int] = 0,
        dpi: int = 180,
        prefer_text: bool = True,
        enable_tables: bool = False,
        enable_formula: bool = False,
        keep_pdfium_convert: bool = False,
        # NEW:
        ocr_backend: Literal["mineru", "easyocr", "off"] = "mineru",
        # Back-compat shim (optional): map old flag if user still passes it
        force_skip_ocr: Optional[bool] = None,
        # EasyOCR extras
        ocr_langs: Tuple[str, ...] = ("en",),
        ocr_dpi: int = 160,
        ocr_grayscale: bool = True,
    ):
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.dpi = dpi
        self.prefer_text = prefer_text
        self.enable_tables = enable_tables
        self.enable_formula = enable_formula
        self.keep_pdfium_convert = keep_pdfium_convert
        self.ocr_langs = ocr_langs
        self.ocr_dpi = ocr_dpi
        self.ocr_grayscale = ocr_grayscale

        # Map legacy flag → new backend
        if force_skip_ocr is not None:
            ocr_backend = "easyocr" if force_skip_ocr else "mineru"
        if ocr_backend not in ("mineru", "easyocr", "off"):
            raise ValueError("ocr_backend must be 'mineru', 'easyocr', or 'off'")
        self.ocr_backend = ocr_backend

        # GPU setup
        if self.use_gpu and torch is not None:
            if self.device_id is not None:
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(self.device_id))
            self.gpu_ok = torch.cuda.is_available()
            if self.gpu_ok:
                try: torch.backends.cudnn.benchmark = True
                except Exception: pass
                try: torch.set_float32_matmul_precision("medium")
                except Exception: pass
                print(f"✅ MinerU Extractor: CUDA available on {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ MinerU Extractor: CUDA not available, using CPU.")
        else:
            self.gpu_ok = False
            print("ℹ️ GPU usage disabled or PyTorch not installed; running on CPU.")

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("MINERU_RENDER_DPI", str(self.dpi))

    # ---- helpers ----
    def _is_listy(self, s: str) -> bool:
        return bool(re.match(r'^\s*(?:[-•*·]|[0-9]+[.)])\s+', s))

    def _reflow_from_boxes(self, items, y_tol=1.2):
        """
        items: list of (bbox, text, conf) from EasyOCR (detail=1, paragraph=False)
        Groups lines by vertical distance; merges lines into paragraphs, keeps bullets as separate lines.
        """
        rows = []
        for box, txt, _ in items:
            txt = (txt or '').strip()
            if not txt:
                continue
            ys = [p[1] for p in box]; xs = [p[0] for p in box]
            y = float(np.mean(ys)); x = float(min(xs))
            h = float(max(ys) - min(ys) + 1e-6)
            rows.append((y, x, h, txt))
        rows.sort(key=lambda r: (r[0], r[1]))

        paras, cur, prev = [], [], None
        for y, x, h, txt in rows:
            if prev is None:
                cur.append(txt); prev = (y, h); continue
            y_gap = abs(y - prev[0])
            new_para = y_gap > y_tol * max(h, prev[1])
            if new_para or (self._is_listy(txt)):
                if cur: paras.append(' '.join(cur)); cur = []
                paras.append(txt)  # bullet or new paragraph starts here
            else:
                if cur and re.search(r'[\-–]$', cur[-1]):      # remem-\nber -> remember
                    cur[-1] = re.sub(r'[\-–]$', '', cur[-1]) + txt
                else:
                    cur.append(txt)
            prev = (y, h)
        if cur: paras.append(' '.join(cur))
        return paras  # list of paragraph strings (bullets included)

    def _normalize_ocr_text(self, text: str) -> str:
        text = text.replace('\r', '')
        text = re.sub(r'(\w+)[\-–]\n(\w+)', r'\1\2', text)     # hyphen wraps
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = re.sub(r'\n[ \t]+', '\n', text)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)           # single NL inside para -> space
        text = re.sub(r' \s*([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])([^\s\n])', r'\1 \2', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        return text.strip()

    def _build_mineru_like_markdown(self, page_paras: list[list[str]]) -> str:
        # page_paras: [[para_or_bullet, ...], ...] per page
        parts = []
        for i, paras in enumerate(page_paras, 1):
            parts.append(f"## Page {i}\n")
            for p in paras:
                if self._is_listy(p):
                    parts.append(p)        # already has bullet/number
                else:
                    parts.append(p)
                parts.append("")           # blank line between blocks
        md = "\n".join(parts).strip()
        return self._normalize_ocr_text(md)

    def _build_middle_json_like(self, page_paras: list[list[str]], filename: str):
        # Minimal “MinerU-like” structure so your app gets the same keys
        pages = []
        for i, paras in enumerate(page_paras, 1):
            blocks = []
            for p in paras:
                blocks.append({
                    "type": "list_item" if self._is_listy(p) else "paragraph",
                    "text": p
                })
            pages.append({"page_no": i, "blocks": blocks})
        return {"pdf_info": {"source": filename, "page_count": len(pages), "pages": pages}}

    def _easyocr_ocr(self, pdf_bytes: bytes):
        reader = _cached_easyocr_reader(tuple(self.ocr_langs), self.gpu_ok and self.use_gpu)
        all_pages = []
        for _, img in self._pdf_pages_to_np_images(pdf_bytes, self.ocr_dpi, self.ocr_grayscale):
            items = reader.readtext(img, detail=1, paragraph=False)  # (bbox, text, conf)
            paras = self._reflow_from_boxes(items)                   # list[str]
            all_pages.append(paras)
        md = self._build_mineru_like_markdown(all_pages)
        mid = self._build_middle_json_like(all_pages, filename="uploaded.pdf")
        return md, mid
    
    def _pdf_pages_to_np_images(self, pdf_bytes: bytes, dpi: int = 160, grayscale: bool = True):
        """
        Render each PDF page to a numpy image for OCR.
        Yields: (page_no, np.ndarray) where img is HxW (grayscale) or HxWx3 (RGB).
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF (pymupdf) is required. pip install pymupdf")

        zoom = dpi / 72.0
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for i, page in enumerate(doc, 1):
                mat = fitz.Matrix(zoom, zoom)
                cs = fitz.csGRAY if grayscale else fitz.csRGB
                pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=cs)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                # squeeze to 2D for grayscale
                if img.ndim == 3 and img.shape[2] == 1:
                    img = img[:, :, 0]
                yield i, img
   
    def _fast_txt_extract(self, pdf_bytes: bytes) -> str:
        """Return markdown from the PDF's existing text layer; '' if none."""
        if fitz is None:
            return ""
        parts = []
        nonempty_pages = 0
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                for i, page in enumerate(doc, 1):
                    # try 'text' first
                    txt = (page.get_text("text") or "").strip()
                    if not txt:
                        # fallback: combine text blocks if any
                        blocks = page.get_text("blocks") or []
                        txt = "\n".join(
                            b[4].strip() for b in blocks
                            if len(b) > 4 and b[4] and b[4].strip()
                        ).strip()
                    if txt:
                        nonempty_pages += 1
                        parts.append(f"## Page {i}\n\n{txt}")
            # if we found NO text at all, signal “no text layer” by returning ""
            return ("\n\n".join(parts).strip()) if nonempty_pages else ""
        except Exception:
            return ""


    # ---- main ----
    def extract(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> dict:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                local_image_dir = os.path.join(temp_dir, "images")
                local_md_dir = os.path.join(temp_dir, "md")
                os.makedirs(local_image_dir, exist_ok=True)
                os.makedirs(local_md_dir, exist_ok=True)

                image_writer = FileBasedDataWriter(local_image_dir)
                md_writer = FileBasedDataWriter(local_md_dir)

                if self.keep_pdfium_convert:
                    pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)

                # --- Backend selector ---
                if self.ocr_backend == "off":
                    md_str = self._fast_txt_extract(pdf_bytes)
                    return {
                        "markdown": md_str or "No extractable text layer (OCR is OFF).",
                        "middle_json": None,
                        "ocr_backend": "off",
                        "source": filename,
                    }

                if self.ocr_backend == "easyocr":
                    # try built-in text first
                    md_str = self._fast_txt_extract(pdf_bytes)
                    middle_json = None

                    if not md_str:
                        if easyocr is None or fitz is None:
                            md_str = "EasyOCR/PyMuPDF not installed; cannot OCR."
                        else:
                            md_res = self._easyocr_ocr(pdf_bytes)
                            # ⬇️ ensure markdown is a string, even if _easyocr_ocr returns (md, mid)
                            if isinstance(md_res, tuple):
                                md_str, middle_json = md_res
                            else:
                                md_str = md_res
                                middle_json = None

                    return {
                        "markdown": md_str or "No text recognized by EasyOCR.",
                        "middle_json": middle_json,
                        "ocr_backend": "easyocr",
                        "source": filename,
                    }




                # --- mineru (default): auto uses text layer; OCR only when needed ---
                p_lang_list = ["en"]
                results, image_lists, pdf_docs, lang_list, ocr_flags = doc_analyze(
                    [pdf_bytes],
                    p_lang_list,
                    parse_method=("auto" if self.prefer_text else "ocr"),
                    formula_enable=self.enable_formula,
                    table_enable=self.enable_tables,
                )

                middle_json = result_to_middle_json(
                    results[0], image_lists[0], pdf_docs[0], image_writer,
                    lang_list[0], ocr_flags[0], formula_enabled=self.enable_formula
                )

                md_str = union_make(
                    middle_json["pdf_info"],
                    MakeMode.MM_MD,
                    os.path.basename(local_image_dir)
                )

                return {
                    "markdown": (md_str or "").strip() or "No content extracted.",
                    "middle_json": middle_json,
                    "ocr_backend": "mineru",
                    "source": filename,
                }

        except Exception as e:
            return {"error": f"❌ MinerU error: {e}"} 
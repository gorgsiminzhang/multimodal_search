import os
import tempfile
from pathlib import Path
import json
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
from mineru.backend.pipeline.pipeline_analyze import doc_analyze
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
from mineru.utils.enum_class import MakeMode
from mineru.data.data_reader_writer import FileBasedDataWriter


class Extractor:
    def __init__(self):
        pass  # Add config options here if needed later

    def extract(self, pdf_bytes: bytes, filename: str = "uploaded.pdf") -> dict:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_stem = Path(filename).stem
                local_image_dir = os.path.join(temp_dir, "images")
                local_md_dir = os.path.join(temp_dir, "md")
                os.makedirs(local_image_dir, exist_ok=True)
                os.makedirs(local_md_dir, exist_ok=True)

                image_writer = FileBasedDataWriter(local_image_dir)
                md_writer = FileBasedDataWriter(local_md_dir)

                pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, 0, None)

                p_lang_list = ["en", "ch"]
                results, image_lists, pdf_docs, lang_list, ocr_flags = doc_analyze(
                    [pdf_bytes], p_lang_list, parse_method="auto", formula_enable=True, table_enable=True
                )

                middle_json = result_to_middle_json(
                    results[0], image_lists[0], pdf_docs[0], image_writer,
                    lang_list[0], ocr_flags[0], formula_enabled=True
                )

                md_str = union_make(middle_json["pdf_info"], MakeMode.MM_MD, os.path.basename(local_image_dir))
                return {
                    "markdown": md_str.strip() or "No content extracted.",
                    "middle_json": middle_json
                }

        except Exception as e:
            return {"error": f"❌ MinerU error: {e}"}

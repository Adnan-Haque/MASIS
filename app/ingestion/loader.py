import io
import json
import logging
from PyPDF2 import PdfReader
from docx import Document
from app.ingestion.vision_processor import process_image_with_vision

logger = logging.getLogger(__name__)

MAX_CHARS = 1500  # safe embedding size


def split_large_text(text, chunk_type="text"):
    """
    Generic fallback splitter for large text blobs.
    Splits by paragraph first, then size.
    """
    paragraphs = text.split("\n")
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) < MAX_CHARS:
            buffer += para + "\n"
        else:
            yield {
                "chunk_type": chunk_type,
                "text": buffer.strip(),
                "structured_data": None,
            }
            buffer = para + "\n"

    if buffer.strip():
        yield {
            "chunk_type": chunk_type,
            "text": buffer.strip(),
            "structured_data": None,
        }


def extract_text_stream(filename, file_bytes):

    filename_lower = filename.lower()

    # ================= IMAGE =================
    if filename_lower.endswith((".png", ".jpg", ".jpeg")):
        try:
            summary, structured = process_image_with_vision(file_bytes)

            yield {
                "chunk_type": structured.get("type", "image_text"),
                "text": summary,
                "structured_data": structured,
            }
        except Exception as e:
            logger.error(f"Vision failed for image {filename}: {str(e)}")
        return

    # ================= PDF =================
    if filename_lower.endswith(".pdf"):
        try:
            reader = PdfReader(io.BytesIO(file_bytes))

            for page_index, page in enumerate(reader.pages):
                text = page.extract_text()

                if text and text.strip():
                    for chunk in split_large_text(text, "text"):
                        chunk["page_number"] = page_index
                        yield chunk

        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
        return

    # ================= DOCX =================
    if filename_lower.endswith(".docx"):
        try:
            doc = Document(io.BytesIO(file_bytes))

            for para in doc.paragraphs:
                if para.text.strip():
                    yield {
                        "chunk_type": "text",
                        "text": para.text,
                        "structured_data": None,
                    }

            for table_index, table in enumerate(doc.tables):
                for row_index, row in enumerate(table.rows):
                    row_text = " | ".join(cell.text for cell in row.cells)

                    if row_text.strip():
                        yield {
                            "chunk_type": "table_row",
                            "text": row_text,
                            "structured_data": None,
                            "table_index": table_index,
                            "row_index": row_index,
                        }

        except Exception as e:
            logger.error(f"DOCX extraction failed: {str(e)}")
        return

    # ================= JSON =================
    if filename_lower.endswith(".json"):
        try:
            text = file_bytes.decode("utf-8")
            data = json.loads(text)

            # If top-level is list → split per item
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    yield {
                        "chunk_type": "json_item",
                        "text": json.dumps(item),
                        "structured_data": item,
                        "item_index": idx,
                    }

            # If dict → split per key
            elif isinstance(data, dict):
                for key, value in data.items():
                    yield {
                        "chunk_type": "json_field",
                        "text": f"{key}: {json.dumps(value)}",
                        "structured_data": {key: value},
                    }

        except Exception as e:
            logger.error(f"JSON decode failed: {str(e)}")
        return

    # ================= XML =================
    if filename_lower.endswith(".xml"):
        try:
            text = file_bytes.decode("utf-8")
            for chunk in split_large_text(text, "xml_block"):
                yield chunk
        except Exception as e:
            logger.error(f"XML decode failed: {str(e)}")
        return

    # ================= TXT / FALLBACK =================
    try:
        text = file_bytes.decode("utf-8", errors="ignore")

        if text.strip():
            for chunk in split_large_text(text):
                yield chunk

    except Exception as e:
        logger.error(f"Fallback decode failed: {str(e)}")
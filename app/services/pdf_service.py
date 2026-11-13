
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed."""


def extract_pdf_pages(file_bytes: bytes) -> list[dict]:
    """
    Extract text from a PDF, keeping the original page numbers.
    """

    # Check whether the file contains a PDF header.
    if b"%PDF-" not in file_bytes[:1024]:
        raise PDFProcessingError("The uploaded file is not a valid PDF.")

    try:
        reader = PdfReader(BytesIO(file_bytes), strict=False)

        # Handle password-protected PDFs.
        if reader.is_encrypted:
            try:
                result = reader.decrypt("")
                if result == 0:
                    raise PDFProcessingError(
                        "This PDF is password-protected."
                    )
            except Exception as error:
                if isinstance(error, PDFProcessingError):
                    raise
                raise PDFProcessingError(
                    "Unable to open this protected PDF."
                ) from error

        extracted_pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            extracted_pages.append({
                "page_number": page_number,
                "text": text.strip()
            })

        return extracted_pages

    except PDFProcessingError:
        raise
    except (PdfReadError, ValueError, OSError) as error:
        raise PDFProcessingError(
            "Unable to read this PDF file."
        ) from error

"""
PDF Parser — Extract text from PDF resumes
============================================
Uses pdfplumber (allowed utility library) to extract raw text from PDF files.
Falls back to PyPDF2 if pdfplumber fails.

Note: This is a UTILITY — not part of the from-scratch ML/NLP pipeline.
      The professor allows libraries for file I/O.
"""

import io


def extract_text_from_pdf(file_input):
    """
    Extract all text content from a PDF file.

    Tries pdfplumber first (better accuracy), falls back to PyPDF2.

    Args:
        file_input: Either a file path (str) or a file-like object
                    (e.g., from Streamlit's file_uploader)

    Returns:
        str: Extracted text from all pages, joined with newlines

    Raises:
        RuntimeError: If text extraction fails with both libraries
    """
    text = _try_pdfplumber(file_input)
    if text and text.strip():
        return text

    text = _try_pypdf2(file_input)
    if text and text.strip():
        return text

    raise RuntimeError("Could not extract text from PDF. The file may be image-based (scanned).")


def _try_pdfplumber(file_input):
    """Attempt extraction with pdfplumber."""
    try:
        import pdfplumber

        if isinstance(file_input, str):
            # File path
            with pdfplumber.open(file_input) as pdf:
                pages = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
                return '\n'.join(pages)
        else:
            # File-like object (e.g., from Streamlit uploader)
            # Need to handle the bytes
            if hasattr(file_input, 'read'):
                content = file_input.read()
                if hasattr(file_input, 'seek'):
                    file_input.seek(0)  # Reset for potential reuse
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    pages = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages.append(page_text)
                    return '\n'.join(pages)
            else:
                # It's raw bytes
                with pdfplumber.open(io.BytesIO(file_input)) as pdf:
                    pages = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages.append(page_text)
                    return '\n'.join(pages)
    except ImportError:
        return None
    except Exception:
        return None


def _try_pypdf2(file_input):
    """Attempt extraction with PyPDF2 as fallback."""
    try:
        from PyPDF2 import PdfReader

        if isinstance(file_input, str):
            reader = PdfReader(file_input)
        elif hasattr(file_input, 'read'):
            content = file_input.read()
            if hasattr(file_input, 'seek'):
                file_input.seek(0)
            reader = PdfReader(io.BytesIO(content))
        else:
            reader = PdfReader(io.BytesIO(file_input))

        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return '\n'.join(pages)
    except ImportError:
        return None
    except Exception:
        return None

import fitz  # PyMuPDF

def load_pdf(pdf_path):
    """
    Reads a PDF file and returns all text as a single string.
    """

    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

        doc.close()

        return text

    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
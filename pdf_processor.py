import fitz  # PyMuPDF

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

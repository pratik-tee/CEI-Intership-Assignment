import os
from utils.pdf_loader import load_pdf

pdf_path = "data/sample.pdf"

print("Current Directory:", os.getcwd())
print("PDF Exists:", os.path.exists(pdf_path))
print("PDF Path:", pdf_path)

text = load_pdf(pdf_path)

print("Length of Extracted Text:", len(text))

if len(text) > 0:
    print("\nFirst 1000 Characters:\n")
    print(text[:1000])
else:
    print("\nNo text extracted from the PDF.")
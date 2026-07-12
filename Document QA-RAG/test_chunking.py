from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text

text = load_pdf("data/sample.pdf")

chunks = split_text(text)

print("=" * 60)
print("Total Chunks:", len(chunks))
print("=" * 60)

print("\nFirst Chunk:\n")
print(chunks[0])

print("\n" + "=" * 60)

print("\nSecond Chunk:\n")
print(chunks[1] if len(chunks) > 1 else "Only one chunk.")
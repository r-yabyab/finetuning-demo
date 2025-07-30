from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

converter = DocumentConverter()
# doc = "data/docling-stuff/sigewinne.md"
doc = converter.convert("data/docling-stuff/sigewinne.md").document
chunker = HybridChunker()
chunks = chunker.chunk(dl_doc=doc)
# result = converter.convert(doc)
# print(result.document.export_to_markdown())


# chunk_iter = chunker.chunk(result.document)

for i, chunk in enumerate(chunks):
    print(f"=== {i} ===")
    print(f"chunk.text:\n{f'{chunk.text[:300]}…'!r}")

    enriched_text = chunker.contextualize(chunk=chunk)
    print(f"chunker.contextualize(chunk):\n{f'{enriched_text[:300]}…'!r}")

    print()
import pymupdf4llm
md_text = pymupdf4llm.to_markdown("raw_documents/1. Nature mental wealth paper.pdf")

import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode())
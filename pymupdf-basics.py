import pymupdf

doc = pymupdf.open("raw_documents/1. Nature mental wealth paper.pdf")
out = open("output.txt", "wb")
for page in doc:
    text = page.get_text().encode("utf8")
    out.write(text)
    out.write(bytes((12,)))
out.close()
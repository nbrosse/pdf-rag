from pathlib import Path

from pypdf import PdfReader

dir_path = Path("/home/nicolas/Documents/projets/pdf-rag/data/pdf-parsing-open-data/")
pdf_path = dir_path / "2023-conocophillips-aim-presentation-1-7.pdf"

pdf_reader = PdfReader(str(pdf_path))
import os
from pathlib import Path
from pdf_rag.readers import VLMPDFReader
from google import genai

from dotenv import load_dotenv
load_dotenv()
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

opio_dir_path = Path("/home/nicolas/Documents/projets/opio/")
shell_dir_path = opio_dir_path / "raw data" / "Shell Dec 5 2024"
small_data_dir_path = opio_dir_path / "small data"
shell_2023_report_path = shell_dir_path / "shell-annual-report-2023.pdf"
shell_2022_report_path = shell_dir_path / "shell-annual-report-2022.pdf"
small_shell_2023_report_path = small_data_dir_path / "shell-annual-report-2023-1-20.pdf"
cache_dir = opio_dir_path / "cache"

doc_path_1 = shell_dir_path / "CMD23-non-gaap-reconciliation.pdf"
doc_path_2 = shell_dir_path / "shell-ets24-print.pdf"

pdfs_paths = list(shell_dir_path.glob("*.pdf"))

vlm_reader = VLMPDFReader(cache_dir=cache_dir, num_workers=32)
documents = vlm_reader.load_data(pdfs_paths)
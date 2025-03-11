import os
from pathlib import Path

from llama_index.core import Settings
from llama_index.llms.gemini import Gemini

from pdf_rag.gemini_wrappers import GeminiEmbeddingUpdated
from pdf_rag.readers import PDFDirectoryReader
from google import genai

from dotenv import load_dotenv

from pdf_rag.transforms import ReformatMarkdownComponent

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

Settings.embed_model = GeminiEmbeddingUpdated(
    model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY")
)
Settings.llm = Gemini(model_name="models/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))

dir_path = Path("/home/nicolas/Documents/projets/pdf-rag/data/")
pdfs_dir_path = dir_path / "pdfs"
cache_dir = dir_path / "cache"

pdf_reader = PDFDirectoryReader(
    root_dir=str(pdfs_dir_path),
    cache_dir=str(cache_dir),
    num_workers=32,
    show_progress=True,
)

documents = pdf_reader.load_data(pdfs_dir_path)
docs_dict = {d.metadata["file_name"]: d for d in documents}
doc = docs_dict["XC9500_CPLD_Family.pdf"]
reformat_markdown = ReformatMarkdownComponent()
doc = reformat_markdown([doc])
import os
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.llms.gemini import Gemini

from pdf_rag.extractors import ContextExtractor, TableOfContentsExtractor, TableOfContentsCreator, \
    LandscapePagesExtractor, StructureExtractor
from pdf_rag.gemini_wrappers import GeminiEmbeddingUpdated
from pdf_rag.readers import PDFDirectoryReader
from google import genai

from dotenv import load_dotenv

from pdf_rag.structure_parsers import parse_structure
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
storage_dir = dir_path / "storage"

# Load the docstore
data_dir = Path("/home/nicolas/Documents/projets/pdf-rag/data/storage/")
docstore = SimpleDocumentStore.from_persist_path(str(data_dir / "processed_docstore_storage.json"))
for k, doc in docstore.docs.items():
    print("#######################################################")
    print(doc.metadata["filename"])
    structure = doc.metadata["structure"]
    parsed_structure = parse_structure(doc)
    print(str(parsed_structure))
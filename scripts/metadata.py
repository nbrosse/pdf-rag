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

pdf_reader = PDFDirectoryReader(
    root_dir=str(pdfs_dir_path),
    cache_dir=str(cache_dir),
    num_workers=32,
    show_progress=True,
)


# docs_dict = {d.metadata["file_name"]: d for d in documents}
# doc = docs_dict["gx-iif-open-data.pdf"]
# reformat_markdown = ReformatMarkdownComponent()
# doc = reformat_markdown([doc])




docstore = SimpleDocumentStore()
processed_docstore = SimpleDocumentStore()
documents = pdf_reader.load_data(pdfs_dir_path)

num_workers = 32
model_name = "gemini-2.0-flash"
pipeline = IngestionPipeline(
    transformations=[
        ReformatMarkdownComponent(model_name=model_name, num_workers=num_workers),
        ContextExtractor(model_name=model_name, num_workers=num_workers),
        TableOfContentsExtractor(model_name=model_name, num_workers=num_workers),
        TableOfContentsCreator(model_name=model_name, num_workers=num_workers),
        LandscapePagesExtractor(model_name=model_name, num_workers=num_workers),
        StructureExtractor(model_name=model_name, num_workers=num_workers),
    ],
    docstore=docstore,
    docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
)

# pipeline.load(persist_dir=str(data_dir / "pipeline_storage_pro"))
# pipeline.docstore = SimpleDocumentStore()

nodes = pipeline.run(documents=documents, show_progress=True)
pipeline.persist(str(storage_dir / "pipeline_storage"))
docstore.persist(str(storage_dir / "docstore_storage.json"))

processed_docstore.add_documents(docs=nodes)
processed_docstore.persist(str(storage_dir / "processed_docstore_storage.json"))

metadata_nodes_str = "\n-------------------\n".join([n.get_metadata_str() for n in nodes])
metadata_file = dir_path / "metadata_str.txt"
metadata_file.write_text(metadata_nodes_str)
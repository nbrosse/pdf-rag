import os
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.gemini import Gemini

from pdf_rag.gemini_wrappers import GeminiEmbeddingUpdated
from pdf_rag.markdown_parsers import MarkdownPageNodeParser
from pdf_rag.node_postprocessors import FullPagePostprocessor
from pdf_rag.readers import PDFDirectoryReader
from google import genai

from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

Settings.embed_model = GeminiEmbeddingUpdated(
    model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY")
)
Settings.llm = Gemini(model_name="models/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))

# Paths
opio_dir_path = Path("/home/nicolas/Documents/projets/opio/")
use_cases_path = opio_dir_path / "raw data" / "use cases"
cache_dir = opio_dir_path / "cache"
storage_dir = opio_dir_path / "storage"

# Doc path
doc_path = use_cases_path / "Shell Dec 5 2024" / "shell-annual-report-2023.pdf"
# Queries
queries = [
    "What is the cash position of Shell at the end of 2023?",
    "Detail Shell's current financing arrangements and the terms associated with them",
    "What are the revenues per business unit in 2023?",
    "What are the Non-GAAP measures reported in Shell's consolidated results in 2023?",
    "What are the revenues per segment earnings in 2023?",
    "What are the key metrics reported in Shell's consolidated results in 2023?"
]

pdf_reader = PDFDirectoryReader(
    cache_dir=cache_dir,
    num_workers=32,
    root_dir=str(use_cases_path),
)

documents = pdf_reader.load_data(doc_path)
document = documents[0]
markdown_parser = MarkdownPageNodeParser()
nodes = markdown_parser.get_nodes_from_documents(documents=documents)

relative_path = document.metadata["relative_path"]
storage_context_persist_dir = storage_dir / relative_path.replace(".pdf", "_vector")
if storage_context_persist_dir.exists():
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_context_persist_dir))
    vector_index: VectorStoreIndex = load_index_from_storage(storage_context=storage_context)
else:
    vector_index: VectorStoreIndex = VectorStoreIndex(nodes=nodes)
    vector_index.storage_context.persist(str(storage_context_persist_dir))

full_page_postprocessor = FullPagePostprocessor(docstore=vector_index.docstore)
retriever = vector_index.as_retriever(
    similarity_top_k=5,
    vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[full_page_postprocessor],
)

outputs_dir = opio_dir_path / "outputs_v4"
outputs_dir.mkdir(parents=True, exist_ok=True)
for i, query in enumerate(queries):
    output = list()
    retrieved_nodes = query_engine.retrieve(QueryBundle(query))
    response = query_engine.query(query)
    output.append(query)
    output.append(str(response))
    for node in retrieved_nodes:
        output.append("-------------------------------------------------")
        output.append(str(node.score))
        output.append(str(node.metadata["page_number"]))
        output.append(node.text)
    output_content = "\n".join(output)
    output_file = outputs_dir / f"output_{i}.md"
    output_file.write_text(output_content)
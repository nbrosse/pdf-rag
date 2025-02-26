import mimetypes
import os
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tiktoken
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, QueryBundle, PromptTemplate, \
    get_response_synthesizer, SummaryIndex, SimpleDirectoryReader, Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine, CitationQueryEngine
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.readers.file.base import get_default_fs, _format_file_timestamp
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Settings
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from pydantic import Field, PrivateAttr, field_validator
from llama_index.core.postprocessor import PrevNextNodePostprocessor

from pdf_rag.gemini_wrappers import GeminiEmbeddingUpdated
from pdf_rag.markdown_parsers import MarkdownPageNodeParser
from pdf_rag.node_postprocessors import FullPagePostprocessor
from pdf_rag.readers import VLMPDFReader, PDFDirectoryReader
from google import genai

from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))






Settings.embed_model = GeminiEmbeddingUpdated(
    model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY")
)
Settings.llm = Gemini(model_name="models/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))

opio_dir_path = Path("/home/nicolas/Documents/projets/opio/")
raw_data_path = opio_dir_path / "raw data"
orano_dir_path =  opio_dir_path / "raw data" / "4 - donnees demo Orano RAA 2016-2023"

shell_dir_path = opio_dir_path / "raw data" / "Shell Dec 5 2024"
small_data_dir_path = opio_dir_path / "small data"
shell_2023_report_path = shell_dir_path / "shell-annual-report-2023.pdf"
shell_2022_report_path = shell_dir_path / "shell-annual-report-2022.pdf"
small_shell_2023_report_path = small_data_dir_path / "shell-annual-report-2023-1-20.pdf"
cache_dir = opio_dir_path / "cache_essai_v2"
cache_dir.mkdir(parents=True, exist_ok=True)
orano_2016_path = orano_dir_path / "orano_RAA_2016.pdf"

# vlm_reader = VLMPDFReader(cache_dir=cache_dir, num_workers=32, show_progress=True)
#
# documents = vlm_reader.load_data(orano_2016_path)

pdf_reader = PDFDirectoryReader(
    root_dir=str(raw_data_path),
    cache_dir=str(cache_dir),
    num_workers=32,
    show_progress=True,
)

documents = pdf_reader.load_data(orano_dir_path)



# simple_directory_reader = SimpleDirectoryReader(
#     input_dir=str(orano_dir_path),
#     recursive=True,
#     required_exts=[".pdf"],
#     file_metadata=partial(file_metadata_func, root_dir_path=raw_data_path),
#     file_extractor={".pdf": vlm_reader},
# )

# documents = simple_directory_reader.load_data(
#     show_progress=True,
#     num_workers=1,
# )


#
#
# doc_path_1 = shell_dir_path / "CMD23-non-gaap-reconciliation.pdf"
# doc_path_2 = shell_dir_path / "shell-ets24-print.pdf"
#
# queries = [
#     "what is the cash position of Shell at the end of 2023 ?",
#     "Detail Shell current financing arrangements and the terms associated with them",
#     "What are the revenues per business unit in 2023 ?",
#     "what are the Non GAAP measures reported in Shell's consolidated results in 2023 ?",
#     "What are the revenues per segment earnings in 2023 ?",
#     "what are the key metrics reported in Shell's consolidated results in 2023 ?",
# ]
#
# query = queries[1]
#
#
# documents = vlm_reader.load_data(shell_2023_report_path)
# markdown_parser = MarkdownPageNodeParser()
# nodes = markdown_parser.get_nodes_from_documents(documents=documents)
# sentence_splitter = SentenceSplitter(chunk_size=1024)
# nodes = sentence_splitter(nodes)
#
# summary_index = SummaryIndex(nodes)
# summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")
# summary = summary_query_engine.query(
#     "Extract a summary of this document"
# )
# print(summary.response)
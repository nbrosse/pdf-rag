import os
from pathlib import Path

import numpy as np
import tiktoken
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Settings
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from pydantic import Field, PrivateAttr
from llama_index.core.postprocessor import PrevNextNodePostprocessor


from pdf_rag.markdown_parsers import MarkdownPageNodeParser
from pdf_rag.readers import VLMPDFReader
from google import genai

from dotenv import load_dotenv
load_dotenv()
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))








class GeminiEmbeddingUpdated(BaseEmbedding):

    api_key: str | None = Field(
        default=None,
        description="API key to access the model. Defaults to None.",
    )

    _client: genai.Client = PrivateAttr()

    def __init__(self,
                 api_key: str,
                 model_name: str = "models/text-embedding-004",
                 ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
        )
        self._client = genai.Client(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "GeminiEmbeddingUpdated"

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        embeddings = await self._client.aio.models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        return [e.values for e in embeddings.embeddings]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._client.models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        return [e.values for e in embeddings.embeddings]

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings."""
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """The asynchronous version of _get_query_embedding."""
        return (await self._aembed(texts=[query]))[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Asynchronously get text embedding."""
        return (await self._aembed(texts=[text]))[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously get text embeddings."""
        return await self._aembed(texts=texts)

Settings.embed_model = GeminiEmbeddingUpdated(
    model_name="models/text-embedding-004", api_key=os.getenv("GEMINI_API_KEY")
)
Settings.llm = Gemini(model_name="models/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))

opio_dir_path = Path("/home/nicolas/Documents/projets/opio/")
shell_dir_path = opio_dir_path / "raw data" / "Shell Dec 5 2024"
small_data_dir_path = opio_dir_path / "small data"
shell_2023_report_path = shell_dir_path / "shell-annual-report-2023.pdf"
shell_2022_report_path = shell_dir_path / "shell-annual-report-2022.pdf"
small_shell_2023_report_path = small_data_dir_path / "shell-annual-report-2023-1-20.pdf"
cache_dir = opio_dir_path / "cache"

queries = [
    "what is the cash position of Shell at the end of 2023 ?",
    "Detail Shell current financing arrangements and the terms associated with them",
    "What are the revenues per business unit in 2023 ?",
    "what are the Non GAAP measures reported in Shell's consolidated results in 2023 ?",
    "What are the revenues per segment earnings in 2023 ?",
    "what are the key metrics reported in Shell's consolidated results in 2023 ?",
]

query = queries[1]

vlm_reader = VLMPDFReader(cache_dir=cache_dir, num_workers=32)
document = vlm_reader.load_data(shell_2023_report_path)
markdown_parser = MarkdownPageNodeParser()
nodes = markdown_parser.get_nodes_from_documents(documents=document)
sentence_splitter = SentenceSplitter(chunk_size=1024)
nodes = sentence_splitter(nodes)

storage_context_persist_dir = opio_dir_path / "2023_shell_storage"

if storage_context_persist_dir.exists():
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_context_persist_dir))
    index = load_index_from_storage(storage_context=storage_context)
else:
    index = VectorStoreIndex(nodes=nodes)
    index.storage_context.persist(str(storage_context_persist_dir))

# We can pass in the index, docstore, or list of nodes to create the retriever
# bm25_retriever = BM25Retriever.from_defaults(
#     nodes=nodes,
#     similarity_top_k=2,
#     # Optional: We can pass in the stemmer and set the language for stopwords
#     # This is important for removing stopwords and stemming the query + text
#     # The default is english for both
#     stemmer=Stemmer.Stemmer("english"),
#     language="english",
# )


# index = VectorStoreIndex(nodes=nodes) # , insert_batch_size=4)
# index.vector_store.persist(str(opio_dir_path / "shell_2023_vector_index_v2"))
# vector_store = SimpleVectorStore.from_persist_path(str(opio_dir_path / "shell_2023_vector_index_v2.json"))
# index = VectorStoreIndex.from_vector_store(vector_store)
# pipeline = IngestionPipeline(
#     docstore=docstore,
#     docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
# )


# vector_store_path = opio_dir_path / "shell_2023_vector_index"
# if vector_store_path.exists():
#     storage_context = StorageContext.from_defaults(persist_dir=str(vector_store_path))
#     index = load_index_from_storage(storage_context=storage_context)
# else:
#     raise ValueError()
#     # index = VectorStoreIndex(nodes=nodes, insert_batch_size=4)
# query_engine = index.as_query_engine()
retriever = index.as_retriever(
    similarity_top_k=5,
    vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
)


postprocessor = PrevNextNodePostprocessor(
    docstore=index.docstore,
    num_nodes=1,  # number of nodes to fetch when looking forawrds or backwards
    mode="both",  # can be either 'next', 'previous', or 'both'
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[postprocessor],
)

response = query_engine.query("How many employees does Shell have in France in 2023 ?")

selected_notes = [n for n in nodes if n.metadata["page_number"] == 34]
# content = "\n\n ------------- \n\n".join([n.text for n in selected_notes])
# target_emb = client.models.embed_content(model="text-embedding-004", contents=target_str).embeddings[0].values
# query_emb = client.models.embed_content(model="text-embedding-004", contents=queries[2]).embeddings[0].values
# target_emb = np.array(target_emb)
# query_emb = np.array(query_emb)
# score = np.dot(query_emb, target_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(target_emb))

# response = query_engine.query(queries[2])

# for i, query in enumerate(queries):
#     output = list()
#     retrieved_nodes = retriever.retrieve(query)
#     response = query_engine.query(query)
#     output.append(query)
#     output.append(str(response))
#     for node in retrieved_nodes:
#         output.append("-------------------------------------------------")
#         output.append(str(node.score))
#         output.append(str(node.metadata["page_number"]))
#         output.append(node.text)
#     output_content = "\n".join(output)
#     output_file = opio_dir_path / "outputs" / f"output_{i}.md"
#     output_file.write_text(output_content)




# for query in queries:
#     print("########")
#     print(query)
#     print("########")
#     response = query_engine.query(query)
#     print(response)

# docstore = SimpleDocumentStore()
# processed_docstore = SimpleDocumentStore()
# documents = parser_cache.load_data(pdfs_files)
#

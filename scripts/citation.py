import os
from pathlib import Path

import numpy as np
import tiktoken
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, QueryBundle, PromptTemplate, \
    get_response_synthesizer
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine, CitationQueryEngine
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Settings
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from pydantic import Field, PrivateAttr
from llama_index.core.postprocessor import PrevNextNodePostprocessor

from pdf_rag.gemini_wrappers import GeminiEmbeddingUpdated
from pdf_rag.markdown_parsers import MarkdownPageNodeParser
from pdf_rag.node_postprocessors import FullPagePostprocessor
from pdf_rag.readers import VLMPDFReader
from google import genai

from dotenv import load_dotenv
load_dotenv()
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))



#
# CITATION_QA_TEMPLATE = PromptTemplate(
#     "Please provide an answer based solely on the provided sources. "
#     "When referencing information from a source, "
#     "cite the appropriate source(s) using their corresponding numbers. "
#     "Every answer should include at least one source citation. "
#     "Only cite a source when you are explicitly referencing it. "
#     "If none of the sources are helpful, you should indicate that. "
#     "For example:\n"
#     "Source 1:\n"
#     "The sky is red in the evening and blue in the morning.\n"
#     "Source 2:\n"
#     "Water is wet when the sky is red.\n"
#     "Query: When is water wet?\n"
#     "Answer: Water will be wet when the sky is red [2], "
#     "which occurs in the evening [1].\n"
#     "Now it's your turn. Below are several numbered sources of information:"
#     "\n------\n"
#     "{context_str}"
#     "\n------\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )
#
# CITATION_REFINE_TEMPLATE = PromptTemplate(
#     "Please provide an answer based solely on the provided sources. "
#     "When referencing information from a source, "
#     "cite the appropriate source(s) using their corresponding numbers. "
#     "Every answer should include at least one source citation. "
#     "Only cite a source when you are explicitly referencing it. "
#     "If none of the sources are helpful, you should indicate that. "
#     "For example:\n"
#     "Source 1:\n"
#     "The sky is red in the evening and blue in the morning.\n"
#     "Source 2:\n"
#     "Water is wet when the sky is red.\n"
#     "Query: When is water wet?\n"
#     "Answer: Water will be wet when the sky is red [2], "
#     "which occurs in the evening [1].\n"
#     "Now it's your turn. "
#     "We have provided an existing answer: {existing_answer}"
#     "Below are several numbered sources of information. "
#     "Use them to refine the existing answer. "
#     "If the provided sources are not helpful, you will repeat the existing answer."
#     "\nBegin refining!"
#     "\n------\n"
#     "{context_msg}"
#     "\n------\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )







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

doc_path_1 = shell_dir_path / "CMD23-non-gaap-reconciliation.pdf"
doc_path_2 = shell_dir_path / "shell-ets24-print.pdf"

queries = [
    "what is the cash position of Shell at the end of 2023 ?",
    "Detail Shell current financing arrangements and the terms associated with them",
    "What are the revenues per business unit in 2023 ?",
    "what are the Non GAAP measures reported in Shell's consolidated results in 2023 ?",
    "What are the revenues per segment earnings in 2023 ?",
    "what are the key metrics reported in Shell's consolidated results in 2023 ?",
]

query = queries[1]

# vlm_reader = VLMPDFReader(cache_dir=cache_dir, num_workers=32)
# documents = vlm_reader.load_data([doc_path_1, doc_path_2])
#
#
#
# markdown_parser = MarkdownPageNodeParser()
# nodes = markdown_parser.get_nodes_from_documents(documents=document)
# sentence_splitter = SentenceSplitter(chunk_size=1024)
# nodes = sentence_splitter(nodes)

# split_nodes = sentence_splitter([nodes[7]])

storage_context_persist_dir = opio_dir_path / "2023_shell_storage"

if storage_context_persist_dir.exists():
    storage_context = StorageContext.from_defaults(persist_dir=str(storage_context_persist_dir))
    index = load_index_from_storage(storage_context=storage_context)
else:
    raise ValueError
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

# docstore = index.docstore

full_page_postprocessor = FullPagePostprocessor(
    docstore=index.docstore,
)

# postprocessor = PrevNextNodePostprocessor(
#     docstore=index.docstore,
#     num_nodes=1,  # number of nodes to fetch when looking forawrds or backwards
#     mode="both",  # can be either 'next', 'previous', or 'both'
# )
#

# query_engine = RetrieverQueryEngine(
#     retriever=retriever,
#     node_postprocessors=[full_page_postprocessor],
# )

query_engine = CitationQueryEngine(
    retriever=retriever,
    # node_postprocessors= [full_page_postprocessor],
    # citation_chunk_size=10000,
    # citation_chunk_overlap=0,
)

prompts = query_engine.get_prompts()
for k, prompt in prompts.items():
    print(k)
    print(prompt.default_template.template)


#
# query_engine = CitationQueryEngine.from_args(
#     index=index,
#     retriever=retriever,
#     node_postprocessors= [full_page_postprocessor],
#     citation_chunk_size=10000,
#     citation_chunk_overlap=0,
# )

# query_engine = RetrieverQueryEngine.from_args(
#     retriever=retriever,
#     node_postprocessors=[full_page_postprocessor],
#     text_qa_template=text_qa_template,
#     refine_template=refine_template,
# )


# retrieved_nodes = query_engine.retrieve(QueryBundle(query))
# response = query_engine.query(query)

#
# selected_notes = [n for n in nodes if n.metadata["page_number"] == 34]
# content = "\n\n ------------- \n\n".join([n.text for n in selected_notes])
# target_emb = client.models.embed_content(model="text-embedding-004", contents=target_str).embeddings[0].values
# query_emb = client.models.embed_content(model="text-embedding-004", contents=queries[2]).embeddings[0].values
# target_emb = np.array(target_emb)
# query_emb = np.array(query_emb)
# score = np.dot(query_emb, target_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(target_emb))

# response = query_engine.query(queries[2])

# outputs_dir = opio_dir_path / "outputs_v4"
# outputs_dir.mkdir(parents=True, exist_ok=True)
#
# for i, query in enumerate(queries):
#     output = list()
#     retrieved_nodes = query_engine.retrieve(QueryBundle(query))
#     response = query_engine.query(query)
#     output.append(query)
#     output.append(str(response))
#     for node in retrieved_nodes:
#         output.append("-------------------------------------------------")
#         output.append(str(node.score))
#         output.append(str(node.metadata["page_number"]))
#         output.append(node.text)
#     output_content = "\n".join(output)
#     output_file = outputs_dir / f"output_{i}.md"
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

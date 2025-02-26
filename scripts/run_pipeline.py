import asyncio
import os
from pathlib import Path

import numpy as np
import tiktoken
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, QueryBundle, PromptTemplate, \
    get_response_synthesizer, SummaryIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.async_utils import run_jobs, asyncio_run
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.objects import ObjectIndex, ObjectRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, CitationQueryEngine, SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from pydantic import Field, PrivateAttr
from llama_index.core.postprocessor import PrevNextNodePostprocessor
from tqdm import tqdm

from pdf_rag.gemini_wrappers import GeminiEmbeddingUpdated
from pdf_rag.markdown_parsers import MarkdownPageNodeParser
from pdf_rag.node_postprocessors import FullPagePostprocessor
from pdf_rag.readers import VLMPDFReader
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
shell_dir_path = opio_dir_path / "raw data" / "Shell Dec 5 2024"
shell_2022_report_path = shell_dir_path / "shell-annual-report-2022.pdf"
shell_2023_report_path = shell_dir_path / "shell-annual-report-2023.pdf"
cache_dir = opio_dir_path / "cache"
storage_dir = opio_dir_path / "storage"
pdf_files = list(shell_dir_path.glob("*.pdf"))
print(len(pdf_files))

queries = [
    "What is the cash position of Shell at the end of 2022?",
    "What is the cash position of Shell at the end of 2023?",
    "What is the cash position of Shell at the end of 2019?",
    "What is the cash flow from operating activities of Shell in 2019?",
    "What is the free cash flow of Shell in 2018?",
    "What are the primary drivers of cash flow variability in Shell business?",
    "What was the percentage of free cash flow that was distributed to shareholders in 2019?",
    "How many employees does Shell have in France in 2023?",
    "How many employees does Shell have in Albania? Look in the 2023 tax report",
    "What are the taxes paid by Shell in Thailand in 2023?",
    "What is the name of the association Shell paid the most in 2023?",
    "What is the position of the Canadian Association of Petroleum Producers on ending routine flaring?",
    "Detail Shell's current financing arrangements and the terms associated with them",
    "What are the revenues per business unit in 2023?",
    "What are the Non-GAAP measures reported in Shell's consolidated results in 2023?",
    "What are the revenues per segment earnings in 2023?",
    "What are the key metrics reported in Shell's consolidated results in 2023?"
]

# query = queries[1]
#
vlm_reader = VLMPDFReader(cache_dir=cache_dir, num_workers=32)
# documents = vlm_reader.load_data(pdf_files)
markdown_parser = MarkdownPageNodeParser()
# nodes = markdown_parser.get_nodes_from_documents(documents=documents)
sentence_splitter = SentenceSplitter(chunk_size=1024)
# nodes = sentence_splitter(nodes)


# define a custom object retriever that adds in a query planning tool
class CustomObjectRetriever(ObjectRetriever):
    def __init__(
        self,
        retriever,
        object_node_mapping,
        node_postprocessors=None,
        llm=None,
        additional_tools=None,
    ):
        super().__init__(
            retriever=retriever,
            object_node_mapping=object_node_mapping,
            node_postprocessors=node_postprocessors,
        )
        self._llm = llm or Settings.llm
        self._additional_tools = additional_tools

    def retrieve(self, query_bundle):
        if isinstance(query_bundle, str):
            query_bundle = QueryBundle(query_str=query_bundle)

        nodes = self._retriever.retrieve(query_bundle)
        for processor in self._node_postprocessors:
            nodes = processor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

        sub_question_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools,
            llm=self._llm,
            question_gen=LLMQuestionGenerator.from_defaults(llm=self._llm),
        )
        sub_question_description = f"""
        Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this
        tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
        """
        sub_question_tool = QueryEngineTool(
            query_engine=sub_question_engine,
            metadata=ToolMetadata(
                name="compare_tool",
                description=sub_question_description
            ),
        )

        return tools + [sub_question_tool] + self._additional_tools



async def abuild_agent_per_doc(pdf_file: Path):
    vector_index_persist_path = storage_dir / f"{pdf_file.stem}_vector"
    summary_index_persist_path = storage_dir / f"{pdf_file.stem}_summary"
    summary_path = storage_dir / f"{pdf_file.stem}_summary.txt"
    if vector_index_persist_path.exists() and summary_index_persist_path.exists():
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(vector_index_persist_path)),
        )
        summary_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(summary_index_persist_path)),
        )
    else:
        documents = await vlm_reader.aload_data(pdf_file)
        nodes = markdown_parser.get_nodes_from_documents(documents=documents)
        nodes = sentence_splitter(nodes)
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(persist_dir=str(vector_index_persist_path))
        summary_index = SummaryIndex(nodes)
        summary_index.storage_context.persist(persist_dir=str(summary_index_persist_path))

    # define query engines
    vector_query_engine = vector_index.as_query_engine(similarity_top_k=5)
    summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")

    # extract a summary
    if not summary_path.exists():
        summary = (
            await summary_query_engine.aquery(
                "Extract a summary of this document"
            )
        ).response
        summary_path.write_text(summary)
    else:
        summary = summary_path.read_text()

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{pdf_file.stem}",
                description=f"Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{pdf_file.stem}",
                description=f"Useful for summarization questions",
            ),
        ),
    ]

    system_prompt = f"""
    You are a specialized agent designed to answer queries about the `{pdf_file.name}` document.
    You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
    """

    # build agent
    agent = ReActAgent.from_tools(
        query_engine_tools,
        verbose=True,
        system_prompt=system_prompt,
    )

    return agent, summary


async def abuild_agents(dir_path: Path):
    pdf_files = list(dir_path.glob("*.pdf"))

    jobs = [
        abuild_agent_per_doc(pdf_file) for pdf_file in pdf_files
    ]

    # results = await asyncio.gather(
    #     *(abuild_agent_per_doc(pdf_file) for pdf_file in pdf_files)
    # )

    results = await run_jobs(
        jobs=jobs,
        show_progress=True,
        workers=4,
        desc="Building agents",
    )

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    for pdf_file, result in zip(pdf_files, results):
        agent, summary = result
        agents_dict[pdf_file.stem] = agent
        # extra_info_dict[pdf_file.stem] = {"summary": summary, "nodes": nodes}
        extra_info_dict[pdf_file.stem] = {"summary": summary}

    return agents_dict, extra_info_dict


def build_global_vector_query_engine(dir_path: Path) -> QueryEngineTool | None:

    def _get_vector_index(pdf_file: Path) -> VectorStoreIndex:
        vector_index_persist_path = storage_dir / f"{pdf_file.stem}_vector"
        if vector_index_persist_path.exists():
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(vector_index_persist_path)),
            )
        else:
            raise ValueError(f"Vector index not found at {str(vector_index_persist_path)}")
        return vector_index

    def _get_nodes_with_embeddings_from_vector_index(vector_index: VectorStoreIndex) -> list[BaseNode]:
        nodes_ids = list(vector_index.docstore.docs.keys())
        nodes = vector_index.docstore.get_nodes(nodes_ids)
        nodes_emb = vector_index._get_node_with_embedding(nodes)
        return nodes_emb

    global_vector_index_persist_path = storage_dir / "global_vector"
    if global_vector_index_persist_path.exists():
        ref_vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=str(global_vector_index_persist_path)),
        )
    else:
        pdf_files = list(dir_path.glob("*.pdf"))
        if len(pdf_files) < 2:
            return None
        ref_vector_index = _get_vector_index(pdf_files[0])
        for pdf_file in pdf_files[1:]:
            vector_index = _get_vector_index(pdf_file)
            nodes_emb = _get_nodes_with_embeddings_from_vector_index(vector_index)
            ref_vector_index.insert_nodes(nodes_emb)
        ref_vector_index.storage_context.persist(str(global_vector_index_persist_path))

    vector_query_engine = ref_vector_index.as_query_engine(similarity_top_k=10)

    global_vector_description = """
    Useful for queries that require extracting specific details from extensive documents.
    ALWAYS use this tool as a last resort if other methods fail. Ensure to call this tool with the original query.
    """

    query_engine_tool = QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name=f"global_vector_tool",
            description=global_vector_description,
        ),
    )

    return query_engine_tool


async def process_query(queries: list[str], agents_dict: dict, extra_info_dict: dict):
    all_tools = []
    for file_base, agent in agents_dict.items():
        summary = extra_info_dict[file_base]["summary"]
        doc_tool = QueryEngineTool(
            query_engine=agent,
            metadata=ToolMetadata(
                name=f"tool_{file_base}",
                description=summary,
            ),
        )
        all_tools.append(doc_tool)

    obj_index = ObjectIndex.from_objects(
        all_tools,
        index_cls=VectorStoreIndex,
    )
    vector_node_retriever = obj_index.as_node_retriever(
        similarity_top_k=10,
    )

    global_query_engine = build_global_vector_query_engine(shell_dir_path)

    custom_obj_retriever = CustomObjectRetriever(
        vector_node_retriever,
        obj_index.object_node_mapping,
        additional_tools=[global_query_engine],
    )

    top_agent = ReActAgent.from_tools(
        tool_retriever=custom_obj_retriever,
        system_prompt="""
        You are an agent designed to answer queries about the documentation.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.
        """,
        verbose=True,
    )
    responses = [await top_agent.aquery(query) for query in queries]

    return responses


async def main():
    agents_dict, extra_info_dict = await abuild_agents(shell_dir_path)

    if not agents_dict:
        raise ValueError("No agents were built.")

    # Process the first query from your queries list
    responses = await process_query(queries, agents_dict, extra_info_dict)
    return responses


if __name__ == "__main__":
    responses = asyncio_run(main())
    for response in responses:
        print(response)


# async def main():
#     agents_dict, extra_info_dict = await abuild_agents(shell_dir_path)
#
#     if not agents_dict:
#         raise ValueError("No agents were built.")
#
#     # define tool for each document agent
#     all_tools = []
#     for file_base, agent in agents_dict.items():
#         summary = extra_info_dict[file_base]["summary"]
#         doc_tool = QueryEngineTool(
#             query_engine=agent,
#             metadata=ToolMetadata(
#                 name=f"tool_{file_base}",
#                 description=summary,
#             ),
#         )
#         all_tools.append(doc_tool)
#
#
#     obj_index = ObjectIndex.from_objects(
#         all_tools,
#         index_cls=VectorStoreIndex,
#     )
#     vector_node_retriever = obj_index.as_node_retriever(
#         similarity_top_k=10,
#     )
#
#     # wrap it with ObjectRetriever to return objects
#     custom_obj_retriever = CustomObjectRetriever(
#         vector_node_retriever,
#         obj_index.object_node_mapping,
#     )
#
#     top_agent = ReActAgent.from_tools(
#         tool_retriever=custom_obj_retriever,
#         system_prompt=""" \
#     You are an agent designed to answer queries about the documentation.
#     Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
#     """,
#         verbose=True,
#     )
#
#     response = await top_agent.aquery(queries[0])
#     return response
#
# # outputs_dir = opio_dir_path / "outputs_multi_v1"
# # outputs_dir.mkdir(parents=True, exist_ok=True)
# #
# # for i, query in enumerate(queries):
# #     output = list()
# #     retrieved_nodes = query_engine.retrieve(QueryBundle(query))
# #     response = query_engine.query(query)
# #     output.append(query)
# #     output.append(str(response))
# #     for node in retrieved_nodes:
# #         output.append("-------------------------------------------------")
# #         output.append(str(node.score))
# #         output.append(str(node.metadata["page_number"]))
# #         output.append(node.text)
# #     output_content = "\n".join(output)
# #     output_file = outputs_dir / f"output_{i}.md"
# #     output_file.write_text(output_content)
#
#
# #
#
# if __name__ == "__main__":
#     response = asyncio_run(main())
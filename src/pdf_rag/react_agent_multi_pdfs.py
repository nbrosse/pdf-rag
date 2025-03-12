"""
https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents-v1/
"""

import os
import re
import time
from functools import cached_property
from pathlib import Path
from typing import Any

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, QueryBundle, SummaryIndex
from llama_index.core.agent import ReActAgent
from llama_index.core.async_utils import run_jobs
from llama_index.core.objects import ObjectIndex, ObjectRetriever
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.gemini import Gemini
from tqdm import tqdm

from pdf_rag.gemini_wrappers import GeminiEmbeddingUpdated
from pdf_rag.markdown_parsers import MarkdownPageNodeParser
from pdf_rag.node_postprocessors import FullPagePostprocessor
from pdf_rag.readers import PDFDirectoryReader

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", None)
if api_key is None:
    raise ValueError("The GEMINI_API_KEY key is required.")

Settings.embed_model = GeminiEmbeddingUpdated(
    model_name="models/text-embedding-004", api_key=api_key
)
Settings.llm = Gemini(model_name="models/gemini-2.0-flash", api_key=api_key)


# define a custom object retriever that adds in a query planning tool
class CustomObjectRetriever(ObjectRetriever):
    def __init__(
        self,
        retriever,
        object_node_mapping,
        node_postprocessors=None,
        llm=Gemini(model_name="models/gemini-2.0-flash", api_key=api_key),
    ):
        super().__init__(
            retriever=retriever,
            object_node_mapping=object_node_mapping,
            node_postprocessors=node_postprocessors,
        )
        self._llm = llm or Settings.llm

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
        retrieved_tools = tools + [sub_question_tool]
        return retrieved_tools


class ReActAgentMultiPdfs:

    def __init__(
            self,
            api_key_gemini: str,
            api_key_mistral: str,
            root_dir: Path,
            pdfs_dir: Path,
            cache_dir: Path,
            storage_dir: Path,
            num_workers: int = 16,
            chunks_top_k: int = 5,
            nodes_top_k: int = 10,
            max_iterations: int = 20,
            verbose: bool = True,
    ) -> None:
        self._root_dir = root_dir
        self._pdfs_dir = pdfs_dir
        self._cache_dir = cache_dir
        self._storage_dir = storage_dir
        self._num_workers = num_workers
        self._chunks_top_k = chunks_top_k
        self._nodes_top_k = nodes_top_k
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._pdf_reader = PDFDirectoryReader(
            api_key_gemini=api_key_gemini,
            api_key_mistral=api_key_mistral,
            cache_dir=str(cache_dir),
            num_workers=num_workers,
            root_dir=str(root_dir),
            show_progress=False,
        )
        self._markdown_parser = MarkdownPageNodeParser()
        #
        assert self._root_dir.exists(), f"{self._root_dir} does not exist"
        assert self._pdfs_dir.exists(), f"{self._pdfs_dir} does not exist"

    def _get_vector_summary_paths(self, pdf_file: Path) -> tuple[str, Path, Path, Path]:
        relative_path = Path(pdf_file).relative_to(self._root_dir)
        storage_name = relative_path.name
        # Clean filename by replacing spaces and parentheses with underscores to ensure compatibility with LLM tool parsing
        storage_name = re.sub(r'[()\s]', '_', storage_name)
        pdf_file_stem_clean = Path(storage_name).stem
        #
        storage_name_vector = storage_name.replace(".pdf", "_vector")
        storage_name_summary = storage_name.replace(".pdf", "_summary")
        storage_name_summary_txt = storage_name.replace(".pdf", "_summary.txt")
        #
        relative_path_vector = relative_path.with_name(storage_name_vector)
        relative_path_summary = relative_path.with_name(storage_name_summary)
        relative_path_summary_txt = relative_path.with_name(storage_name_summary_txt)
        #
        vector_index_persist_path = self._storage_dir / relative_path_vector
        summary_index_persist_path = self._storage_dir / relative_path_summary
        summary_path = self._storage_dir / relative_path_summary_txt
        return pdf_file_stem_clean, vector_index_persist_path, summary_index_persist_path, summary_path

    def build_agent_per_doc(
            self,
            pdf_file: Path,
    ) -> tuple[ReActAgent, str]:
        pdf_file_stem_clean, vector_index_persist_path, summary_index_persist_path, summary_path = self._get_vector_summary_paths(pdf_file)
        if vector_index_persist_path.exists() and summary_index_persist_path.exists():
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(vector_index_persist_path)),
            )
            summary_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(summary_index_persist_path)),
            )
        else:
            documents = self._pdf_reader.load_data(pdf_file)
            nodes = self._markdown_parser.get_nodes_from_documents(documents=documents)
            vector_index = VectorStoreIndex(nodes)
            vector_index.storage_context.persist(persist_dir=str(vector_index_persist_path))
            summary_index = SummaryIndex(nodes)
            summary_index.storage_context.persist(persist_dir=str(summary_index_persist_path))

        # define query engines
        vector_query_engine = vector_index.as_query_engine(similarity_top_k=self._chunks_top_k, node_postprocessors=[FullPagePostprocessor(docstore=vector_index.docstore)])
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")

        # extract a summary
        if not summary_path.exists():
            summary = (
                summary_query_engine.query(
                    "Extract a summary of this document"
                ).response
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary)
        else:
            summary = summary_path.read_text()

        # define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name=f"vector_tool_{pdf_file_stem_clean}",
                    description=f"Useful for questions related to specific facts",
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name=f"summary_tool_{pdf_file_stem_clean}",
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

    async def abuild_agent_per_doc(
            self,
            pdf_file: Path,
    ) -> tuple[ReActAgent, str]:
        pdf_file_stem_clean, vector_index_persist_path, summary_index_persist_path, summary_path = self._get_vector_summary_paths(pdf_file)
        if vector_index_persist_path.exists() and summary_index_persist_path.exists():
            vector_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(vector_index_persist_path)),
            )
            summary_index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=str(summary_index_persist_path)),
            )
        else:
            documents = await self._pdf_reader.aload_data(pdf_file)
            nodes = self._markdown_parser.get_nodes_from_documents(documents=documents)
            vector_index = VectorStoreIndex(nodes, use_async=True)
            vector_index.storage_context.persist(persist_dir=str(vector_index_persist_path))
            summary_index = SummaryIndex(nodes)
            summary_index.storage_context.persist(persist_dir=str(summary_index_persist_path))

        # define query engines
        vector_query_engine = vector_index.as_query_engine(similarity_top_k=self._chunks_top_k, node_postprocessors=[FullPagePostprocessor(docstore=vector_index.docstore)])
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")

        # extract a summary
        if not summary_path.exists():
            summary = (
                await summary_query_engine.aquery(
                    "Extract a summary of this document"
                )
            ).response
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary)
        else:
            summary = summary_path.read_text()

        # define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name=f"vector_tool_{pdf_file_stem_clean}",
                    description=f"Useful for questions related to specific facts",
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name=f"summary_tool_{pdf_file_stem_clean}",
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

    def build_agents(self) -> tuple[dict[str, ReActAgent], dict[str, dict]]:
        pdf_files = list(self._pdfs_dir.rglob("*.pdf"))

        agents_dict = {}
        extra_info_dict = {}

        for pdf_file in tqdm(pdf_files, desc="Building agents"):
            pdf_file_stem_clean, _, _, _ = self._get_vector_summary_paths(pdf_file)
            agent, summary = self.build_agent_per_doc(
                pdf_file=pdf_file,
            )
            agents_dict[pdf_file_stem_clean] = agent
            extra_info_dict[pdf_file_stem_clean] = {"summary": summary}

        return agents_dict, extra_info_dict

    async def abuild_agents(self) -> tuple[dict[str, ReActAgent], dict[str, dict]]:
        pdf_files = list(self._pdfs_dir.rglob("*.pdf"))

        jobs = [
            self.abuild_agent_per_doc(
                pdf_file=pdf_file,
            ) for pdf_file in pdf_files
        ]

        results = await run_jobs(
            jobs=jobs,
            show_progress=True,
            workers=self._num_workers,
            desc="Building agents",
        )

        agents_dict = {}
        extra_info_dict = {}

        for pdf_file, result in zip(pdf_files, results):
            pdf_file_stem_clean, _, _, _ = self._get_vector_summary_paths(pdf_file)
            agent, summary = result
            agents_dict[pdf_file_stem_clean] = agent
            extra_info_dict[pdf_file_stem_clean] = {"summary": summary}

        return agents_dict, extra_info_dict

    @cached_property
    def top_agent(self) -> ReActAgent:
        agents_dict, extra_info_dict = self.build_agents()
        if not agents_dict:
            raise ValueError("No agents were built.")

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
        vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=self._nodes_top_k)

        custom_obj_retriever = CustomObjectRetriever(
            vector_node_retriever,
            obj_index.object_node_mapping,
        )

        top_agent = ReActAgent.from_tools(
            tool_retriever=custom_obj_retriever,
            system_prompt="""
                    You are an agent designed to answer queries about the documentation.
                    Please always use the tools provided to answer a question. Do not rely on prior knowledge.
                    """,
            verbose=self._verbose,
            max_iterations=self._max_iterations,
        )

        return top_agent

    async def abuild_top_agent(self) -> ReActAgent:
        agents_dict, extra_info_dict = await self.abuild_agents()
        if not agents_dict:
            raise ValueError("No agents were built.")

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
        vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=self._nodes_top_k)

        custom_obj_retriever = CustomObjectRetriever(
            vector_node_retriever,
            obj_index.object_node_mapping,
        )

        top_agent = ReActAgent.from_tools(
            tool_retriever=custom_obj_retriever,
            system_prompt="""
                    You are an agent designed to answer queries about the documentation.
                    Please always use the tools provided to answer a question. Do not rely on prior knowledge.
                    """,
            verbose=self._verbose,
            max_iterations=self._max_iterations,
        )

        return top_agent

    def process_queries(self, queries: list[str]) -> list[Any]:
        top_agent = self.top_agent
        responses = list()
        for query in queries:
            try:
                response = top_agent.query(query)
                responses.append(response)
            except Exception as e:
                print(f"Query failed: {query}, with error: {e}")
        return responses

    async def aprocess_queries(self, queries: list[str]) -> list[Any]:
        top_agent = await self.abuild_top_agent()
        try:
            responses = [await top_agent.aquery(query) for query in queries]
        except Exception as e:
            raise Exception("Query failed") from e
        return responses
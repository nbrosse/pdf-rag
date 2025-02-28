import os
import re
from pathlib import Path
from typing import Literal

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.gemini import Gemini

from pdf_rag.evaluation import get_shell_evaluation_df
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
queries_responses_dir = opio_dir_path / "queries_responses_one_doc"

pdf_reader = PDFDirectoryReader(
    cache_dir=cache_dir,
    num_workers=32,
    root_dir=str(use_cases_path),
)


def get_queries_answers_page_numbers(
        case: Literal["Shell", "Thales", "Orano"],
        file: str,
) -> list[list[str, str, list[int]]]:
    match case:
        case "Shell":
            df = get_shell_evaluation_df(csv_path=opio_dir_path / "raw data" / "dataset evaluation - question_answers.csv")
            df = df.query(f"file == '{file}'")
            if len(df) == 0:
                raise ValueError(f"File {file} has no evaluation queries.")
            df = df[["question", "answer", "page_number"]].groupby(["question", "answer"], as_index=False).agg(list)
            return df.values.tolist()
        case "Thales":
            raise NotImplementedError("Thales not implemented.")
        case "Orano":
            raise NotImplementedError("Orano not implemented.")
        case _:
            raise Exception(f"Unknown case: {case}")


def query_one_doc(
        doc_path: Path | str,
        case: Literal["Shell", "Thales", "Orano"],
        queries_answers_page_numbers: list[list[str, str, list[int]]],
):
    doc_path = Path(doc_path)
    documents = pdf_reader.load_data(doc_path)
    document = documents[0]
    markdown_parser = MarkdownPageNodeParser()
    nodes = markdown_parser.get_nodes_from_documents(documents=documents)

    relative_path = Path(document.metadata["relative_path"])
    storage_name = relative_path.name
    storage_name = re.sub(r'[()\s]', '_', storage_name).replace(".pdf", "_vector")
    relative_path = relative_path.with_name(storage_name)
    storage_context_persist_dir = storage_dir / relative_path
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

    outputs_dir = queries_responses_dir / case / doc_path.stem
    outputs_dir.mkdir(parents=True, exist_ok=True)
    for i, query_answer_page_numbers in enumerate(queries_answers_page_numbers):
        query, answer, page_numbers = tuple(query_answer_page_numbers)
        output = list()
        retrieved_nodes = query_engine.retrieve(QueryBundle(query))
        response = query_engine.query(query)
        output.append(f"Query: {query}")
        output.append(f"Reference answer:\n{answer}")
        output.append(f"Page numbers:\n{page_numbers}")
        output.append(f"Predicted answer:\n{str(response)}")
        for node in retrieved_nodes:
            output.append(30 * "-")
            output.append(f"Score:\n{str(node.score)}")
            output.append(f"Page numbers:\n{str(node.metadata['page_number'])}")
            output.append(node.text)
        output_content = "\n".join(output)
        output_file = outputs_dir / f"query_{i}.md"
        output_file.write_text(output_content)


if __name__ == "__main__":
    case = "Shell"
    doc_path = use_cases_path / "Shell Dec 5 2024" / "shell-annual-report-2023.pdf"
    queries_answers_page_numbers = get_queries_answers_page_numbers(file=doc_path.name, case=case)
    query_one_doc(
        doc_path=doc_path,
        case=case,
        queries_answers_page_numbers=queries_answers_page_numbers,
    )

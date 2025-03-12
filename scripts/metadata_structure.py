import logging
import os
import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore
from omegaconf import OmegaConf, ValidationError

from pdf_rag.extractors import ContextExtractor, TableOfContentsExtractor, TableOfContentsCreator, \
    LandscapePagesExtractor, StructureExtractor
from pdf_rag.markdown_parsers import MarkdownLineNodeParser, MarkdownPageNodeParser
from pdf_rag.readers import PDFDirectoryReader

from dotenv import load_dotenv

from pdf_rag.transforms import ReformatMarkdownComponent
from pdf_rag.tree_index import TreeIndex, Neo4jConfig

load_dotenv()

logger = logging.getLogger("metadata_structure")
logging.basicConfig(level=logging.INFO)
root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)
neo4j_logger = logging.getLogger('neo4j')
neo4j_logger.setLevel(logging.ERROR)


@dataclass
class PipelineConfig:
    data_dir: str | Path
    pipeline_step: str = field(metadata={"choices": ["ingest", "index", "all"]})
    erase: bool = False
    api_key_gemini: str | None = None
    api_key_mistral: str | None = None
    num_workers: int = 32
    neo4j_uri: str | None = None
    neo4j_username: str | None = None
    neo4j_password: str | None = None
    neo4j_database: str | None = None
    neo4j_config: Neo4jConfig | None = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if self.neo4j_uri and self.neo4j_username and self.neo4j_password and self.neo4j_database:
            self.neo4j_config = Neo4jConfig(
                uri=self.neo4j_uri,
                username=self.neo4j_username,
                password=self.neo4j_password,
                database=self.neo4j_database,
            )
        elif any([self.neo4j_uri, self.neo4j_username, self.neo4j_password, self.neo4j_database]):
            raise ValueError("Neo4j URI, username, database and password must be provided together.")

        self.api_key_gemini = self.api_key_gemini or os.environ.get("GEMINI_API_KEY")
        self.api_key_mistral = self.api_key_mistral or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key_gemini:
            raise ValueError("Gemini API Key is required. Provide api_key_gemini or set GEMINI_API_KEY environment variable.")
        if not self.api_key_mistral:
            raise ValueError("Mistral API Key is required. Provide api_key_mistral or set MISTRAL_API_KEY environment variable.")

        self.root_dir = self.data_dir / "pdfs"
        self.pdfs_dir = self.data_dir / "pdfs"
        self.cache_dir = self.data_dir / "cache"
        self.storage_dir = self.data_dir / "storage_metadata"

        if self.erase and self.storage_dir.exists():
            logger.info(f"Removed existing storage directory: {self.storage_dir}")
            shutil.rmtree(self.storage_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if not self.pdfs_dir.exists():
            raise ValueError(f"{str(self.pdfs_dir)} does not exist")


def load_and_validate_config(config_path: str) -> PipelineConfig:
    try:
        config = OmegaConf.load(config_path)
        pipeline_config = OmegaConf.structured(PipelineConfig)(config)
        logger.info("Configuration loaded and validated successfully:")
        logger.info(pipeline_config)
        return pipeline_config
    except ValidationError as e:
        raise ValidationError(f"Validation error: {e}")
    except Exception as e:
        raise Exception(f"Error loading configuration: {e}")


def ingestion_pipeline(config: PipelineConfig) -> None:
    """Creates and runs the ingestion pipeline."""

    pdf_reader = PDFDirectoryReader(
        api_key_gemini=config.api_key_gemini,
        api_key_mistral=config.api_key_mistral,
        root_dir=str(config.pdfs_dir),
        cache_dir=str(config.cache_dir),
        num_workers=config.num_workers,
        show_progress=True,
    )

    processed_docstore = SimpleDocumentStore()
    documents = pdf_reader.load_data(config.pdfs_dir)

    pipeline = IngestionPipeline(
        transformations=[
            ReformatMarkdownComponent(api_key=config.api_key_gemini, model_name="gemini-2.0-flash", num_workers=config.num_workers),
            ContextExtractor(api_key=config.api_key_gemini, model_name="gemini-2.0-flash", num_workers=config.num_workers),
            TableOfContentsExtractor(api_key=config.api_key_gemini, model_name="gemini-2.0-flash", num_workers=config.num_workers),
            TableOfContentsCreator(api_key=config.api_key_gemini, model_name="gemini-2.0-flash", num_workers=config.num_workers),
            LandscapePagesExtractor(api_key=config.api_key_gemini, model_name="gemini-2.0-flash", num_workers=config.num_workers),
            StructureExtractor(api_key=config.api_key_gemini, model_name="gemini-2.0-flash", num_workers=config.num_workers),
        ],
        docstore=SimpleDocumentStore(),
        docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
    )

    pipeline_storage_path = config.storage_dir / "pipeline_storage"
    if pipeline_storage_path.exists():
        logger.info(f"Loading existing pipeline storage from: {str(pipeline_storage_path)}")
        pipeline.load(str(pipeline_storage_path))

    nodes = pipeline.run(documents=documents, show_progress=True)
    pipeline.persist(str(config.storage_dir / "pipeline_storage"))

    processed_docstore.add_documents(docs=nodes)
    logger.info(f"Number of docs after ingestion: {len(processed_docstore.docs)}")
    processed_docstore.persist(str(config.storage_dir / "processed_docstore_storage.json"))

    logger.info("Ingestion pipeline completed and persisted.")


def create_tree_index(config: PipelineConfig) -> None:
    """Creates the tree index from the persisted docstore."""
    processed_docstore_path = config.storage_dir / "processed_docstore_storage.json"
    if not processed_docstore_path.exists():
        raise ValueError(f"{str(processed_docstore_path)} does not exist")

    # Load the docstore
    docstore = SimpleDocumentStore.from_persist_path(str(processed_docstore_path))
    markdown_line_parser = MarkdownLineNodeParser()
    markdown_page_parser = MarkdownPageNodeParser(chunk_size=0)

    list_nodes = list()

    for _, doc in docstore.docs.items():
        format = doc.metadata.get("format", "")
        match doc.metadata["format"]:
            case "landscape":
                nodes = markdown_page_parser.get_nodes_from_documents(documents=[doc])
            case "portrait":
                nodes = markdown_line_parser.get_nodes_from_documents(documents=[doc])
            case _:
                raise ValueError(f"Format {format} is not supported")
        list_nodes.extend(nodes)

    list_nodes.extend(list(docstore.docs.values()))
    tree_index = TreeIndex(nodes=list_nodes)
    logger.info("Tree index created.")

    if config.neo4j_config:
        logger.info("Exporting to Neo4j")
        tree_index.export_to_neo4j(config.neo4j_config)


def main():
    parser = argparse.ArgumentParser(description="PDF RAG Metadata Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    pipeline_config = load_and_validate_config(args.config)

    if pipeline_config.pipeline_step in ["ingest", "all"]:
        ingestion_pipeline(
            config=pipeline_config,
        )
    if pipeline_config.pipeline_step in ["tree", "all"]:
        create_tree_index(
            config=pipeline_config,
        )
    if pipeline_config.pipeline_step not in ["index", "tree", "all"]:
        print("Invalid pipeline step. Choose 'ingest', 'index' or 'all'.")


if __name__ == "__main__":
    main()
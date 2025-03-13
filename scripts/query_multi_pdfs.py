import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf, ValidationError

from pdf_rag.react_agent_multi_pdfs import ReActAgentMultiPdfs
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ReActAgentConfig:
    data_dir: Path | str
    api_key_gemini: str | None = None
    api_key_mistral: str | None = None
    num_workers: int = 16
    chunks_top_k: int = 5
    nodes_top_k: int = 10
    max_iterations: int = 20
    verbose: bool = True
    queries: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.root_dir = self.data_dir / "pdfs"
        self.pdfs_dir = self.data_dir / "pdfs"
        self.cache_dir = self.data_dir / "cache"
        self.storage_dir = self.data_dir / "storage_queries"

        self.api_key_gemini = self.api_key_gemini or os.environ.get("GEMINI_API_KEY")
        self.api_key_mistral = self.api_key_mistral or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key_gemini:
            raise ValueError(
                "Gemini API Key is required. Provide api_key_gemini or set GEMINI_API_KEY environment variable."
            )
        if not self.api_key_mistral:
            raise ValueError(
                "Mistral API Key is required. Provide api_key_mistral or set MISTRAL_API_KEY environment variable."
            )


def load_and_validate_config(config_path: str) -> ReActAgentConfig:
    try:
        config = OmegaConf.load(config_path)
        react_agent_schema = OmegaConf.structured(ReActAgentConfig)  # (**config)
        react_agent_config = OmegaConf.merge(react_agent_schema, config)
        react_agent_config = ReActAgentConfig(**react_agent_config)
        print("Configuration loaded and validated successfully:")
        print(str(react_agent_config))
        return react_agent_config
    except ValidationError as e:
        raise ValidationError(f"Validation error: {e}")
    except Exception as e:
        raise Exception(f"Error loading configuration: {e}")


def main():
    parser = argparse.ArgumentParser(description="ReAct Agent RAG Multi PDFs")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = load_and_validate_config(args.config)

    react_agent_multi_pdfs = ReActAgentMultiPdfs(
        api_key_gemini=config.api_key_gemini,
        api_key_mistral=config.api_key_mistral,
        root_dir=config.root_dir,
        pdfs_dir=config.pdfs_dir,
        cache_dir=config.cache_dir,
        storage_dir=config.storage_dir,
        num_workers=config.num_workers,
        chunks_top_k=config.chunks_top_k,
        nodes_top_k=config.nodes_top_k,
        max_iterations=config.max_iterations,
        verbose=config.verbose,
    )

    responses = react_agent_multi_pdfs.process_queries(queries=config.queries)
    for q, r in zip(config.queries, responses):
        print(30 * "-")
        print(f"Query: {q}")
        print(f"Response: {r}")


if __name__ == "__main__":
    main()

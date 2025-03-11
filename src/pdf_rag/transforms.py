import os
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Sequence, Any

from google.generativeai import GenerativeModel
from llama_index.core.async_utils import run_jobs, asyncio_run
from llama_index.core.schema import TransformComponent, BaseNode
from pydantic import Field, PrivateAttr, field_validator
from google import genai

from pdf_rag.config import jinja2_env
from pdf_rag.utils import postprocess_markdown_output


class GeminiTransformComponent(TransformComponent):
    api_key: str = Field(
        default="",
        description="Google API key for Gemini",
        validate_default=True,
    )
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model name to use"
    )

    _model: GenerativeModel = PrivateAttr(default=None, init=False)
    _client: genai.Client = PrivateAttr(default=None, init=False)

    @field_validator("api_key", mode="before", check_fields=True)
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate the API key."""
        if not v:
            api_key = os.getenv("GEMINI_API_KEY", None)
            if api_key is None:
                raise ValueError("The API key is required.")
            return api_key
        return v

    def model_post_init(self, __context: Any) -> None:
        self._client = genai.Client(api_key=self.api_key)

    @abstractmethod
    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """Transform nodes."""

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Async transform nodes."""
        return self.__call__(nodes, **kwargs)


class ReformatMarkdownComponent(GeminiTransformComponent):

    max_iters: int = Field(
        default=50,
        description="Maximum number of iterations for reformatting markdown document.",
    )
    in_place: bool = Field(
        default=True, description="Whether to process nodes in place."
    )
    num_workers: int = Field(
        default=4,
        description="Number of workers to use for concurrent async processing.",
    )
    show_progress: bool = Field(default=True, description="Whether to show progress.")

    async def _aprocess_node(
        self,
        node: BaseNode,
    ) -> BaseNode:
        format = node.metadata.get("format", "")
        # if format != "portrait":
        #     return node
        reformatted = node.metadata.get("is_reformatted", False)
        if reformatted:
            return node
        else:
            cache_dir = node.metadata.get("cache_dir", "")
            relative_path = node.metadata.get("relative_path", "")
            if cache_dir and relative_path:
                cache_dir = Path(cache_dir)
                output_file = cache_dir / f"{relative_path}.reformatted.md"
            else:
                output_file = None
            if output_file and output_file.exists():
                transcription = output_file.read_text()
            else:
                template = jinja2_env.get_template("reformat_markdown.jinja2")
                content = node.text
                transcription = ""
                response = ""
                i = 0
                while "<end>" not in response and i < self.max_iters:
                    request = template.render(document=content, processed=transcription, landscape=(format == "landscape"))
                    print(request)
                    response = await self._client.aio.models.generate_content(
                        contents=request,
                        model=self.model_name,
                    )
                    response = response.text
                    transcription += response.replace("<end>", "")
                    i += 1
                if i == self.max_iters:
                    raise RuntimeError("Maximum number of iterations reached.")
                transcription = postprocess_markdown_output(transcription)
            node.set_content(transcription)
            node.metadata["is_reformatted"] = True
            if output_file and not output_file.exists():
                output_file.write_text(transcription)
            return node

    async def aprocess_nodes(
        self,
        nodes: Sequence[BaseNode],
    ) -> list[BaseNode]:
        if self.in_place:
            new_nodes = nodes
        else:
            new_nodes = [deepcopy(node) for node in nodes]
        jobs = [self._aprocess_node(node) for node in new_nodes]
        processed_nodes: list[BaseNode] = await run_jobs(
            jobs, show_progress=self.show_progress, workers=self.num_workers
        )
        return processed_nodes

    def process_nodes(
        self,
        nodes: Sequence[BaseNode],
    ) -> list[BaseNode]:
        return asyncio_run(self.aprocess_nodes(nodes))

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        """Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process
        """
        return self.process_nodes(nodes)

    async def acall(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        """Post process nodes parsed from documents.

        Allows extractors to be chained.

        Args:
            nodes (List[BaseNode]): nodes to post-process
        """
        return await self.aprocess_nodes(nodes)

import json
import os
import re
from abc import abstractmethod
from pathlib import Path
from typing import Any, Sequence

from google.genai import types
from llama_index.core.async_utils import run_jobs
from llama_index.core.extractors import BaseExtractor
from llama_index.core.schema import BaseNode, MetadataMode
from pydantic import Field, field_validator, PrivateAttr
from google import genai

from pdf_rag.config import jinja2_env


class GeminiBaseExtractor(BaseExtractor):
    api_key: str = Field(
        default="",
        description="Google API key for Gemini",
        validate_default=True,
    )
    temperature: float = Field(
        default=0.1,
        description="Temperature of gemini",
    )
    model_name: str = Field(default="gemini-2.0-flash", description="Gemini model name to use")

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

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "GeminiMetadataExtractor"

    @abstractmethod
    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict]:
        """Extracts metadata for a sequence of nodes, returning a list of
        metadata dictionaries corresponding to each node.

        Args:
            nodes (Sequence[Document]): nodes to extract metadata from

        """
        pass


class ContextExtractor(GeminiBaseExtractor):
    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "ContextExtractor"

    async def _aextract_context_from_node(self, node: BaseNode) -> dict[str, Any]:
        context_str = node.get_content(metadata_mode=MetadataMode.NONE)
        template = jinja2_env.get_template("extract_context.jinja2")
        request = template.render(document=context_str)
        context = await self._client.aio.models.generate_content(
            contents=request, model=self.model_name, config=types.GenerateContentConfig(temperature=self.temperature)
        )
        pattern = r"```json\n(.*?)```"
        match = re.search(pattern, context.text, re.DOTALL)
        if match:
            json_content = match.group(1).strip()
            context_dict = json.loads(json_content)
        else:
            raise ValueError("Could not extract context")
        return context_dict

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, Any]]:
        jobs = [self._aextract_context_from_node(node) for node in nodes]
        contexts = await run_jobs(jobs, show_progress=self.show_progress, workers=self.num_workers)
        return contexts


class TableOfContentsExtractor(GeminiBaseExtractor):
    head_pages: int = Field(
        default=10,
        description="Number of head pages considered for extracting table of contents.",
    )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TableOfContentsExtractor"

    async def _aextract_toc_from_node(self, node: BaseNode) -> str:
        format = node.metadata.get("format", "")
        assert format, "metadata format must be portrait or landscape."
        context_str = node.get_content(metadata_mode=MetadataMode.NONE)
        context_str = context_str.split("--- end page 10")[0]
        template = jinja2_env.get_template("extract_toc.jinja2")
        request = template.render(doc=context_str, format=format)
        toc = await self._client.aio.models.generate_content(
            contents=request, model=self.model_name, config=types.GenerateContentConfig(temperature=self.temperature)
        )
        response = toc.text.strip()
        if response == "<none>":
            response = ""
        return response

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, str]]:
        jobs = [self._aextract_toc_from_node(node) for node in nodes]
        tocs: list[str] = await run_jobs(jobs, show_progress=self.show_progress, workers=self.num_workers)
        return [{"extracted_toc": toc} for toc in tocs]


class TableOfContentsCreator(GeminiBaseExtractor):
    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TableOfContentsCreator"

    @staticmethod
    def _extract_markdown_toc(content: str, file_stem: str) -> str:
        lines = content.splitlines()
        code_block = False
        toc = list()
        for line_number, line in enumerate(lines):
            # Track if we're inside a code block to avoid parsing headers in code
            if line.lstrip().startswith("```"):
                code_block = not code_block
                continue
            # Only parse headers if we're not in a code block
            if not code_block:
                header_match = re.match(r"^(#+)\s(.*)", line)
                if header_match:
                    toc.append(f"{line.strip()} [line {line_number}]")
        if not toc:
            return ""
        if "[line 0]" not in toc[0]:
            toc.insert(0, f"# {file_stem} [line 0]")
        return "\n".join(toc)

    async def _aextract_toc_from_node(self, node: BaseNode) -> str:
        format = node.metadata.get("format", "")
        filename = node.metadata.get("filename", "")
        assert filename
        if format == "portrait":
            is_reformatted = node.metadata.get("is_reformatted", False)
            if not is_reformatted:
                raise Exception(f"The file {filename} must be reformatted.")
            file_stem = Path(filename).stem
            toc = self._extract_markdown_toc(
                content=node.text,
                file_stem=file_stem,
            )
        elif format == "landscape":
            context_str = node.get_content(metadata_mode=MetadataMode.NONE)
            template_create = jinja2_env.get_template("create_landscape_toc.jinja2")
            template_check = jinja2_env.get_template("check_landscape_toc.jinja2")
            request = template_create.render(doc=context_str)
            draft_toc = await self._client.aio.models.generate_content(
                contents=request,
                model=self.model_name,
                config=types.GenerateContentConfig(temperature=self.temperature),
            )
            draft_toc = draft_toc.text.strip()
            request = template_check.render(toc=draft_toc)
            toc = await self._client.aio.models.generate_content(
                contents=request,
                model=self.model_name,
                config=types.GenerateContentConfig(temperature=self.temperature),
            )
            toc = toc.text.strip()
        else:
            raise ValueError(f"Format {format} must be portrait or landscape.")
        return toc

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, str]]:
        jobs = [self._aextract_toc_from_node(node) for node in nodes]
        tocs: list[str] = await run_jobs(jobs, show_progress=self.show_progress, workers=self.num_workers)
        return [{"created_toc": toc} for toc in tocs]


class LandscapePagesExtractor(GeminiBaseExtractor):
    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "LandscapePagesExtractor"

    async def _aextract_pages_from_node(self, node: BaseNode) -> str:
        format = node.metadata.get("format", "")
        if format != "landscape":
            return ""
        extracted_toc = node.metadata.get("extracted_toc", "")
        created_toc = node.metadata.get("created_toc", "")
        toc = extracted_toc if extracted_toc else created_toc
        if not toc:
            raise ValueError("extracted_toc or created_toc are required.")
        context_str = node.get_content(metadata_mode=MetadataMode.NONE)
        template = jinja2_env.get_template("extract_landscape_pages.jinja2")
        request = template.render(doc=context_str, toc=toc)
        pages = await self._client.aio.models.generate_content(
            contents=request, model=self.model_name, config=types.GenerateContentConfig(temperature=self.temperature)
        )
        response = pages.text
        return response

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, str]]:
        jobs = [self._aextract_pages_from_node(node) for node in nodes]
        list_pages: list[str] = await run_jobs(jobs, show_progress=self.show_progress, workers=self.num_workers)
        return [{"pages": pages} for pages in list_pages]


class StructureExtractor(GeminiBaseExtractor):
    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "StructureExtractor"

    # @retry(wait=wait_fixed(15), retry=retry_if_exception_type(ResourceExhausted))
    async def _aextract_structure_from_node(self, node: BaseNode) -> str:
        format = node.metadata.get("format", "")
        filename = node.metadata.get("filename", "")
        assert filename
        if format == "portrait":
            toc = node.metadata.get("created_toc", "")
            return toc
        elif format == "landscape":
            extracted_toc = node.metadata.get("extracted_toc", "")
            created_toc = node.metadata.get("created_toc", "")
            toc = extracted_toc if extracted_toc else created_toc
            if not toc:
                raise ValueError("extracted_toc or created_toc are required.")
            pages = node.metadata.get("pages", "")
            if not pages:
                raise ValueError(f"pages are required for file {filename}.")
            template = jinja2_env.get_template("structure_landscape_pages.jinja2")
            request = template.render(toc=toc, pages=pages)
            structure = await self._client.aio.models.generate_content(
                contents=request,
                model=self.model_name,
                config=types.GenerateContentConfig(temperature=self.temperature),
            )
            response = structure.text.strip()
            return response
        else:
            raise ValueError(f"Format {format} must be portrait or landscape for file {filename}.")

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, str]]:
        jobs = [self._aextract_structure_from_node(node) for node in nodes]
        structures: list[str] = await run_jobs(jobs, show_progress=self.show_progress, workers=self.num_workers)
        return [{"structure": structure} for structure in structures]

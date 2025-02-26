import base64
import logging
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Optional, List, Union, Any

import re

import tenacity
from google.genai import Client, types
from google.genai.errors import ServerError
from jinja2 import Template
from llama_index.core import Document
from llama_index.core.async_utils import run_jobs, asyncio_run

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.readers.file.base import get_default_fs, _format_file_timestamp
from mistralai import Mistral, SDKError
from pdf2image import convert_from_bytes
from pydantic import Field, field_validator, PrivateAttr
import os
from google import genai
from pypdf import PdfReader, PdfWriter

from dotenv import load_dotenv
from tenacity import wait_fixed, retry_if_exception_type, stop_after_attempt

from pdf_rag.config import jinja2_env

load_dotenv()

logger = logging.getLogger(__name__)

WAIT_SECONDS: int = 5

# Asyncio error messages
nest_asyncio_err = "cannot be called from a running event loop"
nest_asyncio_msg = "The event loop is already running. Add `import nest_asyncio; nest_asyncio.apply()` to your code to fix this issue."

FileInput = Path | str


def _postprocess_markdown_output(response: str) -> str:
    pattern = r"```markdown\s*(.*?)(```|$)"
    try:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return response
    except Exception as e:
        return "GEMINI ERROR PARSING"


class VLMPDFReader(BasePydanticReader):
    """PDF reader that uses VLM API for processing."""

    api_key_gemini: str = Field(
        default="",
        description="Google API key for Gemini",
        validate_default=True,
    )
    api_key_mistral: str = Field(
        default="",
        description="Mistral API key",
        validate_default=True,
    )
    model_name_gemini: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model name to use"
    )
    model_name_mistral: str = Field(
        default="pixtral-large-latest",
        description="Mistral model name to use"
    )
    num_workers: int = Field(
        default=4,
        gt=0,
        lt=64,
        description="The number of workers to use sending API requests for parsing.",
    )
    show_progress: bool = Field(
        default=True, description="Show progress"
    )
    cache_dir: str | Path = Field(
        default="",
        description="Cache directory for files.",
        validate_default=True,
    )

    _client_gemini: Client = PrivateAttr(default=None, init=False)
    _client_mistral: Mistral = PrivateAttr(default=None, init=False)
    _template: Template = PrivateAttr(default=None, init=False)

    @field_validator("cache_dir", mode="after", check_fields=True)
    @classmethod
    def validate_cache_dir(cls, value: str) -> Path:
        if not value:
            raise ValueError("Cache directory cannot be empty")
        cache_dir = Path(value)
        if not cache_dir.exists():
            raise ValueError(f"Cache directory {str(cache_dir)} does not exist.")
        return cache_dir

    @field_validator("api_key_gemini", mode="before", check_fields=True)
    @classmethod
    def validate_api_key_gemini(cls, v: str) -> str:
        """Validate the API key."""
        if not v:
            api_key = os.getenv("GEMINI_API_KEY", None)
            if api_key is None:
                raise ValueError("The API key is required.")
            return api_key
        return v

    @field_validator("api_key_mistral", mode="before", check_fields=True)
    @classmethod
    def validate_api_key_mistral(cls, v: str) -> str:
        """Validate the API key."""
        if not v:
            api_key = os.getenv("MISTRAL_API_KEY", None)
            if api_key is None:
                raise ValueError("The API key is required.")
            return api_key
        return v

    def model_post_init(self, __context: Any) -> None:
        self._client_gemini = genai.Client(api_key=self.api_key_gemini)
        self._client_mistral = Mistral(api_key=self.api_key_mistral)
        self._template = jinja2_env.get_template("pdf_to_markdown.jinja2")

    def load_data(
        self,
        file: list[FileInput] | FileInput,
        extra_info: list[dict] | dict | None = None,
    ) -> list[Document]:
        """Load PDF data into Document objects."""
        try:
            return asyncio_run(self.aload_data(file, extra_info))
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    @tenacity.retry(wait=wait_fixed(WAIT_SECONDS), retry=retry_if_exception_type((SDKError, ServerError)), reraise=True, stop=stop_after_attempt(10))
    async def _aload_page(
        self,
        reader: PdfReader,
        page_num: int,
    ):
        page = reader.pages[page_num]
        writer = PdfWriter()
        writer.add_page(page)
        with BytesIO() as bytes_stream:
            writer.write(bytes_stream)
            bytes_stream.seek(0)
            page_content = bytes_stream.read()

        # Convert PDF to image
        images = convert_from_bytes(page_content)
        image = images[0]

        # Save the image to a bytes buffer
        with BytesIO() as image_bytes_stream:
            image.save(image_bytes_stream, format="PNG")
            image_bytes_stream.seek(0)
            image_data = image_bytes_stream.read()

        # Encode the image data
        image_data_base64 = base64.b64encode(image_data).decode("utf-8")

        prompt = self._template.render()
        contents = [
            types.Part.from_bytes(
                data=page_content,
                mime_type="application/pdf",
            ),
            prompt,
        ]
        response = await self._client_gemini.aio.models.generate_content(
            model=self.model_name_gemini,
            contents=contents,
        )
        finish_reason = response.candidates[0].finish_reason
        match finish_reason:
            case "RECITATION":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{image_data_base64}",
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },

                        ]
                    }
                ]
                response = await self._client_mistral.chat.complete_async(
                    model=self.model_name_mistral,
                    messages=messages
                )
                return response.choices[0].message.content
            case _:  # "STOP":
                return response.text
            # case _:
            #     image.save(f"image_{page_num}.png")
            #     print(response.text)
            #     raise RuntimeError(f"Unknown finish reason: {finish_reason}")

    async def _aload_data(
        self,
        file_path: FileInput,
        extra_info: dict | None = None,
        show_progress: bool = False,
    ) -> Document:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {str(file_path)} does not exist")
        if file_path.suffix != ".pdf":
            raise ValueError(f"File {str(file_path)} is not a PDF")
        if extra_info and "relative_path" in extra_info:
            relative_path = extra_info["relative_path"]
        else:
            relative_path = file_path.name
        output_file_path = self.cache_dir / f"{relative_path}.md"
        reader = PdfReader(str(file_path))
        nb_pages = reader.get_num_pages()
        metadata = {"nb_pages": nb_pages, "filename": file_path.name}
        if extra_info:
            metadata.update(extra_info)
        if output_file_path.exists():
            content = output_file_path.read_text()
            return Document(text=content, metadata=metadata)
        jobs = [
            self._aload_page(
                reader=reader,
                page_num=page_num,
            ) for page_num in range(nb_pages)
        ]
        try:
            results = await run_jobs(
                jobs,
                workers=self.num_workers,
                desc=f"Parsing file {file_path.name}",
                show_progress=show_progress,
            )
            results = [_postprocess_markdown_output(r) for r in results]
            transcription = "".join([f"{r}\n\n--- end page {i + 1}\n\n" for i, r in enumerate(results)])
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            output_file_path.write_text(transcription)
            return Document(text=transcription, metadata=metadata)
        except RuntimeError as e:
            if nest_asyncio_err in str(e):
                raise RuntimeError(nest_asyncio_msg)
            else:
                raise e

    async def aload_data(
        self,
        file_path: list[FileInput] | FileInput,
        extra_info: list[dict] | dict | None = None,
    ) -> List[Document]:
        """Load data from the input path."""
        if isinstance(file_path, (str, Path)):
            doc = await self._aload_data(file_path, extra_info=extra_info, show_progress=self.show_progress)
            return [doc]
        elif isinstance(file_path, list):
            if extra_info and len(extra_info) != len(file_path):
                raise ValueError(f"extra_info (length {len(extra_info)} and file_path (length {len(file_path)} must have same length")
            if extra_info is None:
                extra_info = [None] * len(file_path)
            jobs = [
                self._aload_data(
                    f,
                    extra_info=info,
                    show_progress=False,
                )
                for info, f in zip(extra_info, file_path)
            ]
            try:
                results = await run_jobs(
                    jobs,
                    workers=self.num_workers,
                    desc="Parsing files",
                    show_progress=self.show_progress,
                )
                return results
            except RuntimeError as e:
                if nest_asyncio_err in str(e):
                    raise RuntimeError(nest_asyncio_msg)
                else:
                    raise e
        else:
            raise ValueError(
                "The input file_path must be a string or a list of strings."
            )


class PDFDirectoryReader(BasePydanticReader):

    root_dir: str = Field(
        default="",
        description="Root directory of PDF files",
        validate_default=True,
    )
    cache_dir: str | Path = Field(
        default="",
        description="Cache directory for files.",
        validate_default=True,
    )
    num_workers: int = Field(
        default=4,
        gt=0,
        lt=64,
        description="The number of workers to use sending API requests for parsing.",
    )
    show_progress: bool = Field(
        default=True,
        description="Show progress"
    )

    _vlm_reader: VLMPDFReader = PrivateAttr(default=None, init=False)

    @field_validator("cache_dir", mode="after", check_fields=True)
    @classmethod
    def validate_cache_dir(cls, value: str) -> Path:
        if not value:
            raise ValueError("Cache directory cannot be empty")
        cache_dir = Path(value)
        if not cache_dir.exists():
            raise ValueError(f"Cache directory {str(cache_dir)} does not exist.")
        return cache_dir

    @field_validator("root_dir", mode="after", check_fields=True)
    @classmethod
    def validate_root_dir(cls, value: str) -> Path:
        if not value:
            raise ValueError("Root directory cannot be empty")
        root_dir = Path(value)
        if not root_dir.exists():
            raise ValueError(f"Root directory {str(root_dir)} does not exist.")
        return root_dir

    def model_post_init(self, __context: Any) -> None:
        self._vlm_reader = VLMPDFReader(
            cache_dir=self.cache_dir,
            num_workers=self.num_workers,
            show_progress=self.show_progress,
        )

    def _file_metadata_func(self, file_path: str | Path) -> dict:
        fs = get_default_fs()
        file_path = str(file_path)
        stat_result = fs.stat(file_path)

        try:
            file_name = os.path.basename(str(stat_result["name"]))
        except Exception as e:
            file_name = os.path.basename(file_path)

        creation_date = _format_file_timestamp(stat_result.get("created"))
        last_modified_date = _format_file_timestamp(stat_result.get("mtime"))
        last_accessed_date = _format_file_timestamp(stat_result.get("atime"))
        default_meta = {
            "file_path": file_path,
            "file_name": file_name,
            "file_type": mimetypes.guess_type(file_path)[0],
            "file_size": stat_result.get("size"),
            "creation_date": creation_date,
            "last_modified_date": last_modified_date,
            "last_accessed_date": last_accessed_date,
        }
        relative_path = Path(file_path).relative_to(self.root_dir)
        default_meta["relative_path"] = str(relative_path)

        # Return not null value
        return {
            meta_key: meta_value
            for meta_key, meta_value in default_meta.items()
            if meta_value is not None
        }

    def _pre_load_data(self, input_dir: str | Path) -> tuple[list[Path], list[dict]]:
        input_dir = Path(input_dir)
        if not input_dir.is_relative_to(self.root_dir):
            raise ValueError(f"Input directory {str(input_dir)} is not relative to root directory {str(self.root_dir)}")
        pdfs_files = list(input_dir.rglob("*.pdf"))
        extra_info = [self._file_metadata_func(file_path=pdf_file) for pdf_file in pdfs_files]
        return pdfs_files, extra_info

    def load_data(self, input_dir: str | Path) -> list[Document]:
        input_dir = Path(input_dir)
        if not input_dir.is_relative_to(self.root_dir):
            raise ValueError(f"Input directory {str(input_dir)} is not relative to root directory {str(self.root_dir)}")
        pdfs_files = list(input_dir.rglob("*.pdf"))
        if len(pdfs_files) == 0:
            return list()
        extra_info = [self._file_metadata_func(file_path=pdf_file) for pdf_file in pdfs_files]
        documents = self._vlm_reader.load_data(pdfs_files, extra_info=extra_info)
        return documents

    async def aload_data(self, input_dir: str | Path) -> list[Document]:
        pdfs_files, extra_info = self._pre_load_data(input_dir)
        if len(pdfs_files) == 0:
            return list()
        else:
            return await self._vlm_reader.aload_data(pdfs_files, extra_info=extra_info)
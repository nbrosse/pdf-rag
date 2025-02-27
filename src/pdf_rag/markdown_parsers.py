import re
from typing import Any

from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, TextNode, MetadataMode
from pydantic import Field, PrivateAttr


class MarkdownPageNodeParser(MarkdownNodeParser):

    chunk_size: int = Field(
        default=1024,
        description="The token chunk size for each chunk.",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=0,
        description="The token overlap of each chunk when splitting.",
        ge=0,
    )

    _sentence_splitter: SentenceSplitter = PrivateAttr(default=False, init=False)

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = 1024,
        chunk_overlap: int = 0,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: CallbackManager | None = None,
    ) -> "MarkdownPageNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    def model_post_init(self, __context: Any) -> None:
        self._sentence_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def get_nodes_from_node(self, node: BaseNode) -> list[TextNode]:
        """Get nodes from document by splitting on pages."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        split_pattern = r"--- end page \d+"
        page_number_pattern = r"--- end page (\d+)"
        pages = re.split(split_pattern, text)
        page_numbers = re.findall(page_number_pattern, text)
        if not pages[-1].strip():
            pages.pop()
        assert len(page_numbers) == len(pages), f"Page numbers {len(page_numbers)} and pages {len(pages)} do not match"
        page_numbers_split_pages = [(page_number, split_page) for page_number, page in zip(page_numbers, pages) for split_page in self._sentence_splitter.split_text(page)]
        markdown_nodes = [
            self._build_node_from_split_with_page(text_split=split_page, node=node, page_number=int(page_number))
            for (page_number, split_page) in page_numbers_split_pages
        ]
        return markdown_nodes

    def _build_node_from_split_with_page(
        self,
        text_split: str,
        node: BaseNode,
        page_number: int,
    ) -> TextNode:
        """Build node from single text split."""
        node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]
        if self.include_metadata:
            node.metadata["page_number"] = page_number
        return node

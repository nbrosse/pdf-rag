import re
from typing import Any

from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, TextNode, MetadataMode
from pydantic import Field, PrivateAttr


class MarkdownLineNodeParser(MarkdownNodeParser):

    @classmethod
    def from_defaults(
            cls,
            include_metadata: bool = True,
            include_prev_next_rel: bool = True,
            callback_manager: CallbackManager | None = None,
    ) -> "MarkdownLineNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        return cls(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    def get_nodes_from_node(self, node: BaseNode) -> list[TextNode]:
        """Get nodes from document by splitting on headers."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        markdown_nodes = []
        lines = text.splitlines()
        current_section = ""
        current_line_number = 0
        # Keep track of headers at each level
        header_stack: list[str] = []
        code_block = False

        for line_number, line in enumerate(lines):
            # Track if we're inside a code block to avoid parsing headers in code
            if line.lstrip().startswith("```"):
                code_block = not code_block
                current_section += line + "\n"
                continue
            # Only parse headers if we're not in a code block
            if not code_block:
                header_match = re.match(r"^(#+)\s(.*)", line)
                if header_match:
                    # Save the previous section before starting a new one
                    if current_section.strip():
                        markdown_nodes.append(
                            self._build_node_from_split_with_line(
                                text_split=current_section.strip(),
                                node=node,
                                header_path="/".join(header_stack[:-1]) if header_stack else "",
                                line_number=current_line_number,
                            )
                        )

                    level = len(header_match.group(1))
                    header_text = header_match.group(2)

                    # Pop headers of equal or higher level
                    while header_stack and len(header_stack) >= level:
                        header_stack.pop()

                    # Add the new header
                    header_stack.append(header_text)
                    current_section = "#" * level + f" {header_text}\n"
                    current_line_number = line_number
                    continue

            current_section += line + "\n"

        # Add the final section
        if current_section.strip():
            markdown_nodes.append(
                self._build_node_from_split_with_line(
                    text_split=current_section.strip(),
                    node=node,
                    header_path="/".join(header_stack[:-1]) if header_stack else "",
                    line_number=current_line_number,
                )
            )

        return markdown_nodes

    def _build_node_from_split_with_line(
            self,
            text_split: str,
            node: BaseNode,
            header_path: str,
            line_number: int,
    ) -> TextNode:
        """Build node from single text split."""
        node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]
        if self.include_metadata:
            node.metadata["header_path"] = (
                "/" + header_path + "/" if header_path else "/"
            )
            node.metadata["line_number"] = line_number
        return node


class MarkdownPageNodeParser(MarkdownNodeParser):

    chunk_size: int = Field(
        default=1024,
        description="The token chunk size for each chunk. 0 to deactivate.",
    )
    chunk_overlap: int = Field(
        default=0,
        description="The token overlap of each chunk when splitting.",
        ge=0,
    )

    _sentence_splitter: SentenceSplitter | None = PrivateAttr(default=False, init=False)

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
        if self.chunk_size > 0:
            self._sentence_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        else:
            self._sentence_splitter = None

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
        if self._sentence_splitter:
            page_numbers_split_pages = [(page_number, split_page) for page_number, page in zip(page_numbers, pages) for split_page in self._sentence_splitter.split_text(page)]
        else:
            page_numbers_split_pages = list(zip(page_numbers, pages))
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

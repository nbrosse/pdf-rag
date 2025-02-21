import re

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, TextNode, MetadataMode


class MarkdownPageNodeParser(MarkdownNodeParser):
    def get_nodes_from_node(self, node: BaseNode) -> list[TextNode]:
        """Get nodes from document by splitting on pages."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        split_pattern = r"--- end page \d+"
        # Split the string
        pages = re.split(split_pattern, text)
        # Regex pattern to extract page numbers
        page_number_pattern = r"--- end page (\d+)"
        page_numbers = re.findall(page_number_pattern, text)
        if not pages[-1].strip():
            pages.pop()
        assert len(page_numbers) == len(pages)
        markdown_nodes = [
            self._build_node_from_split_with_page(text_split=page, node=node, page_number=int(page_number))
            for page_number, page in zip(page_numbers, pages)
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

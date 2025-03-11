from collections import deque
from pathlib import Path
import logging
from typing import Iterator, Self
import re

from llama_index.core.schema import BaseNode

logger = logging.getLogger(__name__)

DOCUMENT_NODE_NUMBER: int = -1


class TreeNode:
    def __init__(self, name: str, number: int | None = None):
        self.name = name
        self.number = number
        self.children: list[Self] = []
        self.parent: Self | None = None

    def add_child(self, child: Self) -> None:
        self.children.append(child)

    def set_parent(self, parent: Self) -> None:
        if self.parent is not None:
            raise ValueError("parent has already been set")
        else:
            self.parent = parent

    def remove_parent(self) -> None:
        self.parent = None

    def __str__(self, level: int = 0) -> str:
        ret = "  " * level + self.name
        if self.number is not None:
            ret += f" [{self.number}]"
        ret += "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret

    def bfs(self) -> Iterator[Self]:
        """Perform Breadth-First traversal of the tree."""
        queue = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def remove_child(self, child: Self) -> bool:
        if child in self.children:
            child.remove_parent()
            self.children.remove(child)
            return True
        return False

    def __iter__(self):
        return self.bfs()


def parse_landscape_structure(document: BaseNode) -> TreeNode:
    page_pattern = re.compile(r'^-\s*Page\s+(\d+)\s*:\s*(.+)$')
    header_pattern = re.compile(r'^(#+)\s+(.+)$')
    format = document.metadata.get("format", "")
    if format != "landscape":
        raise ValueError(f"Unsupported format {format}")
    number_pages = document.metadata.get("nb_pages", None)
    structure = document.metadata.get("structure", "")
    filename = document.metadata.get("filename", "")
    assert number_pages and structure and filename
    lines = structure.splitlines()
    filestem = Path(filename).stem
    root = TreeNode(name=filestem, number=DOCUMENT_NODE_NUMBER)
    stack = [(root, 0)]  # (node, level) pairs
    abstract_node_number = DOCUMENT_NODE_NUMBER - 1
    processed_page_numbers = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if it's a header
        header_match = header_pattern.match(line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            new_node = TreeNode(name=title, number=abstract_node_number)
            abstract_node_number -= 1
            # Adjust stack for header level
            while stack and stack[-1][1] >= level:
                stack.pop()
            if stack:
                stack[-1][0].add_child(new_node)
                new_node.set_parent(stack[-1][0])
            stack.append((new_node, level))
            continue
        # Check if it's a page entry
        page_match = page_pattern.match(line)
        if page_match:
            page_num = int(page_match.group(1))
            title = page_match.group(2).strip()
            if page_num in processed_page_numbers:
                logger.warning(f"Page {page_num} already processed. Skipping {title}.")
            elif page_num > number_pages:
                logger.warning(f"Page number {page_num} is greater than the number of pages in the document. Skipping {title}.")
            else:
                processed_page_numbers.add(page_num)
                new_node = TreeNode(name=title, number=page_num)
                # Add to last header in stack
                if stack:
                    stack[-1][0].add_child(new_node)
                    new_node.set_parent(stack[-1][0])

    leftout_page_numbers = set(range(1, number_pages + 1)) - processed_page_numbers
    if leftout_page_numbers:
        logger.warning(f"Page numbers {leftout_page_numbers} are not processed.")
        uncategorized_node = TreeNode(name="Uncategorized", number=abstract_node_number)
        abstract_node_number -= 1
        root.add_child(uncategorized_node)
        uncategorized_node.set_parent(root)
        for page_num in leftout_page_numbers:
            new_node = TreeNode(name=f"Page number {page_num}", number=page_num)
            uncategorized_node.add_child(new_node)
            new_node.set_parent(uncategorized_node)

    return root


def parse_portrait_structure(document: BaseNode) -> TreeNode:
    header_pattern = re.compile(r'(#+)\s+(.*?)\s+\[line\s+(\d+)\]')
    format = document.metadata.get("format", "")
    if format != "portrait":
        raise ValueError(f"Unsupported format {format}")
    structure = document.metadata.get("structure", "")
    filename = document.metadata.get("filename", "")
    created_toc = document.metadata.get("created_toc", "")
    assert structure and filename and created_toc
    lines = structure.splitlines()
    filestem = Path(filename).stem
    root = TreeNode(name=filestem, number=DOCUMENT_NODE_NUMBER)
    stack = [(root, 0)]  # (node, level) pairs
    processed_line_numbers = list()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if it's a header
        header_match = header_pattern.match(line)
        if header_match:
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            line_number = int(header_match.group(3))
            processed_line_numbers.append(line_number)
            new_node = TreeNode(name=title, number=line_number)
            # Adjust stack for header level
            while stack and stack[-1][1] >= level:
                stack.pop()
            if stack:
                stack[-1][0].add_child(new_node)
                new_node.set_parent(stack[-1][0])
            stack.append((new_node, level))
            continue

    assert processed_line_numbers[0] == 0 and all(processed_line_numbers[i] <= processed_line_numbers[i+1] for i in range(len(processed_line_numbers) - 1))
    return root


def parse_structure(document: BaseNode) -> TreeNode:
    format = document.metadata.get("format", "")
    match format:
        case "landscape":
            return parse_landscape_structure(document)
        case "portrait":
            return parse_portrait_structure(document)
        case _:
            raise ValueError(f"Unsupported format {format}")
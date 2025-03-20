import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence, Any, Optional, Iterator

from llama_index.core import IndexStructType
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.data_structs.data_structs import IndexStruct
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode, IndexNode, MetadataMode
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from neo4j import GraphDatabase

from pdf_rag.structure_parsers import parse_structure, DOCUMENT_NODE_NUMBER

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    uri: str
    username: str
    password: str
    database: str


class TreeNeo4jExporter:
    def __init__(self, config: Neo4jConfig):
        self.driver = GraphDatabase.driver(config.uri, auth=(config.username, config.password))
        self.database = config.database

    def close(self):
        self.driver.close()

    def create_constraints(self):
        with self.driver.session(database=self.database) as session:
            # Create constraints for unique node IDs
            session.run("""
                CREATE CONSTRAINT node_id IF NOT EXISTS 
                FOR (n:TreeNode) REQUIRE n.node_id IS UNIQUE
            """)

    def create_node(self, node_id: str, content: str, metadata: str, name: str):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MERGE (n:TreeNode {node_id: $node_id})
                SET n.content = $content,
                    n.metadata = $metadata,
                    n.name = $name
            """,
                node_id=node_id,
                content=content,
                metadata=metadata,
                name=name,
            )

    def create_relationship(self, parent_id: str, child_id: str):
        with self.driver.session(database=self.database) as session:
            session.run(
                """
                MATCH (parent:TreeNode {node_id: $parent_id})
                MATCH (child:TreeNode {node_id: $child_id})
                MERGE (parent)-[:HAS_CHILD]->(child)
            """,
                parent_id=parent_id,
                child_id=child_id,
            )

    def clear_database(self):
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n) DETACH DELETE n")


@dataclass
class IndexStructTree(IndexStruct):
    all_nodes: list[str] = field(default_factory=list)
    root_nodes: list[str] = field(default_factory=list)
    node_to_children: dict[str, list[str]] = field(default_factory=dict)
    node_to_parent: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.all_nodes)

    def insert(self, nodes_ids: list[str]) -> None:
        self.all_nodes.extend(nodes_ids)

    def delete_node(self, node_id: str) -> None:
        if node_id in self.all_nodes:
            self.all_nodes.remove(node_id)
        if node_id in self.root_nodes:
            self.root_nodes.remove(node_id)
        if node_id in self.node_to_parent:
            self.node_to_parent.pop(node_id)
        if node_id in self.node_to_children:
            self.node_to_children.pop(node_id)

    def set_root(self, node_id: str) -> None:
        if node_id not in self.all_nodes:
            raise ValueError(f"Node {node_id} not in index")
        self.root_nodes.append(node_id)

    def set_children(self, parent_id: str, children_ids: list[str]) -> None:
        if parent_id not in self.all_nodes or any([child_id not in self.all_nodes for child_id in children_ids]):
            raise ValueError(f"Parent {parent_id} or children {children_ids} not in index")
        self.node_to_children[parent_id] = children_ids

    def get_children(self, parent_id: str) -> list[str]:
        if parent_id not in self.all_nodes:
            raise ValueError(f"Parent {parent_id} not in index")
        return self.node_to_children.get(parent_id, [])

    def get_parent(self, child_id: str) -> str | None:
        if child_id not in self.all_nodes:
            raise ValueError(f"Node {child_id} not in index")
        return self.node_to_parent.get(child_id, None)

    def set_parent(self, child_id: str, parent_id: str) -> None:
        if parent_id not in self.all_nodes or child_id not in self.all_nodes:
            raise ValueError(f"Parent {parent_id} or child {child_id} not in index")
        self.node_to_parent[child_id] = parent_id

    def is_root(self, root_id: str) -> bool:
        return root_id in self.root_nodes

    def is_node(self, node_id: str) -> bool:
        return node_id in self.all_nodes

    def bfs(self, node_id: str) -> Iterator[str]:
        """Perform Breadth-First traversal of the tree."""
        if not self.is_node(node_id):
            raise ValueError(f"Node {node_id} not in index")
        queue = deque([node_id])
        while queue:
            node = queue.popleft()
            yield node
            if children := self.node_to_children.get(node):
                queue.extend(children)

    def dfs_preorder(self, node_id: str) -> Iterator[str]:
        """Perform pre-order (root-left-right) DFS traversal."""
        if not self.is_node(node_id):
            raise ValueError(f"Node {node_id} not in index")
        yield node_id
        if children := self.node_to_children.get(node_id):
            for child in children:
                yield from self.dfs_preorder(child)

    def str_from_node(self, node_id: str, level: int = 0) -> str:
        ret = "  " * level + node_id + "\n"
        if children := self.node_to_children.get(node_id):
            for child in children:
                ret += self.str_from_node(node_id=child, level=level + 1)
        return ret

    @classmethod
    def get_type(cls) -> IndexStructType:
        """Get type."""
        return IndexStructType.TREE


class TreeIndex(BaseIndex[IndexStructTree]):
    index_struct_cls = IndexStructTree

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[IndexStructTree] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            show_progress=show_progress,
            objects=objects,
            **kwargs,
        )

    @property
    def vector_store(self) -> BasePydanticVectorStore:
        return self._vector_store

    def _get_all_ref_doc_info(self, nodes: Sequence[BaseNode]) -> dict[str, RefDocInfo]:
        all_ref_doc_info = {}
        for node in nodes:
            ref_node = node.source_node
            if not ref_node:
                continue
            ref_doc_info = self.docstore.get_ref_doc_info(ref_node.node_id)
            if not ref_doc_info:
                continue
            all_ref_doc_info[ref_node.node_id] = ref_doc_info
        return all_ref_doc_info

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        raise NotImplementedError

    def export_to_neo4j(self, config: Optional[Neo4jConfig] = None) -> None:
        if config is None:
            config = Neo4jConfig()

        exporter = TreeNeo4jExporter(config)

        try:
            # Clear existing data and create constraints
            exporter.clear_database()
            exporter.create_constraints()

            # Create all nodes first
            for node_id in self.index_struct.all_nodes:
                node = self.docstore.get_node(node_id)
                content = node.get_content(metadata_mode=MetadataMode.NONE)
                metadata = node.get_metadata_str() or ""
                name = node.metadata["filename"]
                exporter.create_node(node_id, content, metadata, name)

            # Create relationships
            for parent_id, children_ids in self.index_struct.node_to_children.items():
                for child_id in children_ids:
                    exporter.create_relationship(parent_id, child_id)

        finally:
            exporter.close()

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode], **build_kwargs: Any) -> IndexStructTree:
        self._index_struct = IndexStructTree()
        self._insert(nodes=nodes, **build_kwargs)
        return self.index_struct

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        all_ref_doc_info = self._get_all_ref_doc_info(nodes=nodes)
        if not all_ref_doc_info:
            raise ValueError("No ref doc info")
        for root_id, ref_doc_info in all_ref_doc_info.items():
            format = ref_doc_info.metadata.get("format", "")
            if format == "landscape":
                number_to_node_id = {
                    self.docstore.get_node(node_id).metadata["page_number"]: node_id
                    for node_id in ref_doc_info.node_ids
                }
            elif format == "portrait":
                number_to_node_id = {
                    self.docstore.get_node(node_id).metadata["line_number"]: node_id
                    for node_id in ref_doc_info.node_ids
                }
            else:
                raise ValueError(f"Unknown format: {format}")
            number_to_node_id[DOCUMENT_NODE_NUMBER] = root_id
            ref_doc = self.docstore.get_document(root_id)
            tree = parse_structure(ref_doc)
            abstract_nodes: dict[int, BaseNode] = dict()
            # Check nodes and add abstract nodes
            for tree_node in tree:
                if tree_node.number < DOCUMENT_NODE_NUMBER:
                    new_node = TextNode(
                        text=tree_node.name,
                        metadata=ref_doc.metadata,
                        excluded_embed_metadata_keys=ref_doc.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=ref_doc.excluded_llm_metadata_keys,
                        metadata_template=ref_doc.metadata_template,
                        metadata_seperator=ref_doc.metadata_separator,
                        text_template=ref_doc.text_template,
                        relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
                    )
                    abstract_nodes[tree_node.number] = new_node
                elif tree_node.number not in number_to_node_id:
                    raise ValueError(
                        f"Tree node number {tree_node.number} is not in the index nodes. A node is missing."
                    )
            #
            self.docstore.add_documents(list(abstract_nodes.values()))
            number_to_node_id.update({number: node.node_id for number, node in abstract_nodes.items()})
            #
            self.index_struct.insert(nodes_ids=list(number_to_node_id.values()))
            self.index_struct.set_root(node_id=root_id)
            #
            parent_nodes = list()
            for tree_node in tree:
                node_id = number_to_node_id[tree_node.number]
                node = self.docstore.get_node(node_id)
                if tree_node.parent is not None:
                    parent_id = number_to_node_id[tree_node.parent.number]
                    parent = self.docstore.get_node(parent_id)
                    node.relationships[NodeRelationship.PARENT] = parent.as_related_node_info()
                    self.index_struct.set_parent(child_id=node_id, parent_id=parent_id)
                if tree_node.children:
                    children_ids = [number_to_node_id[child.number] for child in tree_node.children]
                    children = self.docstore.get_nodes(children_ids)
                    node.relationships[NodeRelationship.CHILD] = [child.as_related_node_info() for child in children]
                    self.index_struct.set_children(parent_id=node_id, children_ids=children_ids)
                parent_nodes.append(node)
            #
            self.docstore.add_documents(parent_nodes, allow_update=True)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        self.index_struct.delete_node(node_id=node_id)

    def delete_nodes(
        self,
        node_ids: list[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        raise NotImplementedError("delete_nodes is not implemented. Use delete_ref_doc instead.")

    def _delete_nodes(
        self,
        node_ids: list[str],
        delete_from_docstore: bool = False,
        **delete_kwargs: Any,
    ) -> None:
        """Delete a list of nodes from the index.

        Args:
            doc_ids (List[str]): A list of doc_ids from the nodes to delete

        """
        for node_id in node_ids:
            self._delete_node(node_id, **delete_kwargs)
            if delete_from_docstore:
                self.docstore.delete_document(node_id, raise_error=False)

        self._storage_context.index_store.add_index_struct(self._index_struct)

    def delete_ref_doc(self, ref_doc_id: str, delete_from_docstore: bool = False, **delete_kwargs: Any) -> None:
        """Delete a document and it's nodes by using ref_doc_id."""
        ref_doc_info = self.docstore.get_ref_doc_info(ref_doc_id)
        if ref_doc_info is None:
            logger.warning(f"ref_doc_id {ref_doc_id} not found, nothing deleted.")
            return

        self._delete_nodes(
            ref_doc_info.node_ids + [ref_doc_id],
            delete_from_docstore=False,
            **delete_kwargs,
        )

        if delete_from_docstore:
            self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)
            self.docstore.delete_document(ref_doc_id, raise_error=False)

    @property
    def ref_doc_info(self) -> dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        node_doc_ids = self.index_struct.all_nodes
        nodes = self.docstore.get_nodes(node_doc_ids)
        return self._get_all_ref_doc_info(nodes=nodes)

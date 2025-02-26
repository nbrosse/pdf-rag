from collections import defaultdict
from functools import cached_property

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, TextNode, MetadataMode
from llama_index.core.storage.docstore import BaseDocumentStore


class FullPagePostprocessor(BaseNodePostprocessor):
    docstore: BaseDocumentStore

    @cached_property
    def doc_page_node_ids(self) -> dict[tuple[str, int], list[str]]:
        all_ref_doc_info = self.docstore.get_all_ref_doc_info()
        doc_page_node_ids = defaultdict(list)
        for doc_id, ref_doc_info in all_ref_doc_info.items():
            for node_id in ref_doc_info.node_ids:
                try:
                    node = self.docstore.get_node(node_id)
                    page_number = node.metadata['page_number']
                except KeyError:
                    raise KeyError(f"Node id {node_id} has no page_number metadata")
                doc_page_node_ids[(doc_id, page_number)].append(node_id)
        return doc_page_node_ids

    @classmethod
    def class_name(cls) -> str:
        return "FullPagePostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        doc_pages = set()
        page_nodes_scores: list[NodeWithScore] = list()
        for node in nodes:
            try:
                page_number = node.metadata['page_number']
            except KeyError:
                raise KeyError(f"Node id {node.node_id} has no page_number metadata")
            doc_id = node.node.source_node.node_id
            if (doc_id, page_number) not in doc_pages:  # not already processed
                doc_pages.add((doc_id, page_number))
                page_node_ids = self.doc_page_node_ids[(doc_id, page_number)]
                page_nodes = [self.docstore.get_node(node_id) for node_id in page_node_ids]
                content = "\n".join([n.get_content(metadata_mode=MetadataMode.NONE) for n in page_nodes])
                # filename = page_nodes[0].metadata['filename']
                # source = f"Source {filename}"
                # content = f"Source {len(new_nodes) + 1}:\n{text_chunk}\n"
                page_node = TextNode(
                    text=content,
                    metadata=page_nodes[0].metadata,
                    source_node=page_nodes[0].source_node,
                )
                page_node_score = NodeWithScore(
                    node=page_node,
                    score=node.score,
                )
                page_nodes_scores.append(page_node_score)
        return page_nodes_scores
from __future__ import annotations

import logging

import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from commons.wiki_entity import WikiEntity
from commons.wiki_mapping import WikiMapping

logger = logging.getLogger(__name__)


class SummaryGraphBuilder:
    def __init__(self, seed_nodes: list[str], categories: list[str], min_valid_summary_edges: int, max_threads: int):
        self.seed_nodes = seed_nodes
        self.categories = categories
        self.min_valid_summary_edges = min_valid_summary_edges
        self.max_threads = max_threads
        self.graph_lock = Lock()

    @staticmethod
    def load_node_summaries(seed_node: str) -> tuple[list, WikiEntity | None]:
        summaries = []
        try:
            root_entity = WikiEntity(WikiMapping(wikidata_id=seed_node))
            summaries = root_entity.get_summaries()
            for summary in summaries:
                if summary[0] == summary[2]:
                    logger.warning(f"Self loop detected: {summary}")
            return [(summary[0], summary[2], summary[1], seed_node) for summary in summaries if
                    summary[0] != summary[2]], root_entity
        except Exception as e:
            logger.error(f"Error processing node {seed_node}: {e}")
        return summaries, None

    def build_summary_graph(self) -> nx.MultiDiGraph:
        summary_graph = nx.MultiDiGraph()
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_node = {executor.submit(self.load_node_summaries, self.seed_nodes[i]): i for i in
                              range(len(self.seed_nodes))}
            for future in as_completed(future_to_node):
                seed_node_index = future_to_node[future]
                try:
                    summaries, wiki_entity = future.result()
                    if not wiki_entity:
                        continue
                    seed_node = self.seed_nodes[seed_node_index]
                    if len(summaries) < self.min_valid_summary_edges:
                        logger.warning(
                            f"Seed[{seed_node}] with [{len(summaries)}] summaries is less than"
                            f" {self.min_valid_summary_edges} min allowed summaries. Ignoring...")
                        continue
                    logger.info(
                        f"Building summary graph for seed node[{seed_node}], number of summaries: {len(summaries)}")
                    with self.graph_lock:
                        for s, t, p, root in summaries:
                            summary_graph.add_edge(s, t, key=p, predicate=p, summary_for=root)
                        summary_graph.nodes[seed_node]['is_root'] = True
                        if self.categories:
                            summary_graph.nodes[seed_node]['category'] = self.categories[seed_node_index]
                except Exception as exc:
                    logger.error(f"Node {seed_node} generated an exception: {exc}")
        return summary_graph

import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import networkx as nx
import numpy as np

from commons.neo4j_storage import fetch_random_walk_paths, check_and_create_projection

logger = logging.getLogger(__name__)

class GraphExpander:
    def __init__(self, summary_graph: nx.MultiDiGraph, degree_distribution: dict[str, int], max_threads: int,
                 random_walk_depth_len: int,
                 min_random_walk_number: int, max_random_walk_number: int):
        def get_scaled_random_walk_range() -> dict[str, int]:
            log_degrees = {seed_node: np.log(deg) for seed_node, deg in degree_distribution.items()}

            min_log_deg = min(log_degrees.values())
            max_log_deg = max(log_degrees.values())
            for seed_node, log_deg in log_degrees.items():
                scaled_log_deg = (log_deg - min_log_deg) / (max_log_deg - min_log_deg)
                degree_distribution[seed_node] = int(
                    min_random_walk_number + scaled_log_deg * (
                            max_random_walk_number - min_random_walk_number)
                )
            return degree_distribution

        self.expanded_graph = copy.deepcopy(summary_graph)
        self.walk_plan = get_scaled_random_walk_range()
        self.max_threads = max_threads
        self.random_walk_depth_len = random_walk_depth_len
        self.graph_lock = Lock()
        check_and_create_projection()

    @staticmethod
    def expand_summary_graph_for_node(starting_node: str, random_walk_depth_len: int, walk_count: int) -> list:
        expanded_edges = []
        try:
            walks = fetch_random_walk_paths(
                starting_node, random_walk_depth_len, walk_count
            )
            return [(s, p, t) for walk in walks for s, p, t in walk]
        except Exception as e:
            logger.error(f"Error processing random walk expansion for node {starting_node}: {e}")
        return expanded_edges

    def expand_summary_graph(self) -> nx.MultiDiGraph:
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_node = {
                executor.submit(
                    self.expand_summary_graph_for_node,
                    starting_node,
                    self.random_walk_depth_len,
                    walk_count
                ): starting_node
                for starting_node, walk_count in self.walk_plan.items()
            }
            for future in as_completed(future_to_node):
                starting_node = future_to_node[future]
                try:
                    expanded_edges = future.result()
                    logger.info(f"Expanding graph with {len(expanded_edges)} edges for seed node[{starting_node}]")
                    with self.graph_lock:
                        for s, p, t in expanded_edges:
                            if not self.expanded_graph.has_edge(s, t, key=p):
                                self.expanded_graph.add_edge(s, t, key=p, predicate=p)
                except Exception as exc:
                    logger.error(f"Error during expansion [{starting_node}]: {exc}")

        return self.expanded_graph

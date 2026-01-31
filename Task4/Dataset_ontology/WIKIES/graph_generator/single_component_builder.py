import copy
import logging

import networkx as nx

from commons.neo4j_storage import (
    label_candidates,
    fetch_shortest_pairs_between_components,
    fetch_shortest_path,
    change_component_label_with,
    remove_component_labels, set_node_label
)

logger = logging.getLogger(__name__)


class SingleComponentBuilder:
    def __init__(self, initial_graph: nx.MultiDiGraph, bridge_count: int):
        """
        Initialize the SingleComponentBuilder with a deep copy of the initial graph
        and the number of bridges to connect components.

        :param initial_graph: The initial multi-directed graph to be processed.
        :param bridge_count: The number of bridges (shortest paths) to connect components.
        """
        self.bridge_count = bridge_count
        self.es_graph: nx.MultiDiGraph = copy.deepcopy(initial_graph)

    def _add_paths_between_components(self, top_pairs, current_component_label, target_component_label):
        """
        Add the shortest paths between components to the graph and update component labels.

        :param top_pairs: List of top pairs of nodes representing the shortest paths between components.
        :param current_component_label: Label of the current smaller component.
        :param target_component_label: Label of the target component to which others are being connected.
        """
        connecting_edges_string = ""
        for node_a, node_b in top_pairs:
            for start_node, predicate, end_node in fetch_shortest_path(node_a, node_b):
                if self.es_graph.has_edge(start_node, end_node, key=predicate):
                    continue
                connecting_edges_string += f"({start_node})-[:{predicate}]->({end_node})\n"
                self.es_graph.add_edge(start_node, end_node, key=predicate, predicate=predicate)
                set_node_label(start_node, target_component_label)
        logger.info(
            f"Connecting components [{current_component_label}] and [{target_component_label}] with the following edges:"
            f"\n{connecting_edges_string}"
        )

        change_component_label_with(current_component_label, target_component_label)

    def _connect_components(self) -> None:
        """
        Connect disjoint components in the graph by iteratively adding the shortest paths between them until only one
        component remains.
        """
        weakly_connected_components = list(nx.weakly_connected_components(self.es_graph))
        sorted_components = sorted(weakly_connected_components, key=len, reverse=True)
        component_labels = label_candidates(sorted_components)
        logger.info(f"Assigned labels to the components: [{component_labels}]")
        try:
            largest_component_label = component_labels[0]
            previous_component_count = len(sorted_components)
            search_depth = 0

            # While more than one component exists, keep connecting them
            while len(component_labels) > 1:
                # Increase search depth if no new connections were made in the last iteration
                if len(component_labels) == previous_component_count:
                    search_depth += 1
                    logger.info(f"Connecting components with searching depth [{search_depth}]")

                previous_component_count = len(component_labels)

                # Attempt to connect components to the largest component with K bridge paths
                for label in component_labels[1:]:
                    top_pairs = fetch_shortest_pairs_between_components(
                        label, largest_component_label, search_depth, self.bridge_count
                    )
                    logger.info(f"Top pairs found between [{label}] and [{largest_component_label}]: {top_pairs}")
                    if len(top_pairs) < self.bridge_count:
                        logger.info(f"Ignoring found pairs for [{label}] as they are less than [{self.bridge_count}]")
                        continue

                    self._add_paths_between_components(top_pairs, label, largest_component_label)
                    logger.info(f"Removing component label [{label}]")
                    component_labels.remove(label)
        except Exception as e:
            logger.error(f"Error occurred while connecting components: {e}")
        finally:
            remove_component_labels(component_labels)

    def build_single_component(self) -> nx.MultiDiGraph:
        """
        Build a single connected component from the graph if it's not already connected.

        :return: The graph as a single connected component.
        """
        if nx.number_weakly_connected_components(self.es_graph) == 1:
            logger.info("The graph is already connected.")
            return self.es_graph

        self._connect_components()
        return self.es_graph

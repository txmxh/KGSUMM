from __future__ import annotations

import hashlib
import logging
import os
import pickle
import random

import networkx as nx

from commons.neo4j_storage import fetch_node_degree
from commons.sqlite_storage import create_metadata_tables, insert_root_entity_metadata, load_degree_distribution
from commons.wiki_entity import WikiEntity
from commons.wiki_mapping import WikiMapping
from graph_generator.summary_graph_builder import SummaryGraphBuilder
from graph_generator.metadata_manager import MetadataManager
from graph_generator.graph_expander import GraphExpander
from graph_generator.single_component_builder import SingleComponentBuilder
from graph_generator.config import Config

logger = logging.getLogger(__name__)


class ProcessManager:
    def __init__(self, name: str, seed_nodes: list[str], min_random_walk_number=50, max_random_walk_number=300,
                 categories: list[str] = None):
        self.hashcode = None
        self.metadata_db_path: str = ""
        self.summary_graph_path: str = ""
        self.summary_graph: nx.MultiDiGraph | None = None
        self.expanded_graph_path: str = ""
        self.expanded_graph: nx.MultiDiGraph | None = None
        self.es_graph_path: str = ""
        self.es_graph: nx.MultiDiGraph | None = None
        self.final_graph_path: str = ""
        self.final_graph: nx.MultiDiGraph | None = None
        self.graphml_path: str = ""
        self.seed_nodes: list[str] = seed_nodes
        self.min_random_walk_number: int = min_random_walk_number
        self.max_random_walk_number: int = max_random_walk_number
        self.categories: list[str] = categories
        self.name: str = name
        self.config = Config()
        self._initialize()

    @staticmethod
    def _load_graph_from_file(path):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    @staticmethod
    def _shuffle_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        nodes = list(graph.nodes(data=True))
        edges = list(graph.edges(data=True))

        random.shuffle(nodes)
        random.shuffle(edges)

        shuffled_graph: nx.MultiDiGraph = nx.MultiDiGraph()
        shuffled_graph.add_nodes_from(nodes)
        for s, o, data in edges:
            shuffled_graph.add_edge(s, o, key=data['predicate'], **data)
        return shuffled_graph

    def _update_seed_nodes_additional_info(self):
        summaries = {}
        root_nodes = [node for node in self.summary_graph.nodes if
                      self.summary_graph.nodes[node].get('is_root', False)]
        for s, o, data in self.summary_graph.edges(data=True):
            if data.get('summary_for', False):
                summaries[data['summary_for']] = summaries.get(data['summary_for'], 0) + 1

        for node in root_nodes:
            wiki_entity = WikiEntity(WikiMapping(wikidata_id=node))
            degree = fetch_node_degree(node)
            insert_root_entity_metadata(
                data=(
                    wiki_entity.wikidata_id,
                    wiki_entity.get_wikidata_label(),
                    wiki_entity.get_wikidata_description(),
                    wiki_entity.wikipedia_id,
                    wiki_entity.wikipedia_page_title,
                    self.summary_graph.nodes[node].get('category', None),
                    summaries[node],
                    degree
                ), database_path=self.metadata_db_path)

    def load_summary_graph(self) -> nx.MultiDiGraph:
        """
        Based on the requested seed nodes, look up into our pre-built summary graph and return all the edges connected
        to the seed nodes. This is called the summary graph.
        :return: nx.MultiDiGraph
        Node info:
        'Q43416', data = {
            'is_root': True,
            'category': 'actor'}
        Edge info:
        'Q43416', 'Q3820', data = {
            'predicate': 'P19',
            'summary_for': 'Q43416'}
        """
        if not self.summary_graph:
            self.summary_graph = (
                SummaryGraphBuilder(
                    self.seed_nodes,
                    self.categories,
                    self.config.min_valid_summary_edges,
                    self.config.max_threads
                )
                .build_summary_graph()
            )
            self._update_seed_nodes_additional_info()
            with open(self.summary_graph_path, "wb") as f:
                pickle.dump(self.summary_graph, f)
                logger.info(f"Summary graph saved at {self.summary_graph_path}")

        return self.summary_graph

    def load_expanded_graph(self) -> nx.MultiDiGraph:
        """
        During the summarization process, we might have added all the eligible nodes (if they had enough summaries
        as the min configuration value) and loaded them into our local metadata sqlite database. Now, we will use this
        information to expand our summary graph by adding more edges with the configured path length and size.
        We try to mimic the graph node distribution by scaling the random walk path length based on the degree of the
        seed nodes. However, we have to prevent the nodes with a high degree to dominate the expansion, to do so we have
        a min and max random walk counter configuration.
        :return: nx.MultiDiGraph
        """
        if not self.expanded_graph:
            self.expanded_graph = (
                GraphExpander(
                    self.summary_graph,
                    load_degree_distribution(database_path=self.metadata_db_path),
                    self.config.max_threads,
                    self.config.random_walk_depth_len,
                    self.min_random_walk_number,
                    self.max_random_walk_number
                )
                .expand_summary_graph()
            )
            with open(self.expanded_graph_path, "wb") as f:
                pickle.dump(self.expanded_graph, f)
                logger.info(f"Expanded graph saved at {self.expanded_graph_path}")
        return self.expanded_graph

    def load_es_graph(self) -> nx.MultiDiGraph:
        """
        In this process, we check the connectivity of the expanded graph and create a single connected component graph
        if it is not already connected. This is called the entity summarization graph. First we assign a label to each
        component node and load the info in Neo4j original graph. As we already know, the original graph is a single
        weakly connected component graph, so there is a path between any two nodes. We will use this information to
        connect the components in the expanded graph. Starting from the larger component we first try to find K bridges
        connecting the larger component to the largest component by an immediate connecting edge. If we can't find
        K bridges we continue to the next larger component and try to find K bridges between that component and the
        largest one. When the first iteration is done we will recheck for immediate connecting edge between the largest
        and the new largest component. If still there is no connecting edge we continue by finding paths with
        an intermediate node(shortest path of length 2 edges) and try to connect the components.
        This process repeats the mentioned steps and increases the paths length if it is required until all the
        components are connected. We acknowledge that this process has repetition steps, but because the running time
        of a shorter shortest paths is much faster than the longer one and the possibility of finding a bridge with
        the shorter path is higher in Wikidata knowledge graph, we accepted this trade-off.
        :return: nx.MultiDiGraph
        """
        if not self.es_graph:
            self.es_graph = (
                SingleComponentBuilder(
                    self.expanded_graph,
                    self.config.bridges_number
                )
                .build_single_component()
            )
            with open(self.es_graph_path, "wb") as f:
                pickle.dump(self.es_graph, f)
                logger.info(f"Entity summarization graph saved at {self.es_graph_path}")
        return self.es_graph

    def load_final_graph(self) -> nx.MultiDiGraph:
        """
        Having the entity summarization graph, we will finalize the graph by adding the metadata information to the
        nodes and edges. We will add the wikidata label, description, wikipedia id, and wikipedia title to the nodes and
        the predicate label and description to the edges.
        :return: nx.MultiDiGraph
        'Q43416', data = {
            'is_root': True,
            'category': 'actor',
            'wikidata_label': 'Keanu Reeves',
            'wikidata_desc': 'Canadian actor (born 1964)',
            'wikipedia_title': 'Keanu_Reeves',
            'wikipedia_id': 16603
        }
        'Q3820', data = {
            'wikidata_label': 'Beirut',
            'wikidata_desc': 'capital and largest city of Lebanon',
            'wikipedia_title': 'Beirut',
            'wikipedia_id': 37428
        }
        'Q639669', data = {
            'wikidata_label': 'musician',
            'wikidata_desc': 'person who composes, conducts or performs music',
            'wikipedia_title': 'Musician',
            'wikipedia_id': 38284
        }
        'Q43416', 'Q3820', data = {
            'predicate': 'P19',
            'summary_for': 'Q43416',
            'predicate_label': 'place of birth',
            'predicate_desc': 'location where the subject was born'
         }
        'Q43416', 'Q639669', data = {
            'predicate': 'P106',
            'predicate_label': 'occupation',
            'predicate_desc':
            'occupation of a person; see also "field of work" (Property:P101), "position held" (Property:P39)'
         }
        """
        if not self.final_graph:
            self.final_graph = (
                MetadataManager(
                    self.es_graph,
                    self.metadata_db_path
                )
                .build_final_graph()
            )
            self.final_graph = self._shuffle_graph(self.final_graph)
            with open(self.final_graph_path, "wb") as f:
                pickle.dump(self.final_graph, f)
                logger.info(f"Finalized graph saved at {self.final_graph_path}")
        return self.final_graph

    def _initialize(self):
        logger.info(f"Initializing process for {self.name}...")
        self.hashcode = hashlib.sha256(''.join([seed_node for seed_node in self.seed_nodes]).encode()).hexdigest()
        os.makedirs(f"{self.config.output_path}/{self.name}/", exist_ok=True)
        base_path = f"{self.config.output_path}/{self.name}/"
        self.metadata_db_path = f"{base_path}{self.name}_{self.hashcode}.meta.db"
        create_metadata_tables(database_path=self.metadata_db_path)

        # Load graphs if they exist
        self.summary_graph_path = f"{base_path}summary_graph_{self.name}_{self.hashcode}.pkl"
        self.summary_graph = self._load_graph_from_file(self.summary_graph_path)
        self.expanded_graph_path = f"{base_path}expanded_graph_{self.name}_{self.hashcode}.pkl"
        self.expanded_graph = self._load_graph_from_file(self.expanded_graph_path)
        self.es_graph_path = f"{base_path}es_graph_{self.name}_{self.hashcode}.pkl"
        self.es_graph = self._load_graph_from_file(self.es_graph_path)
        self.final_graph_path = f"{base_path}{self.name}_{self.hashcode}.pkl"
        self.final_graph = self._load_graph_from_file(self.final_graph_path)
        self.graphml_path = f"{base_path}{self.name}_{self.hashcode}.graphml"

    def _finalize(self):
        if not os.path.exists(self.graphml_path):
            nx.write_graphml(self.final_graph, self.graphml_path)
            logger.info(f"GraphML file saved at {self.graphml_path}")
        logger.info(f"Process completed successfully.")

    def run_process(self) -> nx.MultiDiGraph:
        self.load_summary_graph()
        self.load_expanded_graph()
        self.load_es_graph()
        self.load_final_graph()
        self._finalize()
        return self.final_graph

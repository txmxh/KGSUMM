import copy
import logging

import networkx as nx

from commons.sqlite_storage import insert_predicates_metadata, insert_entities_metadata

logger = logging.getLogger(__name__)


class MetadataManager:
    def __init__(self, es_graph: nx.MultiDiGraph, metadata_db_path: str):
        self.final_graph: nx.MultiDiGraph = copy.deepcopy(es_graph)
        self.metadata_db_path = metadata_db_path

    def _insert_entities_metadata(self):

        def insert_callback(entities):
            for entity in entities:
                qid, wikidata_label, wikidata_desc, wikipedia_id, wikipedia_title = entity
                if wikidata_label:
                    self.final_graph.nodes[qid]['wikidata_label'] = wikidata_label
                if wikidata_desc:
                    self.final_graph.nodes[qid]['wikidata_desc'] = wikidata_desc
                if wikipedia_id:
                    self.final_graph.nodes[qid]['wikipedia_id'] = wikipedia_id
                if wikipedia_title:
                    self.final_graph.nodes[qid]['wikipedia_title'] = wikipedia_title

        entity_ids = set(self.final_graph.nodes())
        insert_entities_metadata(
            entity_ids=entity_ids, callback=insert_callback, database_path=self.metadata_db_path
        )

    def _insert_predicates_metadata(self):
        """
        Fetch the labels and descriptions of each used predicates and add it to the local DB
        :es_graph single connected component multi-directed graph called entity summarization graph
        :return path of the generated SQLite file
        """

        def insert_callback(predicates):
            predicates = {predicate: {"label": label, "description": description} for predicate, label, description in
                          predicates}
            for s, t, data in self.final_graph.edges(data=True):
                if data['predicate'] in predicates:
                    label = None if 'label' not in predicates[data['predicate']] else predicates[data['predicate']][
                        'label']
                    description = None if 'description' not in predicates[data['predicate']] else \
                        predicates[data['predicate']]['description']
                    if self.final_graph.has_edge(s, t, data['predicate']):
                        if label:
                            self.final_graph.edges[s, t, data['predicate']]['predicate_label'] = label
                        if description:
                            self.final_graph.edges[s, t, data['predicate']]['predicate_desc'] = description

        predicate_ids = set()
        for s, t, data in self.final_graph.edges(data=True):
            predicate_ids.add(data['predicate'])
        insert_predicates_metadata(
            predicate_ids=predicate_ids, callback=insert_callback, database_path=self.metadata_db_path
        )

    def build_final_graph(self) -> nx.MultiDiGraph:
        self._insert_entities_metadata()
        self._insert_predicates_metadata()
        return self.final_graph

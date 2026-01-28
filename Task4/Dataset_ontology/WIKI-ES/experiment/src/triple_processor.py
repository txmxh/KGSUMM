

class TripleProcessor:

    @staticmethod
    def extract_triples(graph):
        """
        Extract triples from the graph.
        :param graph: Graph from which to extract triples.
        :return: List of triples.
        """
        triples = []
        for u, v, data in graph.edges(data=True):
            triples.append((u, data.get('predicate'), v))
        return triples

    @staticmethod
    def convert_to_label(triple, node_labels, predicate_labels):
        """
        Convert a triple to its labels.
        :param triple: Triple to convert.
        :param node_labels: Dictionary of node labels.
        :param predicate_labels: Dictionary of predicate labels.
        :return: Labeled triple.
        """
        e, p, v = triple
        return node_labels.get(e), predicate_labels.get(p), node_labels.get(v)

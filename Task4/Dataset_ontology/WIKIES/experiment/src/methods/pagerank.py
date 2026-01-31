import numpy as np
import networkx as nx
import random

from src.dataset_loader import DatasetLoader
from src.triple_processor import TripleProcessor
from src.evaluation_metrics import EvaluationMetrics


class Pagerank:

    def __init__(self, dataset_name, data_path):
        """
        Initialize the Pagerank object.
        :param dataset_name: Name of the dataset.
        :param data_path: Path to the dataset.
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.G, self.nodes, self.node_labels, self.root_nodes, self.relation_frequency, self.desc = DatasetLoader.load_dataset(
            data_path)
        self.triples = TripleProcessor.extract_triples(self.G)

    def run(self):
        """
        Initialize PageRank scores with equal values for all nodes.
        :return: Dictionary with initial PageRank scores for each node.
        """
        pagerank_scores = nx.pagerank(self.G)
        return pagerank_scores

    def evaluate(self, topK, pagerank_scores):
        """
        Evaluate the model using the provided parameters.
        :param topK: Top K results.
        :param pagerank_scores: pagerank scores for each node.
        :param results: List to store results.
        """
        f1_list = []
        f1_list_dynamic = []
        average_precisions = []
        average_precisions_dynamic = []

        for root in self.root_nodes:

            all_triples = [(subject, info.get('predicate', None), obj)
                           for subject, obj, info in self.desc[root]]
            random.shuffle(all_triples)

            # Get the ground truth summary nodes based on 'summary_for' attribute
            ground_truth_summary = [(subject, info.get('predicate', None), obj) for subject, obj,
                                    info in self.desc[root] if info.get('summary_for', False) and info.get('summary_for') == root]

            connected_nodes = list(self.G.neighbors(root))
            ranked_nodes = sorted(
                connected_nodes, key=lambda n: pagerank_scores[n], reverse=True)

            predicted_nodes_summary = ranked_nodes[:topK]
            predicted_nodes_summary_dynamic = ranked_nodes[:len(
                ground_truth_summary)]

            predicted_summary = []
            predicted_summary_dynamic = []

            # find triples based on ranked nodes
            for node in predicted_nodes_summary:
                for triple in all_triples:
                    if node == triple[0] or node == triple[2]:
                        predicted_summary.append(triple)
                        break

            for node in predicted_nodes_summary_dynamic:
                for triple in all_triples:
                    if node == triple[0] or node == triple[2]:
                        predicted_summary_dynamic.append(triple)
                        break

            if len(ground_truth_summary) > 0:
                # Calculate Precision, recall, F1
                f1_list.append(EvaluationMetrics.evaluation(
                    predicted_summary, ground_truth_summary, topK))
                f1_list_dynamic.append(EvaluationMetrics.evaluation_f1_dynamic(
                    predicted_summary_dynamic, ground_truth_summary))

                # Calculate Mean Average Precision
                average_precisions.append(EvaluationMetrics.calculate_average_precision(
                    predicted_summary, ground_truth_summary))
                average_precisions_dynamic.append(EvaluationMetrics.calculate_average_precision(
                    predicted_summary_dynamic, ground_truth_summary))

        # Store and print result
        result = {"Method": f"Pagerank ({topK})"}

        print(f"Results for: Pagerank topK={topK}")
        favg = np.mean(f1_list, axis=0)

        print(f"F1 Score: {favg[0]:.10f}")
        result["F1 Score"] = favg[0]

        print(f"Precision: {favg[1]:.10f}")
        result["Precision"] = favg[1]

        print(f"Recall: {favg[2]:.10f}")
        result["Recall"] = favg[2]

        favg_dynamic = np.mean(f1_list_dynamic, axis=0)

        print(f"F1 Score dynamic: {favg_dynamic[0]:.10f}")
        result["F1 Score dynamic"] = favg_dynamic[0]

        if average_precisions:
            avg_precision = sum(average_precisions) / len(average_precisions)
            print(f"Average precision: {avg_precision:.10f}")
            result["Average precision"] = avg_precision
        else:
            print(0)
            result["Average precision"] = 0

        if average_precisions_dynamic:
            avg_precision_dynamic = sum(
                average_precisions_dynamic) / len(average_precisions_dynamic)
            print(f"Average precision dynamic: {avg_precision_dynamic:.10f}")
            result["Average precision dynamic"] = avg_precision_dynamic
        else:
            print(0)
            result["Average precision dynamic"] = 0

        return result

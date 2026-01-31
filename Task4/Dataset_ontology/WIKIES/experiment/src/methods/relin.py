from tqdm import tqdm
import numpy as np
import math

from src.dataset_loader import DatasetLoader
from src.triple_processor import TripleProcessor
from src.evaluation_metrics import EvaluationMetrics
from src.utils import extract_predicate_labels
from src.string_similarity import StringSimilarity


class Relin:

    def __init__(self, dataset_name, data_path):
        """
        Initialize the Relin object.
        :param dataset_name: Name of the dataset.
        :param data_path: Path to the dataset.
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.G, self.nodes, self.node_labels, self.root_nodes, self.relation_frequency, self.desc = DatasetLoader.load_dataset(
            data_path)
        self.triples = TripleProcessor.extract_triples(self.G)
        self.predicate_to_label = extract_predicate_labels(self.G)
        self.relatedness = {}
        self.informativeness = {}
        self.similarity = StringSimilarity()

    def calculate_informativeness(self):
        """
        Calculate informativeness for each triple.
        :return: Dictionary of informativeness scores for each triple.
        """
        triples = [(item[0], item[1], item[2]) for item in self.triples]

        preprocessed_triples = set(triples)

        for triple in tqdm(self.triples, desc="Informativeness"):
            self.informativeness[triple] = self._calculate_informativeness_for_triple(
                triple, preprocessed_triples)

    def _calculate_informativeness_for_triple(self, triple, preprocessed_triples):
        _, p, v = triple

        # Using set for efficient lookup
        relevant_nodes = {(e, p, v) for e in self.G.nodes()} | {
            (v, p, e) for e in self.G.nodes()}

        count = len(relevant_nodes & preprocessed_triples)
        si = - math.log(count / len(self.G.nodes())) if count > 0 else 0
        return si

    def calculate_relatedness(self):
        """
        Calculate relatedness for each pair of triples.
        :return: Dictionary of relatedness scores for each pair of triples.
        """
        for triple in tqdm(self.triples, desc="Processing Triples"):
            e, p, v = triple

            neighbors = [(h, r, t) for (h, r, t) in self.triples if h == e]
            for neighbor in neighbors:
                self.relatedness[(neighbor, triple)] = self._calculate_relatedness_for_pair(
                    TripleProcessor.convert_to_label(
                        neighbor, self.node_labels, self.predicate_to_label),
                    TripleProcessor.convert_to_label(
                        triple, self.node_labels, self.predicate_to_label)
                )

    def _calculate_relatedness_for_pair(self, triple1, triple2):
        # Extract properties and values from triples
        _, p1, v1 = triple1
        _, p2, v2 = triple2

        if (p1 != None and p2 != None and v1 != None and v2 != None):
            # Calculate PMI for properties and values
            PMI_prp = self.similarity.PMI(s1=p1, s2=p2)
            PMI_val = self.similarity.PMI(s1=v1, s2=v2)

            # Calculate the square root of the product of PMI_prp and PMI_val
            pmi_value = math.sqrt(PMI_prp * PMI_val)
            return pmi_value
        else:
            return 0

    def initialize_pagerank(self):
        """
        Initialize PageRank scores with equal values for all triples.
        :return: Dictionary with initial PageRank scores for each triple.
        """
        return {triple: 1/len(self.triples) for triple in self.triples}

    def update_pagerank(self, pagerank, damping_factor):
        """
        Update PageRank scores based on informativeness and relatedness.
        :param pagerank: Current PageRank scores.
        :param damping_factor: Damping factor for PageRank.
        :return: Dictionary with updated PageRank scores.
        """

        N = len(self.triples)
        new_pagerank = {}
        for triple in self.triples:
            e, p, v = triple
            neighbors = [(h, r, t) for (h, r, t) in self.triples if h == e]

            # Calculate jump term based on informativeness
            jump_term = (1 - damping_factor) * self.informativeness[triple]

            # Calculate link term based on relatedness
            link_term = damping_factor * sum(pagerank[t_prime] * self.relatedness[t_prime, triple]
                                             for t_prime in neighbors)
            new_pagerank[triple] = jump_term + link_term
        return new_pagerank

    def normalize_pagerank(self, pagerank):
        """
        Normalize PageRank scores so that they sum up to 1.
        :param pagerank: PageRank scores to be normalized.
        :return: Dictionary with normalized PageRank scores.
        """
        total_rank = sum(pagerank.values())
        return {triple: rank / total_rank for triple, rank in pagerank.items()}

    def run(self, damping_factor=0.85, iterations=100, tol=1e-9):
        """
        Run the Relin algorithm to compute PageRank scores.
        :param damping_factor: Damping factor for PageRank.
        :param iterations: Maximum number of iterations.
        :param tol: Convergence tolerance.
        :return: Final PageRank scores.
        """

        # Initialize PageRank scores
        pagerank = self.initialize_pagerank()

        # Calculate informativeness and relatedness
        self.calculate_informativeness()
        self.calculate_relatedness()

        for _ in range(iterations):

            # Update and normalize PageRank scores
            new_pagerank = self.update_pagerank(pagerank, damping_factor)
            new_pagerank = self.normalize_pagerank(new_pagerank)

            # Check for convergence
            if all(abs(new_pagerank[triple] - pagerank[triple]) < tol for triple in self.triples):
                break

            pagerank = new_pagerank
        return pagerank

    def evaluate(self, topK, relin_pagerank_scores):
        """
        Evaluate the model using the provided parameters.
        :param topK: Top K results.
        :param relin_pagerank_scores: pagerank scores for each node.
        :param results: List to store results.
        """
        f1_list = []
        f1_list_dynamic = []
        average_precisions = []
        average_precisions_dynamic = []

        for root in self.root_nodes:

            # Get the ground truth summary nodes based on 'summary_for' attribute
            ground_truth_summary = [(subject, info.get('predicate', None), obj) for subject, obj,
                                    info in self.desc[root] if info.get('summary_for', False) and info.get('summary_for') == root]

            # Get predicted summary nodes
            related_triples = {triple: score for triple, score in relin_pagerank_scores.items(
            ) if triple[0] == root or triple[2] == root}
            sorted_triples = sorted(
                related_triples.items(), key=lambda item: item[1], reverse=True)

            predicted_summary_all = sorted_triples[:topK]
            predicted_summary_dynamic_all = sorted_triples[:len(
                ground_truth_summary)]

            predicted_summary = [item[0] for item in predicted_summary_all]
            predicted_summary_dynamic = [item[0]
                                         for item in predicted_summary_dynamic_all]

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
        result = {"Method": f"Relin ({topK})"}

        print(f"Results for: Relin topK={topK}")
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

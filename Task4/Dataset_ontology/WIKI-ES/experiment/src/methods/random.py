import numpy as np
from collections import defaultdict
import random


from src.dataset_loader import DatasetLoader
from src.evaluation_metrics import EvaluationMetrics


class RandomSummarizer:

    def __init__(self, dataset_name, data_path, method_type):
        """
        Initialize the RandomSummarizer object.
        :param dataset_name: Name of the dataset.
        :param data_path: Path to the dataset.
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.G, self.nodes, self.node_labels, self.root_nodes, self.relation_frequency, self.desc = DatasetLoader.load_dataset(
            data_path)
        self.predicate_frequency = self.extract_predicate_frequency()
        self.nodes_degree = self.extract_node_degree()
        self.method_type = method_type

    def extract_predicate_frequency(self):
        """
        Extract the frequency of each predicate in the graph.
        :return: Dictionary with predicate frequencies.
        """
        predicate_frequency = defaultdict(int)
        for u, v, data in self.G.edges(data=True):
            predicate = data.get('predicate')
            if predicate:
                predicate_frequency[predicate] += 1

        return predicate_frequency

    def extract_node_degree(self):
        """
        Extract the combined in-degrees and out-degrees of each node.
        :return: Dictionary with node degrees.
        """
        in_degrees = dict(self.G.in_degree())
        out_degrees = dict(self.G.out_degree())

        # Combine in-degrees and out-degrees
        degrees = {node: in_degrees[node] + out_degrees[node]
                   for node in self.G.nodes()}
        return degrees

    def evaluate(self, topK):
        """
        Evaluate the model using the specified method type.
        :param topK: Top K results to consider for evaluation.
        :param method_type: Method type for generating summaries ('random', 'node_frequency', 'reverse_node_frequency', 'relation_frequency', 'reverse_relation_frequency').
        :return: Dictionary with evaluation results.
        """
        random.seed(42)

        f1_list = []
        f1_list_dynamic = []
        average_precisions = []
        average_precisions_dynamic = []

        for root in self.root_nodes:
            all_triples = [(subject, info.get('predicate', None), obj)
                           for subject, obj, info in self.desc[root]]
            random.shuffle(all_triples)

            # Get the ground truth summary nodes based on 'summary_for' attribute
            ground_truth_summary = [(subject, info.get('predicate', None), obj) for subject, obj, info in self.desc[root] if info.get(
                'summary_for', False) and info.get('summary_for') == root]

            if self.method_type == 'random':
                # Pick predicted summary randomly
                predicted_summary = random.sample(all_triples, topK)
                predicted_summary_dynamic = random.sample(
                    all_triples, len(ground_truth_summary))

            elif self.method_type == 'node_frequency':
                # Assign frequency based on the node which is not the root
                triples_with_frequency = []
                for subject, predicate, obj in all_triples:
                    if subject == root:
                        frequency = self.nodes_degree.get(obj, 0)
                    elif obj == root:
                        frequency = self.nodes_degree.get(subject, 0)
                    else:
                        continue
                    triples_with_frequency.append(
                        (subject, predicate, obj, frequency))

                # Sort triples based on the frequency
                sorted_triples_with_frequency = sorted(
                    triples_with_frequency, key=lambda x: x[3], reverse=True)

                sorted_triples = [(subject, predicate, obj) for subject,
                                  predicate, obj, frequency in sorted_triples_with_frequency]

                predicted_summary = sorted_triples[:topK]
                predicted_summary_dynamic = sorted_triples[:len(
                    ground_truth_summary)]

            elif self.method_type == 'reverse_node_frequency':
                # Assign frequency based on the node which is not the root
                triples_with_frequency = []
                for subject, predicate, obj in all_triples:
                    if subject == root:
                        frequency = self.nodes_degree.get(obj, 0)
                    elif obj == root:
                        frequency = self.nodes_degree.get(subject, 0)
                    else:
                        continue
                    triples_with_frequency.append(
                        (subject, predicate, obj, frequency))

                # Sort triples based on the frequency
                sorted_triples_with_frequency = sorted(
                    triples_with_frequency, key=lambda x: x[3], reverse=False)

                sorted_triples = [(subject, predicate, obj) for subject,
                                  predicate, obj, frequency in sorted_triples_with_frequency]

                predicted_summary = sorted_triples[:topK]
                predicted_summary_dynamic = sorted_triples[:len(
                    ground_truth_summary)]

            elif self.method_type == 'relation_frequency':
                # Assign frequency based on the node which is not the root
                triples_with_frequency = []
                for subject, predicate, obj in all_triples:
                    frequency = self.predicate_frequency.get(predicate, 0)
                    triples_with_frequency.append(
                        (subject, predicate, obj, frequency))

                # Sort triples based on the frequency
                sorted_triples_with_frequency = sorted(
                    triples_with_frequency, key=lambda x: x[3], reverse=True)

                sorted_triples = [(subject, predicate, obj) for subject,
                                  predicate, obj, frequency in sorted_triples_with_frequency]

                predicted_summary = sorted_triples[:topK]
                predicted_summary_dynamic = sorted_triples[:len(
                    ground_truth_summary)]

            elif self.method_type == 'reverse_relation_frequency':
                # Assign frequency based on the node which is not the root
                triples_with_frequency = []
                for subject, predicate, obj in all_triples:
                    frequency = self.predicate_frequency.get(predicate, 0)
                    triples_with_frequency.append(
                        (subject, predicate, obj, frequency))

                # Sort triples based on the frequency
                sorted_triples_with_frequency = sorted(
                    triples_with_frequency, key=lambda x: x[3], reverse=False)

                sorted_triples = [(subject, predicate, obj) for subject,
                                  predicate, obj, frequency in sorted_triples_with_frequency]

                predicted_summary = sorted_triples[:topK]
                predicted_summary_dynamic = sorted_triples[:len(
                    ground_truth_summary)]

            else:
                print("choose a model name correctly")

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
        result = {"Method": f"{self.method_type} ({topK})"}

        print(f"Results for: {self.method_type} topK={topK}")
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

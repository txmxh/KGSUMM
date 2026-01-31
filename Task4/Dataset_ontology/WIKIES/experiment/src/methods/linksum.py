from tqdm import tqdm
import requests
import numpy as np
import networkx as nx

from src.dataset_loader import DatasetLoader
from src.triple_processor import TripleProcessor
from src.evaluation_metrics import EvaluationMetrics
from src.backlink_extractor import BacklinkExtractor


class Linksum:

    def __init__(self, dataset_name, data_path, result_path_backlink, alpha=0.5):
        """
        Initialize the Linksum object.
        :param dataset_name: Name of the dataset.
        :param data_path: Path to the dataset.
        """
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.G, self.nodes, self.node_labels, self.root_nodes, self.relation_frequency, self.desc = DatasetLoader.load_dataset(
            data_path)
        self.triples = TripleProcessor.extract_triples(self.G)
        self.result_path_backlink = result_path_backlink
        self.backlinks = self.load_backlink()

    def load_backlink(self):
        extractor = BacklinkExtractor(
            self.data_path, self.result_path_backlink)
        extractor.run()
        return extractor.dict_roots_neighbors

    def initialize_pagerank(self):
        """
        Initialize PageRank scores with equal values for all nodes.
        :return: Dictionary with initial PageRank scores for each node.
        """
        pagerank_scores = nx.pagerank(self.G)
        return pagerank_scores

    def get_wikipedia_title_from_id(self, page_id, session):
        """
        Fetch the Wikipedia page title using the page ID.
        :param page_id: Wikipedia page ID.
        :param session: Requests session.
        :param url: Wikipedia API URL.
        :return: Title of the Wikipedia page.
        """
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "pageids": page_id,
            "format": "json"
        }

        response = session.get(url=url, params=params)
        data = response.json()

        if 'query' in data and 'pages' in data['query']:
            pages = data['query']['pages']
            page = pages.get(str(page_id))
            if page:
                return page.get('title')
        return None

    def get_page_id(self, page_id):
        """
        Extract the page ID from a list or return it directly if it's an integer.
        :param page_id: List of page IDs or a single page ID.
        :return: Single page ID.
        """
        if isinstance(page_id, list) and page_id:
            return page_id[0]
        elif isinstance(page_id, int):
            return page_id

    def check_bidirectional_backlink(self, page_id1, page_id2, session):
        """
        Check if there is a bidirectional backlink between two Wikipedia pages.
        :param page_id1: Wikipedia page ID for the first page.
        :param page_id2: Wikipedia page ID for the second page.
        :param session: Requests session.
        :return: 1 if bidirectional backlink exists, 0 otherwise.
        """
        page_id1 = self.get_page_id(page_id1)
        page_id2 = self.get_page_id(page_id2)

        if (page_id1 == None) or (page_id2 == None):
            return 0

        page_title1 = self.get_wikipedia_title_from_id(page_id1, session)
        page_title2 = self.get_wikipedia_title_from_id(page_id2, session)

        if page_title1 and page_title2:
            backlinks1 = self.backlinks[str(page_id1)]
            backlinks2 = self.backlinks[str(page_id2)]

            return int((page_title1 in backlinks2) and (page_title2 in backlinks1))
        else:
            return 0

    def relation_selection(self, selected_triple):
        """
        Select the most frequent relation from the given triples.
        :param selected_triple: List of triples connected to a specific node.
        :return: The triple with the most frequent relation.
        """
        relation_freq_dict = dict(self.relation_frequency)

        # Extract the relations from the selected_triple
        relations = [triple[1] for triple in selected_triple]

        # Sort relations based on their frequency
        frequent_relation = sorted(
            relations, key=lambda n: relation_freq_dict.get(n, 0), reverse=True)[0]

        # Return the first triple with the most frequent relation
        for triple in selected_triple:
            if triple[1] == frequent_relation:
                return triple

        return None

    def summarize_entity(self, target_node, pagerank_scores, session, alpha=0.5):
        """
        Summarize an entity by selecting relevant triples based on PageRank and backlink scores.
        :param target_node: The target node to summarize.
        :param pagerank_scores: Dictionary of PageRank scores for all nodes.
        :param session: Requests session.
        :return: List of selected triples for the target node.
        """

        # Find all connected nodes to the target node
        des_nodes = [node for node in self.G.neighbors(target_node)]
        des_triples = [(triple[0], triple[1], triple[2])
                       for triple in self.triples if triple[0] == target_node or triple[2] == target_node]

        # Find the max PageRank score among the des nodes
        max_score_among_des_nodes = max(
            (pagerank_scores[n] for n in des_nodes), default=0)

        # Normalize the scores
        if max_score_among_des_nodes > 0:
            normalized_pagerank_scores = {
                n: pagerank_scores[n] / max_score_among_des_nodes for n in des_nodes if n in pagerank_scores}
        else:
            # Handle the case where max score is zero
            normalized_pagerank_scores = {n: 0 for n in des_nodes}

        backlink_scores = {n: self.check_bidirectional_backlink(
            self.G.nodes[n].get('wikipedia_id', None), self.G.nodes[target_node].get('wikipedia_id', None), session) for n in des_nodes}
        combined_scores = {node: alpha * normalized_pagerank_scores[node] + (
            1 - alpha) * backlink_scores[node] for node in des_nodes}

        related_resources = sorted(
            des_nodes, key=lambda n: combined_scores[n], reverse=True)

        selected_relations = []
        for node in related_resources:
            selected_triple = []
            for triple in des_triples:
                if triple[0] == node or triple[2] == node:
                    selected_triple.append(triple)

            # Check if there is more than one ralation betwween two specific nodes
            if len(selected_triple) > 1:
                selected_relations.append(
                    self.relation_selection(selected_triple))
            else:
                selected_relations.append(selected_triple)

        return selected_relations

    def process_node(self, node, pagerank_scores, session):
        """
        Process a node to generate its summary.
        :param node: Node to process.
        :param pagerank_scores: Dictionary of PageRank scores.
        :param session: Requests session.
        :return: Tuple containing the node and its summary.
        """
        return node, self.summarize_entity(node, pagerank_scores, session)

    def run(self):
        """
        Run the Linksum algorithm to summarize entities.
        :return: Dictionary of summaries for each root node.
        """

        session = requests.Session()
        pagerank_scores = self.initialize_pagerank()

        summaries = {}

        # Run the summarization sequentially and display progress
        for node in tqdm(self.root_nodes, desc="Summarizing Entities", leave=True, ncols=100, mininterval=1):
            node, summary = self.process_node(node, pagerank_scores, session)
            summaries[node] = summary

        session.close()

        return summaries

    def evaluate(self, topK, summaries):
        """
        Evaluate the model using the provided parameters.
        :param topK: Top K results.
        :param summaries: Dictionary of summaries for each root node.
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
            predicted_summary_topK = summaries[root][:topK]
            predicted_summary_dynamic = summaries[root][:len(
                ground_truth_summary)]

            # Flatten the lists
            predicted_summary = [
                triple for sublist in predicted_summary_topK for triple in sublist]
            predicted_summary_dynamic = [
                triple for sublist in predicted_summary_dynamic for triple in sublist]

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
        result = {"Method": f"Linksum ({topK})"}

        print(f"Results for: Linksum topK={topK}")
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

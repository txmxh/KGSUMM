import os
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from src.dataset_loader import DatasetLoader


class BacklinkExtractor:
    def __init__(self, data_path, result_path):
        """
        Initialize the BacklinkExtractor.
        :param data_path: Path to the input data.
        :param result_path: Path to save or load the results.
        """
        self.data_path = data_path
        self.result_path = result_path
        self.graph, self.root_nodes = None, None
        self.nodes_for_extraction = set()
        self.dict_roots_neighbors = {}

    def load_graph(self):
        """
        Load the graph data from the dataset.
        """
        self.graph, _, _, self.root_nodes, _, _ = DatasetLoader.load_dataset(
            self.data_path)

    def get_wiki_id(self, node):
        """
        Get the Wikipedia ID of a node.
        :param node: Node to get the Wikipedia ID for.
        :return: Wikipedia ID of the node.
        """
        wiki_id = self.graph.nodes[node].get('wikipedia_id')
        if isinstance(wiki_id, list) and wiki_id:
            return wiki_id[0]
        elif isinstance(wiki_id, int):
            return wiki_id

    def get_wikipedia_title_from_id(self, page_id, session, url):
        """
        Fetch the Wikipedia page title using the page ID.
        :param page_id: Wikipedia page ID.
        :param session: Requests session.
        :param url: Wikipedia API URL.
        :return: Title of the Wikipedia page.
        """
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

    def get_wikipedia_connected_links(self, page_title, session, url):
        """
        Retrieve a list of page titles that link to the given page.
        :param page_title: Title of the Wikipedia page.
        :param session: Requests session.
        :param url: Wikipedia API URL.
        :return: List of page titles that link to the given page.
        """
        blcontinue = None  # Used for handling pagination
        all_links = []

        while True:

            params = {
                "action": "query",
                "format": "json",
                "list": "backlinks",
                "bltitle": page_title,
                "bllimit": "max",
                # {None: all, 0: Article, 1: Talk, 2: User, 3: User talk, 4: Wikipedia, 5: Wikipedia talk, 6: File, 7: File talk, 8: MediaWiki, 9: MediaWiki talk, 10: Template, 11: Template talk, 12: Help, 13: Help talk, 14: Category, 15: Category talk}
                "blnamespace": "0"
            }

            if blcontinue:
                params['blcontinue'] = blcontinue

            response = session.get(url=url, params=params)
            data = response.json()

            all_links.extend([link['title']
                              for link in data['query']['backlinks']])

            if 'continue' in data:
                blcontinue = data['continue']['blcontinue']
            else:
                break

        return all_links

    def collect_nodes_for_extraction(self):
        """
        Collect nodes for extraction based on their Wikipedia IDs.
        """
        for root in self.root_nodes:
            self.nodes_for_extraction.add(
                self.graph.nodes[root].get('wikipedia_id'))

            for in_node in self.graph.predecessors(root):
                self.nodes_for_extraction.add(self.get_wiki_id(in_node))

            for out_node in self.graph.successors(root):
                self.nodes_for_extraction.add(self.get_wiki_id(out_node))

    def run_backlink_extractor(self):
        """
        Run the backlink extraction process using multithreading to speed up API calls.
        """
        session = requests.Session()
        url = "https://en.wikipedia.org/w/api.php"

        nodes_list = list(self.nodes_for_extraction)

        # Use ThreadPoolExecutor to run the tasks in parallel
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(
                self.process_node, page_id, session, url): page_id for page_id in nodes_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Backlink"):
                n, backlinks = future.result()
                if backlinks is not None:
                    self.dict_roots_neighbors[n] = backlinks

        print("Backlink Computation Done!")
        session.close()

    def process_node(self, page_id, session, url):
        """
        Process a node to get its Wikipedia page title and backlinks.
        :param page_id: Wikipedia page ID.
        :param session: Requests session.
        :param url: Wikipedia API URL.
        :return: Tuple containing the page ID and its backlinks.
        """
        page_title = self.get_wikipedia_title_from_id(page_id, session, url)
        if page_title is not None:
            backlinks = self.get_wikipedia_connected_links(
                page_title, session, url)
            return page_id, backlinks
        return page_id, None

    def save_results(self):
        """
        Save the extracted results to a file.
        """
        with open(self.result_path, 'w') as file:
            json.dump(self.dict_roots_neighbors, file)

        print(f"Results saved to {self.result_path}")

    def load_results(self):
        """
        Load the extracted results from a file.
        """
        with open(self.result_path, 'r') as file:
            self.dict_roots_neighbors = json.load(file)
        print(f"Results loaded from {self.result_path}")

    def run(self):
        """
        Run the full extraction process, including loading the graph, collecting nodes, extracting backlinks, and saving the results.
        """
        if os.path.exists(self.result_path):
            self.load_results()
        else:
            self.load_graph()
            self.collect_nodes_for_extraction()
            self.run_backlink_extractor()
            self.save_results()

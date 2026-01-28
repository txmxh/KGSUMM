import os

from rdflib.graph import Graph
from pykeen.triples import TriplesFactory
import networkx as nx
from collections import Counter, defaultdict
import pickle
import numpy as np
import pandas as pd
import json


def import_ESBM(path):
    lst = os.listdir(path)
    len_index = len(lst)
    g = Graph()
    for i in range(1, len_index + 1):
        g_temp = Graph()
        entity_file_name = path + str(i) + "_desc.nt"
        g = g + g_temp.parse(entity_file_name, format='nt')
    return g


def import_ESBM_lmdb(path):
    g = Graph()
    index = [i for i in range(101, 141)]
    index += [i for i in range(166, 176)]
    for i in index:
        g_temp = Graph()
        entity_file_name = path + str(i) + "_desc.nt"
        g = g + g_temp.parse(entity_file_name, format='nt')
    return g


def import_ESBM_dbpedia(path):
    g = Graph()
    index = [i for i in range(1, 101)]
    index += [i for i in range(141, 166)]
    for i in index:
        g_temp = Graph()
        entity_file_name = path + str(i) + "_desc.nt"
        g = g + g_temp.parse(entity_file_name, format='nt')
    return g


def import_FACES(path):
    g = Graph()
    triples = []
    index = range(1, 51)
    for i in index:
        folder_name = path + '/' + str(i) + '/'
        g_temp = Graph()
        entity_file_name = folder_name + str(i) + "_desc.nt"
        g = g + g_temp.parse(entity_file_name, format='nt')

    for s, p, o in g:
        triples.append((s.n3(), p.n3(), o.n3()))

    return triples


def import_INFO(path):
    all_triples = []
    for folder_number in range(0, 100):
        folder_path = os.path.join(path, str(folder_number))
        file_path = os.path.join(folder_path, 'original_description.ttl')

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = file.read()

            subject = None
            triples = []
            current_predicate = None
            parts = data.split(';')

            for part in parts:
                lines = part.strip().split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue

                    if line.startswith('<http') and subject is None:
                        subject = line.split()[0]
                        continue

                    if line.startswith('a'):
                        current_predicate = 'a'
                        objects = line[1:].strip().split(' , ')
                    elif line.startswith('<http'):
                        if i == 0:
                            current_predicate = line.strip()
                            continue
                        else:
                            objects = line.split(' , ')
                    else:
                        objects = line.split(' , ')

                    for obj in objects:
                        obj = obj.strip().rstrip(',').rstrip('.')
                        if obj:
                            triples.append((subject, current_predicate, obj))

            all_triples.extend(triples)
    return all_triples


def import_WIKES(path):
    with open(path, 'rb') as f:
        G: nx.MultiDiGraph = pickle.load(f)

    # Root nodes
    root_nodes = [node for node, data in G.nodes(
        data=True) if data.get('is_root', False)]

    nodes = [n for n, data in G.nodes(data=True)]
    node_labels = {n: data.get('wikidata_label')
                   for n, data in G.nodes(data=True)}

    desc = {n: [] for n in nodes}

    for n in nodes:
        # Outgoing edges
        desc[n].extend([(n, v, d) for n, v, d in G.out_edges(n, data=True)])
        # Incoming edges
        desc[n].extend([(u, n, d) for u, n, d in G.in_edges(n, data=True)])

    # Extract relation frequency in graph
    predicate_frequency = defaultdict(int)
    for _, _, data in G.edges(data=True):
        predicate = data.get('predicate')
        if predicate:
            predicate_frequency[predicate] += 1

    # Create a sorted list of predicates and their frequencies
    relation_frequency = sorted(
        predicate_frequency.items(), key=lambda x: x[1], reverse=True)

    return G, nodes, node_labels, root_nodes, relation_frequency, desc


def extract_triples_from_file(path):
    with open(path, 'r') as file:
        data = file.read()

    subject = None
    triples = []
    current_predicate = None
    parts = data.split(';')

    for part in parts:
        lines = part.strip().split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            if line.startswith('<http') and subject is None:
                subject = line.split()[0]
                continue

            if line.startswith('a'):
                current_predicate = 'a'
                objects = line[1:].strip().split(' , ')
            elif line.startswith('<http'):
                if i == 0:
                    current_predicate = line.strip()
                    continue
                else:
                    objects = line.split(' , ')
            else:
                objects = line.split(' , ')

            for obj in objects:
                obj = obj.strip().rstrip(',').rstrip('.')
                if obj:
                    triples.append((subject, current_predicate, obj))

    return triples


def extract_predicate_labels(graph):
    predicate_to_label = defaultdict(str)
    for u, v, data in graph.edges(data=True):
        predicate = data.get('predicate')
        predicate_label = data.get('predicate_label')
        predicate_to_label[predicate] = predicate_label
    return predicate_to_label


def generate_triples_esbm(dataset_version, dataset_name, data_path):
    path_dataset = f'{data_path}/ESBM_descriptions/'
    triples = []

    if dataset_version == 'esbm':
        if dataset_name == 'dbpedia':
            g = import_ESBM_dbpedia(path_dataset)
            for s, p, o in g:
                triples.append([s.n3(), p.n3(), o.n3()])
            triples = np.array(triples)
            triples_factory = TriplesFactory.from_labeled_triples(triples)
        elif dataset_name == 'lmdb':
            g = import_ESBM_lmdb(path_dataset)
            for s, p, o in g:
                triples.append([s.n3(), p.n3(), o.n3()])
            triples = np.array(triples)
            triples_factory = TriplesFactory.from_labeled_triples(triples)

    triples = triples_factory.triples
    entity_id = triples_factory.entity_to_id
    relation_id = triples_factory.relation_to_id

    return triples, entity_id, relation_id


def generate_entity_dataset_esbm(dataset_name, data_path):
    entitylist = pd.read_csv(
        data_path + '/ESBM-groundtruth/elist.txt', sep='\t', index_col=0)
    if dataset_name == 'lmdb':
        entity_dataset = entitylist[entitylist.dataset == 'lmdb']
    if dataset_name == 'dbpedia':
        entity_dataset = entitylist[entitylist.dataset == 'dbpedia']

    return entity_dataset


def generate_G_esbm(triples):
    G = nx.MultiDiGraph()
    for head, relation, tail in triples:
        G.add_edge(head, tail, relation=relation)
    return G


def load_graph_esbm(dataset_version, dataset_name, data_path):
    # Return triples and graphs
    triples, entity_id, relation_id = generate_triples_esbm(
        dataset_version, dataset_name, data_path)
    entity_dataset = generate_entity_dataset_esbm(dataset_name, data_path)
    G = generate_G_esbm(triples)
    nodes = list(G.nodes())

    entity_node_id = {node: i for i, node in enumerate(nodes)}

    return entity_dataset, G, nodes, entity_node_id, relation_id, triples


def analyze_graph_info(triples):
    """
    Analyze the graph information from the given triples.
    :param triples: List of triples (subject, predicate, object).
    """
    G = nx.MultiDiGraph()

    # Add edges to the graph
    for edge in triples:
        G.add_edge(edge[0], edge[2], label=edge[1])

    G_undirected = nx.to_undirected(G)

    # Check if the graph is connected
    is_connected = nx.is_connected(G_undirected)
    num_connected_components = nx.number_connected_components(G_undirected)

    # Calculate density
    density = nx.density(G)

    # Calculate in-degrees and out-degrees
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())

    # Combine in-degrees and out-degrees
    degrees = {node: in_degrees[node] + out_degrees[node]
               for node in G.nodes()}

    min_degree_node = min(degrees, key=lambda x: degrees[x])
    max_degree_node = max(degrees, key=lambda x: degrees[x])

    # Get the number of nodes and edges
    num_nodes = len(G.nodes)
    num_edges = len(G.edges)

    # Print the analysis results
    print(f"Directed Graph Density: {density}")
    print(f"Is Connected: {is_connected}")
    print(f"Number of Connected Components: {num_connected_components}")
    print(
        f"Node with Minimum Total Degree: ({min_degree_node}, {degrees[min_degree_node]})")
    print(
        f"Node with Maximum Total Degree: ({max_degree_node}, {degrees[max_degree_node]})")
    print(f"Number of Nodes: {num_nodes}")
    print(f"Number of Edges: {num_edges}")


def save_results(model_scores, dataset_name, method):
    """
    Save the model results to a pkl file.
    :param model_scores: model score results to be saved.
    :param dataset_name: Name of the dataset.
    :param method: Method.
    """

    filename = f'./results/{method}_{dataset_name}_unsupervised.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(model_scores, file)

    print(f"Results saved to {filename}")


def save_results_json(result, dataset_name, method):
    """
    Save the model results to a pkl file.
    :param result: model results to be saved.
    :param dataset_name: Name of the dataset.
    :param method: Method.
    """

    filename = f'./results/{method}_{dataset_name}_unsupervised.json'

    with open(filename, 'w') as file:
        json.dump(result, file)

    print(f"Results saved to {filename}")


def save_results_to_csv(results, dataset_name, method, method_type_random, topK):
    """
    Save the evaluation results to a CSV file.
    :param results: Evaluation results to be saved.
    :param dataset_name: Name of the dataset.
    :param method: Method used for evaluation.
    :param topK: Number of top results considered.
    """
    df = pd.DataFrame([results])

    if method_type_random != None:
        csv_file = f'./results/{dataset_name}_{method_type_random}_top{topK}_unsupervised.csv'
    else:
        csv_file = f'./results/{dataset_name}_{method}_top{topK}_unsupervised.csv'
    df.to_csv(csv_file, index=False)

    print(f"Results saved to {csv_file}")

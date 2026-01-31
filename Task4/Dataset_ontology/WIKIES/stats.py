import itertools
import os
import pickle

import networkx as nx


def find_pickle_files(directory):
    print(directory)
    pickle_files = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                if (not file.startswith('es_graph')
                        and not file.startswith('summary_graph')
                        and not file.startswith('expanded_graph')):
                    pickle_files[file[:-len('.pkl')]] = os.path.join(root, file)
    return pickle_files


def process_graph_file(file_path):
    with open(file_path, 'rb') as file:
        G: nx.MultiGraph = pickle.load(file)

    # Extract node and edge information
    nodes_data = list(G.nodes(data=True))
    edges_data = list(G.edges(data=True))

    num_roots = sum(1 for _, data in nodes_data if data.get('is_root', False))
    num_summary_edges = sum(1 for _, _, data in edges_data if 'summary_for' in data)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    predicates = set(data['predicate'] for _, _, data in edges_data if 'predicate' in data)

    category_distribution = {}
    for _, data in nodes_data:
        if data.get('is_root', False):
            category = data.get('category', 'unknown')
            category_distribution[category] = category_distribution.get(category, 0) + 1

    sorted_category_distribution = dict(sorted(category_distribution.items()))
    category_distribution_str = '<br/> '.join([f"{cat}={count}" for cat, count in sorted_category_distribution.items()])

    return {
        "file_path": file_path,
        "num_roots": num_roots,
        "num_summary_edges": num_summary_edges,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_predicates": len(predicates),
        "category_distribution_str": category_distribution_str,
    }


def generate_output(directory, files, base_url, time_report):
    pickle_files = find_pickle_files(directory)
    output_rows = []
    file_urls = ""
    for f in files:
        pickle_file = pickle_files.get(f)
        result = process_graph_file(pickle_file)
        file_url = f"[{f}]({base_url}/{f}.zip"
        file_urls += f"{file_url} \n"
        csv_url = f"[csv]({base_url}/{f}.zip)"
        graphml_url = f"[graphml]({base_url}/{f}.graphml)"
        croissant_url = f"[croissant.json]({base_url}/{f}.croissant.json)"
        output_rows.append(
            f"| {f} </br>{csv_url}, {graphml_url}, {croissant_url} | {result['num_roots']} | {result['num_summary_edges']} | "
            f"{result['num_nodes']} | {result['num_edges']} | {result['num_predicates']} |"
            f" {result['category_distribution_str']} | {time_report[f]}|"
        )

    header = ("| dataset (variant, size, None/train/val/test)                                                                                                                                                                                                                                                                                                                                          | #roots | #smmaries | #nodes | #edges | #labels | roots category distribution                                                                                                                                  | Running Time(sec) |")
    separator = ("|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-----------|--------|--------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|")
    output = f"{header}\n{separator}\n" + "\n".join(output_rows)

    return output, file_urls


if __name__ == "__main__":
    datasets = ['WikiLitArt', 'WikiCinema', 'WikiPro', 'WikiProFem']
    sizes = ['s', 'm', 'l']
    types = ['', '-train', '-val', '-test']
    dataset_files = list(itertools.product(list(itertools.product(datasets, sizes)), types))
    files = []
    for (dataset, size), t in dataset_files:
        files.append(f"{dataset}-{size}{t}")

    time_report = {

        'WikiLitArt-s': 91.934,
        'WikiLitArt-s-train': 66.023,
        'WikiLitArt-s-val': 14.364,
        'WikiLitArt-s-test': 14.6,
        'WikiLitArt-m': 155.368,
        'WikiLitArt-m-train': 111.636,
        'WikiLitArt-m-val': 22.957,
        'WikiLitArt-m-test': 26.187,
        'WikiLitArt-l': 353.113,
        'WikiLitArt-l-train': 244.544,
        'WikiLitArt-l-val': 57.263,
        'WikiLitArt-l-test': 60.466,
        'WikiCinema-s': 118.014,
        'WikiCinema-s-train': 84.364,
        'WikiCinema-s-val': 18.651,
        'WikiCinema-s-test': 19.851,
        'WikiCinema-m': 196.413,
        'WikiCinema-m-train': 142.091,
        'WikiCinema-m-val': 31.722,
        'WikiCinema-m-test': 33.674,
        'WikiCinema-l': 475.679,
        'WikiCinema-l-train': 333.148,
        'WikiCinema-l-val': 68.62,
        'WikiCinema-l-test': 87.07,
        'WikiPro-s': 126.119,
        'WikiPro-s-train': 89.874,
        'WikiPro-s-val': 21.021,
        'WikiPro-s-test': 21.743,
        'WikiPro-m': 208.157,
        'WikiPro-m-train': 141.563,
        'WikiPro-m-val': 36.045,
        'WikiPro-m-test': 36.967,
        'WikiPro-l': 489.409,
        'WikiPro-l-train': 334.864,
        'WikiPro-l-val': 84.089,
        'WikiPro-l-test': 92.545,
        'WikiProFem-s': 177.63,
        'WikiProFem-s-train': 127.614,
        'WikiProFem-s-val': 29.081,
        'WikiProFem-s-test': 27.466,
        'WikiProFem-m': 301.718,
        'WikiProFem-m-train': 217.699,
        'WikiProFem-m-val': 46.793,
        'WikiProFem-m-test': 46.317,
        'WikiProFem-l': 768.99,
        'WikiProFem-l-train': 544.893,
        'WikiProFem-l-val': 116.758,
        'WikiProFem-l-test': 118.524,
    }

    output_path = './data/output/'
    directory = '/home/mks/.wikes_data/1.0.5/'
    base_url = "https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5"
    output, file_urls = generate_output(directory, files, base_url, time_report)
    print(output)

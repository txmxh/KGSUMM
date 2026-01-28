import glob
import itertools
import os
import pickle


def load_graph(root_dir, file_path_pattern):
    files = glob.glob(os.path.join(root_dir, file_path_pattern))
    if not files:
        raise FileNotFoundError(f"No files match the pattern: {file_path_pattern}")
    file = files[0]
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_first_hop_neighbors(graph, root_nodes):
    first_hop_edges = set()
    first_hop_nodes = set()
    for node in root_nodes:
        first_hop_nodes.add(node)
        for neighbor, _, data in graph.in_edges(node, data=True):
            first_hop_nodes.add(neighbor)
            predicate = data.get('predicate')
            if predicate:
                first_hop_edges.add((neighbor, node, predicate))
        for _, neighbor, data in graph.out_edges(node, data=True):
            first_hop_nodes.add(neighbor)
            predicate = data.get('predicate')
            if predicate:
                first_hop_edges.add((node, neighbor, predicate))
    return first_hop_nodes, first_hop_edges


def check_first_hop_leakage(dataset, train_graph, val_graph, test_graph):
    train_roots = [node for node, data in train_graph.nodes(data=True) if data.get('is_root', False)]
    val_roots = [node for node, data in val_graph.nodes(data=True) if data.get('is_root', False)]
    test_roots = [node for node, data in test_graph.nodes(data=True) if data.get('is_root', False)]

    train_nodes, train_edges = get_first_hop_neighbors(train_graph, train_roots)
    val_nodes, val_edges = get_first_hop_neighbors(val_graph, val_roots)
    test_nodes, test_edges = get_first_hop_neighbors(test_graph, test_roots)

    val_edge_leakage = train_edges.intersection(val_edges)
    test_edge_leakage = train_edges.intersection(test_edges)

    val_node_leakage = train_nodes.intersection(val_nodes)
    test_node_leakage = train_nodes.intersection(test_nodes)

    print(f"Dataset {dataset}:")
    print("Edge Leakage:")
    print(f"Train set: {len(train_edges)} total edges")
    print(f"Validation set: {len(val_edge_leakage)} overlapping edges out of {len(val_edges)} total edges")
    print(f"Test set: {len(test_edge_leakage)} overlapping edges out of {len(test_edges)} total edges")

    print("Node Leakage:")
    print(f"Train set: {len(train_nodes)} total nodes")
    print(f"Validation set: {len(val_node_leakage)} overlapping nodes out of {len(val_nodes)} total nodes")
    print(f"Test set: {len(test_node_leakage)} overlapping nodes out of {len(test_nodes)} total nodes")
    print("*" * 50)


if __name__ == '__main__':
    datasets = ['1', '2', '3', '4']
    sizes = ['small', 'medium', 'large']

    output_path = './data/output/'
    for dataset, dataset_size in itertools.product(datasets, sizes):
        try:
            # Load the graphs
            train_graph = load_graph(
                os.path.join(output_path, f'{dataset}_{dataset_size}_train'),
                f'{dataset}_{dataset_size}_train.pkl'
            )
            val_graph = load_graph(
                os.path.join(output_path, f'{dataset}_{dataset_size}_val'),
                f'{dataset}_{dataset_size}_val.pkl'
            )
            test_graph = load_graph(
                os.path.join(output_path, f'{dataset}_{dataset_size}_test'),
                f'{dataset}_{dataset_size}_test.pkl'
            )
            check_first_hop_leakage(f'{dataset}_{dataset_size}', train_graph, val_graph, test_graph)
        except FileNotFoundError as e:
            print(f"Error loading graphs for dataset {dataset}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred for dataset {dataset}: {e}")

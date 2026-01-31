import argparse
from src.methods.relin import Relin
from src.methods.linksum import Linksum
from src.methods.random import RandomSummarizer
from src.methods.pagerank import Pagerank
from src.utils import save_results_to_csv, save_results, save_results_json


def run_method(method, dataset_name, data_path, topK, result_path_backlink=None, method_type_random=None):
    """
    Run the specified method with the provided parameters.
    :param method: The method to run ('random', 'linksum', 'relin', 'pagerank').
    :param dataset_name: Name of the dataset.
    :param data_path_template: Path template to the dataset.
    :param topK: Top K results to consider.
    :param result_path_backlink: Path to save or load backlink results.
    :param method_type_random: Method type for the RandomSummarizer.
    """

    if method == 'relin':
        model = Relin(dataset_name, data_path)
        model_scores = model.run()
        # Save scores results
        save_results(model_scores, dataset_name, method)
        evaluate_results = model.evaluate(topK, model_scores)

    elif method == 'linksum':
        model = Linksum(dataset_name, data_path, result_path_backlink)
        summaries = model.run()
        save_results_json(summaries, dataset_name, method)
        evaluate_results = model.evaluate(topK, summaries)

    elif method == 'pagerank':
        model = Pagerank(dataset_name, data_path)
        pagerank_scores = model.run()
        evaluate_results = model.evaluate(topK, pagerank_scores)

    elif method == 'random':
        if method_type_random is None:
            raise ValueError(
                "method_type_random must be specified for the 'random' method")

        model = RandomSummarizer(dataset_name, data_path, method_type_random)
        evaluate_results = model.evaluate(topK)

    else:
        raise ValueError(f"Unsupported method: {method}")

    # Save results to CSV
    save_results_to_csv(evaluate_results, dataset_name,
                        method, method_type_random, topK)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run summarization method on a dataset.')
    parser.add_argument('--dataset_name', type=str,
                        required=True, help='Name of the dataset')
    parser.add_argument('--method', type=str, required=True,
                        choices=['random', 'linksum', 'relin', 'pagerank'], help='Method to run')
    parser.add_argument('--topK', type=int, required=True,
                        help='Top K results to consider')
    parser.add_argument('--method_type_random', type=str, choices=['random', 'node_frequency', 'reverse_node_frequency',
                        'relation_frequency', 'reverse_relation_frequency'], help='Method type for the RandomSummarizer')

    args = parser.parse_args()

    data_path_template = f"./data/WIKES/{args.dataset_name}_small_unsupervised/{args.dataset_name}_small_unsupervised.pkl"
    result_path_backlink = f"./results/backlink/backlinks_{args.dataset_name}_small_unsupervised.json"

    run_method(args.method, args.dataset_name, data_path_template,
               args.topK, result_path_backlink, args.method_type_random)

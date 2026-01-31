import logging

from graph_generator.process_manager import ProcessManager
from graph_generator.config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

if __name__ == '__main__':
    config = Config()

    name = config.dataset_name
    min_random_walk_number = config.min_random_walk_number
    max_random_walk_number = config.max_random_walk_number
    seed_nodes = config.seed_node_ids
    categories = config.categories
    if not name:
        raise ValueError("Please provide the dataset name.")
    if not min_random_walk_number:
        raise ValueError("Please provide the min_random_walk_number.")
    if not max_random_walk_number:
        raise ValueError("Please provide the max_random_walk_number.")
    if min_random_walk_number >= max_random_walk_number:
        raise ValueError("min_random_walk_number should be less than max_random_walk_number.")
    if not seed_nodes:
        raise ValueError("Please provide the seed_node_ids.")

    ProcessManager(name, seed_nodes, min_random_walk_number, max_random_walk_number, categories).run_process()

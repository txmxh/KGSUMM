import logging
import os
import shutil
import tempfile
from functools import lru_cache

import pandas as pd

from graph_generator.process_manager import ProcessManager
from graph_generator.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.FileHandler('./app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@lru_cache
def load_seed_nodes(path) -> tuple[list[str], list[str]]:
    """
    :param path:  csv file path of the seed nodes
    :return list of the seed nodes based on wikidata_id
    """
    df = pd.read_csv(path, index_col=1)
    return df.index.tolist(), df['level3_main_occ'].tolist()


def compress_datasets(dataset_configs: dict[str, tuple[str, int, int]], output_path: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        for dataset_name in dataset_configs.keys():
            directory = os.path.join(output_path, dataset_name)
            target_dir = os.path.join(temp_dir, dataset_name)
            if os.path.exists(target_dir):
                os.remove(target_dir)
            os.makedirs(target_dir)
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.startswith(dataset_name):
                        new_file_name = file[:len(dataset_name)] + file[len(dataset_name) + 1 + 64:]
                        shutil.copy(os.path.join(root, file), os.path.join(target_dir, new_file_name))

        zip_path = os.path.join(output_path, f"WikiES-datasets")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.make_archive(zip_path, 'zip', temp_dir)


if __name__ == '__main__':
    config = Config()

    dataset_configs = {
        '1_small_unsupervised': ('./data/seed_nodes/1/1.csv', 100, 300),
        '1_small_train': ('./data/seed_nodes/1/1-train.csv', 100, 300),
        '1_small_val': ('./data/seed_nodes/1/1-val.csv', 100, 300),
        '1_small_test': ('./data/seed_nodes/1/1-test.csv', 100, 300),

        '1_medium_unsupervised': ('./data/seed_nodes/1/1.csv', 150, 600),
        '1_medium_train': ('./data/seed_nodes/1/1-train.csv', 150, 600),
        '1_medium_val': ('./data/seed_nodes/1/1-val.csv', 150, 600),
        '1_medium_test': ('./data/seed_nodes/1/1-test.csv', 150, 600),

        '1_large_unsupervised': ('./data/seed_nodes/1/1.csv', 300, 1800),
        '1_large_train': ('./data/seed_nodes/1/1-train.csv', 300, 1800),
        '1_large_val': ('./data/seed_nodes/1/1-val.csv', 300, 1800),
        '1_large_test': ('./data/seed_nodes/1/1-test.csv', 300, 1800),

        '2_small_unsupervised': ('./data/seed_nodes/2/2.csv', 100, 300),
        '2_small_train': ('./data/seed_nodes/2/2-train.csv', 100, 300),
        '2_small_val': ('./data/seed_nodes/2/2-val.csv', 100, 300),
        '2_small_test': ('./data/seed_nodes/2/2-test.csv', 100, 300),

        '2_medium_unsupervised': ('./data/seed_nodes/2/2.csv', 150, 600),
        '2_medium_train': ('./data/seed_nodes/2/2-train.csv', 150, 600),
        '2_medium_val': ('./data/seed_nodes/2/2-val.csv', 150, 600),
        '2_medium_test': ('./data/seed_nodes/2/2-test.csv', 150, 600),

        '2_large_unsupervised': ('./data/seed_nodes/2/2.csv', 300, 1800),
        '2_large_train': ('./data/seed_nodes/2/2-train.csv', 300, 1800),
        '2_large_val': ('./data/seed_nodes/2/2-val.csv', 300, 1800),
        '2_large_test': ('./data/seed_nodes/2/2-test.csv', 300, 1800),

        '3_small_unsupervised': ('./data/seed_nodes/3/3.csv', 100, 300),
        '3_small_train': ('./data/seed_nodes/3/3-train.csv', 100, 300),
        '3_small_val': ('./data/seed_nodes/3/3-val.csv', 100, 300),
        '3_small_test': ('./data/seed_nodes/3/3-test.csv', 100, 300),

        '3_medium_unsupervised': ('./data/seed_nodes/3/3.csv', 150, 600),
        '3_medium_train': ('./data/seed_nodes/3/3-train.csv', 150, 600),
        '3_medium_val': ('./data/seed_nodes/3/3-val.csv', 150, 600),
        '3_medium_test': ('./data/seed_nodes/3/3-test.csv', 150, 600),

        '3_large_unsupervised': ('./data/seed_nodes/3/3.csv', 300, 1800),
        '3_large_train': ('./data/seed_nodes/3/3-train.csv', 300, 1800),
        '3_large_val': ('./data/seed_nodes/3/3-val.csv', 300, 1800),
        '3_large_test': ('./data/seed_nodes/3/3-test.csv', 300, 1800),

        '4_small_unsupervised': ('./data/seed_nodes/4/4.csv', 100, 300),
        '4_small_train': ('./data/seed_nodes/4/4-train.csv', 100, 300),
        '4_small_val': ('./data/seed_nodes/4/4-val.csv', 100, 300),
        '4_small_test': ('./data/seed_nodes/4/4-test.csv', 100, 300),

        '4_medium_unsupervised': ('./data/seed_nodes/4/4.csv', 150, 600),
        '4_medium_train': ('./data/seed_nodes/4/4-train.csv', 150, 600),
        '4_medium_val': ('./data/seed_nodes/4/4-val.csv', 150, 600),
        '4_medium_test': ('./data/seed_nodes/4/4-test.csv', 150, 600),

        '4_large_unsupervised': ('./data/seed_nodes/4/4.csv', 300, 1800),
        '4_large_train': ('./data/seed_nodes/4/4-train.csv', 300, 1800),
        '4_large_val': ('./data/seed_nodes/4/4-val.csv', 300, 1800),
        '4_large_test': ('./data/seed_nodes/4/4-test.csv', 300, 1800),
    }

    for name, (path, min_random_walk_number, max_random_walk_number) in dataset_configs.items():
        seed_nodes, categories = load_seed_nodes(path)
        ProcessManager(name, seed_nodes, min_random_walk_number, max_random_walk_number, categories).run_process()
    compress_datasets(dataset_configs, config.output_path)

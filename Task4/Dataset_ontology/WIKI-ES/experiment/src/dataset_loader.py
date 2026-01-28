from src.utils import import_WIKES


class DatasetLoader:
    @staticmethod
    def load_dataset(data_path):
        """
        Load the dataset.
        :param data_path: Path to the dataset.
        :return: Loaded graph, nodes, node labels, root nodes, description.
        """
        return import_WIKES(data_path)

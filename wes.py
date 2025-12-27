from pathlib import Path
from wikes_toolkit import WikESToolkit, WikESVersions, WikESGraph


WIKIDATA_ENTITY_URL = 'http://www.wikidata.org/entity/'
WIKIDATA_PROPERTY_URL = 'http://www.wikidata.org/prop/direct/'


class WES_DATASET:
    def __init__(self, output_dir='wes'):
        self.root_entities = None
        self.predicates = None
        self.graph = None
        self.output_root_dir = Path(output_dir)

    def read_data(self):
        toolkit = WikESToolkit(save_path="./wes-download-data")  # save_path is optional
        G = toolkit.load_graph(
            WikESGraph,
            # ----- Choose one of the datasets -----
            WikESVersions.V1.WikiCinema.SMALL_TEST,
            # WikESVersions.V1.WikiCinema.SMALL_TRAIN,
            # WikESVersions.V1.WikiLitArt.SMALL_TEST,
            # WikESVersions.V1.WikiCinema.SMALL_TEST,
            entity_formatter=lambda e: f"Entity({e.wikidata_label})",
            predicate_formatter=lambda p: f"Predicate({p.label})",  
            triple_formatter=lambda
                t: f"""<{WIKIDATA_ENTITY_URL}{t.subject_entity.identifier}>
                       <{WIKIDATA_PROPERTY_URL}{t.predicate.label.replace(' ', '_')}>
                       <{WIKIDATA_ENTITY_URL}{t.object_entity.identifier}>"""
            # we assume that the object mapping is only entities if not we can use conditional mapping for literals
        )
        self.root_nodes = G.root_entities()
        self.predicates = G.predicates()
        self.G = G
        print("Graph created successfully")

    def create_directory(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        print(f"created directory: {directory}")

    def create_file(self, file_path):
        file_path.touch()
        print(f"created file: {file_path}")

    def write_file_content(self, file_path, file_content):
        file_content = [t + ' .\n' for t in file_content]
        with file_path.open(mode='w', encoding='utf-8') as content_file:
            content_file.writelines(file_content)
        print(f"finished writing file {file_path} content")

    def format_single_item(self, item, type):
        if type == 'e':
            return f"<{WIKIDATA_ENTITY_URL}{item}>"
        else:
            return f"<{WIKIDATA_PROPERTY_URL}{item}>"

    def get_entity(self, id):
        iterator_entities = (n for n in self.root_nodes if n.identifier == id)
        return next(iterator_entities)

    def get_predicate(self, id):
        iterator_predicates = (p for p in self.predicates if p.predicate_id == id)
        return next(iterator_predicates)

    def get_batch_array(self, data_array, batch_size, max_iterations=None):
        """
        Yields successive batches, stopping after max_iterations if provided.
        """
        N = len(data_array)
        batches_yielded = 0
        for i in range(0, N, batch_size):
            if max_iterations is not None and batches_yielded >= max_iterations:
                # Stop the generator if the limit is reached
                return  
            yield data_array[i : i + batch_size]
            batches_yielded += 1
    

    def format_summary_triples(self, node, t):
        if node.identifier == t.subject_entity.identifier:
            return ' '.join([self.format_single_item(self.G.fetch_entity(t.subject_entity.identifier).identifier,'e'),
                             self.format_single_item(self.G.fetch_predicate(t.predicate.predicate_id).label.replace(' ', '_'), 'p'),
                             self.format_single_item(self.G.fetch_entity(t.object_entity.identifier).identifier, 'e')])
        else:
            return ' '.join([self.format_single_item(self.G.fetch_entity(t.object_entity.identifier).identifier, 'e'),
                             self.format_single_item(self.G.fetch_predicate(t.predicate.predicate_id).label.replace(' ', '_'), 'p'),
                             self.format_single_item(self.G.fetch_entity(t.subject_entity.identifier).identifier,'e')])

    def format_gold_triples(self, node, t):
        if node.identifier == t[0]:
            return ' '.join([self.format_single_item(self.G.fetch_entity(t[0]).identifier,'e'),
                             self.format_single_item(self.G.fetch_predicate(t[1]).label.replace(' ', '_'), 'p'),
                             self.format_single_item(self.G.fetch_entity(t[2]).identifier, 'e')])
        else:
            return ' '.join([self.format_single_item(self.G.fetch_entity(t[2]).identifier, 'e'),
                             self.format_single_item(self.G.fetch_predicate(t[1]).label.replace(' ', '_'), 'p'),
                             self.format_single_item(self.G.fetch_entity(t[0]).identifier,'e')])
    

    def build_dataset(self, name='wes'):
        self.create_directory(self.output_root_dir)
        index = 1

        for node in self.root_nodes:
            node_neightbors = self.G.neighbors(node)
            node_entity_info = self.G.fetch_entity(node.identifier)
            node_desc_all = [self.format_summary_triples(node, t) for t in node_neightbors]
            node_gold = [self.format_gold_triples(node, t) for t in self.G.ground_truths(node)]

            print(f"Working with entity - {node_entity_info.identifier} - {node_entity_info.wikidata_label}")

            entity_dir = self.output_root_dir / str(index)

            # ----- WRITE ENTITY DATA TO FILES -----
            self.create_directory(entity_dir)

            desc_file = str(index) + '_desc.nt'
            entity_desc_file = entity_dir / desc_file
    
            self.create_file(entity_desc_file)
            self.write_file_content(entity_desc_file, node_desc_all)

            # split and structure the gold into batches
            # entity_gold_batch_size = 10
            # number_of_ierations = len(node_gold) // entity_gold_batch_size

            # Cases of fewer gold summaries
            # if number_of_ierations == 0:
            t5file = str(index) + '_gold_top5_0.nt'
            t10file = str(index) + '_gold_top10_0.nt'
            gold_dynamic_file = str(index) + '_gold_dynamic.nt'

            top5_file = entity_dir / t5file
            top10_file = entity_dir / t10file
            dynamic_file = entity_dir / gold_dynamic_file

            self.create_file(top5_file)
            self.create_file(top10_file)
            self.create_file(dynamic_file)

            self.write_file_content(top5_file, node_gold[:5])
            self.write_file_content(top10_file, node_gold[:10])
            self.write_file_content(dynamic_file, node_gold)

            index += 1

        print("--------- Done creating dataset ---------")

if __name__ == '__main__':

    wes_ds = WES_DATASET()
    wes_ds.read_data()
    wes_ds.build_dataset()

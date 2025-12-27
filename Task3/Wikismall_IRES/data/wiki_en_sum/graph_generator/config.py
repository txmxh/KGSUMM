from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    arguments = {
        'min_valid_summary_edges': {
            'name': 'min_valid_summary_edges',
            'default': 5,
            'type': int,
            'help': "Minimum number of valid summaries for a seed ndoe"
        },
        'random_walk_depth_len': {
            'name': 'random_walk_depth_len',
            'default': 3,
            'type': int,
            'help': "Depth length of random walks (number of nodes in each ramdom walk)"
        },
        'bridges_number': {
            'name': 'bridges_number',
            'default': 5,
            'type': int,
            'help': "number of connecting path bridges between components"
        },

        'max_threads': {
            'name': 'max_threads',
            'default': min(8, os.cpu_count() or 1),
            'type': int,
            'help': "Maximum number of threads"
        },
        'output_path': {
            'name': 'output_path',
            'default': "./data/output",
            'help': "Path to save output data"
        },
        'db_name': {
            'name': 'db_name',
            'default': 'wikies',
            'help': 'Database name'
        },
        'db_user': {
            'name': 'db_user',
            'default': 'wikies',
            'help': 'Database user'
        },
        'db_password': {
            'name': 'db_password',
            'default': 'password',
            'help': 'Database password'
        },
        'db_host': {
            'name': 'db_host',
            'default': 'localhost',
            'help': 'Database host'
        },
        'db_port': {
            'name': 'db_port',
            'default': 5432,
            'type': int,
            'help': 'Database port'
        },
        'neo4j_user': {
            'name': 'neo4j_user',
            'default': 'neo4j',
            'help': 'Neo4j user'
        },
        'neo4j_password': {
            'name': 'neo4j_password',
            'default': 'password',
            'help': 'Neo4j password'
        },
        'neo4j_host': {
            'name': 'neo4j_host',
            'default': 'localhost',
            'help': 'Neo4j host'
        },
        'neo4j_port': {
            'name': 'neo4j_port',
            'default': 7687,
            'type': int,
            'help': 'Neo4j port'
        },
    }

    positional_arguments = {
        'dataset_name': {
            'name': 'dataset_name',
            'help': 'The name of the dataset to process (required)',
            'order': 1
        },
        'min_random_walk_number': {
            'name': 'min_random_walk_number',
            'type': int,
            'help': 'Minimum number of random walks for each seed node (required)',
            'order': 2
        },
        'max_random_walk_number': {
            'name': 'max_random_walk_number',
            'type': int,
            'help': 'Maximum number of random walks for each seed node (required)',
            'order': 3
        },
        'seed_node_ids': {
            'name': 'seed_node_ids',
            # 'type': list[str],
            'help': 'Seed node ids in comma separated format (required)',
            'order': 4
        },
        'categories': {
            'name': 'categories',
            # 'type': list[str],
            'help': 'Seed node categories in comma separated format (optional)',
            'order': 5
        }
    }

    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        self.parser = argparse.ArgumentParser()
        for k, v in self.arguments.items():
            default = os.getenv(v['name'].upper(), v['default'])
            arg_type = v['type'] if 'type' in v else str
            self.parser.add_argument(f"--{v['name']}", type=arg_type, default=default, help=v['help'])

        for k, v in sorted(self.positional_arguments.items(), key=lambda item: item[1]['order']):
            arg_type = v['type'] if 'type' in v else str
            if arg_type is list[str]:
                self.parser.add_argument(
                    f"{v['name']}",
                    type=arg_type,
                    nargs='?',
                    help=v['help']
                )
            else:
                self.parser.add_argument(f"{v['name']}", type=arg_type, nargs='?', help=v['help'])

        self.args = self.parser.parse_args()

    @property
    def min_valid_summary_edges(self) -> int:
        return self.args.min_valid_summary_edges

    @property
    def random_walk_depth_len(self) -> int:
        return self.args.random_walk_depth_len

    @property
    def bridges_number(self) -> int:
        return self.args.bridges_number

    @property
    def max_threads(self) -> int:
        return self.args.max_threads

    @property
    def output_path(self) -> str:
        os.makedirs(self.args.output_path, exist_ok=True)
        return self.args.output_path

    @property
    def db_name(self) -> str:
        return self.args.db_name

    @property
    def db_user(self) -> str:
        return self.args.db_user

    @property
    def db_password(self) -> str:
        return self.args.db_password

    @property
    def db_host(self) -> str:
        return self.args.db_host

    @property
    def db_port(self) -> int:
        return self.args.db_port

    @property
    def neo4j_port(self) -> int:
        return self.args.neo4j_port

    @property
    def neo4j_user(self) -> str:
        return self.args.neo4j_user

    @property
    def neo4j_password(self) -> str:
        return self.args.neo4j_password

    @property
    def neo4j_host(self) -> str:
        return self.args.neo4j_host

    @property
    def dataset_name(self) -> str | None:
        return None if not hasattr(self.args, 'dataset_name') else self.args.dataset_name

    @property
    def min_random_walk_number(self) -> int | None:
        return None if not hasattr(self.args, 'min_random_walk_number') else self.args.min_random_walk_number

    @property
    def max_random_walk_number(self) -> int | None:
        return None if not hasattr(self.args, 'max_random_walk_number') else self.args.max_random_walk_number

    @property
    def seed_node_ids(self) -> list[str] | None:
        if not hasattr(self.args, 'seed_node_ids'):
            return None
        return [seed_node.strip() for seed_node in self.args.seed_node_ids.split(',')]

    @property
    def categories(self) -> list[str] | None:
        if not hasattr(self.args, 'categories'):
            return None
        return [category.strip() for category in self.args.categories.split(',')]

    def neo4j_uri(self) -> str:
        return f"bolt://{self.neo4j_host}:{self.neo4j_port}"

    def neo4j_auth(self) -> tuple[str, str]:
        return self.neo4j_user, self.neo4j_password

from __future__ import annotations

import sqlite3
from functools import lru_cache
from functools import wraps

from more_itertools import batched

from commons.pg_storage import fetch_entities_metadata, fetch_predicates_metadata


def sqlite_connection(database_path):
    conn = sqlite3.connect(database_path)
    return conn


def manage_sqlite_connection(func):
    @wraps(func)
    def wrapper(database_path, *args, **kwargs):
        conn = sqlite_connection(database_path)
        cursor = conn.cursor()
        try:
            result = func(*args, cursor=cursor, **kwargs)
            conn.commit()
            return result
        finally:
            cursor.close()
            conn.close()

    return wrapper


@manage_sqlite_connection
def create_metadata_tables(cursor) -> None:
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS root_entities(
                qid VARCHAR PRIMARY KEY, 
                wikidata_label VARCHAR, 
                wikidata_desc TEXT,
                wikipedia_id INTEGER, 
                wikipedia_title VARCHAR,
                category VARCHAR,
                summaries_counter INTEGER,
                degree_counter INTEGER
        )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS entities(
                qid VARCHAR PRIMARY KEY, 
                wikidata_label VARCHAR, 
                wikidata_desc TEXT,
                wikipedia_id INTEGER, 
                wikipedia_title VARCHAR
        )"""
    )
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS predicates(
                property_id VARCHAR PRIMARY KEY,
                property_label VARCHAR,
                description TEXT
        )"""
    )
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_wikipedia_id ON entities (wikipedia_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_entities_wikipedia_title ON entities (wikipedia_title)')


@manage_sqlite_connection
def insert_root_entity_metadata(data: list, cursor) -> None:
    """
    Insert root entities metadata into the database
    :param data:[
                    qid VARCHAR PRIMARY KEY,
                    wikidata_label VARCHAR,
                    wikidata_desc TEXT,
                    wikipedia_id INTEGER,
                    wikipedia_title VARCHAR,
                    category VARCHAR,
                    summaries_counter INTEGER,
                    degree_counter INTEGER
            ]
    :param cursor:
    :return:
    """
    cursor.execute("INSERT INTO root_entities(qid, wikidata_label, wikidata_desc, wikipedia_id, wikipedia_title, "
                   "category, summaries_counter, degree_counter"
                   ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)", data)


@manage_sqlite_connection
def insert_entities_metadata(entity_ids: list[str], callback: callable, cursor) -> None:
    for entities in batched(entity_ids, 1000):
        data = fetch_entities_metadata(list(entities))
        cursor.executemany("INSERT INTO entities(qid, wikidata_label, wikidata_desc, wikipedia_id, wikipedia_title "
                           ") VALUES (?, ?, ?, ?, ?)", data)
        callback(data)


@manage_sqlite_connection
def insert_predicates_metadata(predicate_ids: list[str], callback: callable, cursor) -> None:
    for predicates in batched(predicate_ids, 1000):
        data = fetch_predicates_metadata(list(predicates))
        cursor.executemany("INSERT INTO predicates(property_id, property_label, description"
                           ") VALUES (?, ?, ?)", data)
        callback(data)


@lru_cache
@manage_sqlite_connection
def load_degree_distribution(cursor) -> dict[str, int]:
    cursor.execute("SELECT qid, degree_counter FROM root_entities")
    return {qid: degree for qid, degree in cursor.fetchall()}

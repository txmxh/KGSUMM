from __future__ import annotations


import re
from functools import lru_cache
from functools import wraps

from more_itertools import batched

import psycopg2
from psycopg2 import pool
import atexit

from graph_generator.config import Config

config = Config()

postgresql_pool: psycopg2.pool.ThreadedConnectionPool = None


def close_postgresql_pool():
    if postgresql_pool:
        postgresql_pool.closeall()
        print("PostgreSQL connection pool has been closed.")


def manage_pg_cursor(f):
    global postgresql_pool
    if not postgresql_pool:
        postgresql_pool = psycopg2.pool.ThreadedConnectionPool(
            1, 50,
            user=config.db_user, password=config.db_password,
            host=config.db_host, port=config.db_port,
            database=config.db_name
        )
        atexit.register(close_postgresql_pool)

    @wraps(f)
    def wrapped(*args, **kwargs):

        connection = None
        try:
            connection = postgresql_pool.getconn()
            if connection:
                with connection.cursor() as cursor:
                    result = f(*args, cursor=cursor, **kwargs)
                    return result
        finally:
            if connection:
                connection.commit()
                postgresql_pool.putconn(connection)

    return wrapped


@manage_pg_cursor
def bulk_fetch_wikipedia_titles(wikipedia_titles: list[str], cursor) -> dict[str, tuple[str, str, str]]:
    records = {}
    query = """SELECT wp.id, wp.title, m.wikidata_id
                               FROM wiki_page_to_wiki_data_mappings m
                               RIGHT JOIN wikipedia_pages wp ON m.wikipedia_id = wp.id  WHERE wp.title in %s;"""
    try:
        cursor.execute(query, (tuple(wikipedia_titles),))
        results = cursor.fetchall()
        for result in results:
            records[result[1]] = (result[0], result[1], result[2])
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while fetching abstract", error)
    return records


@manage_pg_cursor
def fetch_wiki_mapping(identifier: str, cursor) -> tuple[str, str, str] | None:
    '''
    Fetch the wikipedia_id, wikipedia_title, and wikidata_id from the database
    :param identifier:
    :param cursor:
    :return:  (wikipedia_id, wikipedia_title, wikidata_id)
    '''
    record = None
    query = """SELECT wp.id, wp.title, m.wikidata_id
                               FROM wiki_page_to_wiki_data_mappings m
                               RIGHT JOIN wikipedia_pages wp ON m.wikipedia_id = wp.id 
                               """
    try:
        if type(identifier) is int:
            query += " WHERE m.wikipedia_id = %s;"
        elif identifier.isdigit():
            query += " WHERE m.wikipedia_id = %s;"
        elif re.match(r'^Q\d+$', identifier) is not None:
            query += " WHERE m.wikidata_id = %s;"
        else:
            query += " WHERE m.wikipedia_title = %s;"
        cursor.execute(query, (identifier,))
        result = cursor.fetchone()
        if result:
            record = (result[0], result[1], result[2])
        elif type(identifier) is str and identifier.startswith("Q"):
            record = (None, None, identifier)
        else:
            record = (None, None, None)
            print("Identifier not found!!!", identifier)
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while fetching abstract", error)
    return record


@lru_cache(maxsize=1024)
@manage_pg_cursor
def fetch_predicate_metadata(predicate_id: str, cursor) -> tuple(str, str) | None:
    record = None
    try:
        cursor.execute("SELECT property_label ,description from predicates where property_id = %s",
                       (predicate_id,))
        result = cursor.fetchone()
        if result:
            record = result[0], result[1]
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while fetching predicate metadata", error)
    return record


@manage_pg_cursor
def fetch_entities_metadata(entity_ids: list[str], cursor) -> set[tuple[str, str, str, int, str]]:
    results = set()
    try:
        for entities in batched(entity_ids, 1000):
            cursor.execute(
                f"""
                SELECT DISTINCT ON (wd.name)
                wd.name,
                wd.label,
                wd.description,
                wp.id,
                wp.title
                FROM
                    subjects wd
                    LEFT JOIN wiki_page_to_wiki_data_mappings m ON wd.name = m.wikidata_id
                    LEFT JOIN wikipedia_pages wp ON m.wikipedia_id = wp.id
                WHERE
                    wd.name IN %s
                ORDER BY
                    wd.name,
                    (wp.id IS NOT NULL) DESC
                """,
                (tuple(entities),)
            )
            records = cursor.fetchall()
            if records:
                for record in records:
                    results.add(
                        (
                            record[0], record[1], record[2], record[3], record[4],
                        )
                    )
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while fetching predicate metadata", error)
    return results


@manage_pg_cursor
def fetch_predicates_metadata(predicate_ids: list[str], cursor) -> list[tuple[str, str, str]]:
    results = []
    try:
        for predicates in batched(predicate_ids, 1000):
            cursor.execute(
                f"""SELECT property_id, property_label, description from predicates where property_id in %s""",
                (tuple(predicates),)
            )
            records = cursor.fetchall()
            if records:
                results.extend(records)
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while fetching predicate metadata", error)

    return results


@lru_cache(maxsize=1024)
@manage_pg_cursor
def fetch_wikidata_metadata(wikidata_id, cursor) -> tuple[str, str] | None:
    '''
    Fetch the label and description of the wikidata_id
    :param wikidata_id:
    :param cursor:
    :return: (label,  description)
    '''
    record = None
    try:
        cursor.execute("SELECT label, description from subjects where name = %s", (wikidata_id,))
        result = cursor.fetchone()
        if result:
            record = (result[0], result[1])
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while fetching wikidata metadata", error)
    return record


@manage_pg_cursor
def fetch_wikipedia_page_content(wikipedia_title, cursor) -> str:
    record = None
    try:
        cursor.execute("SELECT content from wikipedia_pages where title = %s", (wikipedia_title,))
        result = cursor.fetchone()
        if result:
            record = result[0]
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while fetching page content", error)
    return record


@manage_pg_cursor
def fetch_unprocessed_wikipedia_pages(cursor) -> list[tuple[str, str, str]] | None:
    """
    Fetch unprocessed wikipedia pages
    :return: [(wikipedia_id, wikipedia_title, wikidata_id)]
    """
    try:
        cursor.execute(
            """
            SELECT id
            FROM wikipedia_pages
            WHERE processed = FALSE
            order by id
            LIMIT 500 FOR UPDATE SKIP LOCKED;
            """)
        return cursor.fetchall()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while fetching unprocessed pages", error)
    return None


@manage_pg_cursor
def mark_wikipedia_page_processed(root_wikipedia_id, cursor):
    try:
        cursor.execute("UPDATE wikipedia_pages SET processed = TRUE WHERE id = %s",
                       (root_wikipedia_id,))
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while updating processed field", error)


@manage_pg_cursor
def mark_wikipedia_page_process_failed(root_wikipedia_id, cursor):
    try:
        cursor.execute("UPDATE wikipedia_pages SET processed = null WHERE id = %s",
                       (root_wikipedia_id,))
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while updating processed field", error)

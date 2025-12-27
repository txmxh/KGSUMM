from __future__ import annotations

import time
from functools import lru_cache
from functools import wraps

from more_itertools import batched

from neo4j import GraphDatabase, Driver, Session
import atexit

from threading import local
from graph_generator.config import Config

config = Config()

neo4j_driver: Driver = None


def close_neo4j_driver():
    if neo4j_driver:
        neo4j_driver.close()


atexit.register(close_neo4j_driver)

# Thread-local storage for session management
thread_local = local()


def get_neo4j_session():
    if not hasattr(thread_local, 'session'):
        thread_local.session = neo4j_driver.session(database="neo4j")
    return thread_local.session


def close_neo4j_session():
    if hasattr(thread_local, 'session'):
        thread_local.session.close()
        del thread_local.session


def manage_neo4j_session(f):
    global neo4j_driver
    if not neo4j_driver:
        neo4j_driver = GraphDatabase.driver(
            config.neo4j_uri(),
            auth=config.neo4j_auth(),
            max_connection_lifetime=30 * 60,  # 30 minutes
            max_connection_pool_size=50,  # Maximum number of connections in the pool
            connection_acquisition_timeout=5 * 60  # 2 minutes
        )

    @wraps(f)
    def wrapped(*args, **kwargs):
        session = get_neo4j_session()
        try:
            result = f(*args, session=session, **kwargs)
            return result
        finally:
            close_neo4j_session()

    return wrapped


@lru_cache(maxsize=1024)
@manage_neo4j_session
def fetch_relations(subject_qid: str, object_qid: str, session) -> list[tuple[str, str, str]]:
    candidates = set()
    query = """
            MATCH (
                s:WikiEntity {entityName: $subject_qid})-[r:HAS_TYPE]->(t:WikiEntity {entityName: $target_qid}
            )
            RETURN s.entityName as s, r.type as p, t.entityName as t
    """

    result = session.run(query, subject_qid=object_qid, target_qid=subject_qid)
    for record in result:
        candidates.add(record)
    result = session.run(query, subject_qid=subject_qid, target_qid=object_qid)
    for record in result:
        candidates.add(record)

    return list(map(lambda x: (x['s'], x['p'], x['t']), candidates))


@manage_neo4j_session
def fetch_edges_by_candidates(edge_candidates: list[tuple[str, str]], session) -> list[tuple[str, str, str]]:
    query = """UNWIND $candidates AS candidate
    MATCH (s:WikiEntity {entityName: candidate[0]})-[r:HAS_TYPE]->(t:WikiEntity {entityName: candidate[1]})
    RETURN s.entityName as s, r.type as p, t.entityName as t"""
    result = []
    records = session.run(query, candidates=edge_candidates)
    for record in records:
        result.append((record['s'], record['p'], record['t']))
    return result


@manage_neo4j_session
def fetch_first_neighbors(wikidata_id: str, session):
    query = """
    MATCH (s:WikiEntity {entityName: $wikidata_id})-[r:HAS_TYPE]-(t:WikiEntity)
    RETURN STARTNODE(r).entityName AS s, r.type as p, ENDNODE(r).entityName as t
    """
    result = []
    records = session.run(query, wikidata_id=wikidata_id)
    for record in records:
        result.append((record['s'], record['p'], record['t']))
    return result


@manage_neo4j_session
def fetch_node_degree(wikidata_id: str, session) -> int:
    # fetch the degree of the given node
    query = """
        MATCH (s:WikiEntity {entityName: $wikidata_id})-[r:HAS_TYPE]-(t:WikiEntity)
        RETURN count(r)
    """
    result = session.run(query, wikidata_id=wikidata_id).single()
    if result:
        return result[0]
    else:
        return 0


@manage_neo4j_session
def fetch_summaries(wikidata_id, session) -> list[tuple[str, str, str]] | None:
    # fetch the summary edges that has wikidata_id marked as summary_for
    query = """
        MATCH (s:WikiEntity)-[r:SUMMARY {summary_for: $wikidata_id}]->(t:WikiEntity)
        WHERE s.entityName = $wikidata_id OR t.entityName = $wikidata_id
        RETURN s.entityName AS s, r.predicate as p, t.entityName as t
    """
    result = []
    records = session.run(query, wikidata_id=wikidata_id)
    for record in records:
        result.append((record['s'], record['p'], record['t']))
    return result


@manage_neo4j_session
def add_summary_edge(root_wikidata_id, from_entity, predicate, to_entity, session) -> tuple[str, str, str] | None:
    # fetch the edge and append summary_for with root_wikidata_id as an attribute
    query = """
      MATCH (s:WikiEntity {entityName: $from_entity}), (t:WikiEntity {entityName: $to_entity})
        MERGE (s)-[r:SUMMARY {summary_for: $root_wikidata_id, predicate: $predicate}]->(t)
        RETURN s, r, t
    """
    result = session.run(query, from_entity=from_entity, to_entity=to_entity, root_wikidata_id=root_wikidata_id,
                         predicate=predicate)
    return result.single()


@manage_neo4j_session
def label_candidates(candidates, session) -> list[str]:
    label_base_name = f"Candidates_{int(time.time())}_"
    component_labels = []
    for i, candidate in enumerate(candidates):
        candidate_label = f"{label_base_name}{i + 1}"
        component_labels.append(candidate_label)
        for batch_nodes in batched(candidate, 1000):
            session.run(
                f"""
                UNWIND $node_names AS node_name
                MATCH (n:WikiEntity {{entityName: node_name}})
                SET n:{candidate_label}""",
                node_names=batch_nodes
            )
        session.run(f"CREATE INDEX {candidate_label}_index FOR (n:{candidate_label}) ON (n.entityName)")
    return component_labels


@manage_neo4j_session
def change_component_label_with(from_label, to_label, session):
    session.run(f"""
    MATCH (m:WikiEntity:{from_label})
    SET m:{to_label}
    """)
    session.run(f"DROP INDEX {from_label}_index")


@manage_neo4j_session
def set_node_label(entity_id, to_label, session):
    session.run(f"""
    MATCH (m:WikiEntity{{entityName: '{entity_id}'}})
    SET m:{to_label}
    """)


@manage_neo4j_session
def remove_component_labels(component_labels, session):
    for component_label in component_labels:
        session.run(f"""
        MATCH(m:WikiEntity:{component_label})
        REMOVE m:{component_label}
        """, component_label=component_label)
        session.run(f"DROP INDEX {component_label}_index")


@manage_neo4j_session
def fetch_shortest_pairs_between_components(component_label_a, component_label_b, max_depth=1, k=10,
                                            session=None) -> list[tuple[str, str]]:
    k = min(k, min(len(component_label_a), len(component_label_b)))
    top_pairs = []
    results = session.run(
        f"""
            MATCH (a:WikiEntity:{component_label_a}), (b:WikiEntity:{component_label_b}), 
                    path = shortestPath((a)-[:HAS_TYPE*..{max_depth}]-(b))
            WHERE a.entityName IS NOT NULL AND b.entityName IS NOT NULL
            RETURN a.entityName AS a, b.entityName AS b, length(path) AS hops
            ORDER BY hops ASC
            LIMIT $k
        """, k=k
    )
    for record in results:
        if record:
            top_pairs.append((record['a'], record['b']))
    return top_pairs


@manage_neo4j_session
def fetch_shortest_path(a, b, session) -> list[tuple[str, str, str]]:
    record = session.run("""
    MATCH (a:WikiEntity {entityName: $a}), (b:WikiEntity {entityName: $b}) , path = shortestPath((a)-[*]-(b))
    WHERE a <> b
    RETURN [i IN RANGE(0, LENGTH(path)-1) | 
        {
            start_node: STARTNODE(RELATIONSHIPS(path)[i]).entityName, 
            predicate: RELATIONSHIPS(path)[i].type,
            end_node: ENDNODE(RELATIONSHIPS(path)[i]).entityName
        }
    ] AS legs
    """, a=a, b=b)
    path = list()
    for leg in record.single()["legs"]:
        path.append((leg['start_node'], leg['predicate'], leg['end_node']))
    return path


@manage_neo4j_session
def fetch_shortest_paths(pairs: list[tuple[str, str]], session) -> list[list[tuple[str, str, str]]]:
    query = """
    UNWIND $pairs AS pair
    MATCH (a:WikiEntity {entityName: pair.a}), (b:WikiEntity {entityName: pair.b}) , path = shortestPath((a)-[*]-(b))
    WHERE a <> b
    RETURN collect([
        i IN RANGE(0, LENGTH(path)-1) | 
        (
            start_node: STARTNODE(RELATIONSHIPS(path)[i]).entityName, 
            predicate: RELATIONSHIPS(path)[i].type,
            end_node: ENDNODE(RELATIONSHIPS(path)[i]).entityName
        )
    ]) AS legs
    """
    records = session.run(query, pairs=[{'a': a, 'b': b} for a, b in pairs])
    results = []
    for record in records:
        path = list()
        for leg in record["legs"]:
            path.append((leg['start_node'], leg['predicate'], leg['end_node']))
        results.append(path)

    return results


@manage_neo4j_session
def check_and_create_projection(session):
    result = session.run("CALL gds.graph.exists('random_walk_projector') YIELD exists RETURN exists")
    exists = result.single()['exists']

    if not exists:
        create_projection_query = """
        CALL gds.graph.project(
          'random_walk_projector',
          'WikiEntity',
          {
            HAS_TYPE: {
                  type: 'HAS_TYPE',
              orientation: 'UNDIRECTED'
            }
          }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount;
        """
        session.run(create_projection_query)


@manage_neo4j_session
def fetch_random_walk_paths(starting_node: str, walk_length: int = 3, limit: int = 1000, session=None) -> list[
    list[tuple[str, str, str]]]:
    walk_limit = limit + int(limit * 0.2)
    paths = list()
    query = """ 
        MATCH (e:WikiEntity {entityName: $starting_node})
        CALL gds.randomWalk.stream(
          'random_walk_projector',
          {
            sourceNodes: e,
            walkLength: $walk_length,
            walksPerNode: $limit,
            concurrency: 4
          }
        )
        YIELD path
        WITH [node IN nodes(path) | node.entityName] AS path_nodes
        UNWIND range(0, size(path_nodes) - 2) AS i
        WITH DISTINCT path_nodes, path_nodes[i] AS startNode, path_nodes[i+1] AS endNode
        MATCH (s:WikiEntity {entityName: startNode}), (t:WikiEntity {entityName: endNode})
        WITH path_nodes, 
          apoc.coll.randomItem(
            [(s)-[r:HAS_TYPE]->(t) | {s: s.entityName, p: r.type, t: t.entityName}] + 
            [(t)-[r:HAS_TYPE]->(s) | {s: t.entityName, p: r.type, t: s.entityName}]
          ) AS edge
        WITH DISTINCT path_nodes, collect(edge) AS edges
        RETURN {nodes: path_nodes, edges: edges} AS path_info
    """

    records = session.run(query, starting_node=starting_node, walk_length=walk_length, limit=walk_limit)
    counter = 0
    for record in records:
        for p in record:
            counter += 1
            if counter > limit:
                break
            path = set()
            for edge in p['edges']:
                path.add((edge['s'], edge['p'], edge['t']))
            paths.append(path)
    return paths

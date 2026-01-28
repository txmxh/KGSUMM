from commons.neo4j_storage import fetch_random_walk_paths
from commons.pg_storage import fetch_predicates_metadata
from commons.wiki_entity import WikiEntity
from commons.wiki_mapping import WikiMapping
from graph_generator.config import Config

if __name__ == '__main__':
    config = Config()
    entity = WikiEntity(WikiMapping(wikipedia_id='Q76'))
    summaries = entity.get_detailed_summaries()
    print("Summaries:")
    for summary in summaries:
        print(f"{summary['from_wikipedia_title']} - {summary['predicate_label']} -> {summary['to_wikipedia_title']}")
    print()
    walks = fetch_random_walk_paths('Q76', 3, 50)
    print("Random Walks:")
    for walk in walks:
        for edge in walk:
            u = WikiEntity(WikiMapping(wikidata_id=edge[0])).get_wikidata_label()
            p = fetch_predicates_metadata([edge[1]])[0][1]
            v = WikiEntity(WikiMapping(wikidata_id=edge[2])).get_wikidata_label()
            print(f"{u} - {p} -> {v}", end=' | ')
        print()

[![arXiv](https://img.shields.io/badge/arXiv-2406.08435-B31B1B.svg)](https://doi.org/10.48550/arXiv.2406.08435)![GitHub License](https://img.shields.io/github/license/msorkhpar/wiki-entity-summarization)![GitHub Release](https://img.shields.io/github/v/release/msorkhpar/wiki-entity-summarization)

# Wiki Entity Summarization Benchmark (WikES)

This repository leverages
the [wiki-entity-summarization-preprocessor](https://github.com/msorkhpar/wiki-entity-summarization-preprocessor)
project to construct an Entity Summarization Graph based on a given set of nodes. The project tries to
maintain the structure of the Wikidata knowledge graph by performing random walk sampling with a depth of `K,` starting
from seed nodes after all the summary edges have been added to the result.
It then checks if the expanded graph is a single weakly connected component. If not, it finds `B` paths
to connect the components. The final result is a heterogeneous graph consisting of the seed nodes,
their summary edges, (1..K)-hop neighbors of the seed nodes and their edges, and any intermediary nodes added to ensure
graph connectivity. Each node and edge in the graph is enriched with metadata obtained from Wikidata and Wikipedia and
predicate information, providing additional context and details about the entities and their relationships.
<br/>
<br/>
![A single root entity with its summary edges and other expanded edges by random walk](/WikES-example.png)

## Loading the Datasets

### Load Using `wikes-toolkit`

To load the dataset, we have introduced a toolkit that can be used to download, load, work, and evaluate 48
Wiki-Entity-Summarization datasets. The toolkit is available as a Python package and can be installed using pip:

```bash
pip install wikes-toolkit
```

A simple example of how to use the toolkit is as follows:

```python
from wikes_toolkit import WikESToolkit, V1, WikESGraph

toolkit = WikESToolkit(save_path="./data")  # save_path is optional
G = toolkit.load_graph(
    WikESGraph,
    V1.WikiLitArt.SMALL,
    entity_formatter=lambda e: f"Entity({e.wikidata_label})",
    predicate_formatter=lambda p: f"Predicate({p.label})",
    triple_formatter=lambda
        t: f"({t.subject_entity.wikidata_label})-[{t.predicate.label}]-> ({t.object_entity.wikidata_label})"
)

root_nodes = G.root_entities()
nodes = G.entities()

```

Please refer to the [Wiki-Entity-Summarization-Toolkit](https://github.com/msorkhpar/wiki-entity-summarization-toolkit)
repository for more information.

### Using mlcroissant

To load WikES datasets, you can use [mlcorissant](https://github.com/mlcommons/croissant/) as well. You can find the
metadata JSON files in the [dataset details tabel](#Pre-generated-Datasets). </br>

Here is an example of loading our dataset using mlcorissant:

```python
from mlcroissant import Dataset


def print_first_item(record_name):
    for record in dataset.records(record_set=record_name):
        for key, val in record.items():
            if isinstance(val, bytes):
                val = str(val, "utf-8")
            print(f"{key}=[{val}]({type(val)})", end=", ")
        break
    print()


dataset = Dataset(
    jsonld="https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s.json")

print(dataset.metadata.record_sets)

print_first_item("entities")
print_first_item("root-entities")
print_first_item("predicates")
print_first_item("triples")
print_first_item("ground-truths")
""" The output of the above code:
wikes-dataset
[RecordSet(uuid="entities"), RecordSet(uuid="root-entities"), RecordSet(uuid="predicates"), RecordSet(uuid="triples"), RecordSet(uuid="ground-truths")]
id=[0](<class 'int'>), entity=[Q6387338](<class 'str'>), wikidata_label=[Ken Blackwell](<class 'str'>), wikidata_description=[American politician and activist](<class 'str'>), wikipedia_id=[769596](<class 'int'>), wikipedia_title=[Ken_Blackwell](<class 'str'>), 
entity=[9](<class 'int'>), category=[singer](<class 'str'>), 
id=[0](<class 'int'>), predicate=[P1344](<class 'str'>), predicate_label=[participant in](<class 'str'>), predicate_desc=[event in which a person or organization was/is a participant; inverse of P710 or P1923](<class 'str'>), 
subject=[1](<class 'int'>), predicate=[0](<class 'int'>), object=[778](<class 'int'>), 
root_entity=[9](<class 'int'>), subject=[9](<class 'int'>), predicate=[8](<class 'int'>), object=[31068](<class 'int'>), 
"""
```

## Loading the Pre-processed Databases

As described
in [wiki-entity-summarization-preprocessor](https://github.com/msorkhpar/wiki-entity-summarization-preprocessor), we
have imported en-wikidata items as a graph with their summaries into a Neo4j database using Wikipedia and Wikidata XML
dump files. Additionally, all the other related metadata was imported into a Postgres database.

If you want to create your own dataset but do not want to run the pre-processor again, you can download and load the
exported files from these two databases. Please refer to the release notes of the current version `1.0.0` (
enwiki-2023-05-1 and wikidata-wiki-2023-05-1).

- [PostgreSQL-1.0.0 (wiki 2023-05-01)](https://github.com/msorkhpar/wiki-entity-summarization-preprocessor/releases/tag/PostgreSQL-1.0.0)
- [Neo4j-1.0.0 (wiki 2023-05-01)](https://github.com/msorkhpar/wiki-entity-summarization-preprocessor/releases/tag/Neo4j-1.0.0)

## Process Overview

### 1. **Building the Summary Graph**

- Create a summary graph where each seed node is expanded with its summary edges.

### 2. **Expanding the Summary Graph**

- Perform random walks starting from the seed nodes to mimic the structure of the Wikidata graph.
- Scale the number of walks based on the degree of the seed nodes.
- Add new edges to the graph from the random walk results.

### 3. **Connecting Components**

- Check if the expanded graph forms a single weakly connected component.
- If not, iteratively connect smaller components using the shortest paths until a single component is achieved.

### 4. **Adding Metadata**

- Enhance the final graph with additional metadata for each node and edge.
- Include labels, descriptions, and other relevant information from Wikidata, Wikipedia, and predicate information.

### Pre-generated Datasets

We have generated datasets using [A Brief History of Human Time project](https://medialab.github.io/bhht-datascape/).
These datasets contain different sets of seed nodes, categorized by various human arts and professions.

| dataset (variant, size, None/train/val/test)                                                                                                                                                                                                                                                                                                                                          | #roots | #smmaries | #nodes | #edges | #labels | roots category distribution                                                                                                                                  | Running Time(sec) |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-----------|--------|--------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| WikiLitArt-s </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s.json)                          | 494    | 10416     | 85346  | 136950 | 547     | actor=150<br/> composer=35<br/> film=41<br/> novelist=24<br/> painter=59<br/> poet=39<br/> screenwriter=17<br/> singer=72<br/> writer=57                     | 91.934            |
| WikiLitArt-s-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s-train.json)  | 346    | 7234      | 61885  | 96497  | 508     | actor=105<br/> composer=24<br/> film=29<br/> novelist=17<br/> painter=42<br/> poet=27<br/> screenwriter=12<br/> singer=50<br/> writer=40                     | 66.023            |
| WikiLitArt-s-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s-val.json)          | 74     | 1572      | 14763  | 20795  | 340     | actor=23<br/> composer=5<br/> film=6<br/> novelist=4<br/> painter=9<br/> poet=6<br/> screenwriter=2<br/> singer=11<br/> writer=8                             | 14.364            |
| WikiLitArt-s-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-s-test.json)      | 74     | 1626      | 15861  | 22029  | 350     | actor=22<br/> composer=6<br/> film=6<br/> novelist=3<br/> painter=8<br/> poet=6<br/> screenwriter=3<br/> singer=11<br/> writer=9                             | 14.6              |
| WikiLitArt-m </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m.json)                          | 494    | 10416     | 128061 | 220263 | 604     | actor=150<br/> composer=35<br/> film=41<br/> novelist=24<br/> painter=59<br/> poet=39<br/> screenwriter=17<br/> singer=72<br/> writer=57                     | 155.368           |
| WikiLitArt-m-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m-train.json)  | 346    | 7234      | 93251  | 155667 | 566     | actor=105<br/> composer=24<br/> film=29<br/> novelist=17<br/> painter=42<br/> poet=27<br/> screenwriter=12<br/> singer=50<br/> writer=40                     | 111.636           |
| WikiLitArt-m-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m-val.json)          | 74     | 1572      | 22214  | 33547  | 375     | actor=23<br/> composer=5<br/> film=6<br/> novelist=4<br/> painter=9<br/> poet=6<br/> screenwriter=2<br/> singer=11<br/> writer=8                             | 22.957            |
| WikiLitArt-m-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-m-test.json)      | 74     | 1626      | 24130  | 35980  | 394     | actor=22<br/> composer=6<br/> film=6<br/> novelist=3<br/> painter=8<br/> poet=6<br/> screenwriter=3<br/> singer=11<br/> writer=9                             | 26.187            |
| WikiLitArt-l </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l.json)                          | 494    | 10416     | 239491 | 466905 | 703     | actor=150<br/> composer=35<br/> film=41<br/> novelist=24<br/> painter=59<br/> poet=39<br/> screenwriter=17<br/> singer=72<br/> writer=57                     | 353.113           |
| WikiLitArt-l-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l-train.json)  | 346    | 7234      | 176057 | 332279 | 661     | actor=105<br/> composer=24<br/> film=29<br/> novelist=17<br/> painter=42<br/> poet=27<br/> screenwriter=12<br/> singer=50<br/> writer=40                     | 244.544           |
| WikiLitArt-l-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l-val.json)          | 74     | 1572      | 42745  | 71734  | 446     | actor=23<br/> composer=5<br/> film=6<br/> novelist=4<br/> painter=9<br/> poet=6<br/> screenwriter=2<br/> singer=11<br/> writer=8                             | 57.263            |
| WikiLitArt-l-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiLitArt-l-test.json)      | 74     | 1626      | 46890  | 77931  | 493     | actor=22<br/> composer=6<br/> film=6<br/> novelist=3<br/> painter=8<br/> poet=6<br/> screenwriter=3<br/> singer=11<br/> writer=9                             | 60.466            |
| WikiCinema-s </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s.json)                          | 493    | 11750     | 70753  | 126915 | 469     | actor=405<br/> film=88                                                                                                                                       | 118.014           |
| WikiCinema-s-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s-train.json)  | 345    | 8374      | 52712  | 89306  | 437     | actor=284<br/> film=61                                                                                                                                       | 84.364            |
| WikiCinema-s-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s-val.json)          | 73     | 1650      | 13362  | 19280  | 305     | actor=59<br/> film=14                                                                                                                                        | 18.651            |
| WikiCinema-s-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-s-test.json)      | 75     | 1744      | 14777  | 21567  | 313     | actor=62<br/> film=13                                                                                                                                        | 19.851            |
| WikiCinema-m </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m.json)                          | 493    | 11750     | 101529 | 196061 | 541     | actor=405<br/> film=88                                                                                                                                       | 196.413           |
| WikiCinema-m-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m-train.json)  | 345    | 8374      | 75900  | 138897 | 491     | actor=284<br/> film=61                                                                                                                                       | 142.091           |
| WikiCinema-m-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m-val.json)          | 73     | 1650      | 19674  | 30152  | 344     | actor=59<br/> film=14                                                                                                                                        | 31.722            |
| WikiCinema-m-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-m-test.json)      | 75     | 1744      | 22102  | 34499  | 342     | actor=62<br/> film=13                                                                                                                                        | 33.674            |
| WikiCinema-l </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l.json)                          | 493    | 11750     | 185098 | 397546 | 614     | actor=405<br/> film=88                                                                                                                                       | 475.679           |
| WikiCinema-l-train  </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l-train.json) | 345    | 8374      | 139598 | 284417 | 575     | actor=284<br/> film=61                                                                                                                                       | 333.148           |
| WikiCinema-l-val  </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l-val.json)         | 73     | 1650      | 37352  | 63744  | 412     | actor=59<br/> film=14                                                                                                                                        | 68.62             |
| WikiCinema-l-test  </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiCinema-l-test.json)     | 75     | 1744      | 43238  | 74205  | 426     | actor=62<br/> film=13                                                                                                                                        | 87.07             |
| WikiPro-s </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s.json)                                      | 493    | 9853      | 79825  | 125912 | 616     | actor=58<br/> football=156<br/> journalist=14<br/> lawyer=16<br/> painter=23<br/> player=25<br/> politician=125<br/> singer=27<br/> sport=21<br/> writer=28  | 126.119           |
| WikiPro-s-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s-train.json)              | 345    | 6832      | 57529  | 87768  | 575     | actor=41<br/> football=109<br/> journalist=10<br/> lawyer=11<br/> painter=16<br/> player=17<br/> politician=87<br/> singer=19<br/> sport=15<br/> writer=20   | 89.874            |
| WikiPro-s-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s-val.json)                      | 74     | 1548      | 15769  | 21351  | 405     | actor=9<br/> football=23<br/> journalist=2<br/> lawyer=3<br/> painter=3<br/> player=4<br/> politician=19<br/> singer=4<br/> sport=3<br/> writer=4            | 21.021            |
| WikiPro-s-test  </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-s-test.json)                 | 74     | 1484      | 15657  | 21145  | 384     | actor=8<br/> football=24<br/> journalist=2<br/> lawyer=2<br/> painter=4<br/> player=4<br/> politician=19<br/> singer=4<br/> sport=3<br/> writer=4            | 21.743            |
| WikiPro-m </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m.json)                                      | 493    | 9853      | 119305 | 198663 | 670     | actor=58<br/> football=156<br/> journalist=14<br/> lawyer=16<br/> painter=23<br/> player=25<br/> politician=125<br/> singer=27<br/> sport=21<br/> writer=28  | 208.157           |
| WikiPro-m-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m-train.json)              | 345    | 6832      | 86434  | 138676 | 633     | actor=41<br/> football=109<br/> journalist=10<br/> lawyer=11<br/> painter=16<br/> player=17<br/> politician=87<br/> singer=19<br/> sport=15<br/> writer=20   | 141.563           |
| WikiPro-m-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m-val.json)                      | 74     | 1548      | 24230  | 34636  | 463     | actor=9<br/> football=23<br/> journalist=2<br/> lawyer=3<br/> painter=3<br/> player=4<br/> politician=19<br/> singer=4<br/> sport=3<br/> writer=4            | 36.045            |
| WikiPro-m-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-m-test.json)                  | 74     | 1484      | 24117  | 34157  | 462     | actor=8<br/> football=24<br/> journalist=2<br/> lawyer=2<br/> painter=4<br/> player=4<br/> politician=19<br/> singer=4<br/> sport=3<br/> writer=4            | 36.967            |
| WikiPro-l </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l.json)                                      | 493    | 9853      | 230442 | 412766 | 769     | actor=58<br/> football=156<br/> journalist=14<br/> lawyer=16<br/> painter=23<br/> player=25<br/> politician=125<br/> singer=27<br/> sport=21<br/> writer=28  | 489.409           |
| WikiPro-l-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l-train.json)              | 345    | 6832      | 166685 | 290069 | 725     | actor=41<br/> football=109<br/> journalist=10<br/> lawyer=11<br/> painter=16<br/> player=17<br/> politician=87<br/> singer=19<br/> sport=15<br/> writer=20   | 334.864           |
| WikiPro-l-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l-val.json)                      | 74     | 1548      | 48205  | 74387  | 549     | actor=9<br/> football=23<br/> journalist=2<br/> lawyer=3<br/> painter=3<br/> player=4<br/> politician=19<br/> singer=4<br/> sport=3<br/> writer=4            | 84.089            |
| WikiPro-l-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiPro-l-test.json)                  | 74     | 1484      | 47981  | 72845  | 546     | actor=8<br/> football=24<br/> journalist=2<br/> lawyer=2<br/> painter=4<br/> player=4<br/> politician=19<br/> singer=4<br/> sport=3<br/> writer=4            | 92.545            |
| WikiProFem-s </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s.json)                          | 468    | 8338      | 79926  | 123193 | 571     | actor=141<br/> athletic=25<br/> football=24<br/> journalist=16<br/> painter=16<br/> player=32<br/> politician=81<br/> singer=69<br/> sport=18<br/> writer=46 | 177.63            |
| WikiProFem-s-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s-train.json)  | 330    | 5587      | 58329  | 87492  | 521     | actor=98<br/> athletic=18<br/> football=17<br/> journalist=9<br/> painter=13<br/> player=22<br/> politician=57<br/> singer=48<br/> sport=14<br/> writer=34   | 127.614           |
| WikiProFem-s-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s-val.json)          | 68     | 1367      | 14148  | 19360  | 344     | actor=21<br/> athletic=4<br/> football=3<br/> journalist=4<br/> painter=1<br/> player=5<br/> politician=13<br/> singer=11<br/> sport=1<br/> writer=5         | 29.081            |
| WikiProFem-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-s-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-test.json)          | 70     | 1387      | 13642  | 18567  | 360     | actor=22<br/> athletic=3<br/> football=4<br/> journalist=3<br/> painter=2<br/> player=5<br/> politician=11<br/> singer=10<br/> sport=3<br/> writer=7         | 27.466            |
| WikiProFem-m </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m.json)                          | 468    | 8338      | 122728 | 196838 | 631     | actor=141<br/> athletic=25<br/> football=24<br/> journalist=16<br/> painter=16<br/> player=32<br/> politician=81<br/> singer=69<br/> sport=18<br/> writer=46 | 301.718           |
| WikiProFem-m-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m-train.json)  | 330    | 5587      | 89922  | 140505 | 600     | actor=98<br/> athletic=18<br/> football=17<br/> journalist=9<br/> painter=13<br/> player=22<br/> politician=57<br/> singer=48<br/> sport=14<br/> writer=34   | 217.699           |
| WikiProFem-m-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m-val.json)          | 68     | 1367      | 21978  | 31230  | 409     | actor=21<br/> athletic=4<br/> football=3<br/> journalist=4<br/> painter=1<br/> player=5<br/> politician=13<br/> singer=11<br/> sport=1<br/> writer=5         | 46.793            |
| WikiProFem-m-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-m-test.json)      | 70     | 1387      | 21305  | 29919  | 394     | actor=22<br/> athletic=3<br/> football=4<br/> journalist=3<br/> painter=2<br/> player=5<br/> politician=11<br/> singer=10<br/> sport=3<br/> writer=7         | 46.317            |
| WikiProFem-l </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l.json)                          | 468    | 8338      | 248012 | 413895 | 722     | actor=141<br/> athletic=25<br/> football=24<br/> journalist=16<br/> painter=16<br/> player=32<br/> politician=81<br/> singer=69<br/> sport=18<br/> writer=46 | 768.99            |
| WikiProFem-l-train </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l-train.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l-train.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l-train.json)  | 330    | 5587      | 183710 | 297686 | 676     | actor=98<br/> athletic=18<br/> football=17<br/> journalist=9<br/> painter=13<br/> player=22<br/> politician=57<br/> singer=48<br/> sport=14<br/> writer=34   | 544.893           |
| WikiProFem-l-val </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l-val.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l-val.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l-val.json)          | 68     | 1367      | 46018  | 67193  | 492     | actor=21<br/> athletic=4<br/> football=3<br/> journalist=4<br/> painter=1<br/> player=5<br/> politician=13<br/> singer=11<br/> sport=1<br/> writer=5         | 116.758           |
| WikiProFem-l-test </br>[csv](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l-test.zip), [graphml](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l-test.graphml), [croissant.json](https://github.com/msorkhpar/wiki-entity-summarization/releases/download/1.0.5/WikiProFem-l-test.json)      | 70     | 1387      | 44193  | 63563  | 472     | actor=22<br/> athletic=3<br/> football=4<br/> journalist=3<br/> painter=2<br/> player=5<br/> politician=11<br/> singer=10<br/> sport=3<br/> writer=7         | 118.524           |

**Keep in mind that by providing a new set of seed nodes, you can generate the output for your own dataset.**

### Dataset Parameters

| Parameter                     | Value |
|-------------------------------|-------|
| Min valid summary edges       | 5     |
| Random walk depth length      | 3     |
| Min random walk number-small  | 100   |
| Min random walk number-medium | 150   |
| Min random walk number-large  | 300   |
| Max random walk number-small  | 300   |
| Max random walk number-medium | 600   |
| Max random walk number-large  | 1800  |
| Bridges number                | 5     |

## Graph Structure

In the following you can see a sample of the graph format (we highly recommend using our toolkit to load the datasets):

### CSV Format

After unzipping `{variant}-{size}-{dataset_type}.zip` file, you will find the following CSV files:

`{variant}-{size}-{dataset_type}-entities.csv` contains entities. An entity is a Wikidata item (node) in our
dataset.

| Field           | Description                          | datatype |
|-----------------|--------------------------------------|----------| 
| id              | incremental integer starting by zero | int      |
| entity          | Wikidata qid, e.g. `Q76`             | string   |
| wikidata_label  | Wikidata label (nullable)            | string   |
| wikidata_desc   | Wikidata description (nullable)      | string   |
| wikipedia_title | Wikipedia title (nullable)           | string   |
| wikipedia_id    | Wikipedia page id (nullable)         | long     |

`{variant}-{size}-{dataset_type}-root-entities.csv` contains root entities. A root entity is a seed node
described previously.

| Field    | Description                                              | datatype |
|----------|----------------------------------------------------------|----------|
| entity   | id key in `{variant}-{size}-{dataset_type}-entities.csv` | int      |
| category | category                                                 | string   |

`{variant}-{size}-{dataset_type}-predicates.csv` contains predicates. A predicate is a Wikidata property or a
describing
a connection.

| Field           | Description                              | datatype |
|-----------------|------------------------------------------|----------| 
| id              | incremental integer starting by zero     | int      |
| predicate       | Wikidata Property id, e.g. `P121`        | string   |
| predicate_label | Wikidata Property label (nullable)       | string   |
| predicate_desc  | Wikidata Property description (nullable) | string   |

`{variant}-{size}-{dataset_type}-triples.csv` contains triples. A triple is an edge between two entities with a
predicate.

| Field     | Description                                                | datatype |
|-----------|------------------------------------------------------------|----------| 
| subject   | id key in `{variant}-{size}-{dataset_type}-entities.csv`   | int      |
| predicate | id key in `{variant}-{size}-{dataset_type}-predicates.csv` | int      |
| object    | id key in `{variant}-{size}-{dataset_type}-entities.csv`   | int      |

`{viariant}_{size}_{dataset_type}-ground-truths.csv` contains ground truth triples. A ground truth triple is an
edge that
is marked as a summary for a root entity.

| Field       | Description                                                   | datatype |
|-------------|---------------------------------------------------------------|----------| 
| root_entity | entity in `{variant}-{size}-{dataset_type}-root-entities.csv` | int      |
| subject     | id key in `{variant}-{size}-{dataset_type}-entities.csv`      | int      |
| predicate   | id key in `{variant}-{size}-{dataset_type}-predicates.csv`    | int      |
| object      | id key in `{variant}-{size}-{dataset_type}-entities.csv`      | int      |

**Note: for this file one of the columns `subject` or `object` is equal to the `root_entity`.**

### Example of CSV Files

```csv
# entities.csv
id,entity,wikidata_label,wikidata_desc,wikipedia_title,wikipedia_id
0,Q43416,Keanu Reeves,Canadian actor (born 1964),Keanu_Reeves,16603
1,Q3820,Beirut,capital and largest city of Lebanon,Beirut,37428
2,Q639669,musician,person who composes, conducts or performs music,Musician,38284
3,Q219150,Constantine,2005 film directed by Francis Lawrence,Constantine_(film),1210303
```

```csv
# root-entities.csv
entity,category
0,Q43416,actor
```

```csv
# predicates.csv
id,predicate,predicate_label,predicate_desc
0,P19,place of birth,location where the subject was born
1,P106,occupation,occupation of a person; see also "field of work" (Property:P101), "position held" (Property:P39)
2,P161,cast member,actor in the subject production [use "character role" (P453) and/or "name of the character role" (P4633) as qualifiers] [use "voice actor" (P725) for voice-only role]
```

```csv
# triples.csv
subject,predicate,object
0,0,1
0,1,2
3,2,0
```

```csv
# ground-truth.csv
root_entity,subject,predicate,object
0,0,0,1
3,3,2,0
```

### GraphML Example

The same graph can be represented in GraphML format, available in the [dataset details tabel](#Pre-generated-Datasets)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
    <key id="d9" for="edge" attr.name="summary_for" attr.type="string"/>
    <key id="d8" for="edge" attr.name="predicate_desc" attr.type="string"/>
    <key id="d7" for="edge" attr.name="predicate_label" attr.type="string"/>
    <key id="d6" for="edge" attr.name="predicate" attr.type="string"/>
    <key id="d5" for="node" attr.name="category" attr.type="string"/>
    <key id="d4" for="node" attr.name="is_root" attr.type="boolean"/>
    <key id="d3" for="node" attr.name="wikidata_desc" attr.type="string"/>
    <key id="d2" for="node" attr.name="wikipedia_title" attr.type="string"/>
    <key id="d1" for="node" attr.name="wikipedia_id" attr.type="long"/>
    <key id="d0" for="node" attr.name="wikidata_label" attr.type="string"/>
    <graph edgedefault="directed">
        <node id="Q43416">
            <data key="d0">Keanu Reeves</data>
            <data key="d1">16603</data>
            <data key="d2">Keanu_Reeves</data>
            <data key="d3">Canadian actor (born 1964)</data>
            <data key="d4">True</data>
            <data key="d5">actor</data>
        </node>
        <node id="Q3820">
            <data key="d0">Beirut</data>
            <data key="d1">37428</data>
            <data key="d2">Beirut</data>
            <data key="d3">capital and largest city of Lebanon</data>
        </node>
        <node id="Q639669">
            <data key="d0">musician</data>
            <data key="d1">38284</data>
            <data key="d2">Musician</data>
            <data key="d3">person who composes, conducts or performs music</data>
        </node>
        <node id="Q219150">
            <data key="d0">Constantine</data>
            <data key="d1">1210303</data>
            <data key="d2">Constantine_(film)</data>
            <data key="d3">2005 film directed by Francis Lawrence</data>
        </node>
        <edge source="Q43416" target="Q3820" id="P19">
            <data key="d6">P19</data>
            <data key="d7">place of birth</data>
            <data key="d8">location where the subject was born</data>
            <data key="d9">Q43416</data>
        </edge>
        <edge source="Q43416" target="Q639669" id="P106">
            <data key="d6">P106</data>
            <data key="d7">occupation</data>
            <data key="d8">occupation of a person; see also "field of work" (Property:P101), "position held"
                (Property:P39)
            </data>
        </edge>
        <edge source="Q219150" target="Q43416" id="P106">
            <data key="d6">P161</data>
            <data key="d7">cast member</data>
            <data key="d8">actor in the subject production [use "character role" (P453) and/or "name of the character
                role" (P4633) as qualifiers] [use "voice actor" (P725) for voice-only role]
            </data>
            <data key="d9">Q43416</data>
        </edge>
    </graph>
</graphml>
```

## Usage

### Generate a New Dataset

To get started with this project, first clone this repository and install the necessary
dependencies using Poetry.

```bash
git clone https://github.com/yourusername/wiki-entity-summarization.git
cd wiki-entity-summarization
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.in-project true
poetry install
poetry shell

# You can set the parameters via .env file instead of providing command line arguments.
cp .env_sample .env

python3 main.py [-h] [--min_valid_summary_edges MIN_VALID_SUMMARY_EDGES] 
                [--random_walk_depth_len RANDOM_WALK_DEPTH_LEN] [--bridges_number BRIDGES_NUMBER] 
                [--max_threads MAX_THREADS] [--output_path OUTPUT_PATH] [--db_name DB_NAME] [--db_user DB_USER] 
                [--db_password DB_PASSWORD] [--db_host DB_HOST] [--db_port DB_PORT] [--neo4j_user NEO4J_USER] 
                [--neo4j_password NEO4J_PASSWORD] [--neo4j_host NEO4J_HOST] [--neo4j_port NEO4J_PORT]
                [dataset_name] [min_random_walk_number] [max_random_walk_number] [seed_node_ids] [categories]
                
        options:
                -h, --help                Show this help message and exit
                --min_valid_summary_edges Minimum number of valid summaries for a seed ndoe
                --random_walk_depth_len   Depth length of random walks (number of nodes in each random walk)
                --bridges_number          Number of connecting path bridges between components
                --max_threads             Maximum number of threads
                --output_path             Path to save output data
                --db_name                 Database name
                --db_user                 Database user
                --db_password             Database password
                --db_host                 Database host
                --db_port                 Database port
                --neo4j_user              Neo4j user
                --neo4j_password          Neo4j password
                --neo4j_host              Neo4j host
                --neo4j_port              Neo4j port

        Positional arguments:
                dataset_name              The name of the dataset to process (required)
                min_random_walk_number    Minimum number of random walks for each seed node (required)
                max_random_walk_number    Maximum number of random walks for each seed node (required)
                seed_node_ids             Seed node ids in comma-separated format (required)
                categories                Seed node categories in comma-separated format (optional)

```

### Re-generate WikES Dataset

To re-construct our pre-generated datasets, you can use the following command:

```bash 
python3 human_history_dataset.py
```

**This project uses our [pre-processor project](https://github.com/msorkhpar/wiki-entity-summarization-preprocessor)
databases. Make sure you have loaded the data and run the databases properly.**

## Citation

If you use this project in your research, please cite the following paper:

```bibtex
@misc{javadi2024wiki,
    title = {Wiki Entity Summarization Benchmark},
    author = {Saeedeh Javadi and Atefeh Moradan and Mohammad Sorkhpar and Klim Zaporojets and Davide Mottin and Ira Assent},
    year = {2024},
    eprint = {2406.08435},
    archivePrefix = {arXiv},
    primaryClass = {cs.IR}
}
```

## License

This project and its released datasets are licensed under the CC BY 4.0 License. See the [LICENSE](LICENSE)
file for details.

In the following, you can check other licenses that we used as external services, libraries, or software.
By using this project, you accept the third parties' licenses.

1. Wikipedia:
    - https://www.gnu.org/licenses/fdl-1.3.html
    - https://creativecommons.org/licenses/by-sa/3.0/
    - https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use
2. Wikidata:
    - https://creativecommons.org/publicdomain/zero/1.0/
    - https://creativecommons.org/licenses/by-sa/3.0/
3. Python:
    - https://docs.python.org/3/license.html#psf-license
    - https://docs.python.org/3/license.html#bsd0
    - https://docs.python.org/3/license.html#otherlicenses
4. DistilBERT:
    - https://github.com/RayWilliam46/FineTune-DistilBERT/blob/main/LICENSE
5. Networkx:
    - https://github.com/networkx/nx-guides/blob/main/LICENSE
6. Postgres:
    - https://opensource.org/license/postgresql
7. Neo4j:
    - https://www.gnu.org/licenses/quick-guide-gplv3.html
8. Docker:
    - https://github.com/moby/moby/blob/master/LICENSE
9. PyTorch:
    - https://github.com/intel/torch/blob/master/LICENSE.md
10. Scikit-learn:
    - https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
11. Pandas:
    - https://github.com/pandas-dev/pandas/blob/main/LICENSE
12. Numpy:
    - https://numpy.org/doc/stable/license.html
13. Java-open:
    - https://github.com/openjdk/jdk21/blob/master/LICENSE
14. Spring framework:
    - https://github.com/spring-projects/spring-boot/blob/main/LICENSE.txt
15. Other libraries:
    - https://github.com/tatuylonen/wikitextprocessor/blob/main/LICENSE
    - https://github.com/aaronsw/html2text/blob/master/COPYING
    - https://github.com/earwig/mwparserfromhell/blob/main/LICENSE
    - https://github.com/more-itertools/more-itertools/blob/master/LICENSE
    - https://github.com/siznax/wptools/blob/master/LICENSE
    - https://github.com/tqdm/tqdm/blob/master/LICENCE

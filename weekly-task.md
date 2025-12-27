# KGSUMM-PG Weekly Tasks

Note: If you need to have an in-person/group discussion, you can ping me on Element.

## Week 1 Task (Thursday, 16 October 2025, 10:00–12:00, Room FU.136.)

Task objective: Understand the entity summarization task by implementing existing models.

1. Choose each model of entity summarization from unsupervised and supervised learning methods (see https://github.com/asep-fajar-firmansyah/Entity-Summarization-Timeline)
2. Reproduce the entity summarization models on ESBM v1.2 and FACES datasets
3. Evaluate all models
4. Report findings
5. Submit the report through this [link](https://umfragen.uni-paderborn.de/index.php/871557)

## Week 2 Task (Wednesday, 22 October 2025, 14:00–15:00, Room FU.136/504) 

Task objective: Evaluate the model and present the findings.

1. Evaluate the selected models:
   - (a) Measure the model performance using F-Measure, NDCG score, and MAP.
   - (b) Conduct a statistical test using the Wilcoxon signed rank test.
   - (c) Evaluate the runtime between models.
3.  Conduct qualitative analysis.
4.  Conduct error analysis if needed.
5.  Write an experimental report in PDF that describes the experimental setup, the implementation, and the findings.
6.  Submit the report through this [link](https://umfragen.uni-paderborn.de/index.php/871557)

## Week 3 Task (Wednesday, 29 October 2025, 14:00–16:00, Room FU.504)

Task objective: Complete tasks 1 and 2

1. Submit the report through this [link](https://umfragen.uni-paderborn.de/index.php/871557)

## Week 4 Task (Wednesday, 6-12 November 2025, 14:00–16:00, Room FU.504)

Task objective: Complete tasks 1 and 2

1. Please complete both tasks (1 and 2).    
2. Submit the report through this [link](https://umfragen.uni-paderborn.de/index.php/871557)

## Week 5-6 Task (Wednesday, 13-26 November 2025, 14:00–16:00, Room FU.504)

Task objective: Evaluate the existing models on Wiki Entity Summarization Datasets (WikiES)

1. Decide which subgroup you will join: unsupervised or supervised learning.
2. Evaluate one existing model (choose either one unsupervised or one supervised model) on the WikiES-SMALL dataset.
3. Report your findings.   
4. Submit the report through this [link](https://umfragen.uni-paderborn.de/index.php/871557).

## Week 7 Task (Wednesday, 27 November - 03 December 2025, 14:00–16:00, Room FU.504)

Task objective: Identify your model and compare it with other models.

Note: The task is both an individual task and a group task; each student is responsible for their main task and also for contributing to the group outputs. 

1. Identify the strengths and limitations of your model by conducting a deep analysis.
2. Compare your model with other models
3. Report your findings.   
4. Submit the report through this [link](https://umfragen.uni-paderborn.de/index.php/871557).

## Week 8 Task (Wednesday, 04 December - 17 December 2025, Sub-group discussion)

Task objective: Develop a new approach to address the limitations of the existing models.

Note: The task is both an individual task and a group task; each student is responsible for their main task and also for contributing to the group outputs. 

1. Propose a new approach to address the limitations of the existing models.
2. Distribute the tasks needed to run the experiments.
3. Present your plan (on 17 December 2025).
4. Report your findings.   
5. Submit the report through this [link](https://umfragen.uni-paderborn.de/index.php/871557).

## Week 9 Task (Wednesday, 04 December 2025 - 21 January 2026, 14:00–16:00, Room FU.504)

Task objective: Implement your approach.

Note: The task is both an individual task and a group task; each student is responsible for their main task and also for contributing to the group outputs.

1. Implement the new approach by running the experiments.
3. Submit the report weekly through this [link](https://umfragen.uni-paderborn.de/index.php/871557).

# Group Tasks
## Sub Groups
### Sub Group Supervised Learning
1. Alireza Rahnama
2. Praveen Kumar Neswi Prakasha
3. Mrunal Sudhir patil
4. Shreyasri Ghosal
5. Swaranjali Arvind Mahadik

### Sub Group Unsupervised Learning
1. Vraj Rakesh Patel
2. Adam Satria
3. Eugene Agbor Egbe
4. Dhruv Mukeshbhai Sonani
5. Tayyaba Munawar
   
Note: You need to complete Tasks 1 and 2 before joining a subgroup. 

## Task Group (Wednesday, 27 November 2025 - 06 February 2026, 14:00–16:00, Room FU.504)

Task objective: Build the team, collaboratively analyze the existing models, and implement the proposed approach.

1. Create two sub-groups: Unsupervised and Supervised learning.
2. Produce comparative findings:
   - Compare the reproduced methods using the same datasets/evaluation protocol.
   - Identify the pros and cons of each method.
   - Discuss limitations.
   - Discuss future work:
      - How to Address the limitations.
      - Concrete ideas to improve and propose a new entity summarization model.
3. Implement the proposed approach.
4. Report findings.
5. Present your findings.

# Technical aspects
## The structure entity summaries outputs
```
├── outputs
│   ├── dbpedia
│   │   ├── <entity_id>
│   │   │   ├── <entity_id>_rank_top10.nt
│   │   │   ├── <entity_id>_rank_top5.nt
│   │   │   ├── <entity_id>_top10.nt
│   │   │   └── <entity_id>_top5.nt
│   ├── lmdb
│   │   ├── <entity_id>
│   │   │   ├── <entity_id>_rank_top10.nt
│   │   │   ├── <entity_id>_rank_top5.nt
│   │   │   ├── <entity_id>_top10.nt
│   │   │   └── <entity_id>_top5.nt
│   ├── faces
│   │   ├── <entity_id>
│   │   │   ├── <entity_id>_rank_top10.nt
│   │   │   ├── <entity_id>_rank_top5.nt
│   │   │   ├── <entity_id>_top10.nt
│   │   │   └── <entity_id>_top5.nt
│   ├── <dataset_name>
│   │   ├── <entity_id>
│   │   │   ├── <entity_id>_rank_top10.nt
│   │   │   ├── <entity_id>_rank_top5.nt
│   │   │   ├── <entity_id>_top10.nt
│   │   │   └── <entity_id>_top5.nt
```
Note: 
<entity_id>_rank_topK.nt is required if you evaluate the model using MAP and NDCG metrics.

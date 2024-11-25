## Energy and Performance-Aware Task Scheduling in a Mobile Cloud Computing Environment

### How applications are modeled in their MCC task scheduling system

1. Task Graph Representation:
- Applications are represented as a Directed Acyclic Graph (DAG) G = (V,E)
- V = set of tasks (nodes)
- E = edges showing precedence constraints between tasks

2. Graph Structure:
- Each node v ∈ V represents an individual task
- A directed edge (vi,vj) ∈ E represents a precedence constraint: Task vi must complete its execution before task vj can start
- The graph contains N total tasks

3. Special Task Types:
- Entry Task: A task with no parent nodes (no incoming edges)
- Exit Task: A task with no child nodes (no outgoing edges)

4. Data Requirements:
For each task v_i, two key parameters are defined:
- data_{i}: Amount of task specification and input data needed to upload to cloud
- data'_{i}: Amount of data needed to download from cloud after execution of task v_i is offloaded onto the cloud
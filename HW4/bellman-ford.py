import networkx as nx
import matplotlib.pyplot as plt

# Bellman-Ford Algorithm Implementation
def bellman_ford(src, V, G_adj_list):
    # Distance from the source
    d = [float('inf')] * V
    d[src] = 0

    # Relaxation step
    for _ in range(V - 1):
        for u in range(V):
            for v, w in G_adj_list[u]:
                if d[u] != float('inf') and d[u] + w < d[v]:
                    d[v] = d[u] + w

    # Negative cycle detection
    negative_cycle = False
    for u in range(V):
        for v, w in G_adj_list[u]:
            if d[u] != float('inf') and d[u] + w < d[v]:
                negative_cycle = True
                print("Graph contains a negative weight cycle.")
                return None

    # Print distances
    print("Vertex\tDistance from Source")
    for vertex, distance in enumerate(d):
        print(f"{vertex}\t{'INF' if distance == float('inf') else distance}")

    return d

# Define the graph using an edge list
edge_list = [
    (1, 3, 2),
    (4, 3, -1),
    (2, 4, 1),
    (1, 2, 1),
    (0, 1, 5),
]

# Number of vertices (based on the maximum vertex index in the edge list)
V = 5

# Convert edge list to adjacency list
G_adj_list = {i: [] for i in range(V)}
for u, v, w in edge_list:
    G_adj_list[u].append((v, w))

# Run Bellman-Ford Algorithm
src = 0
distances = bellman_ford(src, V, G_adj_list)

# Visualization using NetworkX
G = nx.DiGraph()
for u, v, w in edge_list:
    G.add_edge(u, v, weight=w)

pos = nx.spring_layout(G)  # Layout for better visualization

# Determine the Shortest Path Tree (SPT)
SPT_edges = []
if distances is not None:
    for u in range(V):
        for v, w in G_adj_list[u]:
            if distances[v] == distances[u] + w:
                SPT_edges.append((u, v))

# Draw the graph
plt.figure(figsize=(12, 6))

# Draw the original graph
plt.subplot(1, 2, 1)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Graph with Edge Weights")

# Draw only the Shortest Path Tree (SPT) edges
if distances is not None:
    SPT_G = nx.DiGraph()
    for u, v in SPT_edges:
        SPT_G.add_edge(u, v, weight=G[u][v]['weight'])  # Add edges from SPT

    plt.subplot(1, 2, 2)
    nx.draw(SPT_G, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10)
    sp_edge_labels = nx.get_edge_attributes(SPT_G, 'weight')
    nx.draw_networkx_edge_labels(SPT_G, pos, edge_labels=sp_edge_labels)
    plt.title("Shortest Path Tree (SPT)")

plt.tight_layout()
plt.show()
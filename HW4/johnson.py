import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

INF = float('inf')

def visualize_full_process(original_graph, reweighted_graph, potentials, V):
    """Visualizes the full process: Original Graph, Bellman-Ford Potentials, and Reweighted Graph."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot Original Graph
    G1 = nx.DiGraph()
    for u in range(V):
        if u in original_graph:
            for v, w in original_graph[u]:
                G1.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G1, seed=42)
    edge_labels = nx.get_edge_attributes(G1, 'weight')
    nx.draw(G1, pos, with_labels=True, ax=axes[0], node_color='lightblue', node_size=2000, font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=edge_labels, font_size=10, ax=axes[0])
    axes[0].set_title("Original Graph with Negative Edges Weights")

    # Plot Reweighted Graph
    G3 = nx.DiGraph()
    for u in range(V):
        if u in reweighted_graph:
            for v, w in reweighted_graph[u]:
                G3.add_edge(u, v, weight=w)

    edge_labels_reweighted = nx.get_edge_attributes(G3, 'weight')
    nx.draw(G3, pos, with_labels=True, ax=axes[1], node_color='lightblue', node_size=2000, font_size=12, font_weight='bold')
    nx.draw_networkx_edge_labels(G3, pos, edge_labels=edge_labels_reweighted, font_size=10, ax=axes[1])
    axes[1].set_title("Transformed Graph with No Negative Edge Weights")

    plt.tight_layout()
    plt.show()

def bellman_ford(src, V, G):
    d = [INF] * V
    d[src] = 0

    for _ in range(V - 1):
        for u in range(V):
            if u in G:
                for v, w in G[u]:
                    if d[u] != INF and d[u] + w < d[v]:
                        d[v] = d[u] + w

    for u in range(V):
        if u in G:
            for v, w in G[u]:
                if d[u] != INF and d[u] + w < d[v]:
                    print("Graph contains a negative weight cycle.")
                    return False, []

    return True, d

def johnson(V, G):
    original_graph = G.copy()

    G[V] = [(u, 0) for u in range(V)]

    has_no_neg_cycle, h = bellman_ford(V, V + 1, G)
    if not has_no_neg_cycle:
        print("Negative weight cycle detected. Algorithm cannot proceed.")
        return

    del G[V]

    reweighted_graph = defaultdict(list)
    for u in range(V):
        for v, w in G[u]:
            new_w = w + h[u] - h[v]
            reweighted_graph[u].append((v, new_w))

    visualize_full_process(original_graph, reweighted_graph, h, V)

if __name__ == "__main__":
    V = 4
    graph = [
        [0, -5, 2, 3],
        [0,  0, 4, 0],
        [0,  0, 0, 1],
        [0,  0, 0, 0]
    ]

    G = defaultdict(list)
    for u in range(V):
        for v in range(V):
            if graph[u][v] != 0:
                G[u].append((v, graph[u][v]))

    johnson(V, G)
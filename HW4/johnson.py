import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq

INT_MAX = float('inf')

def visualize_graph(G, edge_labels=None):
    """
    Visualize the graph using NetworkX.
    """
    pos = nx.spring_layout(G)  # Layout for visualization
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=10, font_weight="bold")
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def dijkstra_networkx(graph, src):
    """
    Dijkstra's algorithm using NetworkX's built-in functionality for a single source.
    """
    return nx.single_source_dijkstra(graph, src)

def johnson_with_visualization(V, edges):
    """
    Implement Johnson's Algorithm with visualization at key steps.
    """
    # Step 1: Create a graph
    G = nx.DiGraph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    print("Original Graph:")
    edge_labels = {(u, v): f"{w}" for u, v, w in edges}
    visualize_graph(G, edge_labels=edge_labels)

    # Step 2: Add a temporary vertex for reweighting
    reweighting_graph = G.copy()
    reweighting_graph.add_node(V)  # Temporary node
    for node in range(V):
        reweighting_graph.add_edge(V, node, weight=0)

    print("Graph with reweighting node added:")
    visualize_graph(reweighting_graph)

    # Step 3: Bellman-Ford for reweighting
    try:
        h = nx.single_source_bellman_ford_path_length(reweighting_graph, V)
    except nx.NetworkXUnbounded:
        print("Graph contains a negative weight cycle.")
        return

    print(f"Reweighting values (h): {h}")

    # Step 4: Reweight the graph
    reweighted_graph = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        reweighted_graph.add_edge(u, v, weight=data['weight'] + h[u] - h[v])

    print("Reweighted Graph:")
    edge_labels = {(u, v): f"{data['weight']}" for u, v, data in reweighted_graph.edges(data=True)}
    print(f"Edge Labels: {edge_labels}")
    visualize_graph(reweighted_graph, edge_labels=edge_labels)

    # Step 5: Compute shortest paths using Dijkstra's algorithm
    for src in range(V):
        distances, paths = dijkstra_networkx(reweighted_graph, src)
        print(f"Shortest paths from vertex {src}:")
        for dest in range(V):
            if dest in distances:
                # Convert back to original weights
                original_dist = distances[dest] - h[src] + h[dest]
                print(f"  To vertex {dest}: {original_dist}")
            else:
                print(f"  To vertex {dest}: inf")

if __name__ == "__main__":
    V = 4
    edges = [
        (0, 1, -5),
        (0, 2, 2),
        (0, 3, 3),
        (1, 2, 4),
        (2, 3, 1)
    ]

    johnson_with_visualization(V, edges)
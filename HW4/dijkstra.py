import heapq
import networkx as nx
import matplotlib.pyplot as plt

def dijkstra(src, V, G_adj_list):
    Q = []
    d = [float('inf')] * V

    d[src] = 0
    heapq.heappush(Q, (0, src))

    while Q:
        dist, u = heapq.heappop(Q)

        if dist > d[u]:
            continue

        for v, w in G_adj_list[u]:
            if d[u] + w < d[v]:
                d[v] = d[u] + w
                heapq.heappush(Q, (d[v], v))

    print("Vertex\tDistance from Source")
    for vertex, distance in enumerate(d):
        print(f"{vertex}\t{'INF' if distance == float('inf') else distance}")
    
    return d

V = 8
G_adj_list = {
    0: [(1, 3), (3, 7)],
    1: [(2, 1), (3, 4)],
    2: [(3, 2), (4, 5)],
    3: [(4, 1)],
    4: [(5, 7), (6, 3)],
    5: [(6, 2), (7, 4)],
    6: [(7, 6)],
    7: [],
}

src = 0
distances = dijkstra(src, V, G_adj_list)

G = nx.DiGraph()
for u, neighbors in G_adj_list.items():
    for v, w in neighbors:
        G.add_edge(u, v, weight=w)

pos = nx.spring_layout(G)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Original Graph")

SP_tree = nx.DiGraph()
for u in range(V):
    for v, w in G_adj_list[u]:
        if distances[v] == distances[u] + w:
            SP_tree.add_edge(u, v, weight=w)

plt.subplot(1, 2, 2)
nx.draw(SP_tree, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10)
sp_edge_labels = nx.get_edge_attributes(SP_tree, 'weight')
nx.draw_networkx_edge_labels(SP_tree, pos, edge_labels=sp_edge_labels)
plt.title("Shortest Path Tree (SPT from Source)")

plt.tight_layout()
plt.show()

import heapq
import networkx as nx
import matplotlib.pyplot as plt

def prim(G_adj_list, V):
    key = [float('inf')] * V
    MST = [-1] * V
    selected = [False] * V

    # Priority queue implemented using heapq
    Q = []
    heapq.heappush(Q, (0, 0))  # (weight, vertex)
    key[0] = 0

    while Q:
        _, u = heapq.heappop(Q)

        if selected[u]:
            continue

        selected[u] = True

        for w, v in G_adj_list[u]:
            if not selected[v] and w < key[v]:
                key[v] = w
                MST[v] = u
                heapq.heappush(Q, (key[v], v))

    return MST

# Graph adjacency list
V = 6  # Total number of vertices
G_adj_list = {
    0: [(6, 1), (1, 2), (5, 3)],
    1: [(6, 0), (5, 2), (3, 4)],
    2: [(1, 0), (5, 1), (2, 3), (6, 4), (4, 5)],
    3: [(5, 0), (2, 2), (4, 5)],
    4: [(3, 1), (6, 2), (6, 5)],
    5: [(4, 2), (4, 3), (6, 4)],
}

# Running Prim's algorithm
mst = prim(G_adj_list, V)

# Creating the original graph using NetworkX
G = nx.Graph()
for u in G_adj_list:
    for w, v in G_adj_list[u]:
        G.add_edge(u, v, weight=w)

# Creating the MST graph using NetworkX
MST_G = nx.Graph()
for v, u in enumerate(mst):
    if u != -1:
        for w, neighbor in G_adj_list[u]:
            if neighbor == v:
                MST_G.add_edge(u, v, weight=w)

# Visualization of the Original Graph
pos = nx.spring_layout(G)  # Layout for better visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Original Graph")

# Visualization of the MST
plt.subplot(1, 2, 2)
nx.draw(MST_G, pos, with_labels=True, node_color='lightgreen', edge_color='black', node_size=500, font_size=10)
mst_edge_labels = nx.get_edge_attributes(MST_G, 'weight')
nx.draw_networkx_edge_labels(MST_G, pos, edge_labels=mst_edge_labels)
plt.title("Minimum Spanning Tree (MST)")

plt.tight_layout()
plt.show()

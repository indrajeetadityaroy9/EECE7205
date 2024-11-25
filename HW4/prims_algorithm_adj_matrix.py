import networkx as nx
import matplotlib.pyplot as plt

def extract_min(key, selected):
    min_val = float('inf')
    min_idx = -1
    for i in range(len(key)):
        if not selected[i] and key[i] < min_val:
            min_val = key[i]
            min_idx = i
    return min_idx

def prim(G_matrix, V):
    key = [float('inf')] * V
    MST = [-1] * V
    selected = [False] * V

    key[0] = 0

    for _ in range(V - 1):
        u = extract_min(key, selected)
        if u == -1:
            return None
        selected[u] = True

        for v in range(len(G_matrix[u])):
            if G_matrix[u][v] > 0 and not selected[v] and G_matrix[u][v] < key[v]:
                key[v] = G_matrix[u][v]
                MST[v] = u

    return MST

V = 5
G_matrix = [
    [ 0, 4, 0, 7, 2 ],
[ 4, 0, 3, 0, 1 ],
[ 0, 3, 0, 5, 6 ],
[ 7, 0, 5, 0, 4 ],
[ 2, 1, 6, 4, 0 ]
]

mst = prim(G_matrix, V)

G = nx.Graph()
for i in range(V):
    for j in range(i + 1, V):
        if G_matrix[i][j] > 0:
            G.add_edge(i, j, weight=G_matrix[i][j])

MST_G = nx.Graph()
for v, u in enumerate(mst):
    if u != -1:
        MST_G.add_edge(u, v, weight=G_matrix[u][v])

pos = nx.spring_layout(G)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Original Graph")

plt.subplot(1, 2, 2)
nx.draw(MST_G, pos, with_labels=True, node_color='lightgreen', edge_color='black', node_size=500, font_size=10)
mst_edge_labels = nx.get_edge_attributes(MST_G, 'weight')
nx.draw_networkx_edge_labels(MST_G, pos, edge_labels=mst_edge_labels)
plt.title("Minimum Spanning Tree (MST)")

plt.tight_layout()
plt.show()
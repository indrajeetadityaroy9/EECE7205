#include <iostream>
#include <vector>
#include <climits>
#include <map>
#include <iomanip>
using namespace std;

vector<int> bf_dist;

bool bellmanFord(int src, int V, map<int, vector<pair<int, int>>>& G) {
    vector<int> d(V, INT_MAX);
    d[src] = 0;

    for (int i = 0; i < V - 1; ++i) {
        for (int u = 0; u < V; ++u) {
            if (G.find(u) != G.end()) {
                for (auto& e : G[u]) {
                    int v = e.first;
                    int w = e.second;
                    if (d[u] != INT_MAX && d[u] + w < d[v]) {
                        d[v] = d[u] + w;
                    }
                }
            }
        }
    }

    for (int u = 0; u < V; ++u) {
        if (G.find(u) != G.end()) {
            for (auto& e : G[u]) {
                int v = e.first;
                int w = e.second;
                if (d[u] != INT_MAX && d[u] + w < d[v]) {
                    cout << "Graph contains a negative weight cycle." << endl;
                    return false;
                }
            }
        }
    }
    bf_dist = d;
    return true;
}

vector<int> dijkstra(int src, int V, map<int, vector<pair<int, int>>>& G) {
    vector<int> d(V, INT_MAX);
    vector<bool> visited(V, false);
    vector<pair<int, int>> Q;

    d[src] = 0;
    Q.push_back({0, src});

    while (!Q.empty()) {
        int min_idx = 0;
        for (int i = 1; i < Q.size(); i++) {
            if (Q[i].first < Q[min_idx].first) {
                min_idx = i;
            }
        }

        int dist = Q[min_idx].first;
        int u = Q[min_idx].second;
        Q.erase(Q.begin() + min_idx);

        if (visited[u]) {
            continue;
        }
        visited[u] = true;

        if (G.find(u) != G.end()) {
            for (auto& e : G[u]) {
                int v = e.first;
                int w = e.second;

                if (!visited[v] && d[u] + w < d[v]) {
                    d[v] = d[u] + w;
                    Q.push_back({d[v], v});
                }
            }
        }
    }
    return d;
}

void johnson(int V, map<int, vector<pair<int, int>>>& G) {
    for (int u = 0; u < V; ++u) {
        G[V].emplace_back(u, 0);
    }

    if (!bellmanFord(V, V + 1, G)) {
        cout << "Negative weight cycle detected. Algorithm cannot proceed." << endl;
        return;
    }

    vector<int> h = bf_dist;
    map<int, vector<pair<int, int>>> new_G;

    for (int u = 0; u < V; ++u) {
        for (auto& [v, w] : G[u]) {
            // w'(u,v) = w(u,v) + h(u) - h(v)
            int new_w = w + h[u] - h[v];
            new_G[u].emplace_back(v, new_w);
        }
    }
    cout << endl;

    cout << "New Graph (Graph after adjusting edge weights to eliminate negative weights):" << endl;
    for (int u = 0; u < V; ++u) {
        for (int v = 0; v < V; ++v) {
            int w = INT_MAX;
            if (new_G.find(u) != new_G.end()) {
                for (const auto& e : new_G[u]) {
                    if (e.first == v) {
                        w = e.second;
                        break;
                    }
                }
            }
            cout << setw(5) << (w == INT_MAX ? "0" : to_string(w));
        }
        cout << endl;
    }

    vector<vector<int>> all_pairs_distances(V, vector<int>(V, INT_MAX));
    for (int u = 0; u < V; ++u) {
        vector<int> distances = dijkstra(u, V, new_G);

        for (int v = 0; v < V; ++v) {
            if (distances[v] < INT_MAX) {
                all_pairs_distances[u][v] = distances[v] + h[v] - h[u];
            }
        }
    }

    cout << "Original Graph Shortest Paths Between All Pairs (After Conversion from Reweighted Graph):" << endl;
    for (int u = 0; u < V; ++u) {
        for (int v = 0; v < V; ++v) {
            cout << setw(5) << (all_pairs_distances[u][v] == INT_MAX ? "inf" : to_string(all_pairs_distances[u][v]));
        }
        cout << endl;
    }
}

int main() {
    int V = 4;
    vector<vector<int>> graph = {
        {0, -5, 2, 3},
        {0,  0, 4, 0},
        {0,  0, 0, 1},
        {0,  0, 0, 0}
    };

    map<int, vector<pair<int, int>>> G;
    for (int u = 0; u < V; ++u) {
        for (int v = 0; v < V; ++v) {
            if (graph[u][v] != 0) {
                G[u].emplace_back(v, graph[u][v]);
            }
        }
    }

    johnson(V, G);
    return 0;
}
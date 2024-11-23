#include <iostream>
#include <vector>
#include <climits>

using namespace std;

bool bellmanFord(int src, int V, vector<vector<pair<int, int>>>& G) {
    vector<int> d(V, INT_MAX);
    d[src] = 0;

    for (int i = 1; i <= V - 1; ++i) {
        for (int u = 0; u < V; ++u) {
            for (auto& e : G[u]) {
                int v = e.first;
                int w = e.second;

                if (d[u] != INT_MAX && d[u] + w < d[v]) {
                    d[v] = d[u] + w;
                }
            }
        }
    }

    for (int u = 0; u < V; ++u) {
        for (auto& edge : G[u]) {
            int v = edge.first;
            int w = edge.second;

            if (d[u] != INT_MAX && d[u] + w < d[v]) {
                cout << "Graph contains a negative weight cycle.\n";
                return false;
            }
        }
    }

    cout << "Vertex\tDistance from Source\n";
    int vertex = 0;
    for (auto it = d.begin(); it != d.end(); ++it) {
        cout << vertex++ << "\t";
        if (*it == INT_MAX) {
            cout << "INF\n";
        } else {
            cout << *it << "\n";
        }
    }

    return true;
}

int main() {
    int V = 5;
    vector<vector<pair<int, int>>> G(V);

    G[0].push_back({1, -1});
    G[0].push_back({2, 4});
    G[1].push_back({2, 3});
    G[1].push_back({3, 2});
    G[1].push_back({4, 2});
    G[3].push_back({2, 5});
    G[3].push_back({1, 1});
    G[4].push_back({3, -3});

    int src = 0;

    if (!bellmanFord(src, V, G)) {
        cout << "Negative-weight cycle detected.\n";
    }

    return 0;
}
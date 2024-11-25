#include <iostream>
#include <vector>
#include <climits>
#include <map>
#include <set>
using namespace std;

bool bellmanFord(int src, int V, map<int, vector<pair<int, int>>>& G) {
    vector<int> d(V, INT_MAX);
    d[src] = 0;

    for (int i = 0; i < V - 1; ++i) {
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

    cout << "Vertex\tDistance from Source" << endl;
    for (int i = 0; i < V; ++i) {
        if (d[i] == INT_MAX) {
            cout << i << "\tINF" << endl;
        } else {
            cout << i << "\t" << d[i] << endl;
        }
    }
    return true;
}

int main() {
    int V = 5;
    vector<tuple<int, int, int>> edges = {
        {1, 3, 2},
        {4, 3, -1},
        {2, 4, 1},
        {1, 2, 1},
        {0, 1, 5},
    };

    map<int, vector<pair<int, int>>> G;
    for (auto& edge : edges) {
        int u, v, w;
        tie(u, v, w) = edge;
        G[u].emplace_back(v, w);
    }

    int src = 0;
    bellmanFord(src, V, G);
    return 0;
}
#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <climits>
#include <utility>
using namespace std;

void dijkstra(int src, int V, map<int, vector<pair<int, int>>>& G) {
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

    cout << "Vertex\tDistance from Source\n";
    for (int i = 0; i < V; i++) {
        cout << i << "\t";
        if (d[i] == INT_MAX) {
            cout << "INF\n";
        } else {
            cout << d[i] << "\n";
        }
    }
}

int main() {
    int V = 8;
    vector<tuple<int, int, int>> edges = {
        {0, 1, 3},
        {0, 3, 7},
        {1, 2, 1},
        {1, 3, 4},
        {2, 3, 2},
        {2, 4, 5},
        {3, 4, 1},
        {4, 5, 7},
        {4, 6, 3},
        {5, 6, 2},
        {5, 7, 4},
        {6, 7, 6},
    };

    map<int, vector<pair<int, int>>> G;
    for (auto& edge : edges) {
        int u, v, w;
        tie(u, v, w) = edge;
        G[u].emplace_back(v, w);
    }

    int src = 0;
    dijkstra(src, V, G);
    return 0;
}
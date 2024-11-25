#include <iostream>
#include <vector>
#include <tuple>
#include <map>
#include <climits>
#include <utility>
using namespace std;

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

int main() {
    int V = 8;
    map<int, vector<pair<int, int>>> G;
    G[0] = {{1, 3}, {3, 7}};
    G[1] = {{2, 1}, {3, 4}};
    G[2] = {{3, 2}, {4, 5}};
    G[3] = {{4, 1}};
    G[4] = {{5, 7}, {6, 3}};
    G[5] = {{6, 2}, {7, 4}};
    G[6] = {{7, 6}};
    G[7] = {};

    int src = 0;
    vector<int> distances = dijkstra(src, V, G);

    cout << "Vertex\tDistance from Source\n";
    for (int i = 0; i < distances.size(); i++) {
        cout << i << "\t" << distances[i] << "\n";
    }
    return 0;
}
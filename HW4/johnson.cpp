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
        bool relaxed = false;
        for (int u = 0; u < V; ++u) {
            for (auto& e : G[u]) {
                int v = e.first;
                int w = e.second;
                if (d[u] != INT_MAX && d[u] + w < d[v]) {
                    d[v] = d[u] + w;
                    relaxed = true;
                }
            }
        }
        if (!relaxed){
            break;
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
        return;
    }

    vector<int> h = bf_dist;
    map<int, vector<pair<int, int>>> new_G;

    cout << left << setw(10) << "Edge" << setw(20) << "Original Weight" << setw(20) << "New Weight" << endl;
    for (int u = 0; u < V; ++u) {
        for (auto& [v, w] : G[u]) {
            int new_w = w + h[u] - h[v];
            new_G[u].emplace_back(v, new_w);
            cout << setw(20) << "(" + to_string(u) + " -> " + to_string(v) + ")"
                 << setw(15) << w
                 << setw(20) << new_w << endl;
        }
    }
    cout << endl;

    vector<vector<int>> all_pairs_distances(V, vector<int>(V, INT_MAX));
    for (int u = 0; u < V; ++u) {
        vector<int> distances = dijkstra(u, V, new_G);

        for (int v = 0; v < V; ++v) {
            if (distances[v] < INT_MAX) {
                all_pairs_distances[u][v] = distances[v] + h[v] - h[u];
            }
        }
    }

    cout << "All Pairs Shortest Paths in Original Graph G:" << endl;
    cout << setw(6) << " " << " ";
    for (int i = 0; i < V; ++i) {
        cout << setw(8) << ("V" + to_string(i));
    }
    cout << endl;
    for (int u = 0; u < V; ++u) {
        cout << setw(6) << ("V" + to_string(u)) << " ";
        for (int v = 0; v < V; ++v) {
            if (all_pairs_distances[u][v] == INT_MAX)
                cout << setw(8) << "inf";
            else
                cout << setw(8) << all_pairs_distances[u][v];
        }
        cout << endl;
    }
}

int main() {
    int V = 5;
    map<int, vector<pair<int, int>>> G;
    G[0] = {{1, 3}, {2, 8}};
    G[1] = {{3, 1}, {4, -4}};
    G[2] = {{4, 2}};
    G[3] = {{0, 2}, {2, -5}};
    G[4] = {{3, 6}};

    johnson(V, G);
    return 0;
}
#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

int extract_min(priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>>& Q) {
    int vertex = Q.top().second;
    Q.pop();
    return vertex;
}

void prim(const vector<vector<pair<int, int>>>& G, int V) {
    vector<int> key(V, INT_MAX);
    vector<int> MST(V, -1);
    vector<bool> selected(V, false);

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> Q;
    key[0] = 0;
    Q.push({0, 0});

    while (!Q.empty()) {
        int u = extract_min (Q);

        if (selected[u]) {
            continue;
        }
        selected[u] = true;

        for (const auto& [v, w] : G[u]) {
            if (!selected[v] && w < key[v]) {
                key[v] = w;
                MST[v] = u;
                Q.push({key[v], v});
            }
        }
    }

    cout << "Edge \tWeight\n";
    for (int i = 1; i < V; ++i) {
        if (MST[i] != -1) {
            cout << MST[i] << " - " << i << "\t" << key[i] << "\n";
        }
    }
}

int main() {
    int V = 6;
    vector<vector<pair<int, int>>> G(V);
    G[0] = {{1, 6}, {2, 1}, {3, 5}};
    G[1] = {{0, 6}, {2, 5}, {4, 3}};
    G[2] = {{0, 1}, {1, 5}, {3, 2}, {4, 6}, {5, 4}};
    G[3] = {{0, 5}, {2, 2}, {5, 4}};
    G[4] = {{1, 3}, {2, 6}, {5, 6}};
    G[5] = {{2, 4}, {3, 4}, {4, 6}};

    prim(G, V);
    return 0;
}
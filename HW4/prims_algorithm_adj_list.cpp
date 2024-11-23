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
        int u = extract_min(Q);

        if (selected[u]) {
            continue;
        }

        selected[u] = true;

        for (const auto& [w, v] : G[u]) {
            if (!selected[v] && w < key[v]) {
                key[v] = w;
                MST[v] = u;
                Q.push({key[v], v});
            }
        }
    }

    cout << "Edge \tWeight\n";
    for (int i = 1; i < V; ++i) {
        if (MST[i] != -1)
            cout << MST[i] << " - " << i << "\t" << key[i] << "\n";
    }
}

int main() {
    int V = 6;
    vector<vector<pair<int, int>>> G(V);

    G[0].push_back({6, 1});
    G[0].push_back({1, 2});
    G[0].push_back({5, 3});

    G[1].push_back({6, 0});
    G[1].push_back({5, 2});
    G[1].push_back({3, 4});
    
    G[2].push_back({1, 0});
    G[2].push_back({5, 1});
    G[2].push_back({2, 3});
    G[2].push_back({6, 4});
    G[2].push_back({4, 5});

    G[3].push_back({5, 0});
    G[3].push_back({2, 2});
    G[3].push_back({4, 5});

    G[4].push_back({3, 1});
    G[4].push_back({6, 2});
    G[4].push_back({6, 5});

    G[5].push_back({4, 2});
    G[5].push_back({4, 3});
    G[5].push_back({6, 4});

    prim(G, V);
    return 0;
}
#include <iostream>
#include <vector>
#include <climits>

using namespace std;

int extract_min(const vector<int>& key, const vector<bool>& selected) {
    int min = INT_MAX;
    int min_idx = -1;

    for (int i = 0; i < key.size(); ++i) {
        if (!selected[i] && key[i] < min) {
            min = key[i];
            min_idx = i;
        }
    }
    return min_idx;
}

void prim(const vector<vector<int>>& G, int V) {
    vector<int> key(V, INT_MAX);
    vector<int> MST(V, -1);
    vector<bool> selected(V, false);

    key[0] = 0;

    for (int i = 0; i < V - 1; ++i) {
        int u = extract_min(key, selected);
        if (u == -1) {
            return;
        }
        selected[u] = true;

        for (int v = 0; v < G[u].size(); ++v) {
            if (G[u][v] > 0 && !selected[v] && G[u][v] < key[v]) {
                key[v] = G[u][v];
                MST[v] = u;
            }
        }
    }

    cout << "Edge \tWeight\n";
    for (int i = 1; i < MST.size(); ++i) {
        if (MST[i] != -1)
            cout << MST[i] << " - " << i << "\t" << G[i][MST[i]] << "\n";
    }
}

int main() {
    int V = 5;
    vector<vector<int>> G = {
        { 0, 4, 0, 7, 2 },
        { 4, 0, 3, 0, 1 },
        { 0, 3, 0, 5, 6 },
        { 7, 0, 5, 0, 4 },
        { 2, 1, 6, 4, 0 }
    };

    prim(G, V);
    return 0;
}
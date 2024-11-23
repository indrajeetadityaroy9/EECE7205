#include <iostream>
#include <vector>
#include <queue>
#include <climits>

using namespace std;

void dijkstra(int src, int V, vector<vector<pair<int, int>>>& G) {

    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> Q;
    vector<int> d(V, INT_MAX);

    d[src] = 0;
    Q.push({0, src});

    while (!Q.empty()) {
        int dist = Q.top().first;
        int u = Q.top().second;
        Q.pop();

        if (dist > d[u]){
            continue;
        }

        for (auto& e : G[u]) {
            int v = e.first;
            int w = e.second;

            if (d[u] + w < d[v]) {
                d[v] = d[u] + w;
                Q.push({d[v], v});
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
}   

int main() {
    int V = 8;
    vector<vector<pair<int, int>>> G(V);

    G[0].push_back({1, 3});
    G[0].push_back({3, 7});
    G[1].push_back({2, 1});
    G[1].push_back({3, 4});
    G[2].push_back({3, 2});
    G[2].push_back({4, 5});
    G[3].push_back({4, 1});
    G[4].push_back({5, 7});
    G[4].push_back({6, 3});
    G[5].push_back({6, 2});
    G[5].push_back({7, 4});
    G[6].push_back({7, 6});

    int src = 0;
    dijkstra(src, V, G);

    return 0;
}
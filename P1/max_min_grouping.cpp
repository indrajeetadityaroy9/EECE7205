#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <numeric>
using namespace std;

vector<int> max_min_grouping(const vector<int>& A, int N, int M) {
    // Initialize DP table C[i][j] which stores the maximum
    // minimal sum that can be achieved by partitioning the i elements into j groups
    vector<vector<int>> C(N + 1, vector<int>(M + 1, INT_MIN));
    // K table to store the partition points for reconstructing the optimal groups
    vector<vector<int>> K(N + 1, vector<int>(M + 1, -1));
    // Base case: For 0 elements and 0 groups, the value max infinity as maximizing the minimum values
    C[0][0] = INT_MAX;
    // Sum array P to quickly calculate sums over any subarray of A
    vector<int> P(N + 1, 0);
    for (int i = 1; i <= N; ++i) {
        P[i] = P[i - 1] + A[i - 1];
    }
    // DP base case: Only 1 group (j=1)
    for (int i = 1; i <= N; ++i) {
        C[i][1] = P[i];
        K[i][1] = 0;
    }

    // DP table populate using the recurrence relation for each possible number of groups j (from 2 to M) and elements i (from j to N)
    // Try multiple partition points from k until (j-1) to partition the first i elements into j groups by considering different
    for (int j = 2; j <= M; ++j) {
        for (int i = j; i <= N; ++i) {
            for (int k = j - 1; k < i; ++k) {
                // Sum of elements from A[k+1] to A[i]
                int sum = P[i] - P[k];
                // Minimum value from partitioning the first k elements into (j-1) groups and including the new sum
                int min_sum = min(C[k][j - 1], sum);
                // Update the DP table if minimum value is larger minimum sum
                if (min_sum > C[i][j]) {
                    C[i][j] = min_sum;
                    K[i][j] = k; // Store the partition point for backtracking
                }
            }
        }
    }

    // Backtrack to reconstruct the optimal grouping G[1..M] by using the partition points stored in K
    vector<int> G(M, 0);
    int idx = N;
    for (int j = M; j >= 1; --j) {
        int k = K[idx][j]; // Get the partition point for group
        G[j - 1] = idx - k; // Store the size of the j-th group
        idx = k; // Move to the previous partition point
    }

    // Return the grouping sizes G[1..M], where each G[j] represents the number of elements
    return G;
}

int main() {
    vector<int> A = {3,1,4,1,5,9,2,6,5,3};
    int N = A.size();
    int M = 3;

    vector<int> G = max_min_grouping(A, N, M);

    cout << "Optimal Grouping G: ";
    for (int g : G) {
        cout << g << " ";
    }
    cout << endl;

    return 0;
}

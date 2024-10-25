#include <iostream>
#include <vector>
#include <string_view>
#include <algorithm>

using namespace std;

// Function to find the length of the LCS and also return the LCS itself
pair<int, string> lcs(string_view s1, string_view s2) {
    int n = s1.size();
    int m = s2.size();

    // DP table to store LCS length for different prefixes of s1 and s2
    vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));

    // Fill DP table
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (s1[i - 1] == s2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    int index = dp[n][m];
    string lcs;
    lcs.reserve(index);

    int i = n, j = m;
    while (i > 0 && j > 0) {
        if (s1[i - 1] == s2[j - 1]) {
            lcs.push_back(s1[i - 1]);
            --i;
            --j;
        } else if (dp[i - 1][j] > dp[i][j - 1]) {
            --i;
        } else {
            --j;
        }
    }

    reverse(lcs.begin(), lcs.end());

    return {dp[n][m], lcs};
}

int main() {
    string s1, s2;
    cout << "Enter first string: ";
    cin >> s1;
    cout << "Enter second string: ";
    cin >> s2;

    auto [length, lcs] = lcs(s1, s2);  // Structured binding
    cout << "Length of LCS: " << length << "\nLCS: " << lcs << endl;

    return 0;
}

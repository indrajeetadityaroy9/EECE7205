#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_set>

using namespace std;

void print_all_lcs(const vector<vector<int>>& C, const vector<vector<char>>& b, const string& s1, int i, int j, string curr_lcs, unordered_set<string>& all_lcs) {

    if (i == 0 || j == 0) {
        reverse(curr_lcs.begin(), curr_lcs.end());
        all_lcs.insert(curr_lcs);
        return;
    }

    if (b[i][j] == 'D') {
        curr_lcs.push_back(s1[i - 1]);
        print_all_lcs(C, b, s1, i - 1, j - 1, curr_lcs, all_lcs);
    } else {
        if (b[i][j] == 'U' || C[i - 1][j] == C[i][j]) {
            print_all_lcs(C, b, s1, i - 1, j, curr_lcs, all_lcs);
        }
        if (b[i][j] == 'L' || C[i][j - 1] == C[i][j]) {
            print_all_lcs(C, b, s1, i, j - 1, curr_lcs, all_lcs);
        }
    }
}

pair<int, unordered_set<string>> lcs(const string& s1, const string& s2) {
    int n = s1.size();
    int m = s2.size();

    vector<vector<int>> C(n + 1, vector<int>(m + 1, 0));
    vector<vector<char>> b(n + 1, vector<char>(m + 1, ' '));

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (s1[i - 1] == s2[j - 1]) {
                C[i][j] = C[i - 1][j - 1] + 1;
                b[i][j] = 'D';
            } else if (C[i - 1][j] >= C[i][j - 1]) {
                C[i][j] = C[i - 1][j];
                b[i][j] = 'U';
            } else {
                C[i][j] = C[i][j - 1];
                b[i][j] = 'L';
            }
        }
    }

    unordered_set<string> all_lcs;
    print_all_lcs(C, b, s1, n, m, "", all_lcs);
    return {C[n][m], all_lcs};
}

int main() {
    string s1 = "ABCBDAB";
    string s2 = "BDCABA";

    auto [lcs_length, all_lcs] = lcs(s1, s2);
    cout << "Length of LCS: " << lcs_length << "\nAll LCS:\n";
    for (const auto& lcs_str : all_lcs) {
        cout << lcs_str << endl;
    }
    return 0;
}

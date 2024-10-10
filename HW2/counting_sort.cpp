#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

using namespace std;

void counting_sort(vector<int>& A) {
    if (A.empty()) return;

    const auto [min, max] = minmax_element(A.begin(), A.end());
    const int range = *max - *min + 1;

    vector<int> count(range, 0);
    for (const auto& num : A) {
        count[num - *min]++;
    }
    for (size_t i = 1; i < count.size(); ++i) {
        count[i] += count[i - 1];
    }

    vector<int> output(A.size());
    for (auto it = A.rbegin(); it != A.rend(); ++it) {
        output[--count[*it - *min]] = *it;
    }

    A = move(output);
}

int main() {
    vector<int> arr = {20, 18, 5, 7, 16, 10, 9, 3, 12, 14, 0};
    counting_sort(arr);
    cout << "Sorted array: ";
    copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));
    cout << endl;

    return 0;
}
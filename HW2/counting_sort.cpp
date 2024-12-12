#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
using namespace std;

void counting_sort(vector<int>& arr) {
    const auto [min, max] = minmax_element(arr.begin(), arr.end());
    const int range = *max - *min + 1;
    int n = arr.size();

    vector<int> B(n);
    vector<int> C(range, 0);
    
    for (const auto& num : arr) {
        C[num - *min]++;
    }

    for (size_t i = 1; i < C.size(); ++i) {
        C[i] += C[i - 1];
    }

    for (auto it = arr.rbegin(); it != arr.rend(); ++it) {
        B[--C[*it - *min]] = *it;
    }

    arr = move(B);
}

int main() {
    vector<int> arr = {20, 18, 5, 7, 16, 10, 9, 3, 12, 14, 0};
    counting_sort(arr);
    cout << "Sorted array: ";
    copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));
    cout << endl;

    return 0;
}

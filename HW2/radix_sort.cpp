#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

void digit_counting_sort(vector<int>& arr, int x) {
    int n = arr.size();
    vector<int> B(n);
    vector<int> C(10, 0);

    for (int i = 0; i < n; i++) {
        int digit = (arr[i] / x) % 10;
        C[digit]++;
    }

    for (int i = 1; i < 10; i++) {
        C[i] += C[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
        int digit = (arr[i] / x) % 10;
        B[--C[digit]] = arr[i];
    }

    for (int i = 0; i < n; i++) {
        arr[i] = B[i];
    }
}

void radix_sort(vector<int>& arr) {
    int max = *max_element(arr.begin(), arr.end());

    for (int x = 1; max / x > 0; x *= 10) {
        digit_counting_sort(arr, x);
    }
}

int main() {
    vector<int> arr = {329, 457, 657, 839, 436, 720, 353};
    radix_sort(arr);

    cout << "Sorted array: ";
    copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));
    cout << endl;

    return 0;
}
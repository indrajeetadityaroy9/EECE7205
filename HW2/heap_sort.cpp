#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <ctime>
#include <iterator>
using namespace std;

void heapify(vector<int>& arr, int n, int i) {
    int parent = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[parent])
        parent = left;

    if (right < n && arr[right] > arr[parent])
        parent = right;

    if (parent != i) {
        swap(arr[i], arr[parent]);
        heapify(arr, n, parent);
    }
}

void heap_sort(vector<int>& arr) {
    int n = arr.size();

    // Build heap
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(arr, n, i);
    }

    // Sort
    for (int i = n - 1; i > 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

int main() {
    vector<int> arr(100);
    for (int i = 0; i < 100; ++i) {
        arr[i] = i + 1;
    }

    random_device rd;
    mt19937 g(rd());
    shuffle(arr.begin(), arr.end(), g);

    cout << "Random permutation of A: ";
    copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));
    cout << endl;

    heap_sort(arr);

    cout << "Sorted array: ";
    copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));
    cout << endl;

    return 0;
}

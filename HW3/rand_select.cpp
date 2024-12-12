#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
using namespace std;

int randomized_partition(vector<int>& arr, int left, int right, mt19937& g) {
    uniform_int_distribution<int> dist(left, right);
    int pivot_idx = dist(g);
    
    int pivot = arr[pivot_idx];
    swap(arr[pivot_idx], arr[right]);
    int partition_idx = left;

    for (int i = left; i < right; i++) {
        if (arr[i] < pivot) {
            swap(arr[i], arr[partition_idx]);
            partition_idx++;
        }
    }
    swap(arr[partition_idx], arr[right]);

    return partition_idx;
}

int rand_select(vector<int>& arr, int p, int q, int i, mt19937& g) {
    if (p == q)
        return arr[p];

    int r = randomized_partition(arr, p, q, g);
    int k = r - p + 1;

    if (i == k)
        return arr[r];
    else if (i < k)
        return rand_select(arr, p, r - 1, i, g);
    else
        return rand_select(arr, r + 1, q, i - k, g);
}

int kth_smallest_element(vector<int>& arr, int k) {
    random_device rd;
    mt19937 g(rd());
    return rand_select(arr, 0, arr.size() - 1, k, g);
}

int main() {
    vector<int> arr(100);
    for (int i = 0; i < 100; i++) {
        arr[i] = i + 1;
    }

    random_device rd;
    mt19937 g(rd());
    shuffle(arr.begin(), arr.end(), g);

    std::cout << "Shuffled input array A: ";
    std::copy(arr.begin(), arr.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    for (int k = 1; k <= 10; k++) {
        int result = kth_smallest_element(arr, k);
        if (result != -1) {
            std::cout << "The " << k << "-th smallest element: " << result << std::endl;
        }
    }
    return 0;
}

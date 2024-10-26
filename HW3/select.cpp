#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <random>

using namespace std;

int partition(vector<int>& arr, int left, int right, int pivot) {
    int pivot_idx = left;
    for (int i = left; i <= right; i++) {
        if (arr[i] == pivot) {
            pivot_idx = i;
            break;
        }
    }
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

int median(vector<int>& A, int left, int right) {
    sort(A.begin() + left, A.begin() + right + 1);
    int mid = (left + right) / 2;
    return A[mid];
}

int median_of_medians(vector<int>& arr, int p, int q) {
    int n = q - p + 1;
    if (n <= 5) {
        return median(arr, p, q);
    }

    int groups = (n + 4) / 5;
    vector<int> medians(groups);

    for (int i = 0; i < groups; i++) {
        int l = p + i * 5;
        int r = min(l + 4, q);
        medians[i] = median(arr, l, r);
    }

    return median_of_medians(medians, 0, groups - 1);
}

int select(vector<int>& arr, int p, int q, int i) {
    if (p == q)
        return arr[p];
    
    int x = median_of_medians(arr, p, q);
    int pivot_idx = partition(arr, p, q, x);
    int k = pivot_idx - p + 1;

    if (i == k)
        return arr[pivot_idx];
    else if (i < k)
        return select(arr, p, pivot_idx - 1, i);
    else
        return select(arr, pivot_idx + 1, q, i - k);
}

int kth_smallest_element(vector<int>& arr, int k) {
    return select(arr, 0, arr.size() - 1, k);
}

int main() {
    vector<int> arr(100);
    for (int i = 0; i < 100; i++) {
        arr[i] = i + 1;
    }

    random_device rd;
    mt19937 g(rd());
    shuffle(arr.begin(), arr.end(), g);

    cout << "Shuffled input array A: ";
    copy(arr.begin(), arr.end(), ostream_iterator<int>(cout, " "));
    cout << endl;

    for (int k = 1; k <= 10; k++) {
        int result = kth_smallest_element(arr, k);
        if (result != -1) {
            cout << "The " << k << "-th smallest element: " << result << endl;
        }
    }
    return 0;
}
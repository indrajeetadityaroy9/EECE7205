#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <random>

using namespace std;

int partition(vector<int> &arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

int median(vector<int> &A, int low, int high) {
    sort(A.begin() + low, A.begin() + high + 1);
    int mid = (low + high) / 2;
    return A[mid];
}

int medianOfMedians(vector<int> &A, int low, int high) {
    int n = high - low + 1;
    
    // Divide the array into groups of 5
    vector<int> medians;
    for (int i = 0; i < n / 5; i++) {
        int groupLow = low + i * 5;
        int groupHigh = min(groupLow + 4, high);
        medians.push_back(median(A, groupLow, groupHigh));
    }

    // If the array size is not a multiple of 5, handle the remainder
    if (n % 5 != 0) {
        int groupLow = low + (n / 5) * 5;
        int groupHigh = high;
        medians.push_back(median(A, groupLow, groupHigh));
    }

    // Recursively find the median of medians
    if (medians.size() <= 5) {
        return median(medians, 0, medians.size() - 1);
    } else {
        return medianOfMedians(medians, 0, medians.size() - 1);
    }
}

// Select function (worst-case linear time)
int select(vector<int> &A, int low, int high, int i) {
    if (low == high) {
        return A[low];
    }

    // Find a good pivot using the median of medians
    int x = medianOfMedians(A, low, high);
    
    // Partition the array around the pivot
    int x_idx = partition(A, low, high);

    int k = x_idx - low + 1;  // Number of elements in the left partition

    if (i == k) {
        return A[x_idx];
    } else if (i < k) {
        return select(A, low, x_idx - 1, i);
    } else {
        return select(A, x_idx + 1, high, i - k);
    }
}

int main() {
    // Seed for random number generation
    srand(time(0));

    // Generate the array A = {1, 2, 3, ..., 100}
    vector<int> A(100);
    for (int i = 0; i < 100; i++) {
        A[i] = i + 1;
    }

    // Create a random number generator
    random_device rd;
    mt19937 g(rd());  // Mersenne Twister random number generator

    // Shuffle the array using std::shuffle (C++11 and later)
    shuffle(A.begin(), A.end(), g);

    // Test Select (Median of Medians) to find the 50th smallest element
    int selectResult = select(A, 0, A.size() - 1, 50);
    cout << "50th smallest element using Select (Median of Medians): " << selectResult << endl;

    return 0;
}
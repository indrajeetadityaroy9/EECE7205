#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace chrono;

template<typename Func>
double measure_time(Func sort_function, vector<int>& arr) {
    auto start = high_resolution_clock::now();
    sort_function(arr, 0, arr.size() - 1);
    auto end = high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

int partition(vector<int> &A, int low, int high) {
    int pivot = A[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (A[j] <= pivot) {
            i++;
            swap(A[i], A[j]);
        }
    }
    swap(A[i + 1], A[high]);
    return i + 1;
}

void quick_sort(vector<int> &A, int low, int high) {
    if (low < high) {
        int pi = partition(A, low, high);
        quick_sort(A, low, pi - 1);
        quick_sort(A, pi + 1, high);
    }
}

int main() {
    srand(time(0));
    vector<int> repeated_arr_1(100000, 1);  // 100,000 elements, all set to 1
    vector<int> repeated_arr_2(100000);     // 100,000 elements, first half set to 1, rest random
    vector<int> repeated_arr_3(100000);     // 100,000 elements, alternating between 2 and 20

    for (int i = 0; i < 50000; ++i) {
        repeated_arr_2[i] = 1;
    }
    for (int i = 50000; i < 100000; ++i) {
        repeated_arr_2[i] = rand() % 100;
    }
    for (int i = 0; i < 100000; ++i) {
        repeated_arr_3[i] = (i % 2 == 0) ? 2 : 20;
    }

    vector<int> input_arr = repeated_arr_1;
    double time_taken1 = measure_time(quick_sort, input_arr);
    cout << "Input (100,000 repeated elements, all 1): " << time_taken1 << " seconds" << endl;
    input_arr = repeated_arr_2;
    double time_taken2 = measure_time(quick_sort, input_arr);
    cout << "Input (50,000 repeated elements and random elements split): " << time_taken2 << " seconds" << endl;
    input_arr = repeated_arr_3;
    double time_taken3 = measure_time(quick_sort, input_arr);
    cout << "Input (elements alternating between 2 and 20): " << time_taken3 << " seconds" << endl;

    return 0;
}
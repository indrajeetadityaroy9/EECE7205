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

void partition(vector<int> &arr, int low, int high, int &l_pivot, int &g_pivot) {
    int pivot = arr[high];
    l_pivot = low;
    g_pivot = high;
    int i = low;

    while (i <= g_pivot) {
        if (arr[i] < pivot) {
            swap(arr[l_pivot], arr[i]);
            l_pivot++;
            i++;
        } else if (arr[i] > pivot) {
            swap(arr[i], arr[g_pivot]);
            g_pivot--;
        } else {
            i++;
        }
    }
}

void quick_sort(vector<int> &arr, int low, int high) {
    if (low < high) {
        int l_p, g_p;
        partition(arr, low, high, l_p, g_p);
        quick_sort(arr, low, l_p - 1);
        quick_sort(arr, g_p + 1, high);
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
    cout << "Time taken (100,000 repeated elements, all 1): " << time_taken1 << " seconds" << endl;
    input_arr = repeated_arr_2;
    double time_taken2 = measure_time(quick_sort, input_arr);
    cout << "Time taken (50,000 repeated elements and random elements split): " << time_taken2 << " seconds" << endl;
    input_arr = repeated_arr_3;
    double time_taken3 = measure_time(quick_sort, input_arr);
    cout << "Time taken (elements alternating between 2 and 20): " << time_taken3 << " seconds" << endl;

    return 0;
}
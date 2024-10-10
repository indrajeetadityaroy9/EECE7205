#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

template<typename Func>
double measure_time(Func sort, vector<int>& arr) {
    auto start = chrono::high_resolution_clock::now();
    sort(arr);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

int partition(vector<int> &A, int low, int high) {
    int random_index = low + rand() % (high - low + 1);
    swap(A[random_index], A[high]);
    
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

void randomized_quick_sort(vector<int> &A, int low, int high) {
    if (low < high) {
        int pi = partition(A, low, high);
        randomized_quick_sort(A, low, pi - 1);
        randomized_quick_sort(A, pi + 1, high);
    }
}

auto quicksort_wrapper = [](vector<int>& arr) {
    randomized_quick_sort(arr, 0, arr.size() - 1);
};

int main() {
    srand(time(0));

    vector<int> A(100);
    for (int i = 0; i < 100; i++) {
        A[i] = i + 1;
    }
    vector<double> run_times;
    double total_time = 0.0;
    
    for (int run = 1; run <= 5; run++) {
        vector<int> input_arr = A;
        double run_time = measure_time(quicksort_wrapper, input_arr);
        total_time += run_time;
        run_times.push_back(run_time);
        cout << "Run " << run << ": " << run_time << " seconds" << endl;
    }
    
    double average_time = total_time / 5;
    cout << "Average running time: " << average_time << " seconds" << endl;
    return 0;
}
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

int partition(vector<int> &arr, int low, int high) {
    int rand_idx = low + rand() % (high - low + 1);
    swap(arr[rand_idx], arr[high]);
    
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

void randomized_quick_sort(vector<int> &arr, int low, int high) {
    if (low < high) {
        int p = partition(arr, low, high);
        randomized_quick_sort(arr, low, p - 1);
        randomized_quick_sort(arr, p + 1, high);
    }
}

auto quicksort_wrapper = [](vector<int>& arr) {
    randomized_quick_sort(arr, 0, arr.size() - 1);
};

int main() {
    srand(time(0));

    vector<int> arr(100);
    for (int i = 0; i < 100; i++) {
        arr[i] = i + 1;
    }
    vector<double> run_times;
    double total_time = 0.0;
    
    for (int run = 1; run <= 5; run++) {
        vector<int> input_arr = arr;
        double run_time = measure_time(quicksort_wrapper, input_arr);
        total_time += run_time;
        run_times.push_back(run_time);
        cout << "Run " << run << ": " << run_time << " seconds" << endl;
    }
    
    double average_time = total_time / 5;
    cout << "Average running time: " << average_time << " seconds" << endl;
    return 0;
}

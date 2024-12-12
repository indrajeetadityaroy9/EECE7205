#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <matplot/matplot.h>
using namespace std;
using namespace matplot;

template<typename Func>
double measure_time(Func sort_function, vector<int>& arr) {
    auto start = chrono::high_resolution_clock::now();
    sort_function(arr, 0, arr.size() - 1);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

int partition(vector<int>& arr, int left, int right) {
    int mid = arr[right];
    int i = left - 1;

    for (int j = left; j <right; ++j) {
        if (arr[j] <= mid) {
            ++i;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[right]);
    return i + 1;
}

void quick_sort(vector<int>& arr, int left, int right) {
    if (left <right) {
        int mid = partition(arr, left,right);
        quick_sort(arr, left, mid - 1);
        quick_sort(arr, mid + 1,right);
    }
}

int main() {
    int n = 10000;
    vector<int> worst_case_arr(n);
    iota(worst_case_arr.begin(), worst_case_arr.end(), 1);

    vector<int> worst_case_shuffled_arr = worst_case_arr;
    random_device rd;
    shuffle(worst_case_shuffled_arr.begin(), worst_case_shuffled_arr.end(), g(rd()));

    double worst_case_time = measure_time(quick_sort, worst_case_arr);
    double worst_case_shuffled_time = measure_time(quick_sort, worst_case_shuffled_arr);

    vector<string> cases = {"Worst Case", "Worst Case Shuffled"};
    vector<double> times = {worst_case_time, worst_case_shuffled_time};
    vector<double> x_values = {1, 2};

    auto fig = figure(true);
    bar(x_values, times);
    gca()->xticklabels(cases);
    ylabel("Time (seconds)");
    title("Quick Sort Performance: Worst Case vs Worst Case Shuffled");
    gca()->grid(false);
    show();
    return 0;
}

#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <matplot/matplot.h>
#include <functional>

using namespace std;
using namespace matplot;

template<typename Func>
double measure_time(Func sort, vector<int>& arr) {
  auto start = chrono::high_resolution_clock::now();
  sort(arr);
  auto end = chrono::high_resolution_clock::now();
  chrono::duration<double> elapsed = end - start;
  return elapsed.count();
}

void insertion_sort(vector<int> arr) {
  for (int i = 1; i < arr.size(); ++i) {
    int temp = arr[i];
    int j = i;
    while (j > 0 && arr[j - 1] > temp) {
      arr[j] = arr[j - 1];
      --j;
    }
    arr[j] = temp;
  }
}

void merge(vector<int>& arr, int left, int mid, int right) {
  int idx1 = mid - left + 1;
  int idx2 = right - mid;
  vector<int> Left(idx1);
  vector<int> Right(idx2);

  for (int i = 0; i < idx1; ++i) {
    Left[i] = arr[left + i];
  }
  for (int i = 0; i < idx2; ++i) {
    Right[i] = arr[mid + 1 + i];
  }

  int i = 0;
  int j = 0; 
  int k = left;
  
  while (i < idx1 && j < idx2) {
    if (Left[i] <= Right[j]) {
      arr[k] = Left[i];
      ++i;
    } else {
      arr[k] = Right[j];
      ++j;
    }
    ++k;
  }

  while (i < idx1) {
    arr[k] = Left[i];
    ++i;
    ++k;
  }

  while (j < idx2) {
    arr[k] = Right[j];
    ++j;
    ++k;
  }
}

void merge_sort(vector<int>& arr, int left, int right) {
  if (left < right) {
    int mid = left + (right - left) / 2;
    merge_sort(arr, left, mid);
    merge_sort(arr, mid + 1, right);
    merge(arr, left, mid, right);
  }
}

int main() {
    vector<int> input_n_sizes = {1000, 2000, 4000, 8000, 10000, 15000, 18000, 20000};
    vector<double> insertion_sort_times;
    vector<double> merge_sort_times;

    for (int n : input_n_sizes) {
      vector<int> arr1(n);
      iota(arr1.begin(), arr1.end(), 1);
      reverse(arr1.begin(), arr1.end());
      vector<int> arr2 = arr1;
      
      double insertion_sort_time = measure_time([](auto& arr) { insertion_sort(arr); }, arr1);
      insertion_sort_times.push_back(insertion_sort_time);

      double merge_sort_time = measure_time([](auto& arr) { merge_sort(arr, 0, arr.size() - 1); }, arr2);
      merge_sort_times.push_back(merge_sort_time);
    }

    auto fig = figure(true);
    auto insertion_sort_p = plot(input_n_sizes, insertion_sort_times);
    insertion_sort_p->marker("o");
    insertion_sort_p->marker_face_color("auto");

    auto merge_sort_p = plot(input_n_sizes, merge_sort_times);
    merge_sort_p->marker("o");
    merge_sort_p->marker_face_color("auto");

    xlabel("Input Size (n)");
    ylabel("Time (seconds)");
    grid(true);
    show();
    return 0;
}

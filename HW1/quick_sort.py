import time
import sys
import matplotlib.pyplot as plt
import random

sys.setrecursionlimit(1000000)

def quick_sort(arr, low, high):
    if low < high:
        pivot = partition(arr, low, high)
        quick_sort(arr, low, pivot - 1)
        quick_sort(arr, pivot + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def running_time(sort_function, arr):
    start_time = time.perf_counter()
    sort_function(arr, 0, len(arr) - 1)
    end_time = time.perf_counter()
    return end_time - start_time

def main():
    n = 10000
    worst_case_arr = list(range(1, n + 1))
    worst_case_shuffled_arr = worst_case_arr.copy()
    random.shuffle(worst_case_shuffled_arr)

    worst_case_time = running_time(quick_sort, worst_case_arr)
    worst_case_shuffled_time = running_time(quick_sort, worst_case_shuffled_arr)
    cases = ['Worst Case', 'Worst Case Shuffled']
    times = [worst_case_time, worst_case_shuffled_time]

    plt.figure(figsize=(8, 6))
    plt.bar(cases, times, color=['red', 'green'])
    plt.ylabel('Time (seconds)')
    plt.show()

if __name__ == "__main__":
    main()

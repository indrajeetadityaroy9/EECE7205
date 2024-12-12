import time
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.setrecursionlimit(1000000)

def insertion_sort(arr):
    """
    Sorts an array in place using the Insertion Sort algorithm.
    """
    for i in range(1, len(arr)):
        temp = arr[i]
        j = i
        while j > 0 and arr[j - 1] > temp:
            arr[j] = arr[j - 1]
            j -= 1
        arr[j] = temp

def merge_sort(arr):
    """
    Sorts an array using the Merge Sort algorithm.
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]
        R = arr[mid:]

        # Recursively sort both halves
        merge_sort(L)
        merge_sort(R)

        # Merge the sorted halves
        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] <= R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Check if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

def generate_worst_case_input(n):
    """
    Generates a list of size n sorted in decreasing order.
    """
    return list(range(n, 0, -1))

def measure_time(sort_function, n):
    """
    Measures the time taken to sort an array of size n using the specified sorting function.
    """
    arr = generate_worst_case_input(n)
    start_time = time.perf_counter()
    sort_function(arr)
    end_time = time.perf_counter()
    return end_time - start_time

def main():
    input_n_sizes = [1000, 2000, 4000, 8000, 10000, 15000]
    insertion_sort_times = []
    merge_sort_times = []

    for n in input_n_sizes:
        time_taken_insertion = measure_time(insertion_sort, n)
        insertion_sort_times.append(time_taken_insertion)

        time_taken_merge = measure_time(merge_sort, n)
        merge_sort_times.append(time_taken_merge)

    input_n_sizes = np.array(input_n_sizes)
    insertion_sort_times = np.array(insertion_sort_times)
    merge_sort_times = np.array(merge_sort_times)

    sort_time_crossover_point_index = np.where(merge_sort_times < insertion_sort_times)[0]
    if len(sort_time_crossover_point_index) > 0:
        crossover_n = input_n_sizes[sort_time_crossover_point_index[0]]
        crossover_insertion_sort = insertion_sort_times[sort_time_crossover_point_index[0]]
        crossover_merge_sort = merge_sort_times[sort_time_crossover_point_index[0]]
        print(f"\nCrossover point occurs at n = {crossover_n}")
    else:
        crossover_n = None
        print("\nNo crossover point found in the tested range.")

    plt.figure(figsize=(12, 6))
    plt.plot(input_n_sizes, insertion_sort_times, marker='o', label='Insertion Sort')
    plt.plot(input_n_sizes, merge_sort_times, marker='o', label='Merge Sort')

    if crossover_n is not None:
        plt.axvline(x=crossover_n, color='gray', linestyle='--', label=f'Crossover at n = {crossover_n}')
        plt.scatter(crossover_n, crossover_insertion_sort, color='red', zorder=5)
        plt.scatter(crossover_n, crossover_merge_sort, color='red', zorder=5)

    plt.xlabel('Input Size (n)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

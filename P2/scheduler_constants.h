#ifndef SCHEDULER_CONSTANTS_H
#define SCHEDULER_CONSTANTS_H

#include <array>
#include <map>
#include <vector>

enum class SchedulingState {
    UNSCHEDULED = 0,      // Initial state before scheduling
    SCHEDULED = 1,        // Task has been scheduled in initial phase
    KERNEL_SCHEDULED = 2   // Task has been processed by kernel algorithm
};

namespace scheduler_constants {
    constexpr int NUM_CORES = 3;
    const std::array<int, 3> CLOUD_EXECUTION_TIMES = {3, 1, 1};
    const std::map<int, std::array<int, NUM_CORES>> CORE_EXECUTION_TIMES = {
        {1,  {9, 7, 5}},
        {2,  {8, 6, 5}},
        {3,  {6, 5, 4}},
        {4,  {7, 5, 3}},
        {5,  {5, 4, 2}},
        {6,  {7, 6, 4}},
        {7,  {8, 5, 3}},
        {8,  {6, 4, 2}},
        {9,  {5, 3, 2}},
        {10, {7, 4, 2}},
        {11, {12, 3, 3}},
        {12, {12, 8, 4}},
        {13, {11, 3, 2}},
        {14, {12, 11, 4}},
        {15, {13, 4, 2}},
        {16, {9, 7, 3}},
        {17, {9, 3, 3}},
        {18, {13, 9, 2}},
        {19, {10, 5, 3}},
        {20, {12, 5, 4}}
    };
}

// Structure to track migration decisions
struct TaskMigrationState {
    int time;              // Total completion time after migration
    int energy;            // Total energy consumption after migration
    double efficiency;     // Energy reduction per unit time
    int task_index;        // Task selected for migration
    int target_exec_unit;  // Target execution unit (core or cloud)

    TaskMigrationState(
        int t = 0, 
        int e = 0, 
        double eff = 0.0, 
        int task = -1, 
        int target = -1
    ) : time(t), energy(e), efficiency(eff), 
        task_index(task), target_exec_unit(target) {}

    // Support for structured bindings
    template<typename T>
    friend struct as_tuple_t;
};

// Enable structured bindings support
template<typename T>
struct [[nodiscard]] as_tuple_t;

template<>
struct as_tuple_t<TaskMigrationState> {
    // For const access
    static auto apply(const TaskMigrationState& s) {
        return std::make_tuple(s.time, s.energy, s.efficiency, s.task_index, s.target_exec_unit);
    }
    
    // For modifiable access
    static auto apply(TaskMigrationState& s) {
        return std::tie(s.time, s.energy, s.efficiency, s.task_index, s.target_exec_unit);
    }
};

#endif // SCHEDULER_CONSTANTS_H
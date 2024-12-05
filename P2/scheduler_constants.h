#ifndef SCHEDULER_CONSTANTS_H
#define SCHEDULER_CONSTANTS_H
#include <array>
#include <map>
#include <vector>
using namespace std;

enum class SchedulingState {
    UNSCHEDULED = 0,
    SCHEDULED = 1,
    KERNEL_SCHEDULED = 2
};

namespace scheduler_constants {
    constexpr int NUM_CORES = 3;
    const array<int, 3> CLOUD_EXECUTION_TIMES = {3, 1, 1};
    const map<int, array<int, NUM_CORES>> CORE_EXECUTION_TIMES = {
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

struct TaskMigrationState {
    int time;              
    double energy;          
    double efficiency;     
    int task_index;        
    int target_exec_unit;

    TaskMigrationState(
        int t = 0, 
        double e = 0.0, 
        double eff = 0.0, 
        int task = -1, 
        int target = -1
    ) : time(t), energy(e), efficiency(eff), 
        task_index(task), target_exec_unit(target) {}

    template<typename T>
    friend struct as_tuple_t;
};

template<typename T>
struct [[nodiscard]] as_tuple_t;

template<>
struct as_tuple_t<TaskMigrationState> {
    static auto apply(const TaskMigrationState& s) {
        return make_tuple(s.time, s.energy, s.efficiency, s.task_index, s.target_exec_unit);
    }
    
    static auto apply(TaskMigrationState& s) {
        return tie(s.time, s.energy, s.efficiency, s.task_index, s.target_exec_unit);
    }
};

#endif // SCHEDULER_CONSTANTS_H
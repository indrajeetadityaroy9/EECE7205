#ifndef MCC_H
#define MCC_H
#include <vector>
#include <numeric>
#include "task.h"
#include "initial_task_scheduler.h"
using namespace std;

class MCCScheduler {
public:
    MCCScheduler(vector<Task>& tasks, int num_cores = 3);
    vector<vector<int>> selectExecutionUnits();
    int totalTime() const;
    double calculateTaskEnergyConsumption(const Task& task, const vector<int>& core_powers, double cloud_sending_power) const;
    double totalEnergy(const vector<int>& core_powers, double cloud_sending_power) const;
    void primaryAssignment();
    void taskPrioritizing();
    int calculatePriority(Task& task, vector<int>& w,map<int, int>& computed_priority_scores);
    vector<Task>& tasks;
    int num_cores;
};

#endif // MCC_H
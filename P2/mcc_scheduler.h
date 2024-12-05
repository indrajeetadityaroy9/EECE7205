#ifndef MCC_H
#define MCC_H
#include <vector>
#include <numeric>
#include <queue>
#include <tuple>
#include <vector>   // For std::vector
#include <queue>    // For std::priority_queue
#include <tuple>    // For std::tuple
#include <limits>   // For std::numeric_limits
#include <optional> // For std::optional
#include <algorithm>// For std::max
#include "task.h"
#include "initial_task_scheduler.h"
#include "task_migration_scheduler.h"
using namespace std;

class MCCScheduler {
public:
    MCCScheduler(vector<Task>& tasks, int num_cores = 3);

    vector<Task>& tasks;
    int num_cores;

    vector<vector<int>> selectExecutionUnits();
    int totalTime(const vector<Task>& tasks);
    double  totalEnergy(const vector<Task>& tasks, const vector<int>& core_powers, double cloud_sending_power);
    void primaryAssignment();
    void taskPrioritizing();
    int calculatePriority(Task& task, vector<int>& w,map<int, int>& computed_priority_scores);
    std::vector<Task> KernalAlgorithm(std::vector<Task>& tasks, std::vector<std::vector<int>>& sequences);

    std::tuple<int, double> evaluateMigration(
    const std::vector<Task>& tasks,
    const std::vector<std::vector<int>>& seqs,
    int task_idx,
    int target_execution_unit,
    std::map<std::tuple<int, int, std::vector<int>>, std::tuple<int, double>>& migration_cache,
    const std::vector<int>& core_powers,
    double cloud_sending_power
    );

    std::vector<std::vector<bool>> initializeMigrationChoices(const std::vector<Task>& tasks);

    TaskMigrationState* identifyOptimalMigration(
    const std::vector<std::tuple<int, int, int, double>>& migration_trials_results,
    int T_final,
    double E_total,
    int T_max
);

std::pair<std::vector<Task>, std::vector<std::vector<int>>> optimizeTaskScheduling(
    std::vector<Task>& tasks,
    std::vector<std::vector<int>> sequence,
    int T_final,
    const std::vector<int>& core_powers,
    double cloud_sending_power
);

private:
    double  calculateEnergyConsumption(const Task& task, const vector<int>& core_powers, double cloud_sending_power);
    std::tuple<int, int, std::vector<int>> GenerateCacheKey(const std::vector<Task>& tasks, int task_idx, int target_execution_unit);
    vector<vector<int>> constructSequence(vector<Task>& tasks, int task_id, int execution_unit, vector<vector<int>> original_sequence);
};

#endif // MCC_H
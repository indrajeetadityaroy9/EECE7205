#include "mcc_scheduler.h"
#include <numeric>
#include <queue>
#include <tuple>
using namespace std;

 MCCScheduler:: MCCScheduler(vector<Task>& tasks, int num_cores)
    : tasks(tasks)
    , num_cores(num_cores) 
{
}

vector<vector<int>>  MCCScheduler::selectExecutionUnits() {
    InitialTaskScheduler scheduler(tasks, num_cores);
    vector<int> priority_ordered_tasks = scheduler.getPriorityOrderedTasks();
    auto [entry_tasks, non_entry_tasks] = scheduler.classifyEntryTasks(priority_ordered_tasks);
    scheduler.scheduleEntryTasks(entry_tasks);
    scheduler.scheduleNonEntryTasks(non_entry_tasks);
    return scheduler.sequences;
}

int MCCScheduler::totalTime(const vector<Task>& tasks) {
    // Implementation of total completion time calculation T_total from equation (10):
    // T_total = max(max(FTi^l, FTi^wr))
    //           vi∈exit_tasks
    
    vector<int> exit_task_times;
    
    // For each exit task vi, compute max(FTi^l, FTi^wr) where:
    // - FTi^l: Finish time if task executes on local core
    // - FTi^wr: Finish time if task executes on cloud (when results are received)
    for (const Task& task : tasks) {
        // Only consider exit tasks (tasks with no successors)
        if (task.getSuccTasks().empty()) {
            exit_task_times.push_back(
                std::max(
                    task.getFinishTimeLocal(),        // FTi^l: local execution finish time
                    task.getFinishTimeWirelessReceive() // FTi^wr: cloud execution finish time
                )
            );
        }
    }
    
    // Return the maximum completion time among all exit tasks
    return *std::max_element(exit_task_times.begin(), exit_task_times.end());
}

double  MCCScheduler::calculateEnergyConsumption(const Task& task, const vector<int>& core_powers, double cloud_sending_power) {
    // Calculate energy consumption for a single task vi
    // Based on equations (7) and (8) from Section II.D
    
    if (task.isCoreTask()) {
        // Implement equation (7): Ei,k^l = Pk × Ti,k^l where:
        // - Pk: Power consumption of k-th core (core_powers[k])
        // - Ti,k^l: Execution time of task vi on core k
        // - task.assignment: Selected core k for execution
        return static_cast<double>(core_powers[task.getAssignment()]) * 
               static_cast<double>(task.getCoreExecutionTimes()[task.getAssignment()]);
    } else {
        // Implement equation (8): Ei^c = P^s × Ti^s where:
        // - P^s: Power consumption of RF component for sending
        // - Ti^s: Time required to send task vi to cloud
        return cloud_sending_power * static_cast<double>(task.getCloudExecutionTimes()[0]);
    }
}

double  MCCScheduler::totalEnergy(const vector<Task>& tasks, const vector<int>& core_powers, double cloud_sending_power) {
    return accumulate(
        tasks.begin(),
        tasks.end(),
        0.0,
        [this, &core_powers, cloud_sending_power](double current_sum, const Task& task) {
            return current_sum + calculateEnergyConsumption(task, core_powers, cloud_sending_power);
        }
    );
}

void  MCCScheduler::primaryAssignment() {
    for (Task& task : tasks) {
        int t_l_min = *min_element(
            task.getCoreExecutionTimes().begin(),
            task.getCoreExecutionTimes().end()
        );

        const auto& cloud_times = task.getCloudExecutionTimes();
        int t_re = cloud_times[0] + cloud_times[1] + cloud_times[2];
        task.setIsCoreTask(!(t_re < t_l_min));
    }
}

void  MCCScheduler::taskPrioritizing() {
    vector<int> w(tasks.size(), 0);
    for (size_t i = 0; i < tasks.size(); i++) {
        Task& task = tasks[i];
        if (!task.isCoreTask()) {
            const auto& times = task.getCloudExecutionTimes();
            w[i] = times[0] + times[1] + times[2];
        } else {
            const auto& times = task.getCoreExecutionTimes();
            int sum = accumulate(times.begin(), times.end(), 0);
            w[i] = sum / times.size();
        }
    }
    
    map<int, int> computed_priority_scores;
    
    for (Task& task : tasks) {
        calculatePriority(task, w, computed_priority_scores);
    }
    
    for (Task& task : tasks) {
        task.setPriorityScore(computed_priority_scores[task.getId()]);
    }
}

int  MCCScheduler::calculatePriority(Task& task, vector<int>& w,map<int, int>& computed_priority_scores) {
    if (computed_priority_scores.count(task.getId()) > 0) {
        return computed_priority_scores[task.getId()];
    }

    if (task.getSuccTasks().empty()) {
        int priority = w[task.getId() - 1];
        computed_priority_scores[task.getId()] = priority;
        return priority;
    }
    
    int max_successor_priority = -1;
    for (Task* successor : task.getSuccTasks()) {
        int successor_priority = calculatePriority(*successor, w, computed_priority_scores);
        max_successor_priority = max(max_successor_priority, successor_priority);
    }
    
    int task_priority = w[task.getId() - 1] + max_successor_priority;
    computed_priority_scores[task.getId()] = task_priority;
    return task_priority;
}

std::vector<Task> MCCScheduler::KernalAlgorithm(std::vector<Task>& tasks, std::vector<std::vector<int>>& sequences) {
    TaskMigrationScheduler scheduler(tasks, sequences);
    std::deque<Task*> queue = scheduler.initialize_queue();
    
    while (!queue.empty()) {
        Task* current_task = queue.front();
        queue.pop_front();
        
        current_task->setSchedulingState(SchedulingState::KERNEL_SCHEDULED);
        
        if (current_task->isCoreTask()) {
            scheduler.schedule_local_task(*current_task);
        } else {
            scheduler.schedule_cloud_task(*current_task);
        }
        
        for (Task& task : tasks) {
            scheduler.update_task_state(task);
            
            if (scheduler.dependency_ready[task.getId() - 1] == 0 &&
                scheduler.sequence_ready[task.getId() - 1] == 0 &&
                task.getSchedulingState() != SchedulingState::KERNEL_SCHEDULED) {
                
                auto it = std::find_if(queue.begin(), queue.end(),
                    [&task](const Task* t) { return t->getId() == task.getId(); });
                
                if (it == queue.end()) {
                    queue.push_back(&task);
                }
            }
        }
    }
    
    for (Task& task : tasks) {
        task.setSchedulingState(SchedulingState::UNSCHEDULED);
    }
    
    return tasks;
}

vector<vector<int>> MCCScheduler::constructSequence(vector<Task>& tasks, int task_id, int execution_unit, vector<vector<int>> original_sequence){
    unordered_map<int, Task*> task_id_to_task;
    for (Task& task : tasks) {
        task_id_to_task[task.getId()] = &task;
    }

    Task* target_task = task_id_to_task[task_id];
    if (!target_task) {
        throw runtime_error("Target task not found");
    }

    int target_task_rt;
    if (target_task->isCoreTask()) {
        target_task_rt = target_task->getReadyTimeLocal();
    } else {
        target_task_rt = target_task->getReadyTimeWirelessSend();
    }

    int original_assignment = target_task->getAssignment();
    auto& original_task_sequence = original_sequence[original_assignment];
    original_task_sequence.erase(
        remove(original_task_sequence.begin(), 
               original_task_sequence.end(), 
               target_task->getId()),
        original_task_sequence.end()
    );

    auto& new_sequence_task_list = original_sequence[execution_unit];
    vector<int> start_times;
    start_times.reserve(new_sequence_task_list.size());

    for (int task_id : new_sequence_task_list) {
        Task* task = task_id_to_task[task_id];
        start_times.push_back(
            task->getExecutionUnitTaskStartTime(execution_unit)
        );
    }

    auto insertion_point = lower_bound(
        start_times.begin(), 
        start_times.end(), 
        target_task_rt
    );

    size_t insertion_index = distance(start_times.begin(), insertion_point);

    new_sequence_task_list.insert(
        new_sequence_task_list.begin() + insertion_index,
        target_task->getId()
    );

    target_task->setAssignment(execution_unit);
    target_task->setIsCoreTask(execution_unit != 3);

    return original_sequence;
}

std::tuple<int, int, std::vector<int>> MCCScheduler::GenerateCacheKey(const std::vector<Task>& tasks, int task_idx, int target_execution_unit) {

    std::vector<int> task_assignments;
    task_assignments.reserve(tasks.size());
    for (const Task& task : tasks) {
        task_assignments.push_back(task.getAssignment());
    }

    return std::make_tuple(task_idx, target_execution_unit, task_assignments);
}

std::tuple<int, double> MCCScheduler::evaluateMigration(
    const std::vector<Task>& tasks,
    const std::vector<std::vector<int>>& seqs,
    int task_idx,
    int target_execution_unit,
    std::map<std::tuple<int, int, std::vector<int>>, std::tuple<int, double>>& migration_cache,
    const std::vector<int>& core_powers,
    double cloud_sending_power) {
    // Generate cache key for this migration scenario
    auto cache_key = GenerateCacheKey(tasks, task_idx, target_execution_unit);
                
    // Check cache for previously evaluated scenario
    auto cache_it = migration_cache.find(cache_key);
    if (cache_it != migration_cache.end()) {
        return cache_it->second;
    }

    // Create copies to avoid modifying original state
    std::vector<std::vector<int>> sequence_copy;
    for (const auto& seq : seqs) {
        sequence_copy.push_back(seq);
    }
    std::vector<Task> tasks_copy = tasks;

    // Apply migration and recalculate schedule
    sequence_copy = constructSequence(
        tasks_copy, 
        task_idx + 1,  // Convert to 1-based task ID
        target_execution_unit, 
        sequence_copy
    );
    
    KernalAlgorithm(tasks_copy, sequence_copy);

    // Calculate new metrics
    int migration_T = totalTime(tasks_copy);
    double migration_E = totalEnergy(tasks_copy, core_powers, cloud_sending_power);

    // Cache results
    auto result = std::make_tuple(migration_T, migration_E);
    migration_cache[cache_key] = result;
    return result;
}

std::vector<std::vector<bool>> MCCScheduler::initializeMigrationChoices(const std::vector<Task>& tasks) {
    std::vector<std::vector<bool>> migration_choices(
        tasks.size(),              // Number of rows (tasks)
        std::vector<bool>(4, false) // Each row has 4 columns, initialized to false
    );
    
    // Iterate through each task to set valid migration targets
    for (size_t i = 0; i < tasks.size(); i++) {
        const Task& task = tasks[i];
        
        if (task.getAssignment() == 3) {
            for (int j = 0; j < 4; j++) {
                migration_choices[i][j] = true;
            }
        } else {
            migration_choices[i][task.getAssignment()] = true;
        }
    }
    
    return migration_choices;
}

TaskMigrationState* MCCScheduler::identifyOptimalMigration(
    const std::vector<std::tuple<int, int, int, double>>& migration_trials_results,
    int T_final,
    double E_total,
    int T_max
) {
    using MigrationTuple = std::tuple<double, int, int, int, double>;
    
    double best_energy_reduction = 0;
    std::optional<MigrationTuple> best_migration;

    for (const auto& trial : migration_trials_results) {
        const auto& [task_idx, resource_idx, time, energy] = trial;
        
        if (time > T_max) {
            continue;
        }
        
        double energy_reduction = E_total - energy;
    
        if (time <= T_final && energy_reduction > 0) {
            if (energy_reduction > best_energy_reduction) {
                best_energy_reduction = energy_reduction;
                // Construct the correct tuple type
                best_migration = std::make_tuple(
                    energy_reduction,  // Store energy_reduction as first element
                    task_idx,
                    resource_idx,
                    time,
                    energy
                );
            }
        }
    }

    if (best_migration) {
        // Properly decompose the 5-element tuple
        const auto& [efficiency, task_idx, resource_idx, time, energy] = *best_migration;
        return new TaskMigrationState(
            time,
            energy,
            efficiency,
            task_idx + 1,
            resource_idx + 1
        );
    }

    // Second phase: Using std::priority_queue with built-in tuple comparison
    // The < operator for tuples compares elements lexicographically
    std::priority_queue<MigrationTuple> migration_candidates;

    for (const auto& trial : migration_trials_results) {
        const auto& [task_idx, resource_idx, time, energy] = trial;
        
        if (time > T_max) {
            continue;
        }
        
        double energy_reduction = E_total - energy;
        if (energy_reduction > 0) {
            int time_increase = std::max(0, time - T_final);
            double efficiency;
            
            if (time_increase == 0) {
                efficiency = std::numeric_limits<double>::infinity();
            } else {
                efficiency = energy_reduction / time_increase;
            }
        
            migration_candidates.push(std::make_tuple(
                -efficiency,  // Negative for correct ordering
                task_idx,
                resource_idx,
                time,
                energy
            ));
        }
    }

    if (migration_candidates.empty()) {
        return nullptr;
    }
    
    const auto& [neg_ratio, n_best, k_best, T_best, E_best] = migration_candidates.top();
    return new TaskMigrationState(
        T_best,
        E_best,
        -neg_ratio,
        n_best + 1,
        k_best + 1
    );
}

std::pair<std::vector<Task>, std::vector<std::vector<int>>> MCCScheduler::optimizeTaskScheduling(
    std::vector<Task>& tasks,
    std::vector<std::vector<int>> sequence,
    int T_final,
    const std::vector<int>& core_powers,
    double cloud_sending_power
) {
    // Create a cache for storing migration evaluations
    // Using std::map as it automatically handles tuple keys
    std::map<std::tuple<int, int, std::vector<int>>, std::tuple<int, double>> migration_cache;

    // Calculate initial energy consumption E_total (equation 9)
    double current_iteration_energy = totalEnergy(tasks, core_powers, cloud_sending_power);

    // Iterative improvement loop
    // We continue until we can't reduce energy consumption further
    bool energy_improved = true;
    while (energy_improved) {
        // Store current energy for comparison
        double previous_iteration_energy = current_iteration_energy;

        // Get current schedule metrics
        int current_time = totalTime(tasks);  // T_total (equation 10)
        int T_max = static_cast<int>(T_final * 1.5);  // Allow scheduling flexibility

        // Initialize N×K migration possibilities matrix
        auto migration_choices = initializeMigrationChoices(tasks);

        // Evaluate all valid migration options
        std::vector<std::tuple<int, int, int, double>> migration_trials_results;
        for (size_t task_idx = 0; task_idx < tasks.size(); task_idx++) {
            for (int possible_execution_unit = 0; possible_execution_unit < 4; possible_execution_unit++) {
                if (migration_choices[task_idx][possible_execution_unit]) {
                    continue;
                }

                // Calculate T_total and E_total after migration
                auto [migration_trial_time, migration_trial_energy] = evaluateMigration(
                    tasks,
                    sequence,
                    task_idx,
                    possible_execution_unit,
                    migration_cache,
                    core_powers,
                    cloud_sending_power
                );

                migration_trials_results.emplace_back(
                    task_idx,
                    possible_execution_unit,
                    migration_trial_time,
                    migration_trial_energy
                );
            }
        }

        // Select best migration using two-step criteria
        TaskMigrationState* best_migration = identifyOptimalMigration(
            migration_trials_results,
            current_time,
            previous_iteration_energy,
            T_max
        );

        // Exit if no valid migrations remain
        if (best_migration == nullptr) {
            energy_improved = false;
            delete best_migration;
            break;
        }

        // Apply selected migration:
        // 1. Construct new sequences
        sequence = constructSequence(
            tasks,
            best_migration->task_index,
            best_migration->target_execution_unit - 1,
            sequence
        );

        // 2. Apply kernel algorithm for O(N) rescheduling
        KernalAlgorithm(tasks, sequence);

        // Calculate new energy consumption
        current_iteration_energy = totalEnergy(tasks, core_powers, cloud_sending_power);
        energy_improved = current_iteration_energy < previous_iteration_energy;

        // Clean up allocated memory
        delete best_migration;

        // Manage cache size for memory efficiency
        if (migration_cache.size() > 1000) {
            migration_cache.clear();
        }
    }

    return {tasks, sequence};
}
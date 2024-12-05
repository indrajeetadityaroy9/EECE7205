#include <iostream>
#include <map>
#include <array>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <deque>
#include <vector>
#include <map>
#include <tuple>
#include <algorithm> // for std::copy
#include <utility>   // for std::pair
#include <queue>
#include <cmath>
#include <limits>
#include <iomanip>
#include <string>

// Equivalent of Python's enum for SchedulingState
enum class SchedulingState {
    UNSCHEDULED = 0,    // Task not yet processed
    SCHEDULED = 1,      // After initial scheduling (Step 1: minimal-delay scheduling)
    KERNEL_SCHEDULED = 2 // After task migration (Step 2: energy optimization)
};

// Global dictionary storing execution times for tasks on different cores
// This implements T_i^l_k from Section II.B of the referenced paper.
// Key: task ID (1-20)
// Value: Array of execution times [core1_time, core2_time, core3_time]
static const std::map<int, std::array<int,3>> core_execution_times = {
    {1, {9, 7, 5}},
    {2, {8, 6, 5}},
    {3, {6, 5, 4}},
    {4, {7, 5, 3}},
    {5, {5, 4, 2}},
    {6, {7, 6, 4}},
    {7, {8, 5, 3}},
    {8, {6, 4, 2}},
    {9, {5, 3, 2}},
    {10,{7, 4, 2}},
    {11,{12,3, 3}},
    {12,{12,8, 4}},
    {13,{11,3, 2}},
    {14,{12,11,4}},
    {15,{13,4, 2}},
    {16,{9, 7, 3}},
    {17,{9, 3, 3}},
    {18,{13,9, 2}},
    {19,{10,5, 3}},
    {20,{12,5, 4}}
};

// Cloud execution parameters from Section II.B of the paper:
// [T_send (T_i^s), T_cloud (T_i^c), T_receive (T_i^r)]
static const std::array<int,3> cloud_execution_times = {3, 1, 1};

// Equivalent of Python's dataclass for TaskMigrationState
struct TaskMigrationState {
    int time;         
    double energy;      
    double efficiency; 
    int task_index;    
    int target_execution_unit; 
};

class Task {
public:
    int id;
    std::vector<int> pred_tasks;
    std::vector<int> succ_task;

    std::array<int,3> core_execution_times;  
    std::array<int,3> cloud_execution_times; 

    int FT_l;   
    int FT_ws;  
    int FT_c;   
    int FT_wr;  

    int RT_l;   
    int RT_ws;  
    int RT_c;   
    int RT_wr;  

    double priority_score;

    int assignment;   
    bool is_core_task;

    std::vector<int> execution_unit_task_start_times;
    int execution_finish_time;

    SchedulingState is_scheduled;

    Task(int task_id,
         const std::map<int, std::array<int,3>>& core_times_map,
         const std::array<int,3>& cloud_times)
        : id(task_id),
          FT_l(0), FT_ws(0), FT_c(0), FT_wr(0),
          RT_l(-1), RT_ws(-1), RT_c(-1), RT_wr(-1),
          priority_score(-1.0),
          assignment(-2),
          is_core_task(false),
          execution_finish_time(-1),
          is_scheduled(SchedulingState::UNSCHEDULED)
    {
        auto it = core_times_map.find(id);
        if (it != core_times_map.end()) {
            core_execution_times = it->second;
        } else {
            core_execution_times = {0,0,0};
        }
        cloud_execution_times = cloud_times;
    }
};

int total_time(const std::vector<Task>& tasks) {
    int max_completion_time = 0;
    for (const auto& task : tasks) {
        if (task.succ_task.empty()) {
            int completion = std::max(task.FT_l, task.FT_wr);
            if (completion > max_completion_time) {
                max_completion_time = completion;
            }
        }
    }
    return max_completion_time;
}

double calculate_energy_consumption(const Task& task, const std::vector<int>& core_powers, double cloud_sending_power) {
    if (task.is_core_task) {
        int core_index = task.assignment;
        return static_cast<double>(core_powers[core_index]) * static_cast<double>(task.core_execution_times[core_index]);
    } else {
        return static_cast<double>(cloud_sending_power) * static_cast<double>(task.cloud_execution_times[0]);
    }
}

double total_energy(const std::vector<Task>& tasks, const std::vector<int>& core_powers, double cloud_sending_power) {
    double total = 0.0;
    for (const auto& task : tasks) {
        total += calculate_energy_consumption(task, core_powers, cloud_sending_power);
    }
    return total;
}

void primary_assignment(std::vector<Task>& tasks, int k) {
    for (auto& task : tasks) {
        // T_i^l_min = min( T_i,k^l ) over all k cores
        int t_l_min = *std::min_element(task.core_execution_times.begin(), task.core_execution_times.end());

        // T_i^re = T_i^s + T_i^c + T_i^r
        int t_re = task.cloud_execution_times[0] + // T_i^s (send)
                    task.cloud_execution_times[1] + // T_i^c (cloud)
                    task.cloud_execution_times[2];   // T_i^r (receive)

        // If T_i^re < T_i^l_min, offload to cloud
        if (t_re < t_l_min) {
            task.is_core_task = false;
            task.assignment = k; // k represents the cloud
        } else {
            // Otherwise, assign locally
            task.is_core_task = true;
            // Assignment to a specific core is done later
        }
    }
}

double calculate_priority(const Task& task,
                          const std::vector<Task>& tasks,
                          const std::vector<double>& w,
                          std::map<int,double>& computed_priority_scores);

void task_prioritizing(std::vector<Task>& tasks) {
    // w[i] = computation cost of task i
    // i corresponds to tasks[i], but task IDs are 1-based
    std::vector<double> w(tasks.size(), 0.0);

    // Step 1: Calculate computation costs (w_i) for each task
    for (size_t i = 0; i < tasks.size(); i++) {
        const Task& task = tasks[i];
        if (!task.is_core_task) {
            // Cloud task: w_i = T_i^re = T_i^s + T_i^c + T_i^r
            w[i] = static_cast<double>(task.cloud_execution_times[0] + 
                                       task.cloud_execution_times[1] +
                                       task.cloud_execution_times[2]);
        } else {
            // Local task: w_i = average local execution time across all cores
            double sum_local = static_cast<double>(
                std::accumulate(task.core_execution_times.begin(),
                                 task.core_execution_times.end(), 0));
            w[i] = sum_local / static_cast<double>(task.core_execution_times.size());
        }
    }

    // Cache for memoization of priority calculations
    std::map<int,double> computed_priority_scores;

    // Compute priorities for all tasks
    for (auto& task : tasks) {
        calculate_priority(task, tasks, w, computed_priority_scores);
    }

    // Update priority scores in the task objects
    for (auto& task : tasks) {
        task.priority_score = computed_priority_scores[task.id];
    }
}

double calculate_priority(const Task& task,
                          const std::vector<Task>& tasks,
                          const std::vector<double>& w,
                          std::map<int,double>& computed_priority_scores) {
    // Check if we have already computed this task's priority
    auto it = computed_priority_scores.find(task.id);
    if (it != computed_priority_scores.end()) {
        return it->second;
    }

    // Base case: Exit tasks
    // priority(vi) = w_i if vi is an exit task
    if (task.succ_task.empty()) {
        double priority_val = w[task.id - 1];
        computed_priority_scores[task.id] = priority_val;
        return priority_val;
    }

    // Recursive case: Non-exit tasks
    // priority(vi) = w_i + max(vj in succ(vi)) priority(vj)
    double max_successor_priority = -std::numeric_limits<double>::infinity();
    for (int succ_id : task.succ_task) {
        // succ_id is the ID of the successor task
        const Task& succ_task = tasks[succ_id - 1];
        double succ_priority = calculate_priority(succ_task, tasks, w, computed_priority_scores);
        if (succ_priority > max_successor_priority) {
            max_successor_priority = succ_priority;
        }
    }

    double task_priority = w[task.id - 1] + max_successor_priority;
    computed_priority_scores[task.id] = task_priority;
    return task_priority;
}

class InitialTaskScheduler {
public:
    InitialTaskScheduler(std::vector<Task>& tasks, int num_cores=3)
        : tasks(tasks), k(num_cores), ws_ready(0), wr_ready(0)
    {
        core_earliest_ready.resize(k, 0);
        sequences.resize(k+1); // k local cores + 1 cloud channel
    }

    std::vector<int> get_priority_ordered_tasks() {
        // Exactly same logic as Python:
        // Sort tasks by priority_score descending
        std::vector<std::pair<double,int>> task_priority_list;
        for (auto &t : tasks) {
            task_priority_list.emplace_back(t.priority_score, t.id);
        }
        std::sort(task_priority_list.begin(), task_priority_list.end(),
                  [](const auto &a, const auto &b){return a.first > b.first;});

        std::vector<int> result;
        for (auto &item : task_priority_list) {
            result.push_back(item.second);
        }
        return result;
    }

    std::pair<std::vector<Task*>, std::vector<Task*>> classify_entry_tasks(const std::vector<int>& priority_order) {
        // Same logic as Python
        std::vector<Task*> entry_tasks;
        std::vector<Task*> non_entry_tasks;

        for (int task_id : priority_order) {
            Task &task = tasks[task_id - 1];
            if (task.pred_tasks.empty()) {
                entry_tasks.push_back(&task);
            } else {
                non_entry_tasks.push_back(&task);
            }
        }
        return {entry_tasks, non_entry_tasks};
    }

    struct CoreChoice {
        int core;
        int start_time;
        int finish_time;
    };

    CoreChoice identify_optimal_local_core(Task &task, int ready_time=0) {
        // Exactly same logic: choose core that minimizes finish time
        int best_finish_time = std::numeric_limits<int>::max();
        int best_core = -1;
        int best_start_time = std::numeric_limits<int>::max();

        for (int core = 0; core < k; core++) {
            int start_time = std::max(ready_time, core_earliest_ready[core]);
            int finish_time = start_time + task.core_execution_times[core];

            if (finish_time < best_finish_time) {
                best_finish_time = finish_time;
                best_core = core;
                best_start_time = start_time;
            }
        }

        return {best_core, best_start_time, best_finish_time};
    }

    void schedule_on_local_core(Task &task, int core, int start_time, int finish_time) {
        // Exactly as Python:
        task.FT_l = finish_time;
        task.execution_finish_time = finish_time;

        task.execution_unit_task_start_times.assign(k+1, -1);
        task.execution_unit_task_start_times[core] = start_time;

        core_earliest_ready[core] = finish_time;

        // In Python: "task.assignment = core"
        task.assignment = core;

        task.is_scheduled = SchedulingState::SCHEDULED;

        sequences[core].push_back(task.id);
    }

    struct CloudTiming {
        int send_ready;
        int send_finish;
        int cloud_ready;
        int cloud_finish;
        int receive_ready;
        int receive_finish;
    };

    CloudTiming calculate_cloud_phases_timing(Task &task) {
        // Same logic as Python:
        int send_ready = task.RT_ws;
        int send_finish = send_ready + task.cloud_execution_times[0];

        int cloud_ready = send_finish;
        int cloud_finish = cloud_ready + task.cloud_execution_times[1];

        int receive_ready = cloud_finish;
        int receive_finish = std::max(wr_ready, receive_ready) + task.cloud_execution_times[2];

        return {send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish};
    }

    void schedule_on_cloud(Task &task,
                           int send_ready, int send_finish,
                           int cloud_ready, int cloud_finish,
                           int receive_ready, int receive_finish) {
        // Exactly as Python code:
        task.RT_ws = send_ready;
        task.FT_ws = send_finish;

        task.RT_c = cloud_ready;
        task.FT_c = cloud_finish;

        task.RT_wr = receive_ready;
        task.FT_wr = receive_finish;

        task.execution_finish_time = receive_finish;
        task.FT_l = 0;

        task.execution_unit_task_start_times.assign(k+1, -1);
        task.execution_unit_task_start_times[k] = send_ready;  // cloud start time

        // Python: "task.assignment = self.k" for cloud
        task.assignment = k;

        task.is_scheduled = SchedulingState::SCHEDULED;

        ws_ready = send_finish;
        wr_ready = receive_finish;

        sequences[k].push_back(task.id);
    }

    void schedule_entry_tasks(std::vector<Task*> &entry_tasks) {
        // Same two-phase logic as Python:
        // First local tasks, then cloud tasks
        std::vector<Task*> cloud_entry_tasks;

        for (auto* task : entry_tasks) {
            if (task->is_core_task) {
                auto choice = identify_optimal_local_core(*task);
                schedule_on_local_core(*task, choice.core, choice.start_time, choice.finish_time);
            } else {
                cloud_entry_tasks.push_back(task);
            }
        }

        for (auto* task : cloud_entry_tasks) {
            task->RT_ws = ws_ready;
            auto timing = calculate_cloud_phases_timing(*task);
            schedule_on_cloud(*task, timing.send_ready, timing.send_finish,
                              timing.cloud_ready, timing.cloud_finish,
                              timing.receive_ready, timing.receive_finish);
        }
    }

    void calculate_non_entry_task_ready_times(Task &task) {
        // Matches Python code EXACTLY:
        // RTi^l = max( max(FTj^l, FTj^wr) for j in pred(vi) ), and at least 0
        int max_pred_finish_l_wr = 0;
        for (int pred_id : task.pred_tasks) {
            Task &pred = tasks[pred_id - 1];
            int val = std::max(pred.FT_l, pred.FT_wr);
            if (val > max_pred_finish_l_wr) {
                max_pred_finish_l_wr = val;
            }
        }
        task.RT_l = std::max(max_pred_finish_l_wr, 0);

        // RTi^ws = max( max(FTj^l, FTj^ws) for j in pred(vi), ws_ready )
        int max_pred_finish_l_ws = 0;
        for (int pred_id : task.pred_tasks) {
            Task &pred = tasks[pred_id - 1];
            int val = std::max(pred.FT_l, pred.FT_ws);
            if (val > max_pred_finish_l_ws) {
                max_pred_finish_l_ws = val;
            }
        }
        task.RT_ws = std::max(max_pred_finish_l_ws, ws_ready);
    }

    void schedule_non_entry_tasks(std::vector<Task*> &non_entry_tasks) {
        // Process in priority order, as given
        for (auto* task : non_entry_tasks) {
            calculate_non_entry_task_ready_times(*task);

            if (!task->is_core_task) {
                // Cloud task
                auto timing = calculate_cloud_phases_timing(*task);
                schedule_on_cloud(*task, timing.send_ready, timing.send_finish,
                                  timing.cloud_ready, timing.cloud_finish,
                                  timing.receive_ready, timing.receive_finish);
            } else {
                // Local task: compare local and cloud finish times
                auto local_choice = identify_optimal_local_core(*task, task->RT_l);
                auto timing = calculate_cloud_phases_timing(*task);
                int cloud_finish_time = timing.receive_finish;

                // Choose whichever finishes earlier
                if (local_choice.finish_time <= cloud_finish_time) {
                    schedule_on_local_core(*task, local_choice.core, local_choice.start_time, local_choice.finish_time);
                } else {
                    task->is_core_task = false;
                    schedule_on_cloud(*task, timing.send_ready, timing.send_finish,
                                      timing.cloud_ready, timing.cloud_finish,
                                      timing.receive_ready, timing.receive_finish);
                }
            }
        }
    }

    std::vector<std::vector<int>> get_sequences() const {
        return sequences; // Assuming sequences is a member of InitialTaskScheduler
    }

    std::vector<Task> &tasks;
    int k; // number of cores

    std::vector<int> core_earliest_ready;
    int ws_ready;
    int wr_ready;

    std::vector<std::vector<int>> sequences;
};

std::vector<std::vector<int>> execution_unit_selection(std::vector<Task>& tasks) {
    // Initialize scheduler with tasks and K=3 cores
    InitialTaskScheduler scheduler(tasks, 3);

    // Order tasks by priority score
    std::vector<int> priority_orderered_tasks = scheduler.get_priority_ordered_tasks();

    // Classify tasks into entry and non-entry based on dependencies
    auto [entry_tasks, non_entry_tasks] = scheduler.classify_entry_tasks(priority_orderered_tasks);

    // Schedule entry tasks:
    // 1. Local tasks first
    // 2. Then cloud tasks (as per the Python logic)
    scheduler.schedule_entry_tasks(entry_tasks);

    // Schedule non-entry tasks:
    // - Calculate ready times
    // - Compare local vs. cloud execution
    // - Choose the path that minimizes finish time
    scheduler.schedule_non_entry_tasks(non_entry_tasks);

    // Return sequences Sk for each execution unit
    return scheduler.get_sequences();
}

std::vector<std::vector<int>> construct_sequence(
    std::vector<Task>& tasks,
    int task_id,
    int execution_unit,
    std::vector<std::vector<int>> original_sequence
) {
    // Step 1: O(1) access to tasks is already given by vector indexing (task_id - 1)
    // No need for a separate map/dictionary since we have direct indexing.
    
    // Step 2: Get the target task v_tar for migration
    Task &target_task = tasks[task_id - 1];

    // Step 3: Get ready time for insertion
    // RTi^l if core task, else RTi^ws for cloud task
    int target_task_rt = target_task.is_core_task ? target_task.RT_l : target_task.RT_ws;

    // Step 4: Remove task from original sequence
    int original_assignment = target_task.assignment;
    auto &old_seq = original_sequence[original_assignment];
    // Remove the task_id from old_seq
    old_seq.erase(std::remove(old_seq.begin(), old_seq.end(), target_task.id), old_seq.end());

    // Step 5: Get sequence for new execution unit
    auto &new_seq = original_sequence[execution_unit];

    // Get start times for tasks in new sequence
    std::vector<int> start_times;
    start_times.reserve(new_seq.size());
    for (int tid : new_seq) {
        // tid is the task ID in the new sequence
        Task &t = tasks[tid - 1];
        // execution_unit_task_start_times stores start times per unit
        start_times.push_back(t.execution_unit_task_start_times[execution_unit]);
    }

    // Step 6: Find insertion point using binary search (like bisect_left)
    // Insert target_task_rt into start_times to keep it sorted
    auto it = std::lower_bound(start_times.begin(), start_times.end(), target_task_rt);
    int insertion_index = static_cast<int>(std::distance(start_times.begin(), it));

    // Step 7: Insert task at the correct position in new_seq
    new_seq.insert(new_seq.begin() + insertion_index, target_task.id);

    // Step 8: Update task execution information
    target_task.assignment = execution_unit;
    // is_core_task = True if not cloud (execution_unit != 3), False if cloud
    target_task.is_core_task = (execution_unit != 3);

    return original_sequence;
}

class KernelScheduler {
public:
    KernelScheduler(std::vector<Task>& tasks, std::vector<std::vector<int>>& sequences)
        : tasks(tasks), sequences(sequences)
    {
        // As per the Python logic:
        RT_ls = {0,0,0};  // Local cores ready times
        cloud_phases_ready_times = {0,0,0}; // RTi^ws, RTi^c, RTi^wr

        std::tie(dependency_ready, sequence_ready) = initialize_task_state();
    }

    std::pair<std::vector<int>, std::vector<int>> initialize_task_state() {
        // ready1 = dependency_ready
        // Count how many predecessors each task has initially
        std::vector<int> dependency_ready(tasks.size(), 0);
        for (size_t i = 0; i < tasks.size(); i++) {
            dependency_ready[i] = static_cast<int>(tasks[i].pred_tasks.size());
        }

        // ready2 = sequence_ready
        // -1: not in sequence
        //  0: ready to execute
        //  1: waiting for predecessor in sequence
        std::vector<int> sequence_ready(tasks.size(), -1);

        // Mark first task in each sequence as ready2=0
        for (auto &seq : sequences) {
            if (!seq.empty()) {
                sequence_ready[seq[0] - 1] = 0;
            }
        }

        return {dependency_ready, sequence_ready};
    }

    void update_task_state(Task &task) {
        // Only update if not KERNEL_SCHEDULED
        if (task.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {
            // Update dependency_ready (ready1): count unscheduled preds
            int unsched_preds = 0;
            for (int pred_id : task.pred_tasks) {
                Task &pred = tasks[pred_id - 1];
                if (pred.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {
                    unsched_preds++;
                }
            }
            dependency_ready[task.id - 1] = unsched_preds;

            // Update sequence_ready (ready2)
            // Find which sequence this task is in
            for (auto &seq : sequences) {
                auto it = std::find(seq.begin(), seq.end(), task.id);
                if (it != seq.end()) {
                    int idx = static_cast<int>(std::distance(seq.begin(), it));
                    if (idx > 0) {
                        // Has a predecessor in sequence
                        int prev_task_id = seq[idx - 1];
                        Task &prev_task = tasks[prev_task_id - 1];
                        sequence_ready[task.id - 1] = 
                            (prev_task.is_scheduled != SchedulingState::KERNEL_SCHEDULED) ? 1 : 0;
                    } else {
                        // First in sequence => ready2=0
                        sequence_ready[task.id - 1] = 0;
                    }
                    break;
                }
            }
        }
    }

    void schedule_local_task(Task &task) {
        // Compute RT_l from preds (eq. 3)
        if (task.pred_tasks.empty()) {
            task.RT_l = 0;
        } else {
            int max_finish = 0;
            for (int pred_id : task.pred_tasks) {
                Task &pred = tasks[pred_id - 1];
                int val = std::max(pred.FT_l, pred.FT_wr);
                if (val > max_finish) {
                    max_finish = val;
                }
            }
            task.RT_l = max_finish;
        }

        int core_index = task.assignment;
        task.execution_unit_task_start_times.assign(4, -1);

        int start_time = std::max(RT_ls[core_index], task.RT_l);
        task.execution_unit_task_start_times[core_index] = start_time;

        task.FT_l = start_time + task.core_execution_times[core_index];

        // Update core availability
        RT_ls[core_index] = task.FT_l;

        // Local task: no cloud phases
        task.FT_ws = -1;
        task.FT_c = -1;
        task.FT_wr = -1;
    }

    void schedule_cloud_task(Task &task) {
        // Compute RT_ws
        if (task.pred_tasks.empty()) {
            task.RT_ws = 0;
        } else {
            int max_finish = 0;
            for (int pred_id : task.pred_tasks) {
                Task &pred = tasks[pred_id - 1];
                int val = std::max(pred.FT_l, pred.FT_ws);
                if (val > max_finish) {
                    max_finish = val;
                }
            }
            task.RT_ws = max_finish;
        }

        task.execution_unit_task_start_times.assign(4, -1);

        // Cloud is at index 3
        int send_start = std::max(cloud_phases_ready_times[0], task.RT_ws);
        task.execution_unit_task_start_times[3] = send_start;

        // Phase 1: Sending
        task.FT_ws = send_start + task.cloud_execution_times[0];
        cloud_phases_ready_times[0] = task.FT_ws;

        // Phase 2: Cloud computing (eq. 5)
        int max_pred_c = 0;
        for (int pred_id : task.pred_tasks) {
            Task &pred = tasks[pred_id - 1];
            if (pred.FT_c > max_pred_c) {
                max_pred_c = pred.FT_c;
            }
        }
        task.RT_c = std::max(task.FT_ws, max_pred_c);
        task.FT_c = std::max(cloud_phases_ready_times[1], task.RT_c) + task.cloud_execution_times[1];
        cloud_phases_ready_times[1] = task.FT_c;

        // Phase 3: Receiving
        task.RT_wr = task.FT_c;
        task.FT_wr = std::max(cloud_phases_ready_times[2], task.RT_wr) + task.cloud_execution_times[2];
        cloud_phases_ready_times[2] = task.FT_wr;

        // Cloud task: no local finish time
        task.FT_l = -1;
    }

    std::deque<Task*> initialize_queue() {
        std::deque<Task*> dq;
        for (auto &t : tasks) {
            if (sequence_ready[t.id - 1] == 0) {
                // Check if all preds are kernel-scheduled
                bool all_preds_sched = true;
                for (int pred_id : t.pred_tasks) {
                    Task &pred = tasks[pred_id - 1];
                    if (pred.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {
                        all_preds_sched = false;
                        break;
                    }
                }
                if (all_preds_sched) {
                    dq.push_back(&t);
                }
            }
        }
        return dq;
    }

    std::vector<Task> &tasks;
    std::vector<std::vector<int>> &sequences;

    std::array<int,3> RT_ls;
    std::array<int,3> cloud_phases_ready_times;

    std::vector<int> dependency_ready;
    std::vector<int> sequence_ready;
};

std::vector<Task>& kernel_algorithm(std::vector<Task>& tasks, std::vector<std::vector<int>>& sequences) {
    // Initialize kernel scheduler with tasks and sequences
    KernelScheduler scheduler(tasks, sequences);

    // Initialize LIFO stack (in Python code, they use deque and popleft(),
    // which actually behaves like a queue (FIFO). We mirror that here.
    std::deque<Task*> queue = scheduler.initialize_queue();

    // Main scheduling loop
    while (!queue.empty()) {
        // Pop next ready task from front of the queue (mimicking popleft())
        Task* current_task = queue.front();
        queue.pop_front();

        // Mark as KERNEL_SCHEDULED
        current_task->is_scheduled = SchedulingState::KERNEL_SCHEDULED;

        // Schedule based on execution type
        if (current_task->is_core_task) {
            // Schedule local
            scheduler.schedule_local_task(*current_task);
        } else {
            // Schedule cloud
            scheduler.schedule_cloud_task(*current_task);
        }

        // Update state for all tasks
        for (auto &task : tasks) {
            scheduler.update_task_state(task);
        }

        // Add newly ready tasks to the queue
        // Condition: ready1[j] = 0 and ready2[j] = 0, not kernel_scheduled, not already in queue
        for (auto &task : tasks) {
            int idx = task.id - 1;
            if (scheduler.dependency_ready[idx] == 0 &&
                scheduler.sequence_ready[idx] == 0 &&
                task.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {

                // Check if task is not already in the queue
                bool in_queue = false;
                for (auto* tptr : queue) {
                    if (tptr == &task) {
                        in_queue = true;
                        break;
                    }
                }

                if (!in_queue) {
                    queue.push_back(&task);
                }
            }
        }
    }

    // Reset scheduling state for next iteration
    for (auto &task : tasks) {
        task.is_scheduled = SchedulingState::UNSCHEDULED;
    }

    return tasks;
}

struct MigrateKey {
    int task_idx;
    int target_execution_unit;
    std::vector<int> assignments;

    bool operator<(const MigrateKey& other) const {
        if (task_idx != other.task_idx) return task_idx < other.task_idx;
        if (target_execution_unit != other.target_execution_unit) return target_execution_unit < other.target_execution_unit;
        return assignments < other.assignments;
    }
};

// Generate cache key function
MigrateKey generate_cache_key(const std::vector<Task>& tasks, int task_idx, int target_execution_unit) {
    MigrateKey key;
    key.task_idx = task_idx;
    key.target_execution_unit = target_execution_unit;
    key.assignments.reserve(tasks.size());
    for (const auto& t : tasks) {
        key.assignments.push_back(t.assignment);
    }
    return key;
}

// evaluate_migration function
std::pair<int,double> evaluate_migration(
    std::vector<Task>& tasks,
    const std::vector<std::vector<int>>& seqs,
    int task_idx,
    int target_execution_unit,
    std::map<MigrateKey, std::pair<int,double>>& migration_cache, // Changed to int,double
    const std::vector<int>& core_powers = {1,2,4},
    double cloud_sending_power = 0.5
) {
    // Generate cache key
    MigrateKey cache_key = generate_cache_key(tasks, task_idx, target_execution_unit);

    // Check cache
    auto it = migration_cache.find(cache_key);
    if (it != migration_cache.end()) {
        return it->second; // pair<int,double>
    }

    // Copy tasks and sequences to avoid modifying original state
    std::vector<std::vector<int>> sequence_copy = seqs;
    std::vector<Task> tasks_copy = tasks;

    // Apply migration and recalculate schedule
    sequence_copy = construct_sequence(tasks_copy, task_idx + 1, target_execution_unit, sequence_copy);
    kernel_algorithm(tasks_copy, sequence_copy);

    // Calculate new metrics
    int migration_T = total_time(tasks_copy); // total_time now returns int
    double migration_E = total_energy(tasks_copy, core_powers, cloud_sending_power);

    // Cache results
    migration_cache[cache_key] = std::make_pair(migration_T, migration_E);

    return {migration_T, migration_E};
}

std::vector<std::array<bool,4>> initialize_migration_choices(const std::vector<Task>& tasks) {
    // Create Nx4 matrix of booleans, initialized to false
    std::vector<std::array<bool,4>> migration_choices(tasks.size(), std::array<bool,4>{false,false,false,false});

    for (size_t i = 0; i < tasks.size(); i++) {
        const Task& task = tasks[i];
        if (task.assignment == 3) {
            // Cloud-assigned tasks
            // All four columns (3 cores + cloud) can be considered
            // Mark entire row as true
            for (int j = 0; j < 4; j++) {
                migration_choices[i][j] = true;
            }
        } else {
            // Locally-assigned tasks
            // Only the current assignment is marked as true.
            // To exactly match the Python code, we only mark the current assignment:
            migration_choices[i][task.assignment] = true;
        }
    }

    return migration_choices;
}

TaskMigrationState* identify_optimal_migration(
    const std::vector<std::tuple<int,int,int,double>>& migration_trials_results,
    int T_final,
    double E_total,
    int T_max
) {
    double best_energy_reduction = 0.0;
    TaskMigrationState* best_migration_state = nullptr;

    // Step 1: Find migrations that reduce energy without increasing time
    for (auto &res : migration_trials_results) {
        int task_idx, resource_idx, time_int;
        double energy;
        std::tie(task_idx, resource_idx, time_int, energy) = res;

        int time = time_int;

        // Skip if violates T_max
        if (time > T_max) {
            continue;
        }

        double energy_reduction = E_total - energy;
        if (time <= T_final && energy_reduction > 0.0) {
            // Check if this reduces energy more than best found so far
            if (energy_reduction > best_energy_reduction) {
                best_energy_reduction = energy_reduction;
                if (best_migration_state) {
                    delete best_migration_state;
                }
                best_migration_state = new TaskMigrationState{
                    time,
                    energy,
                    best_energy_reduction,
                    task_idx + 1,     // +1 to match Python logic
                    resource_idx + 1  // +1 to match Python logic
                };
            }
        }
    }

    // If we found a direct energy-reducing migration, return it
    if (best_migration_state) {
        return best_migration_state;
    }

    // Step 2: No direct energy reduction found
    // Select based on efficiency ratio ΔE/ΔT

    struct Candidate {
        double neg_efficiency;
        int task_idx;
        int resource_idx;
        int time;
        double energy;

        bool operator<(const Candidate& other) const {
            // We want the smallest neg_efficiency on top which corresponds to the largest efficiency
            return neg_efficiency > other.neg_efficiency;
        }
    };

    std::priority_queue<Candidate> migration_candidates;

    for (auto &res : migration_trials_results) {
        int task_idx, resource_idx, time_int;
        double energy;
        std::tie(task_idx, resource_idx, time_int, energy) = res;

        int time = time_int;
        if (time > T_max) {
            continue;
        }

        double energy_reduction = E_total - energy;
        if (energy_reduction > 0.0) {
            int time_increase = std::max(0, time - T_final);
            double efficiency;
            if (time_increase == 0) {
                efficiency = std::numeric_limits<double>::infinity();
            } else {
                efficiency = energy_reduction / static_cast<double>(time_increase);
            }

            // Push candidate with neg_efficiency
            migration_candidates.push(Candidate{-efficiency, task_idx, resource_idx, time, energy});
        }
    }

    if (migration_candidates.empty()) {
        return nullptr;
    }

    // Get best candidate (largest efficiency)
    Candidate best = migration_candidates.top();
    double efficiency = -best.neg_efficiency;
    return new TaskMigrationState{
        best.time,
        best.energy,
        efficiency,
        best.task_idx + 1,
        best.resource_idx + 1
    };
}

std::pair<std::vector<Task>, std::vector<std::vector<int>>>
optimize_task_scheduling(
    std::vector<Task> tasks,
    std::vector<std::vector<int>> sequence,
    int T_final,
    std::vector<int> core_powers = {1, 2, 4},
    double cloud_sending_power = 0.5
) {
    // Convert core_powers if needed; here we can just use them as is.
    // In Python, numpy was used, in C++ we can use std::vector directly.

    std::map<MigrateKey, std::pair<int,double>> migration_cache;

    double current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power);

    bool energy_improved = true;
    while (energy_improved) {
        double previous_iteration_energy = current_iteration_energy;

        int current_time = total_time(tasks);  
        int T_max = static_cast<int>(std::floor(T_final * 1.5));

        auto migration_choices = initialize_migration_choices(tasks);

        std::vector<std::tuple<int,int,int,double>> migration_trials_results;

        for (size_t task_idx = 0; task_idx < tasks.size(); task_idx++) {
            for (int possible_execution_unit = 0; possible_execution_unit < 4; possible_execution_unit++) {
                // Same logic as Python:
                // If migration_choices[task_idx, possible_execution_unit] == True, skip
                // In C++: migration_choices[task_idx][possible_execution_unit]
                if (migration_choices[task_idx][possible_execution_unit]) {
                    continue;
                }

                // Evaluate migration
                auto [migration_trial_time, migration_trial_energy] = evaluate_migration(
                    tasks, sequence, static_cast<int>(task_idx), possible_execution_unit,
                    migration_cache, core_powers, cloud_sending_power
                );
                migration_trials_results.push_back(
                    std::make_tuple(static_cast<int>(task_idx), possible_execution_unit, migration_trial_time, migration_trial_energy)
                );
            }
        }

        TaskMigrationState* best_migration = identify_optimal_migration(
            migration_trials_results,
            current_time,
            previous_iteration_energy,
            T_max
        );

        if (!best_migration) {
            energy_improved = false;
            break;
        }

        // Apply chosen migration
        sequence = construct_sequence(
            tasks,
            best_migration->task_index,
            best_migration->target_execution_unit - 1,
            sequence
        );

        kernel_algorithm(tasks, sequence);

        // Recalculate energy
        current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power);
        energy_improved = (current_iteration_energy < previous_iteration_energy);

        delete best_migration; // Clean up

        // Manage cache size
        if (migration_cache.size() > 1000) {
            migration_cache.clear();
        }
    }

    return {tasks, sequence};
}

void print_task_schedule(const std::vector<Task>& tasks) {
    // Assignment mapping
    static const char* ASSIGNMENT_MAPPING[] = {
        "Core 1", "Core 2", "Core 3", "Cloud"
    };

    std::cout << "\nTask Scheduling Details:\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& task : tasks) {
        std::cout << "\n";
        // Task ID
        std::cout << std::setw(15) << std::left << "Task ID:" << task.id << "\n";

        // Assignment
        std::string assignment_str;
        if (task.assignment >= 0 && task.assignment <= 3) {
            assignment_str = ASSIGNMENT_MAPPING[task.assignment];
        } else if (task.assignment == -2) {
            assignment_str = "Not Scheduled";
        } else {
            assignment_str = "Unknown";
        }

        std::cout << std::setw(15) << std::left << "Assignment:" << assignment_str << "\n";

        if (task.is_core_task) {
            // Local core execution timing
            int start_time = task.execution_unit_task_start_times[task.assignment];
            double start_double = static_cast<double>(start_time);
            double end_double = static_cast<double>(start_time + task.core_execution_times[task.assignment]);

            std::cout << std::setw(15) << std::left << "Execution Window:" 
                      << std::fixed << std::setprecision(2) << start_double 
                      << " → " << end_double << "\n";
        } else {
            // Cloud execution phases
            int send_start = task.execution_unit_task_start_times[3];
            double send_start_d = static_cast<double>(send_start);
            double send_end_d = static_cast<double>(send_start + task.cloud_execution_times[0]);

            double RT_c_d = static_cast<double>(task.RT_c);
            double cloud_end_d = RT_c_d + task.cloud_execution_times[1];

            double RT_wr_d = static_cast<double>(task.RT_wr);
            double receive_end_d = RT_wr_d + task.cloud_execution_times[2];

            std::cout << std::setw(15) << std::left << "Send Phase:" 
                      << std::fixed << std::setprecision(2) << send_start_d 
                      << " → " << send_end_d << "\n";

            std::cout << std::setw(15) << std::left << "Cloud Phase:" 
                      << std::fixed << std::setprecision(2) << RT_c_d 
                      << " → " << cloud_end_d << "\n";

            std::cout << std::setw(15) << std::left << "Receive Phase:" 
                      << std::fixed << std::setprecision(2) << RT_wr_d 
                      << " → " << receive_end_d << "\n";
        }

        std::cout << std::string(40, '-') << "\n";
    }
}

std::tuple<bool, std::vector<std::string>> check_schedule_constraints(const std::vector<Task>& tasks) {
    std::vector<std::string> violations;

    // Check wireless sending channel
    auto check_sending_channel = [&]() {
        std::vector<Task> cloud_tasks;
        for (const auto& task : tasks) {
            if (!task.is_core_task) {
                cloud_tasks.push_back(task);
            }
        }

        std::sort(cloud_tasks.begin(), cloud_tasks.end(), [](const Task& a, const Task& b) {
            return a.execution_unit_task_start_times[3] < b.execution_unit_task_start_times[3];
        });

        for (size_t i = 0; i < cloud_tasks.size() - 1; ++i) {
            const Task& current = cloud_tasks[i];
            const Task& next_task = cloud_tasks[i + 1];

            if (current.FT_ws > next_task.execution_unit_task_start_times[3]) {
                violations.push_back("Wireless Sending Channel Conflict: Task " + std::to_string(current.id) +
                                     " sending ends at " + std::to_string(current.FT_ws) +
                                     " but Task " + std::to_string(next_task.id) +
                                     " starts at " + std::to_string(next_task.execution_unit_task_start_times[3]));
            }
        }
    };

    // Check cloud computing channel
    auto check_computing_channel = [&]() {
        std::vector<Task> cloud_tasks;
        for (const auto& task : tasks) {
            if (!task.is_core_task) {
                cloud_tasks.push_back(task);
            }
        }

        std::sort(cloud_tasks.begin(), cloud_tasks.end(), [](const Task& a, const Task& b) {
            return a.RT_c < b.RT_c;
        });

        for (size_t i = 0; i < cloud_tasks.size() - 1; ++i) {
            const Task& current = cloud_tasks[i];
            const Task& next_task = cloud_tasks[i + 1];

            if (current.FT_c > next_task.RT_c) {
                violations.push_back("Cloud Computing Conflict: Task " + std::to_string(current.id) +
                                     " computing ends at " + std::to_string(current.FT_c) +
                                     " but Task " + std::to_string(next_task.id) +
                                     " starts at " + std::to_string(next_task.RT_c));
            }
        }
    };

    // Check wireless receiving channel
    auto check_receiving_channel = [&]() {
        std::vector<Task> cloud_tasks;
        for (const auto& task : tasks) {
            if (!task.is_core_task) {
                cloud_tasks.push_back(task);
            }
        }

        std::sort(cloud_tasks.begin(), cloud_tasks.end(), [](const Task& a, const Task& b) {
            return a.RT_wr < b.RT_wr;
        });

        for (size_t i = 0; i < cloud_tasks.size() - 1; ++i) {
            const Task& current = cloud_tasks[i];
            const Task& next_task = cloud_tasks[i + 1];

            if (current.FT_wr > next_task.RT_wr) {
                violations.push_back("Wireless Receiving Channel Conflict: Task " + std::to_string(current.id) +
                                     " receiving ends at " + std::to_string(current.FT_wr) +
                                     " but Task " + std::to_string(next_task.id) +
                                     " starts at " + std::to_string(next_task.RT_wr));
            }
        }
    };

    // Add other checks (pipelined dependencies, core execution, etc.) as needed

    // Execute checks
    check_sending_channel();
    check_computing_channel();
    check_receiving_channel();

    return {violations.empty(), violations};
}

// Function to print validation report
void print_validation_report(const std::vector<Task>& tasks) {
    auto [is_valid, violations] = check_schedule_constraints(tasks);

    std::cout << "\nSchedule Validation Report\n";
    std::cout << std::string(50, '=') << "\n";

    if (is_valid) {
        std::cout << "Schedule is valid with all pipelining constraints satisfied!\n";
    } else {
        std::cout << "Found constraint violations:\n";
        for (const auto& violation : violations) {
            std::cout << violation << "\n";
        }
    }
}

void printFinalSequences(const std::vector<std::vector<int>>& sequences) {
    std::cout << "\nExecution Sequences:\n";
    std::cout << std::string(40, '-') << "\n";
    
    for (std::size_t i = 0; i < sequences.size(); i++) {
        // Label each sequence appropriately
        std::string label = (i < 3) ? "Core " + std::to_string(i + 1) : "Cloud";
        std::cout << std::setw(12) << std::left << label << ": ";
        
        // Print task sequence
        std::cout << "[";
        for (std::size_t j = 0; j < sequences[i].size(); j++) {
            if (j > 0) std::cout << ", ";
            std::cout << sequences[i][j];
        }
        std::cout << "]\n";
    }
}

int main() {
    // Create tasks

static const std::map<int, std::array<int,3>> core_times_map = {
    {1, {9, 7, 5}},
    {2, {8, 6, 5}},
    {3, {6, 5, 4}},
    {4, {7, 5, 3}},
    {5, {5, 4, 2}},
    {6, {7, 6, 4}},
    {7, {8, 5, 3}},
    {8, {6, 4, 2}},
    {9, {5, 3, 2}},
    {10,{7, 4, 2}},
    {11,{12,3, 3}},
    {12,{12,8, 4}},
    {13,{11,3, 2}},
    {14,{12,11,4}},
    {15,{13,4, 2}},
    {16,{9, 7, 3}},
    {17,{9, 3, 3}},
    {18,{13,9, 2}},
    {19,{10,5, 3}},
    {20,{12,5, 4}}
};

// Cloud execution parameters from Section II.B of the paper:
// [T_send (T_i^s), T_cloud (T_i^c), T_receive (T_i^r)]
static const std::array<int,3> cloud_times = {3, 1, 1};

Task task10(10, core_times_map, cloud_times);
Task task9(9, core_times_map, cloud_times);
Task task8(8, core_times_map, cloud_times);
Task task7(7, core_times_map, cloud_times);
Task task6(6, core_times_map, cloud_times);
Task task5(5, core_times_map, cloud_times);
Task task4(4, core_times_map, cloud_times);
Task task3(3, core_times_map, cloud_times);
Task task2(2, core_times_map, cloud_times);
Task task1(1, core_times_map, cloud_times);

// Set successors using task IDs
task9.succ_task.push_back(task10.id);
task8.succ_task.push_back(task10.id);
task7.succ_task.push_back(task10.id);
task6.succ_task.push_back(task8.id);
task5.succ_task.push_back(task9.id);
task4.succ_task.push_back(task8.id);
task4.succ_task.push_back(task9.id);
task3.succ_task.push_back(task7.id);
task2.succ_task.push_back(task8.id);
task2.succ_task.push_back(task9.id);
task1.succ_task.push_back(task2.id);
task1.succ_task.push_back(task3.id);
task1.succ_task.push_back(task4.id);
task1.succ_task.push_back(task5.id);
task1.succ_task.push_back(task6.id);

// Set predecessors using task IDs
task10.pred_tasks = {task7.id, task8.id, task9.id};
task9.pred_tasks = {task2.id, task4.id, task5.id};
task8.pred_tasks = {task2.id, task4.id, task6.id};
task7.pred_tasks = {task3.id};
task6.pred_tasks = {task1.id};
task5.pred_tasks = {task1.id};
task4.pred_tasks = {task1.id};
task3.pred_tasks = {task1.id};
task2.pred_tasks = {task1.id};
task1.pred_tasks = {};

    std::vector<Task> tasks = {task1, task2, task3, task4, task5, task6, task7, task8, task9, task10};

    primary_assignment(tasks, 3);
    task_prioritizing(tasks);
    auto sequence = execution_unit_selection(tasks);

    int T_final = total_time(tasks);
    double E_total = total_energy(tasks, {1,2,4}, 0.5);
    std::cout << "INITIAL SCHEDULING APPLICATION COMPLETION TIME: " << T_final << "\n";
    std::cout << "INITIAL APPLICATION ENERGY CONSUMPTION: " << E_total << "\n";
    std::cout << "INITIAL TASK SCHEDULE:\n";
    print_task_schedule(tasks);
    print_validation_report(tasks);
    printFinalSequences(sequence);

    auto [tasks2, sequence2] = optimize_task_scheduling(tasks, sequence, T_final, {1,2,4}, 0.5);


    int T_final_after = total_time(tasks2);
    double E_final = total_energy(tasks2, {1,2,4}, 0.5);
    std::cout << "FINAL SCHEDULING APPLICATION COMPLETION TIME: " << T_final_after << "\n";
    std::cout << "FINAL APPLICATION ENERGY CONSUMPTION: " << E_final << "\n";
    std::cout << "FINAL TASK SCHEDULE:\n";
    print_task_schedule(tasks2);
    print_validation_report(tasks2);
    printFinalSequences(sequence2);

    return 0;
}
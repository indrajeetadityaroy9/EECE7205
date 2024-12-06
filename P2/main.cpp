#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// SchedulingState tracks the scheduling progress of each task
enum class SchedulingState {
    UNSCHEDULED = 0,    // Initial state before scheduling
    SCHEDULED = 1,      // Task has been scheduled in initial phase
    KERNEL_SCHEDULED = 2 // Task has been processed by kernel algorithm during migration
};

// Task class represents a node in the directed acyclic task graph (DAG)
class Task {
public:
    int id;  // Unique task identifier
    // pred_tasks and succ_tasks implement the precedence constraints
    // described where each edge (vi,vj) represents that
    // task vi must complete before task vj starts
    std::vector<int> pred_tasks;  // Immediate predecessors in the task graph
    std::vector<int> succ_tasks;  // Immediate successors in the task graph

    // Execution times for different processing units
    // where execution time T_i^l is inversely proportional to core frequency f_k
    std::array<int,3> core_execution_times;  // T_i^l for each local core k
    // Cloud execution phases:
    // [0] = sending time (T_i^s)
    // [1] = cloud computation time (T_i^c)
    // [2] = receiving time (T_i^r)
    std::array<int,3> cloud_execution_times;

    // Finish times for different execution phases
    // These track when each phase of task execution completes
    int FT_l;   // Local core finish time
    int FT_ws;  // Wireless sending finish time 
    int FT_c;   // Cloud computation finish time
    int FT_wr;  // Wireless receiving finish time

    // Ready times for different execution phases
    // These represent the earliest times when a task can start on each resource
    int RT_l;   // Ready time for local execution (Equation 3)
    int RT_ws;  // Ready time for wireless sending (Equation 4)
    int RT_c;   // Ready time for cloud execution (Equation 5)
    int RT_wr;  // Ready time for receiving results (Equation 6)

    // Priority score used in the initial scheduling phase
    // Calculated according to (Equations 15-16)
    double priority_score;

    // Task assignment: -1 for cloud, 0...K-1 for local cores
    // This implements the execution location tracking
    int assignment;

    // Flags whether task is initially assigned to core (vs cloud)
    // Used in primary assignment phase
    bool is_core_task;

    // Tracks start times for task on different execution units
    // Used for scheduling and migration decisions
    std::vector<int> execution_unit_task_start_times;
    int execution_finish_time;

    // Current state in the scheduling process
    SchedulingState is_scheduled;

    // Constructor initializes a task with its execution time requirements
    Task(int task_id,
         const std::map<int, std::array<int,3>>& core_exec_times_map,
         const std::array<int,3>& cloud_exec_times_input)
        : id(task_id),
          FT_l(0),
          FT_ws(0),
          FT_c(0),
          FT_wr(0),
          RT_l(-1),
          RT_ws(-1),
          RT_c(-1),
          RT_wr(-1),
          priority_score(-1.0),
          assignment(-2),
          is_core_task(false),
          execution_finish_time(-1),
          is_scheduled(SchedulingState::UNSCHEDULED)
    {
        // Set local core execution times if available
        auto it = core_exec_times_map.find(id);
        if (it != core_exec_times_map.end()) {
            this->core_execution_times = it->second;
        } else {
            this->core_execution_times = {0,0,0};
        }

        // Set cloud execution phase times
        this->cloud_execution_times = cloud_exec_times_input;
    }
};


int total_time(const std::vector<Task>& tasks) {
    int max_completion_time = 0;
    for (const auto& task : tasks) {
        if (task.succ_tasks.empty()) {
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
        int t_l_min = *std::min_element(task.core_execution_times.begin(), task.core_execution_times.end());

        int t_re = task.cloud_execution_times[0] + task.cloud_execution_times[1]+ task.cloud_execution_times[2];

        if (t_re < t_l_min) {
            task.is_core_task = false;
            task.assignment = k;
        } else {
            task.is_core_task = true;
        }
    }
}

double calculate_priority(const Task& task, const std::vector<Task>& tasks, const std::vector<double>& w, std::map<int,double>& computed_priority_scores);

void task_prioritizing(std::vector<Task>& tasks) {
    std::vector<double> w(tasks.size(), 0.0);

    for (size_t i = 0; i < tasks.size(); i++) {
        const Task& task = tasks[i];
        if (!task.is_core_task) {
            w[i] = static_cast<double>(task.cloud_execution_times[0] + task.cloud_execution_times[1] + task.cloud_execution_times[2]);
        } else {
            double sum_local = static_cast<double>(std::accumulate(task.core_execution_times.begin(), task.core_execution_times.end(), 0));
            w[i] = sum_local / static_cast<double>(task.core_execution_times.size());
        }
    }

    std::map<int,double> computed_priority_scores;

    for (auto& task : tasks) {
        calculate_priority(task, tasks, w, computed_priority_scores);
    }

    for (auto& task : tasks) {
        task.priority_score = computed_priority_scores[task.id];
    }
}

double calculate_priority(const Task& task, const std::vector<Task>& tasks, const std::vector<double>& w, std::map<int,double>& computed_priority_scores) {
    auto it = computed_priority_scores.find(task.id);
    if (it != computed_priority_scores.end()) {
        return it->second;
    }

    if (task.succ_tasks.empty()) {
        double priority_val = w[task.id - 1];
        computed_priority_scores[task.id] = priority_val;
        return priority_val;
    }

    double max_successor_priority = -std::numeric_limits<double>::infinity();
    for (int succ_id : task.succ_tasks) {
        const Task& succ_tasks = tasks[succ_id - 1];
        double succ_priority = calculate_priority(succ_tasks, tasks, w, computed_priority_scores);
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
        sequences.resize(k+1);
    }

    std::vector<int> get_priority_ordered_tasks() {
        std::vector<std::pair<double,int>> task_priority_list;
        for (auto &t : tasks) {
            task_priority_list.emplace_back(t.priority_score, t.id);
        }
        std::sort(task_priority_list.begin(), task_priority_list.end(), [](const auto &a, const auto &b){return a.first > b.first;});

        std::vector<int> result;
        for (auto &item : task_priority_list) {
            result.push_back(item.second);
        }
        return result;
    }

    std::pair<std::vector<Task*>, std::vector<Task*>> classify_entry_tasks(const std::vector<int>& priority_order) {
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
        task.FT_l = finish_time;
        task.execution_finish_time = finish_time;

        task.execution_unit_task_start_times.assign(k+1, -1);
        task.execution_unit_task_start_times[core] = start_time;

        core_earliest_ready[core] = finish_time;

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
        int send_ready = task.RT_ws;
        int send_finish = send_ready + task.cloud_execution_times[0];

        int cloud_ready = send_finish;
        int cloud_finish = cloud_ready + task.cloud_execution_times[1];

        int receive_ready = cloud_finish;
        int receive_finish = std::max(wr_ready, receive_ready) + task.cloud_execution_times[2];

        return {send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish};
    }

    void schedule_on_cloud(Task &task, int send_ready, int send_finish, int cloud_ready, int cloud_finish, int receive_ready, int receive_finish) {
        task.RT_ws = send_ready;
        task.FT_ws = send_finish;

        task.RT_c = cloud_ready;
        task.FT_c = cloud_finish;

        task.RT_wr = receive_ready;
        task.FT_wr = receive_finish;

        task.execution_finish_time = receive_finish;
        task.FT_l = 0;

        task.execution_unit_task_start_times.assign(k+1, -1);
        task.execution_unit_task_start_times[k] = send_ready;

        task.assignment = k;

        task.is_scheduled = SchedulingState::SCHEDULED;

        ws_ready = send_finish;
        wr_ready = receive_finish;

        sequences[k].push_back(task.id);
    }

    void schedule_entry_tasks(std::vector<Task*> &entry_tasks) {
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
            schedule_on_cloud(*task, timing.send_ready, timing.send_finish, timing.cloud_ready, timing.cloud_finish, timing.receive_ready, timing.receive_finish);
        }
    }

    void calculate_non_entry_task_ready_times(Task &task) {
        int max_pred_finish_l_wr = 0;
        for (int pred_id : task.pred_tasks) {
            Task &pred = tasks[pred_id - 1];
            int val = std::max(pred.FT_l, pred.FT_wr);
            if (val > max_pred_finish_l_wr) {
                max_pred_finish_l_wr = val;
            }
        }
        task.RT_l = std::max(max_pred_finish_l_wr, 0);

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
        for (auto* task : non_entry_tasks) {
            calculate_non_entry_task_ready_times(*task);

            if (!task->is_core_task) {
                auto timing = calculate_cloud_phases_timing(*task);
                schedule_on_cloud(*task, timing.send_ready, timing.send_finish, timing.cloud_ready, timing.cloud_finish, timing.receive_ready, timing.receive_finish);
            } else {
                auto local_choice = identify_optimal_local_core(*task, task->RT_l);
                auto timing = calculate_cloud_phases_timing(*task);
                int cloud_finish_time = timing.receive_finish;

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
        return sequences;
    }

    std::vector<Task> &tasks;
    int k;

    std::vector<int> core_earliest_ready;
    int ws_ready;
    int wr_ready;

    std::vector<std::vector<int>> sequences;
};

std::vector<std::vector<int>> execution_unit_selection(std::vector<Task>& tasks) {
    InitialTaskScheduler scheduler(tasks, 3);
    std::vector<int> priority_orderered_tasks = scheduler.get_priority_ordered_tasks();
    auto [entry_tasks, non_entry_tasks] = scheduler.classify_entry_tasks(priority_orderered_tasks);
    scheduler.schedule_entry_tasks(entry_tasks);
    scheduler.schedule_non_entry_tasks(non_entry_tasks);
    return scheduler.get_sequences();
}

std::vector<std::vector<int>> construct_sequence(std::vector<Task>& tasks, int task_id, int execution_unit, std::vector<std::vector<int>> original_sequence) {

    Task &target_task = tasks[task_id - 1];

    int target_task_rt = target_task.is_core_task ? target_task.RT_l : target_task.RT_ws;

    int original_assignment = target_task.assignment;
    auto &old_seq = original_sequence[original_assignment];
    old_seq.erase(std::remove(old_seq.begin(), old_seq.end(), target_task.id), old_seq.end());

    auto &new_seq = original_sequence[execution_unit];

    std::vector<int> start_times;
    start_times.reserve(new_seq.size());
    for (int tid : new_seq) {
        Task &t = tasks[tid - 1];
        start_times.push_back(t.execution_unit_task_start_times[execution_unit]);
    }

    auto it = std::lower_bound(start_times.begin(), start_times.end(), target_task_rt);
    int insertion_index = static_cast<int>(std::distance(start_times.begin(), it));
    new_seq.insert(new_seq.begin() + insertion_index, target_task.id);
    target_task.assignment = execution_unit;
    target_task.is_core_task = (execution_unit != 3);

    return original_sequence;
}

class KernelScheduler {
public:
    KernelScheduler(std::vector<Task>& tasks, std::vector<std::vector<int>>& sequences)
        : tasks(tasks), sequences(sequences)
    {
        RT_ls = {0,0,0};
        cloud_phases_ready_times = {0,0,0};

        std::tie(dependency_ready, sequence_ready) = initialize_task_state();
    }

    std::pair<std::vector<int>, std::vector<int>> initialize_task_state() {
        std::vector<int> dependency_ready(tasks.size(), 0);
        for (size_t i = 0; i < tasks.size(); i++) {
            dependency_ready[i] = static_cast<int>(tasks[i].pred_tasks.size());
        }

        std::vector<int> sequence_ready(tasks.size(), -1);
        for (auto &seq : sequences) {
            if (!seq.empty()) {
                sequence_ready[seq[0] - 1] = 0;
            }
        }

        return {dependency_ready, sequence_ready};
    }

    void update_task_state(Task &task) {
        if (task.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {
            int unsched_preds = 0;
            for (int pred_id : task.pred_tasks) {
                Task &pred = tasks[pred_id - 1];
                if (pred.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {
                    unsched_preds++;
                }
            }
            dependency_ready[task.id - 1] = unsched_preds;

            for (auto &seq : sequences) {
                auto it = std::find(seq.begin(), seq.end(), task.id);
                if (it != seq.end()) {
                    int idx = static_cast<int>(std::distance(seq.begin(), it));
                    if (idx > 0) {
                        int prev_task_id = seq[idx - 1];
                        Task &prev_task = tasks[prev_task_id - 1];
                        sequence_ready[task.id - 1] = 
                            (prev_task.is_scheduled != SchedulingState::KERNEL_SCHEDULED) ? 1 : 0;
                    } else {
                        sequence_ready[task.id - 1] = 0;
                    }
                    break;
                }
            }
        }
    }

    void schedule_local_task(Task &task) {
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

        RT_ls[core_index] = task.FT_l;

        task.FT_ws = -1;
        task.FT_c = -1;
        task.FT_wr = -1;
    }

    void schedule_cloud_task(Task &task) {
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

        int send_start = std::max(cloud_phases_ready_times[0], task.RT_ws);
        task.execution_unit_task_start_times[3] = send_start;

        task.FT_ws = send_start + task.cloud_execution_times[0];
        cloud_phases_ready_times[0] = task.FT_ws;

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

        task.RT_wr = task.FT_c;
        task.FT_wr = std::max(cloud_phases_ready_times[2], task.RT_wr) + task.cloud_execution_times[2];
        cloud_phases_ready_times[2] = task.FT_wr;

        task.FT_l = -1;
    }

    std::deque<Task*> initialize_queue() {
        std::deque<Task*> dq;
        for (auto &t : tasks) {
            if (sequence_ready[t.id - 1] == 0) {
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
    KernelScheduler scheduler(tasks, sequences);

    std::deque<Task*> queue = scheduler.initialize_queue();

    while (!queue.empty()) {
        Task* current_task = queue.front();
        queue.pop_front();
        current_task->is_scheduled = SchedulingState::KERNEL_SCHEDULED;
        if (current_task->is_core_task) {
            scheduler.schedule_local_task(*current_task);
        } else {
            scheduler.schedule_cloud_task(*current_task);
        }

        for (auto &task : tasks) {
            scheduler.update_task_state(task);
        }

        for (auto &task : tasks) {
            int idx = task.id - 1;
            if (scheduler.dependency_ready[idx] == 0 &&
                scheduler.sequence_ready[idx] == 0 &&
                task.is_scheduled != SchedulingState::KERNEL_SCHEDULED) {

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

    for (auto &task : tasks) {
        task.is_scheduled = SchedulingState::UNSCHEDULED;
    }

    return tasks;
}

struct MigrationKey {
    int task_idx;
    int target_execution_unit;
    std::vector<int> assignments;

    bool operator<(const MigrationKey& other) const {
        if (task_idx != other.task_idx) return task_idx < other.task_idx;
        if (target_execution_unit != other.target_execution_unit) return target_execution_unit < other.target_execution_unit;
        return assignments < other.assignments;
    }
};

MigrationKey generate_cache_key(const std::vector<Task>& tasks, int task_idx, int target_execution_unit) {
    MigrationKey key;
    key.task_idx = task_idx;
    key.target_execution_unit = target_execution_unit;
    key.assignments.reserve(tasks.size());
    for (const auto& t : tasks) {
        key.assignments.push_back(t.assignment);
    }
    return key;
}

std::pair<int,double> evaluate_migration(
    std::vector<Task>& tasks,
    const std::vector<std::vector<int>>& seqs,
    int task_idx,
    int target_execution_unit,
    std::map<MigrationKey, std::pair<int,double>>& migration_cache,
    const std::vector<int>& core_powers = {1,2,4},
    double cloud_sending_power = 0.5
) {
    MigrationKey cache_key = generate_cache_key(tasks, task_idx, target_execution_unit);

    auto it = migration_cache.find(cache_key);
    if (it != migration_cache.end()) {
        return it->second;
    }

    std::vector<std::vector<int>> sequence_copy = seqs;
    std::vector<Task> tasks_copy = tasks;

    sequence_copy = construct_sequence(tasks_copy, task_idx + 1, target_execution_unit, sequence_copy);
    kernel_algorithm(tasks_copy, sequence_copy);

    int migration_T = total_time(tasks_copy);
    double migration_E = total_energy(tasks_copy, core_powers, cloud_sending_power);

    migration_cache[cache_key] = std::make_pair(migration_T, migration_E);

    return {migration_T, migration_E};
}

std::vector<std::array<bool,4>> initialize_migration_choices(const std::vector<Task>& tasks) {
    std::vector<std::array<bool,4>> migration_choices(tasks.size(), std::array<bool,4>{false,false,false,false});

    for (size_t i = 0; i < tasks.size(); i++) {
        const Task& task = tasks[i];
        if (task.assignment == 3) {
            for (int j = 0; j < 4; j++) {
                migration_choices[i][j] = true;
            }
        } else {
            migration_choices[i][task.assignment] = true;
        }
    }

    return migration_choices;
}

struct TaskMigrationState {
    int time;         
    double energy;      
    double efficiency; 
    int task_index;    
    int target_execution_unit; 
};

TaskMigrationState* identify_optimal_migration(
    const std::vector<std::tuple<int,int,int,double>>& migration_trials_results,
    int T_final,
    double E_total,
    int T_max
) {
    double best_energy_reduction = 0.0;
    TaskMigrationState* best_migration_state = nullptr;

    for (auto &res : migration_trials_results) {
        int task_idx, resource_idx, time_int;
        double energy;
        std::tie(task_idx, resource_idx, time_int, energy) = res;

        int time = time_int;

        if (time > T_max) {
            continue;
        }

        double energy_reduction = E_total - energy;
        if (time <= T_final && energy_reduction > 0.0) {
            if (energy_reduction > best_energy_reduction) {
                best_energy_reduction = energy_reduction;
                if (best_migration_state) {
                    delete best_migration_state;
                }
                best_migration_state = new TaskMigrationState{
                    time,
                    energy,
                    best_energy_reduction,
                    task_idx + 1,
                    resource_idx + 1
                };
            }
        }
    }

    if (best_migration_state) {
        return best_migration_state;
    }

    struct MigrationCandidate {
        double neg_efficiency;
        int task_idx;
        int resource_idx;
        int time;
        double energy;

        bool operator<(const MigrationCandidate& other) const {
            return neg_efficiency > other.neg_efficiency;
        }
    };

    std::priority_queue<MigrationCandidate> migration_candidates;

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

            migration_candidates.push(MigrationCandidate{-efficiency, task_idx, resource_idx, time, energy});
        }
    }

    if (migration_candidates.empty()) {
        return nullptr;
    }

    MigrationCandidate best = migration_candidates.top();
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
    std::map<MigrationKey, std::pair<int,double>> migration_cache;

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
                if (migration_choices[task_idx][possible_execution_unit]) {
                    continue;
                }

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

        sequence = construct_sequence(
            tasks,
            best_migration->task_index,
            best_migration->target_execution_unit - 1,
            sequence
        );

        kernel_algorithm(tasks, sequence);

        current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power);
        energy_improved = (current_iteration_energy < previous_iteration_energy);

        delete best_migration;

        if (migration_cache.size() > 1000) {
            migration_cache.clear();
        }
    }

    return {tasks, sequence};
}

void printScheduleTasks(const std::vector<Task>& tasks) {
    static const char* ASSIGNMENT_MAPPING[] = {
        "Core 1", "Core 2", "Core 3", "Cloud"
    };

    // Define column widths
    const int width_id = 8;        // "TaskID" fits well within 8 chars
    const int width_assignment = 12;
    const int width_start = 12;
    const int width_execwindow = 25;
    const int width_send = 20;
    const int width_cloud = 20;
    const int width_receive = 20;

    std::cout << "\nTask Scheduling Summary (Horizontal Format):\n";
    // Print a line separator wide enough for all columns
    std::cout << std::string(130, '-') << "\n";

    // Print Header Row
    std::cout << std::left
              << std::setw(width_id) << "TaskID"
              << std::setw(width_assignment) << "Assignment"
              << std::setw(width_start) << "StartTime"
              << std::setw(width_execwindow) << "ExecWindow(Core)"
              << std::setw(width_send) << "SendPhase(Cloud)"
              << std::setw(width_cloud) << "CloudPhase(Cloud)"
              << std::setw(width_receive) << "ReceivePhase(Cloud)"
              << "\n";

    std::cout << std::string(130, '-') << "\n";

    for (const auto& task : tasks) {
        std::string assignment_str;
        if (task.assignment >= 0 && task.assignment <= 3) {
            assignment_str = ASSIGNMENT_MAPPING[task.assignment];
        } else if (task.assignment == -2) {
            assignment_str = "Not Scheduled";
        } else {
            assignment_str = "Unknown";
        }

        std::string execution_window = "-";
        std::string send_phase = "-";
        std::string cloud_phase = "-";
        std::string receive_phase = "-";

        int display_start_time = -1;

        if (task.is_core_task && task.assignment >= 0 &&
            static_cast<size_t>(task.assignment) < task.execution_unit_task_start_times.size()) {
            int start_time = task.execution_unit_task_start_times[task.assignment];
            int end_time = start_time + task.core_execution_times[task.assignment];
            execution_window = std::to_string(start_time) + "=>" + std::to_string(end_time);
            display_start_time = start_time;
        }

        if (!task.is_core_task && task.execution_unit_task_start_times.size() > 3) {
            int send_start = task.execution_unit_task_start_times[3];
            int send_end = send_start + task.cloud_execution_times[0];
            send_phase = std::to_string(send_start) + "=>" + std::to_string(send_end);

            int RT_c_val = task.RT_c;
            int cloud_end = RT_c_val + task.cloud_execution_times[1];
            cloud_phase = std::to_string(RT_c_val) + "=>" + std::to_string(cloud_end);

            int RT_wr_val = task.RT_wr;
            int receive_end = RT_wr_val + task.cloud_execution_times[2];
            receive_phase = std::to_string(RT_wr_val) + "=>" + std::to_string(receive_end);

            // For cloud tasks, consider the send start time as the primary start time
            display_start_time = send_start;
        }

        std::cout << std::left
                  << std::setw(width_id) << task.id
                  << std::setw(width_assignment) << assignment_str
                  << std::setw(width_start) << ((display_start_time >= 0) ? std::to_string(display_start_time) : "-")
                  << std::setw(width_execwindow) << execution_window
                  << std::setw(width_send) << send_phase
                  << std::setw(width_cloud) << cloud_phase
                  << std::setw(width_receive) << receive_phase
                  << "\n";
    }

    std::cout << std::string(130, '-') << "\n";
}

std::tuple<bool, std::vector<std::string>> validate_schedule_constraints(const std::vector<Task>& tasks) {
    std::vector<std::string> violations;

    // Build a helper map from task ID to index
    std::map<int, size_t> id_to_index;
    for (size_t i = 0; i < tasks.size(); ++i) {
        id_to_index[tasks[i].id] = i;
    }

    auto get_task = [&](int task_id) -> const Task& {
        return tasks[id_to_index.at(task_id)];
    };

    // Separate cloud and core tasks
    std::vector<const Task*> cloud_tasks;
    std::vector<const Task*> core_tasks;

    for (auto &t : tasks) {
        if (t.is_core_task) {
            core_tasks.push_back(&t);
        } else {
            cloud_tasks.push_back(&t);
        }
    }

    std::cout << "[INFO] Validating Wireless Sending Channel Constraints...\n";
    // 1. Check Wireless Sending Channel Conflicts
    {
        auto sorted = cloud_tasks;
        std::sort(sorted.begin(), sorted.end(), [](const Task* a, const Task* b) {
            return a->execution_unit_task_start_times[3] < b->execution_unit_task_start_times[3];
        });

        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            const Task* current = sorted[i];
            const Task* next_task = sorted[i + 1];
            if (current->FT_ws > next_task->execution_unit_task_start_times[3]) {
                violations.push_back("Wireless Sending Channel Conflict: Task " +
                                     std::to_string(current->id) +
                                     " sending ends at " + std::to_string(current->FT_ws) +
                                     " but Task " + std::to_string(next_task->id) +
                                     " starts at " + std::to_string(next_task->execution_unit_task_start_times[3]));
            }
        }
    }

    std::cout << "[INFO] Validating Cloud Computing Constraints...\n";
    // 2. Check Cloud Computing Channel Conflicts
    {
        auto sorted = cloud_tasks;
        std::sort(sorted.begin(), sorted.end(), [](const Task* a, const Task* b) {
            return a->RT_c < b->RT_c;
        });

        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            const Task* current = sorted[i];
            const Task* next_task = sorted[i + 1];
            if (current->FT_c > next_task->RT_c) {
                violations.push_back("Cloud Computing Conflict: Task " +
                                     std::to_string(current->id) +
                                     " computing ends at " + std::to_string(current->FT_c) +
                                     " but Task " + std::to_string(next_task->id) +
                                     " starts at " + std::to_string(next_task->RT_c));
            }
        }
    }

    std::cout << "[INFO] Validating Wireless Receiving Channel Constraints...\n";
    // 3. Check Wireless Receiving Channel Conflicts
    {
        auto sorted = cloud_tasks;
        std::sort(sorted.begin(), sorted.end(), [](const Task* a, const Task* b) {
            return a->RT_wr < b->RT_wr;
        });

        for (size_t i = 0; i + 1 < sorted.size(); ++i) {
            const Task* current = sorted[i];
            const Task* next_task = sorted[i + 1];
            if (current->FT_wr > next_task->RT_wr) {
                violations.push_back("Wireless Receiving Channel Conflict: Task " +
                                     std::to_string(current->id) +
                                     " receiving ends at " + std::to_string(current->FT_wr) +
                                     " but Task " + std::to_string(next_task->id) +
                                     " starts at " + std::to_string(next_task->RT_wr));
            }
        }
    }

    std::cout << "[INFO] Validating Task Dependency Constraints...\n";
    // 4. Check Pipelined Dependencies
    {
        for (auto &task : tasks) {
            if (!task.is_core_task) {
                // Cloud task dependencies
                int task_send_start = task.execution_unit_task_start_times[3];
                for (int pred_id : task.pred_tasks) {
                    const Task &pred = get_task(pred_id);
                    if (pred.is_core_task) {
                        // Core predecessor must finish local exec before this cloud task sends
                        if (pred.FT_l > task_send_start) {
                            violations.push_back("Core-Cloud Dependency Violation: Parent Task " +
                                                 std::to_string(pred.id) + " finishes at " +
                                                 std::to_string(pred.FT_l) + " but Cloud Task " +
                                                 std::to_string(task.id) + " starts sending at " +
                                                 std::to_string(task_send_start));
                        }
                    } else {
                        // Cloud predecessor must finish sending before this one starts sending
                        if (pred.FT_ws > task_send_start) {
                            violations.push_back("Cloud Pipeline Dependency Violation: Parent Task " +
                                                 std::to_string(pred.id) + " sending ends at " +
                                                 std::to_string(pred.FT_ws) + " but Task " +
                                                 std::to_string(task.id) + " starts sending at " +
                                                 std::to_string(task_send_start));
                        }
                    }
                }
            } else {
                // Core task dependencies
                int core_id = task.assignment;
                int task_start = task.execution_unit_task_start_times[core_id];
                for (int pred_id : task.pred_tasks) {
                    const Task &pred = get_task(pred_id);
                    int pred_finish = pred.is_core_task ? pred.FT_l : pred.FT_wr;
                    if (pred_finish > task_start) {
                        violations.push_back("Core Task Dependency Violation: Parent Task " +
                                             std::to_string(pred.id) + " finishes at " +
                                             std::to_string(pred_finish) + " but Core Task " +
                                             std::to_string(task.id) + " starts at " +
                                             std::to_string(task_start));
                    }
                }
            }
        }
    }

    std::cout << "[INFO] Validating Core Execution Conflicts...\n";
    // 5. Check Core Execution Conflicts
    {
        int num_cores = 3; // Adjust as needed
        for (int core_id = 0; core_id < num_cores; ++core_id) {
            std::vector<const Task*> core_specific;
            for (auto t : core_tasks) {
                if (t->assignment == core_id) {
                    core_specific.push_back(t);
                }
            }
            std::sort(core_specific.begin(), core_specific.end(), [=](const Task* a, const Task* b) {
                return a->execution_unit_task_start_times[core_id] < b->execution_unit_task_start_times[core_id];
            });
            for (size_t i = 0; i + 1 < core_specific.size(); ++i) {
                const Task* current = core_specific[i];
                const Task* next_task = core_specific[i + 1];
                if (current->FT_l > next_task->execution_unit_task_start_times[core_id]) {
                    violations.push_back("Core " + std::to_string(core_id) + " Execution Conflict: Task " +
                                         std::to_string(current->id) + " finishes at " +
                                         std::to_string(current->FT_l) + " but Task " +
                                         std::to_string(next_task->id) + " starts at " +
                                         std::to_string(next_task->execution_unit_task_start_times[core_id]));
                }
            }
        }
    }

    bool is_valid = violations.empty();
    return {is_valid, violations};
}

void print_schedule_validation_report(const std::vector<Task>& tasks) {
    auto [is_valid, violations] = validate_schedule_constraints(tasks);

    std::cout << "\nSchedule Validation Report\n";
    std::cout << std::string(50, '=') << "\n";

    if (is_valid) {
        std::cout << "Schedule is valid with all constraints satisfied!\n";
    } else {
        std::cout << "Found constraint violations:\n";
        for (const auto& violation : violations) {
            std::cout << violation << "\n";
        }
    }
}

void printScheduleSequences(const std::vector<std::vector<int>>& sequences) {
    std::cout << "\nExecution Sequences:\n";
    std::cout << std::string(40, '-') << "\n";
    
    for (std::size_t i = 0; i < sequences.size(); i++) {
        std::string label = (i < 3) ? "Core " + std::to_string(i + 1) : "Cloud";
        std::cout << std::setw(12) << std::left << label << ": ";
        
        std::cout << "[";
        for (std::size_t j = 0; j < sequences[i].size(); j++) {
            if (j > 0) std::cout << ", ";
            std::cout << sequences[i][j];
        }
        std::cout << "]\n";
    }
}

std::vector<Task> create_task_graph(
    const std::vector<int>& task_ids,
    const std::map<int, std::array<int,3>>& core_exec_times,
    const std::array<int,3>& cloud_exec_times,
    const std::vector<std::pair<int,int>>& edges
) {
    std::vector<Task> tasks;
    tasks.reserve(task_ids.size());

    for (int tid : task_ids) {
        tasks.emplace_back(tid, core_exec_times, cloud_exec_times);
    }

    std::map<int,int> id_to_index;
    for (size_t i = 0; i < tasks.size(); ++i) {
        id_to_index[tasks[i].id] = static_cast<int>(i);
    }

    for (auto &edge : edges) {
        int from = edge.first;
        int to   = edge.second;
        tasks[id_to_index[from]].succ_tasks.push_back(to);
        tasks[id_to_index[to]].pred_tasks.push_back(from);
    }

    return tasks;
}

int main() {
    static const std::map<int, std::array<int,3>> core_execution_times = {
        {1, {9, 7, 5}}, {2, {8, 6, 5}}, {3, {6, 5, 4}}, {4, {7, 5, 3}},
        {5, {5, 4, 2}}, {6, {7, 6, 4}}, {7, {8, 5, 3}}, {8, {6, 4, 2}},
        {9, {5, 3, 2}}, {10,{7,4,2}},  {11,{10,7,4}}, {12,{11,8,5}},
        {13,{9,6,3}}, {14,{12,8,4}}, {15,{10,7,3}}, {16,{11,7,4}},
        {17,{9,6,3}}, {18,{12,8,5}}, {19,{10,7,4}}, {20,{11,8,5}}
    };

    static const std::array<int,3> cloud_execution_times = {3, 1, 1};

    std::vector<int> ten_task_graph_task_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> twenty_task_graph_task_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    // Graph 1
    std::vector<std::pair<int,int>> graph1_edges = {
        {1,2}, {1,3}, {1,4}, {1,5}, {1,6},
        {2,8}, {2,9},
        {3,7},
        {4,8}, {4,9},
        {5,9},
        {6,8},
        {7,10},
        {8,10},
        {9,10}
    };

    //Graph 2
    std::vector<std::pair<int,int>> graph2_edges = {
        {1,2}, {1,3},
        {2,4}, {2,5},
        {3,5}, {3,6},
        {4,6},
        {5,7},
        {6,7}, {6,8},
        {7,8}, {7,9},
        {8,10},
        {9,10}
    };

    //Graph 3
    std::vector<std::pair<int,int>> graph3_edges = {
        {1,2}, {1,3}, {1,4}, {1,5}, {1,6},
        {2,7}, {2,8},
        {3,7}, {3,8},
        {4,8}, {4,9},
        {5,9}, {5,10},
        {6,10}, {6,11},
        {7,12},
        {8,12}, {8,13},
        {9,13}, {9,14},
        {10,11}, {10,15},
        {11,15}, {11,16},
        {12,17},
        {13,17}, {13,18},
        {14,18}, {14,19},
        {15,19},
        {16,19},
        {17,20},
        {18,20},
        {19,20}
    };

    //Graph 4
    std::vector<std::pair<int,int>> graph4_edges = {
        {1,7},
        {2,7},
        {3,7}, {3,8},
        {4,8}, {4,9},
        {5,9}, {5,10},
        {6,10}, {6,11},
        {7,12},
        {8,12}, {8,13},
        {9,13}, {9,14},
        {10,11}, {10,15},
        {11,15}, {11,16},
        {12,17},
        {13,17}, {13,18},
        {14,18}, {14,19},
        {15,19},
        {16,19},
        {17,20},
        {18,20},
        {19,20}
    };

    //Graph 5
    std::vector<std::pair<int,int>> graph5_edges = {
    {1,4}, {1,5}, {1,6},
    {2,7}, {2,8},
    {3,7}, {3,8},
    {4,8}, {4,9},
    {5,9}, {5,10},
    {6,10}, {6,11},
    {7,12},
    {8,12}, {8,13},
    {9,13}, {9,14},
    {10,11}, {10,15},
    {11,15}, {11,16},
    {12,17},
    {13,17}, {13,18},
    {14,18},
    {15,19},
    {16,19},
    {18,20},
    };

    std::vector<std::vector<Task>> all_graphs;
    all_graphs.push_back(create_task_graph(ten_task_graph_task_ids, core_execution_times, cloud_execution_times, graph1_edges));
    all_graphs.push_back(create_task_graph(ten_task_graph_task_ids, core_execution_times, cloud_execution_times, graph2_edges));
    all_graphs.push_back(create_task_graph(twenty_task_graph_task_ids, core_execution_times, cloud_execution_times, graph3_edges));
    all_graphs.push_back(create_task_graph(twenty_task_graph_task_ids, core_execution_times, cloud_execution_times, graph4_edges));
    all_graphs.push_back(create_task_graph(twenty_task_graph_task_ids, core_execution_times, cloud_execution_times, graph5_edges));

    // Process each graph
    for (size_t i = 0; i < all_graphs.size(); ++i) {
        std::cout << "\nProcessing Graph " << (i+1) << ":\n";
        auto &tasks = all_graphs[i];

        primary_assignment(tasks, 3);
        task_prioritizing(tasks);
        auto sequence = execution_unit_selection(tasks);

        int T_final = total_time(tasks);
        double E_total = total_energy(tasks, {1,2,4}, 0.5);
        std::cout << "INITIAL SCHEDULING APPLICATION COMPLETION TIME: " << T_final << "\n";
        std::cout << "INITIAL APPLICATION ENERGY CONSUMPTION: " << E_total << "\n";
        std::cout << "INITIAL TASK SCHEDULE:\n";
        printScheduleTasks(tasks);
        print_schedule_validation_report(tasks);
        printScheduleSequences(sequence);

        auto [tasks2, sequence2] = optimize_task_scheduling(tasks, sequence, T_final, {1,2,4}, 0.5);

        int T_final_after = total_time(tasks2);
        double E_final = total_energy(tasks2, {1,2,4}, 0.5);
        std::cout << "MAXIMUM APPLICATION COMPLETION TIME: " << T_final*1.5 << "\n";
        std::cout << "FINAL SCHEDULING APPLICATION COMPLETION TIME: " << T_final_after << "\n";
        std::cout << "FINAL APPLICATION ENERGY CONSUMPTION: " << E_final << "\n";
        std::cout << "FINAL TASK SCHEDULE:\n";
        printScheduleTasks(tasks2);
        print_schedule_validation_report(tasks2);
        printScheduleSequences(sequence2);
    }

    return 0;
}
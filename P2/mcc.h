# pragma once

#include <vector>
#include <unordered_set>
#include <map>
#include <array>
#include <tuple>
#include <queue>
#include <optional>
#include <numeric>
using namespace std;


extern const std::map<int, std::vector<float>> CORE_EXECUTION_TIMES;
extern const std::vector<float> CLOUD_EXECUTION_TIMES;

class Task;
using TaskSet = std::unordered_set<Task*>;
using MigrationTrial = std::tuple<int, int, float, float>;

enum class SchedulingState {
    UNSCHEDULED = 0,
    SCHEDULED = 1,
    KERNEL_SCHEDULED = 2
};

struct TaskMigrationState {
    float time;
    float energy;
    float efficiency_ratio;
    int node_index;
    int target_execution_unit;
};

class Task {
private:
    int id;
    TaskSet parents;
    TaskSet children;
    std::vector<float> core_execution_times;
    std::vector<float> cloud_execution_times;
    float local_core_finish_time;
    float wireless_sending_finish_time;
    float remote_cloud_finish_time;
    float wireless_recieving_finish_time;
    float local_core_ready_time;
    float wireless_sending_ready_time;
    float remote_cloud_ready_time;
    float wireless_recieving_ready_time;
    float priority_score;
    float execution_finish_time; 
    int assignment;
    bool is_core_task;
    std::array<int, 4> execution_unit_start_times;
    SchedulingState scheduling_state;

public:
    Task(int task_id, TaskSet* parent_tasks = nullptr, TaskSet* child_tasks = nullptr);

    Task(const Task& other);
    
    int getId() const;
    const TaskSet& getParents() const;
    const TaskSet& getChildren() const;
    const std::vector<float>& getCoreExecutionTimes() const;
    const std::vector<float>& getCloudExecutionTimes() const;
    float getLocalCoreFinishTime() const;
    float getWirelessSendingFinishTime() const;
    float getRemoteCloudFinishTime() const;
    float getWirelessRecievingFinishTime() const;
    float getLocalCoreReadyTime() const;
    float getWirelessSendingReadyTime() const;
    float getRemoteCloudReadyTime() const;
    float getWirelessRecievingReadyTime() const;
    float getPriorityScore() const;
    float getExecutionFinishTime() const;
    int getAssignment() const;
    bool isCoreTask() const;
    const std::array<int, 4>& getExecutionUnitStartTimes() const;
    SchedulingState getSchedulingState() const;

    void setCoreExecutionTimes(const std::vector<float>& times);
    void setCloudExecutionTimes(const std::vector<float>& times);
    void addParent(Task* parent);
    void addChild(Task* child);
    void removeParent(Task* parent);
    void removeChild(Task* child);
    void setLocalCoreFinishTime(float time);
    void setWirelessSendingFinishTime(float time);
    void setRemoteCloudFinishTime(float time);
    void setWirelessRecievingFinishTime(float time);
    void setLocalCoreReadyTime(float time);
    void setWirelessSendingReadyTime(float time);
    void setRemoteCloudReadyTime(float time);
    void setWirelessRecievingReadyTime(float time);
    void setPriorityScore(float score);
    void setExecutionFinishTime(float time);
    void setAssignment(int newAssignment);
    void setCoreTask(bool isCore);
    void setExecutionUnitStartTime(int index, int time);
    void setSchedulingState(SchedulingState state);
};

float calculate_energy_consumption(const Task* node, const vector<float>& core_powers, float cloud_sending_power);
float total_energy(const vector<Task*>& nodes, const vector<float>& core_powers, float cloud_sending_power);
float total_time(const vector<Task*>& nodes);
void primary_assignment(const vector<Task*>& nodes);
void task_prioritizing(const vector<Task*>& nodes);
vector<vector<int>> execution_unit_selection(vector<Task*>& nodes);

// Task Groups structure
struct TaskGroups {
    vector<Task*> entry_tasks;
    vector<Task*> non_entry_tasks;
    vector<Task*> cloud_entry_tasks;
};

// Function declarations
vector<int> create_priority_order(const vector<Task*>& nodes);

TaskGroups classify_tasks(const vector<Task*>& nodes, const vector<int>& priority_order);

void schedule_core_entry_task(Task* task, 
                            vector<float>& core_earliest_ready,
                            vector<vector<int>>& sequences,
                            const int k);

void schedule_cloud_entry_tasks(vector<Task*>& cloud_tasks,
                              float& wireless_send_ready,
                              float& wireless_receive_ready,
                              vector<vector<int>>& sequences,
                              const int k);

void schedule_non_entry_tasks(vector<Task*>& non_entry_tasks,
                            vector<float>& core_earliest_ready,
                            float& wireless_send_ready,
                            float& wireless_receive_ready,
                            vector<vector<int>>& sequences,
                            const int k);

vector<vector<int>> execution_unit_selection(vector<Task*>& nodes);

vector<vector<int>> construct_sequence(const vector<Task*>& nodes, 
                                     int targetNodeId, 
                                     int targetLocation, 
                                     vector<vector<int>> seq);
                                
pair<vector<int>, vector<int>> initialize_readiness_tracking(
    const vector<Task*>& nodes, 
    const vector<vector<int>>& sequences
);

void schedule_local_task(
    Task* node,
    vector<float>& local_core_ready_times
);

void schedule_cloud_task(
    Task* node,
    vector<float>& cloud_stage_ready_times
);

void update_node_readiness(
    Task* node,
    const vector<Task*>& nodes,
    const vector<vector<int>>& sequences,
    vector<int>& dependency_ready,
    vector<int>& sequence_ready
);

vector<Task*> kernel_algorithm(
    vector<Task*>& nodes,
    vector<vector<int>>& sequences
);

optional<TaskMigrationState> find_best_migration(
    const vector<tuple<int, int, float, float>>& migration_trials_results,
    float T_init,
    float E_init,
    float T_max_constraint
);

vector<vector<bool>> initialize_migration_choices(const vector<Task*>& nodes);

tuple<float, float> evaluate_migration(
    const vector<Task*>& nodes,
    const vector<vector<int>>& sequences,
    int node_idx,
    int target_execution_unit,
    unordered_map<string, pair<float, float>>& migration_cache
);

string get_cache_key(int node_idx, int target_execution_unit, const vector<Task*>& nodes);

pair<vector<Task*>, vector<vector<int>>> optimize_task_scheduling(
    vector<Task*>& nodes,
    vector<vector<int>>& sequence,
    float T_init_pre_kernel,
    const vector<float>& core_powers = {1.0f, 2.0f, 4.0f},
    float cloud_sending_power = 0.5f
);
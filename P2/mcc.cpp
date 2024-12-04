#include "mcc.h"
#include <limits>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <assert.h>
#include <string>
#include <vector>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <sstream>
using namespace std;

const map<int, vector<float>> CORE_EXECUTION_TIMES = {
    {1, {9, 7, 5}},
    {2, {8, 6, 5}},
    {3, {6, 5, 4}},
    {4, {7, 5, 3}},
    {5, {5, 4, 2}},
    {6, {7, 6, 4}},
    {7, {8, 5, 3}},
    {8, {6, 4, 2}},
    {9, {5, 3, 2}},
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

const vector<float> CLOUD_EXECUTION_TIMES = {3, 1, 1};

Task::Task(int task_id, TaskSet* parent_tasks, TaskSet* child_tasks)
    : id(task_id),
      parents(parent_tasks ? *parent_tasks : TaskSet()),
      children(child_tasks ? *child_tasks : TaskSet()),
      core_execution_times(CORE_EXECUTION_TIMES.at(task_id)),
      cloud_execution_times(CLOUD_EXECUTION_TIMES),
      local_core_finish_time(0.0f),
      wireless_sending_finish_time(0.0f),
      remote_cloud_finish_time(0.0f),
      wireless_recieving_finish_time(0.0f),
      local_core_ready_time(-1.0f),
      wireless_sending_ready_time(-1.0f),
      remote_cloud_ready_time(-1.0f),
      wireless_recieving_ready_time(-1.0f),
      priority_score(0.0f),
      execution_finish_time(0.0f),
      assignment(-2),
      is_core_task(false),
      execution_unit_start_times{-1, -1, -1, -1},
      scheduling_state(SchedulingState::UNSCHEDULED) {}

Task::Task(const Task& other)
    : id(other.id),
      parents(other.parents),  // shallow copy of parents
      children(other.children),  // shallow copy of children
      core_execution_times(other.core_execution_times),
      cloud_execution_times(other.cloud_execution_times),
      local_core_finish_time(other.local_core_finish_time),
      wireless_sending_finish_time(other.wireless_sending_finish_time),
      remote_cloud_finish_time(other.remote_cloud_finish_time),
      wireless_recieving_finish_time(other.wireless_recieving_finish_time),
      local_core_ready_time(other.local_core_ready_time),
      wireless_sending_ready_time(other.wireless_sending_ready_time),
      remote_cloud_ready_time(other.remote_cloud_ready_time),
      wireless_recieving_ready_time(other.wireless_recieving_ready_time),
      priority_score(other.priority_score),
      execution_finish_time(other.execution_finish_time),
      assignment(other.assignment),
      is_core_task(other.is_core_task),
      execution_unit_start_times(other.execution_unit_start_times),
      scheduling_state(other.scheduling_state){}

int Task::getId() const { 
    return id; 
}

const TaskSet& Task::getParents() const { 
    return parents; 
}

const TaskSet& Task::getChildren() const { 
    return children; 
}

const vector<float>& Task::getCoreExecutionTimes() const { 
    return core_execution_times; 
}

const vector<float>& Task::getCloudExecutionTimes() const { 
    return cloud_execution_times; 
}

float Task::getLocalCoreFinishTime() const { 
    return local_core_finish_time; 
}

float Task::getWirelessSendingFinishTime() const { 
    return wireless_sending_finish_time; 
}

float Task::getRemoteCloudFinishTime() const { 
    return remote_cloud_finish_time; 
}

float Task::getWirelessRecievingFinishTime() const { 
    return wireless_recieving_finish_time; 
}

float Task::getLocalCoreReadyTime() const { 
    return local_core_ready_time; 
}

float Task::getWirelessSendingReadyTime() const { 
    return wireless_sending_ready_time; 
}

float Task::getRemoteCloudReadyTime() const { 
    return remote_cloud_ready_time; 
}

float Task::getWirelessRecievingReadyTime() const { 
    return wireless_recieving_ready_time; 
}

float Task::getPriorityScore() const { 
    return priority_score; 
}

int Task::getAssignment() const { 
    return assignment; 
}

bool Task::isCoreTask() const { 
    return is_core_task; 
}

const array<int, 4>& Task::getExecutionUnitStartTimes() const { 
    return execution_unit_start_times; 
}

void Task::setCoreExecutionTimes(const vector<float>& times) {
    core_execution_times = times;
}

void Task::setCloudExecutionTimes(const vector<float>& times) {
    cloud_execution_times = times;
}

void Task::addParent(Task* parent) {
    if (parent && parent != this) {
        parents.insert(parent);
        parent->children.insert(this);
    }
}

void Task::addChild(Task* child) {
    if (child && child != this) {
        children.insert(child);
        child->parents.insert(this);
    }
}

void Task::removeParent(Task* parent) {
    if (parent) {
        parents.erase(parent);
        parent->children.erase(this);
    }
}

void Task::removeChild(Task* child) {
    if (child) {
        children.erase(child);
        child->parents.erase(this);
    }
}

void Task::setLocalCoreFinishTime(float time) {
    if (time >= 0.0f) {
        local_core_finish_time = time;
    }
}

void Task::setWirelessSendingFinishTime(float time) {
    if (time >= 0.0f) {
        wireless_sending_finish_time = time;
    }
}

void Task::setRemoteCloudFinishTime(float time) {
    if (time >= 0.0f) {
        remote_cloud_finish_time = time;
    }
}

void Task::setWirelessRecievingFinishTime(float time) {
    if (time >= 0.0f) {
        wireless_recieving_finish_time = time;
    }
}

void Task::setLocalCoreReadyTime(float time) {
    local_core_ready_time = time;
}

void Task::setWirelessSendingReadyTime(float time) {
    wireless_sending_ready_time = time;
}

void Task::setRemoteCloudReadyTime(float time) {
    remote_cloud_ready_time = time;
}

void Task::setWirelessRecievingReadyTime(float time) {
    wireless_recieving_ready_time = time;
}

void Task::setPriorityScore(float score) {
    priority_score = score;
}

void Task::setAssignment(int newAssignment) {
    assignment = newAssignment;
}

void Task::setCoreTask(bool isCore) {
    is_core_task = isCore;
}

void Task::setExecutionUnitStartTime(int index, int time) {
    if (index >= 0 && index < 4) {
        execution_unit_start_times[index] = time;
    }
}

SchedulingState Task::getSchedulingState() const {
    return scheduling_state;
}

void Task::setSchedulingState(SchedulingState state) {
    scheduling_state = state;
}

float Task::getExecutionFinishTime() const {
    return execution_finish_time;
}

void Task::setExecutionFinishTime(float time) {
    execution_finish_time = time;
}

float total_time(const vector<Task*>& tasks) {
    float max_time = 0.0f;
    for (const auto& task : tasks) {
        if (task->getChildren().empty()) {
            float task_time = max(
                task->getLocalCoreFinishTime(),
                task->getWirelessRecievingFinishTime()
            );
            if (task_time > max_time) {
                max_time = task_time;
            }
        }
    }
    return max_time;
}

float calculate_energy_consumption(const Task* node, const vector<float>& core_powers, float cloud_sending_power) {
    if (node->isCoreTask()) {
        return core_powers[node->getAssignment()] * node->getCoreExecutionTimes()[node->getAssignment()];
    } else {
        return cloud_sending_power * node->getCloudExecutionTimes()[0];
    }
}

float total_energy(const vector<Task*>& nodes, const vector<float>& core_powers, float cloud_sending_power) {
    float total = 0.0f;
    for (const auto& node : nodes) {
        total += calculate_energy_consumption(node, core_powers, cloud_sending_power);
    }
    return total;
}

void primary_assignment(const vector<Task*>& nodes) {
    for (auto& node : nodes) {
        // Calculate minimum local execution time
        float t_l_min = *min_element(
            node->getCoreExecutionTimes().begin(),
            node->getCoreExecutionTimes().end()
        );
        // Calculate total remote execution time
        float t_re = node->getCloudExecutionTimes()[0] + node->getCloudExecutionTimes()[1] + node->getCloudExecutionTimes()[2];
        // If remote execution is faster, assign to cloud
        node->setCoreTask(!(t_re < t_l_min));
    }
}

void task_prioritizing(const vector<Task*>& nodes) {
    // Calculate computation costs (wi)
    vector<float> w(nodes.size(), 0.0f);
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        if (!node->isCoreTask()) {  // Cloud task
            // Equation (13): wi = Tre_i for cloud tasks
            const auto& cloud_times = node->getCloudExecutionTimes();
            w[i] = cloud_times[0] + cloud_times[1] + cloud_times[2];
        } else {  // Local task
            // Equation (14): wi = avg(1≤k≤K) Tl,k_i for local tasks
            const auto& core_times = node->getCoreExecutionTimes();
            w[i] = accumulate(core_times.begin(), core_times.end(), 0.0f) / 
                   core_times.size();
        }
    }
    unordered_map<int, float> computed_priority_scores;

    // Recursive lambda for priority calculation
    function<float(Task*)> calculate_priority = [&](Task* task) -> float {
        // Check if already calculated
        if (auto it = computed_priority_scores.find(task->getId()); 
            it != computed_priority_scores.end()) {
            return it->second;
        }

        // Base case: exit task
        // Equation (16): priority(vi) = wi for exit tasks
        if (task->getChildren().empty()) {
            float priority = w[task->getId() - 1];
            computed_priority_scores[task->getId()] = priority;
            return priority;
        }

        // Recursive case: Equation (15)
        // priority(vi) = wi + max(vj∈succ(vi)) priority(vj)
        float max_successor_priority = 0.0f;
        for (const auto& successor : task->getChildren()) {
            max_successor_priority = max(
                max_successor_priority, 
                calculate_priority(successor)
            );
        }

        float task_priority = w[task->getId() - 1] + max_successor_priority;
        computed_priority_scores[task->getId()] = task_priority;
        return task_priority;
    };

    // Calculate priorities for all nodes
    for (const auto& task : nodes) {
        calculate_priority(task);
    }

    // Update priority scores
    for (auto& node : nodes) {
        node->setPriorityScore(computed_priority_scores[node->getId()]);
    }
}

// Creates priority ordered list of task IDs
vector<int> create_priority_order(const vector<Task*>& nodes) {
    vector<pair<float, int>> node_priority_list;
    node_priority_list.reserve(nodes.size());
    
    for (const auto& node : nodes) {
        node_priority_list.emplace_back(node->getPriorityScore(), node->getId());
    }
    
    sort(node_priority_list.begin(), node_priority_list.end(), greater<>());
    
    vector<int> priority_order;
    priority_order.reserve(node_priority_list.size());
    for (const auto& [_, id] : node_priority_list) {
        priority_order.push_back(id);
    }
    return priority_order;
}

// Separates tasks into entry and non-entry groups
TaskGroups classify_tasks(const vector<Task*>& nodes, const vector<int>& priority_order) {
    TaskGroups groups;
    
    for (int node_id : priority_order) {
        Task* node = nodes[node_id - 1];
        if (node->getParents().empty()) {
            groups.entry_tasks.push_back(node);
        } else {
            groups.non_entry_tasks.push_back(node);
        }
    }
    
    return groups;
}

void schedule_core_entry_task(Task* task, vector<float>& core_earliest_ready, vector<vector<int>>& sequences, const int k) {
    // For local core tasks, find the best core and earliest possible start time
    float best_finish_time = numeric_limits<float>::infinity();
    int best_core = -1;
    float best_start_time = numeric_limits<float>::infinity();
    
    for (int core = 0; core < k; ++core) {
        // Task must start after core's previous task finishes
        float start_time = core_earliest_ready[core];
        float finish_time = start_time + task->getCoreExecutionTimes()[core];
        
        if (finish_time < best_finish_time) {
            best_finish_time = finish_time;
            best_core = core;
            best_start_time = start_time;
        }
    }
    
    // Schedule task on selected core
    task->setLocalCoreFinishTime(best_finish_time);
    task->setExecutionFinishTime(best_finish_time);
    
    // Set all execution unit start times to -1 first
    for (int i = 0; i < 4; ++i) {
        task->setExecutionUnitStartTime(i, -1);
    }
    // Set the selected core's start time
    task->setExecutionUnitStartTime(best_core, best_start_time);
    
    core_earliest_ready[best_core] = best_finish_time;
    task->setAssignment(best_core);
    task->setSchedulingState(SchedulingState::SCHEDULED);
    sequences[best_core].push_back(task->getId());
}

void schedule_cloud_entry_tasks(vector<Task*>& cloud_tasks, float& wireless_send_ready, 
                              float& wireless_receive_ready, vector<vector<int>>& sequences, const int k) {
    for (auto* task : cloud_tasks) {
        // Schedule sending phase
        task->setWirelessSendingReadyTime(wireless_send_ready);
        task->setWirelessSendingFinishTime(wireless_send_ready + task->getCloudExecutionTimes()[0]);
        wireless_send_ready = task->getWirelessSendingFinishTime();
        
        // Schedule cloud computation
        task->setRemoteCloudReadyTime(task->getWirelessSendingFinishTime());
        task->setRemoteCloudFinishTime(task->getRemoteCloudReadyTime() + task->getCloudExecutionTimes()[1]);
        
        // Schedule receiving phase
        task->setWirelessRecievingReadyTime(task->getRemoteCloudFinishTime());
        task->setWirelessRecievingFinishTime(
            max(wireless_receive_ready, task->getWirelessRecievingReadyTime()) + 
            task->getCloudExecutionTimes()[2]
        );
        wireless_receive_ready = task->getWirelessRecievingFinishTime();
        
        // Update task parameters
        task->setExecutionFinishTime(task->getWirelessRecievingFinishTime());
        task->setLocalCoreFinishTime(0.0f);
        
        // Set all execution unit start times to -1 first
        for (int i = 0; i < 4; ++i) {
            task->setExecutionUnitStartTime(i, -1);
        }
        // Set the cloud start time
        task->setExecutionUnitStartTime(k, task->getWirelessSendingReadyTime());
        
        task->setAssignment(k);
        task->setSchedulingState(SchedulingState::SCHEDULED);
        sequences[k].push_back(task->getId());
    }
}

void schedule_non_entry_tasks(vector<Task*>& non_entry_tasks, 
                           vector<float>& core_earliest_ready,
                           float& wireless_send_ready,
                           float& wireless_receive_ready,
                           vector<vector<int>>& sequences,
                           const int k) {

   for (auto* task : non_entry_tasks) {
       // Calculate ready times based on parent task completions
       float local_core_ready = 0.0f;
       for (const auto* parent : task->getParents()) {
           local_core_ready = max(local_core_ready, 
                                max(parent->getLocalCoreFinishTime(), 
                                    parent->getWirelessRecievingFinishTime()));
       }
       task->setLocalCoreReadyTime(local_core_ready);

       float wireless_send_ready_time = wireless_send_ready;
       for (const auto* parent : task->getParents()) {
           wireless_send_ready_time = max(wireless_send_ready_time,
                                        max(parent->getLocalCoreFinishTime(),
                                            parent->getWirelessSendingFinishTime()));
       }
       task->setWirelessSendingReadyTime(wireless_send_ready_time);

       // Calculate wireless sending finish time
       task->setWirelessSendingFinishTime(
           task->getWirelessSendingReadyTime() + task->getCloudExecutionTimes()[0]);

       // Calculate cloud ready and finish times
       float remote_cloud_ready = task->getWirelessSendingFinishTime();
       for (const auto* parent : task->getParents()) {
           remote_cloud_ready = max(remote_cloud_ready, 
                                  parent->getRemoteCloudFinishTime());
       }
       task->setRemoteCloudReadyTime(remote_cloud_ready);
       
       task->setRemoteCloudFinishTime(
           task->getRemoteCloudReadyTime() + task->getCloudExecutionTimes()[1]);
           
       task->setWirelessRecievingReadyTime(task->getRemoteCloudFinishTime());
       task->setWirelessRecievingFinishTime(
           max(wireless_receive_ready, task->getWirelessRecievingReadyTime()) + 
           task->getCloudExecutionTimes()[2]);

       if (!task->isCoreTask()) {
           // Schedule on cloud
           task->setExecutionFinishTime(task->getWirelessRecievingFinishTime());
           task->setLocalCoreFinishTime(0.0f);
           
           // Reset all execution unit start times to -1
           for (int i = 0; i < 4; ++i) {
               task->setExecutionUnitStartTime(i, -1);
           }
           task->setExecutionUnitStartTime(k, task->getWirelessSendingReadyTime());
           
           task->setAssignment(k);
           wireless_send_ready = task->getWirelessSendingFinishTime();
           wireless_receive_ready = task->getWirelessRecievingFinishTime();
           sequences[k].push_back(task->getId());
       } else {
           // Find best local core
           float best_finish_time = numeric_limits<float>::infinity();
           int best_core = -1;
           float best_start_time = numeric_limits<float>::infinity();
           
           for (int core = 0; core < k; ++core) {
               float start_time = max(task->getLocalCoreReadyTime(), core_earliest_ready[core]);
               float finish_time = start_time + task->getCoreExecutionTimes()[core];
               
               if (finish_time < best_finish_time) {
                   best_finish_time = finish_time;
                   best_core = core;
                   best_start_time = start_time;
               }
           }
           
           // Compare with potential cloud execution time
           if (best_finish_time <= task->getWirelessRecievingFinishTime()) {
               // Execute locally
               task->setLocalCoreFinishTime(best_finish_time);
               task->setExecutionFinishTime(best_finish_time);
               task->setWirelessRecievingFinishTime(0.0f);
               
               // Reset all execution unit start times to -1
               for (int i = 0; i < 4; ++i) {
                   task->setExecutionUnitStartTime(i, -1);
               }
               task->setExecutionUnitStartTime(best_core, best_start_time);
               
               core_earliest_ready[best_core] = best_finish_time;
               task->setAssignment(best_core);
               sequences[best_core].push_back(task->getId());
           } else {
               // Execute on cloud
               task->setExecutionFinishTime(task->getWirelessRecievingFinishTime());
               task->setLocalCoreFinishTime(0.0f);
               
               // Reset all execution unit start times to -1
               for (int i = 0; i < 4; ++i) {
                   task->setExecutionUnitStartTime(i, -1);
               }
               task->setExecutionUnitStartTime(k, task->getWirelessSendingReadyTime());
               
               task->setAssignment(k);
               task->setCoreTask(false);
               wireless_send_ready = task->getWirelessSendingFinishTime();
               wireless_receive_ready = task->getWirelessRecievingFinishTime();
               sequences[k].push_back(task->getId());
           }
       }
       
       task->setSchedulingState(SchedulingState::SCHEDULED);
   }
}

vector<vector<int>> execution_unit_selection(vector<Task*>& nodes) {
   const int k = 3;  // Number of cores
   vector<vector<int>> sequences(k + 1);  // k cores + cloud
   
   // Track resource availability
   vector<float> core_earliest_ready(k, 0.0f);
   float wireless_send_ready = 0.0f;
   float wireless_receive_ready = 0.0f;
   
   // Create and sort priority list
   auto priority_order = create_priority_order(nodes);
   
   // Classify tasks
   auto task_groups = classify_tasks(nodes, priority_order);
   
   // Handle core entry tasks
   for (auto* task : task_groups.entry_tasks) {
       if (task->isCoreTask()) {
           schedule_core_entry_task(task, core_earliest_ready, sequences, k);
       } else {
           task_groups.cloud_entry_tasks.push_back(task);
       }
   }
   
   // Handle cloud entry tasks
   schedule_cloud_entry_tasks(task_groups.cloud_entry_tasks, 
                            wireless_send_ready, 
                            wireless_receive_ready, 
                            sequences, k);
   
   // Handle non-entry tasks
   schedule_non_entry_tasks(task_groups.non_entry_tasks,
                          core_earliest_ready,
                          wireless_send_ready,
                          wireless_receive_ready,
                          sequences,
                          k);

    for (const auto* task : nodes) {
        assert(task->getSchedulingState() == SchedulingState::SCHEDULED);
    }
   
   return sequences;
}

vector<vector<int>> construct_sequence(const vector<Task*>& nodes, 
                                     int targetNodeId, 
                                     int targetLocation, 
                                     vector<vector<int>> seq) {
    // Step 1: Map node IDs to node objects for quick lookup
    unordered_map<int, Task*> node_id_to_node;
    for (auto* node : nodes) {
        node_id_to_node[node->getId()] = node;
    }

    // Step 2: Validate inputs and locate the target node
    auto target_node = node_id_to_node[targetNodeId];

    // Step 3: Determine the ready time of the target node
    float target_node_rt = target_node->isCoreTask() ? 
                          target_node->getLocalCoreReadyTime() : 
                          target_node->getWirelessSendingReadyTime();

    // Step 4: Remove the target node from its original sequence
    int original_assignment = target_node->getAssignment();
    auto& original_seq = seq[original_assignment];
    original_seq.erase(
        remove(original_seq.begin(), original_seq.end(), target_node->getId()),
        original_seq.end()
    );

    // Step 5: Prepare the new sequence for insertion
    auto& new_sequence_nodes_list = seq[targetLocation];

    // Precompute start times for the new sequence's nodes
    vector<float> start_times;
    start_times.reserve(new_sequence_nodes_list.size());
    for (int node_id : new_sequence_nodes_list) {
        start_times.push_back(
            node_id_to_node[node_id]->getExecutionUnitStartTimes()[targetLocation]
        );
    }

    // Step 6: Use lower_bound to find the insertion index (C++ equivalent of bisect.bisect_left)
    auto insertion_iter = lower_bound(start_times.begin(), 
                                    start_times.end(), 
                                    target_node_rt);
    size_t insertion_index = insertion_iter - start_times.begin();

    // Step 7: Insert the target node at the correct index
    new_sequence_nodes_list.insert(
        new_sequence_nodes_list.begin() + insertion_index,
        target_node->getId()
    );

    // Step 8: Update the target node's assignment and status
    target_node->setAssignment(targetLocation);
    target_node->setCoreTask(targetLocation != 3);  // Location 3 is the cloud

    return seq;
}

void update_node_readiness(
    Task* node,
    const vector<Task*>& nodes,
    const vector<vector<int>>& sequences,
    vector<int>& dependency_ready,
    vector<int>& sequence_ready
) {
    if (node->getSchedulingState() != SchedulingState::KERNEL_SCHEDULED) {
        // Update dependency readiness
        int count = 0;
        for (const auto& parent : node->getParents()) {
            if (parent->getSchedulingState() != SchedulingState::KERNEL_SCHEDULED) {
                count++;
            }
        }
        dependency_ready[node->getId() - 1] = count;

        // Update sequence readiness
        for (const auto& sequence : sequences) {
            auto it = find(sequence.begin(), sequence.end(), node->getId());
            if (it != sequence.end()) {
                if (it != sequence.begin()) {
                    Task* prev_node = nodes[*(it - 1) - 1];
                    sequence_ready[node->getId() - 1] = 
                        (prev_node->getSchedulingState() != SchedulingState::KERNEL_SCHEDULED) ? 1 : 0;
                } else {
                    sequence_ready[node->getId() - 1] = 0;
                }
                break;
            }
        }
    }
}

void schedule_local_task(Task* node, vector<float>& local_core_ready_times) {
   // Calculate ready time based on parent completion
   float parent_ready_time = 0.0f;
   if (!node->getParents().empty()) {
       for (const auto& parent : node->getParents()) {
           parent_ready_time = max(
               parent_ready_time,
               max(parent->getLocalCoreFinishTime(), 
                   parent->getWirelessRecievingFinishTime())
           );
       }
   }
   node->setLocalCoreReadyTime(parent_ready_time);
   
   // Schedule on assigned core
   int core_index = node->getAssignment();
   
   // Reset execution unit start times
   for (int i = 0; i < 4; ++i) {
       node->setExecutionUnitStartTime(i, -1);
   }
   
   // Set start time on assigned core
   float start_time = max(local_core_ready_times[core_index], 
                         node->getLocalCoreReadyTime());
   node->setExecutionUnitStartTime(core_index, start_time);
   
   // Calculate and set finish time
   float finish_time = start_time + node->getCoreExecutionTimes()[core_index];
   node->setLocalCoreFinishTime(finish_time);
   
   // Update core ready time
   local_core_ready_times[core_index] = finish_time;
   
   // Clear cloud-related timings
   node->setWirelessSendingFinishTime(-1);
   node->setRemoteCloudFinishTime(-1);
   node->setWirelessRecievingFinishTime(-1);
}

void schedule_cloud_task(Task* node, vector<float>& cloud_stage_ready_times) {
   // Calculate wireless sending ready time
   float send_ready_time = 0.0f;
   if (!node->getParents().empty()) {
       for (const auto& parent : node->getParents()) {
           send_ready_time = max(
               send_ready_time,
               max(parent->getLocalCoreFinishTime(),
                   parent->getWirelessSendingFinishTime())
           );
       }
   }
   node->setWirelessSendingReadyTime(send_ready_time);

   // Initialize start times
   for (int i = 0; i < 4; ++i) {
       node->setExecutionUnitStartTime(i, -1);
   }
   // Set cloud start time (index 3)
   float cloud_start = max(cloud_stage_ready_times[0], 
                          node->getWirelessSendingReadyTime());
   node->setExecutionUnitStartTime(3, cloud_start);

   // Schedule wireless sending
   float send_finish_time = cloud_start + node->getCloudExecutionTimes()[0];
   node->setWirelessSendingFinishTime(send_finish_time);
   cloud_stage_ready_times[0] = send_finish_time;

   // Schedule cloud processing
   float cloud_ready_time = send_finish_time;
   if (!node->getParents().empty()) {
       for (const auto& parent : node->getParents()) {
           cloud_ready_time = max(cloud_ready_time,
                                parent->getRemoteCloudFinishTime());
       }
   }
   node->setRemoteCloudReadyTime(cloud_ready_time);
   
   float cloud_finish = max(cloud_stage_ready_times[1], cloud_ready_time) + 
                       node->getCloudExecutionTimes()[1];
   node->setRemoteCloudFinishTime(cloud_finish);
   cloud_stage_ready_times[1] = cloud_finish;

   // Schedule wireless receiving
   node->setWirelessRecievingReadyTime(cloud_finish);
   float receive_finish = max(cloud_stage_ready_times[2], cloud_finish) + 
                         node->getCloudExecutionTimes()[2];
   node->setWirelessRecievingFinishTime(receive_finish);
   cloud_stage_ready_times[2] = receive_finish;

   // Clear local timing
   node->setLocalCoreFinishTime(-1);
}

pair<vector<int>, vector<int>> initialize_readiness_tracking(
   const vector<Task*>& nodes, 
   const vector<vector<int>>& sequences
) {
   // Initialize dependency_ready with the number of parent tasks for each node
   vector<int> dependency_ready;
   dependency_ready.reserve(nodes.size());
   for (const auto& node : nodes) {
       dependency_ready.push_back(node->getParents().size());
   }

   // Initialize sequence_ready to -1 for all nodes
   vector<int> sequence_ready(nodes.size(), -1);

   // Mark first node in each sequence as ready in terms of sequence order
   for (const auto& sequence : sequences) {
       if (!sequence.empty()) {
           sequence_ready[sequence[0] - 1] = 0;
       }
       else {
           // Handle empty sequences if necessary
           continue;
       }
   }
   
   return {dependency_ready, sequence_ready};
}

vector<Task*> kernel_algorithm(vector<Task*>& nodes, vector<vector<int>>& sequences) {
   // Initialize timing trackers
   vector<float> local_core_ready_times(3, 0.0f);  // [0] * 3 
   vector<float> cloud_stage_ready_times(3, 0.0f);  // [0] * 3

   // Initialize readiness tracking
   auto [dependency_ready, sequence_ready] = initialize_readiness_tracking(nodes, sequences);
   
   // Initialize processing queue with ready nodes
   deque<Task*> queue;
   for (auto* node : nodes) {
       if (sequence_ready[node->getId() - 1] == 0) {
           bool all_parents_scheduled = true;
           for (const auto* parent : node->getParents()) {
               if (parent->getSchedulingState() != SchedulingState::KERNEL_SCHEDULED) {
                   all_parents_scheduled = false;
                   break;
               }
           }
           if (all_parents_scheduled) {
               queue.push_back(node);
           }
       }
   }
   
   // Main scheduling loop
   while (!queue.empty()) {
       Task* current_node = queue.front();
       queue.pop_front();
       current_node->setSchedulingState(SchedulingState::KERNEL_SCHEDULED);
       
       // Schedule task based on type
       if (current_node->isCoreTask()) {
           schedule_local_task(current_node, local_core_ready_times);
       } else {
           schedule_cloud_task(current_node, cloud_stage_ready_times);
       }
       
       // Update readiness status for remaining nodes
       for (auto* node : nodes) {
           update_node_readiness(node, nodes, sequences, dependency_ready, sequence_ready);
           
           // Add newly ready nodes to queue
           if (dependency_ready[node->getId() - 1] == 0 && 
               sequence_ready[node->getId() - 1] == 0 && 
               node->getSchedulingState() != SchedulingState::KERNEL_SCHEDULED) {
                   
               // Check if node is already in queue
               if (find(queue.begin(), queue.end(), node) == queue.end()) {
                   queue.push_back(node);
               }
           }
       }
   }
   
   // Reset scheduling status
   for (auto* node : nodes) {
       node->setSchedulingState(SchedulingState::UNSCHEDULED);
   }
   
   return nodes;
}

string get_cache_key(int node_idx, int target_execution_unit, const vector<Task*>& nodes) {
    ostringstream key;
    key << node_idx << ',' << target_execution_unit << ',';
    for (const auto* node : nodes) {
        key << node->getAssignment() << ',';
    }
    return key.str();
}

tuple<float, float> evaluate_migration(
    const vector<Task*>& nodes,
    const vector<vector<int>>& sequences,
    int node_idx,
    int target_execution_unit,
    unordered_map<string, pair<float, float>>& migration_cache
) {
    // Generate cache key
    string cache_key = get_cache_key(node_idx, target_execution_unit, nodes);
    
    // Check cache
    auto cache_it = migration_cache.find(cache_key);
    if (cache_it != migration_cache.end()) {
        return {cache_it->second.first, cache_it->second.second};
    }
    
    // Create copies for evaluation
    auto nodes_copy = nodes; // Shallow copy is sufficient as we only modify scheduling state
    auto seq_copy = sequences;
    
    // Apply migration and evaluate
    seq_copy = construct_sequence(nodes_copy, node_idx + 1, target_execution_unit, seq_copy);
    kernel_algorithm(nodes_copy, seq_copy);
    
    float current_T = total_time(nodes_copy);
    float current_E = total_energy(nodes_copy, core_powers, cloud_sending_power);
    
    // Cache result
    migration_cache[cache_key] = {current_T, current_E};
    
    return {current_T, current_E};
}

vector<vector<bool>> initialize_migration_choices(const vector<Task*>& nodes) {
    vector<vector<bool>> choices(nodes.size(), vector<bool>(4, false));
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i]->getAssignment() == 3) {  // Cloud-assigned node
            for (int j = 0; j < 4; ++j) {
                choices[i][j] = true;
            }
        } else {
            choices[i][nodes[i]->getAssignment()] = true;
        }
    }
    
    return choices;
}

std::tuple<float, float> evaluate_migration(
    const std::vector<Task*>& nodes,
    const std::vector<std::vector<int>>& seqs,
    int node_idx,
    int target_execution_unit
) {
    // Check cache
    std::string cache_key = get_cache_key(node_idx, target_execution_unit);
    auto it = migration_cache.find(cache_key);
    if (it != migration_cache.end()) {
        return it->second;
    }
    
    // Create copies
    std::vector<std::vector<int>> seq_copy;
    seq_copy.reserve(seqs.size());
    for (const auto& seq : seqs) {
        seq_copy.push_back(seq);
    }
    
    std::vector<Task*> nodes_copy;
    nodes_copy.reserve(nodes.size());
    for (const auto* node : nodes) {
        nodes_copy.push_back(new Task(*node));
    }
    
    seq_copy = construct_sequence(nodes_copy, node_idx + 1, target_execution_unit, seq_copy);
    kernel_algorithm(nodes_copy, seq_copy);
    
    float current_T = total_time(nodes_copy);
    float current_E = total_energy(nodes_copy, core_powers, cloud_sending_power);
    
    // Cache result
    migration_cache[cache_key] = std::make_pair(current_T, current_E);
    
    // Cleanup
    for (auto* node : nodes_copy) {
        delete node;
    }
    
    return {current_T, current_E};
}

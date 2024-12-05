#include "initial_task_scheduler.h"
#include <algorithm>

InitialTaskScheduler::InitialTaskScheduler(std::vector<Task>& tasks, int num_cores)
   : tasks(tasks)
   , k(num_cores)
{
   // Resource timing tracking (Section II.B and II.C)
   core_earliest_ready = std::vector<int>(k, 0);
   ws_ready = 0;
   wr_ready = 0;
   
   // Sk sequence sets from Section III.B
   // Tracks task execution sequences for each resource (cores + cloud)
   sequences = std::vector<std::vector<int>>(k + 1);
}

std::vector<int> InitialTaskScheduler::getPriorityOrderedTasks() const {
   std::vector<std::pair<int, int>> task_priority_list;
   
   // Create list of (priority_score, id) pairs
   for (const auto& task : tasks) {
       task_priority_list.push_back({task.getPriorityScore(), task.getId()});
   }
   
   // Sort in reverse order, matching Python tuple comparison behavior
   std::sort(task_priority_list.begin(), task_priority_list.end(), std::greater<std::pair<int, int>>());
   
   // Extract just the task IDs
   std::vector<int> result;
   for (const auto& pair : task_priority_list) {
       result.push_back(pair.second);
   }
   
   return result;
}

std::pair<std::vector<Task*>, std::vector<Task*>> InitialTaskScheduler::classifyEntryTasks(const std::vector<int>& priority_order) {
    std::vector<Task*> entry_tasks;
    std::vector<Task*> non_entry_tasks;

    for (int task_id : priority_order) {
        Task& task = tasks[task_id - 1];
        
        if (task.getPredTasks().empty()) {
            entry_tasks.push_back(&task);
        } else {
            non_entry_tasks.push_back(&task);
        }
    }
    return {entry_tasks, non_entry_tasks};
}

std::tuple<int, int, int> InitialTaskScheduler::identifyOptimalLocalCore(Task& task, int ready_time) {
    int best_finish_time = std::numeric_limits<int>::max();
    int best_core = -1;
    int best_start_time = std::numeric_limits<int>::max();

    const auto& core_times = task.getCoreExecutionTimes(); // Get reference once
    for (int core = 0; core < k; core++) {
        int start_time = std::max(ready_time, core_earliest_ready[core]);
        int finish_time = start_time + core_times[core];
        
        if (finish_time < best_finish_time) {
            best_finish_time = finish_time;
            best_core = core;
            best_start_time = start_time;
        }
    }

    return std::make_tuple(best_core, best_start_time, best_finish_time);
}

void InitialTaskScheduler::scheduleOnLocalCore(Task& task, int core, int start_time, int finish_time) {
    task.setFinishTimeLocal(finish_time);
    task.setExecutionFinishTime(finish_time);
    
    // Match Python's direct initialization and setting
    for (int i = 0; i < k + 1; i++) {
        task.setExecutionUnitTaskStartTime(i, -1);
    }
    task.setExecutionUnitTaskStartTime(core, start_time);
    
    core_earliest_ready[core] = finish_time;
    task.setAssignment(core);
    task.setSchedulingState(SchedulingState::SCHEDULED);
    sequences[core].push_back(task.getId());
}

std::tuple<int, int, int, int, int, int> InitialTaskScheduler::calculateCloudPhaseTiming(Task& task) {
    int send_ready = task.getReadyTimeWirelessSend();
    const auto& cloud_times = task.getCloudExecutionTimes();
    int send_finish = send_ready + cloud_times[0];
    int cloud_ready = send_finish;
    int cloud_finish = cloud_ready + cloud_times[1];
    int receive_ready = cloud_finish;
    int receive_finish = std::max(wr_ready, receive_ready) + cloud_times[2];

    return std::make_tuple(send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish);
}

void InitialTaskScheduler::scheduleOnCloud(Task& task, int send_ready, int send_finish, int cloud_ready, int cloud_finish, int receive_ready, int receive_finish) {
    task.setReadyTimeWirelessSend(send_ready);    
    task.setFinishTimeWirelessSend(send_finish);  
    task.setReadyTimeCloud(cloud_ready);     
    task.setFinishTimeCloud(cloud_finish);   
    task.setReadyTimeWirelessReceive(receive_ready);  
    task.setFinishTimeWirelessReceive(receive_finish);
    task.setExecutionFinishTime(receive_finish);
    task.setFinishTimeLocal(0);
    
    for (int i = 0; i < k + 1; i++) {
        task.setExecutionUnitTaskStartTime(i, -1);
    }
    task.setExecutionUnitTaskStartTime(k, send_ready);
    
    task.setAssignment(k);
    task.setSchedulingState(SchedulingState::SCHEDULED);
    ws_ready = send_finish;
    wr_ready = receive_finish;
    sequences[k].push_back(task.getId());
}

void InitialTaskScheduler::scheduleEntryTasks(const std::vector<Task*>& entry_tasks) {
    std::vector<Task*> cloud_entry_tasks;

    for (Task* task : entry_tasks) {
        if (task->isCoreTask()) {
            auto [core, start_time, finish_time] = identifyOptimalLocalCore(*task);
            scheduleOnLocalCore(*task, core, start_time, finish_time);
        } else {
            cloud_entry_tasks.push_back(task);
        }
    }

    for (Task* task : cloud_entry_tasks) {
        task->setReadyTimeWirelessSend(ws_ready);
        auto timing = calculateCloudPhaseTiming(*task);
        std::apply([this, task](auto... args) {
            scheduleOnCloud(*task, args...);
        }, timing);
    }
}

void InitialTaskScheduler::calculateNonEntryTaskReadyTimes(Task& task) {
    const auto& pred_tasks = task.getPredTasks();
    
    // Calculate local core ready time RTi^l (equation 3)
    int local_ready = 0;
    if (!pred_tasks.empty()) {
        for (const Task* pred_task : pred_tasks) {
            local_ready = std::max(local_ready, 
                std::max(pred_task->getFinishTimeLocal(), 
                        pred_task->getFinishTimeWirelessReceive()));
        }
    }
    task.setReadyTimeLocal(std::max(local_ready, 0));

    // Calculate cloud sending ready time RTi^ws (equation 4)
    int cloud_ready = ws_ready;
    if (!pred_tasks.empty()) {
        for (const Task* pred_task : pred_tasks) {
            cloud_ready = std::max(cloud_ready,
                std::max(pred_task->getFinishTimeLocal(),
                        pred_task->getFinishTimeWirelessSend()));
        }
    }
    task.setReadyTimeWirelessSend(cloud_ready);
}

void InitialTaskScheduler::scheduleNonEntryTasks(const std::vector<Task*>& non_entry_tasks) {
    for (Task* task : non_entry_tasks) {
        calculateNonEntryTaskReadyTimes(*task);
        
        if (!task->isCoreTask()) {
            auto timing = calculateCloudPhaseTiming(*task);
            std::apply([this, task](auto... args) {
                scheduleOnCloud(*task, args...);
            }, timing);
        } else {
            auto [core, start_time, finish_time] = 
                identifyOptimalLocalCore(*task, task->getReadyTimeLocal());
            
            auto timing = calculateCloudPhaseTiming(*task);
            int cloud_finish_time = std::get<5>(timing);
            
            if (finish_time <= cloud_finish_time) {
                scheduleOnLocalCore(*task, core, start_time, finish_time);
            } else {
                task->setIsCoreTask(false);
                std::apply([this, task](auto... args) {
                    scheduleOnCloud(*task, args...);
                }, timing);
            }
        }
    }
}

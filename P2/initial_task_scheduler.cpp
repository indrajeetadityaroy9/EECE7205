#include <algorithm>
#include "initial_task_scheduler.h"
using namespace std;

InitialTaskScheduler::InitialTaskScheduler(vector<Task>& tasks, int num_cores)
   : tasks(tasks)
   , k(num_cores)
{
   core_earliest_ready = vector<int>(k, 0);
   ws_ready = 0;
   wr_ready = 0;
   sequences = vector<vector<int>>(k + 1);
}

vector<int> InitialTaskScheduler::getPriorityOrderedTasks() const {
   vector<pair<int, int>> task_priority_list;
   
   for (const auto& task : tasks) {
       task_priority_list.push_back({task.getPriorityScore(), task.getId()});
   }
   
   sort(task_priority_list.begin(), task_priority_list.end(), greater<pair<int, int>>());

   vector<int> result;
   for (const auto& pair : task_priority_list) {
       result.push_back(pair.second);
   }
   
   return result;
}

pair<vector<Task*>, vector<Task*>> InitialTaskScheduler::classifyEntryTasks(const vector<int>& priority_order) {
    vector<Task*> entry_tasks;
    vector<Task*> non_entry_tasks;

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

tuple<int, int, int> InitialTaskScheduler::identifyOptimalLocalCore(Task& task, int ready_time) {
    int best_finish_time = numeric_limits<int>::max();
    int best_core = -1;
    int best_start_time = numeric_limits<int>::max();

    const auto& core_times = task.getCoreExecutionTimes();
    for (int core = 0; core < k; core++) {
        int start_time = max(ready_time, core_earliest_ready[core]);
        int finish_time = start_time + core_times[core];
        
        if (finish_time < best_finish_time) {
            best_finish_time = finish_time;
            best_core = core;
            best_start_time = start_time;
        }
    }

    return make_tuple(best_core, best_start_time, best_finish_time);
}

void InitialTaskScheduler::scheduleOnLocalCore(Task& task, int core, int start_time, int finish_time) {
    task.setFinishTimeLocal(finish_time);
    task.setExecutionFinishTime(finish_time);
    
    for (int i = 0; i < k + 1; i++) {
        task.setExecutionUnitTaskStartTime(i, -1);
    }
    task.setExecutionUnitTaskStartTime(core, start_time);
    
    core_earliest_ready[core] = finish_time;
    task.setAssignment(core);
    task.setSchedulingState(SchedulingState::SCHEDULED);
    sequences[core].push_back(task.getId());
}

tuple<int, int, int, int, int, int> InitialTaskScheduler::calculateCloudPhaseTiming(Task& task) {
    int send_ready = task.getReadyTimeWirelessSend();
    const auto& cloud_times = task.getCloudExecutionTimes();
    int send_finish = send_ready + cloud_times[0];
    int cloud_ready = send_finish;
    int cloud_finish = cloud_ready + cloud_times[1];
    int receive_ready = cloud_finish;
    int receive_finish = max(wr_ready, receive_ready) + cloud_times[2];
    return make_tuple(send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish);
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

void InitialTaskScheduler::scheduleEntryTasks(const vector<Task*>& entry_tasks) {
    vector<Task*> cloud_entry_tasks;

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
        apply([this, task](auto... args) {
            scheduleOnCloud(*task, args...);
        }, timing);
    }
}

void InitialTaskScheduler::calculateNonEntryTaskReadyTimes(Task& task) {
    const auto& pred_tasks = task.getPredTasks();
    
    int local_ready = 0;
    if (!pred_tasks.empty()) {
        for (const Task* pred_task : pred_tasks) {
            local_ready = max(local_ready, 
                max(pred_task->getFinishTimeLocal(), 
                        pred_task->getFinishTimeWirelessReceive()));
        }
    }
    task.setReadyTimeLocal(max(local_ready, 0));

    int cloud_ready = ws_ready;
    if (!pred_tasks.empty()) {
        for (const Task* pred_task : pred_tasks) {
            cloud_ready = max(cloud_ready,
                max(pred_task->getFinishTimeLocal(),
                        pred_task->getFinishTimeWirelessSend()));
        }
    }
    task.setReadyTimeWirelessSend(cloud_ready);
}

void InitialTaskScheduler::scheduleNonEntryTasks(const vector<Task*>& non_entry_tasks) {
    for (Task* task : non_entry_tasks) {
        calculateNonEntryTaskReadyTimes(*task);
        
        if (!task->isCoreTask()) {
            auto timing = calculateCloudPhaseTiming(*task);
            apply([this, task](auto... args) {
                scheduleOnCloud(*task, args...);
            }, timing);
        } else {
            auto [core, start_time, finish_time] = 
                identifyOptimalLocalCore(*task, task->getReadyTimeLocal());
            
            auto timing = calculateCloudPhaseTiming(*task);
            int cloud_finish_time = get<5>(timing);
            
            if (finish_time <= cloud_finish_time) {
                scheduleOnLocalCore(*task, core, start_time, finish_time);
            } else {
                task->setIsCoreTask(false);
                apply([this, task](auto... args) {
                    scheduleOnCloud(*task, args...);
                }, timing);
            }
        }
    }
}

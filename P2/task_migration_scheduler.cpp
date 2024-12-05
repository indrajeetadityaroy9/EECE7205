#include <stack>
#include "task_migration_scheduler.h"

TaskMigrationScheduler::TaskMigrationScheduler(std::vector<Task>& tasks, std::vector<std::vector<int>> sequences)
    : tasks(tasks)
    , sequences(sequences)
    , RT_ls(3, 0)
    , cloud_phases_ready_times(3, 0)
{
    auto [dep_ready, seq_ready] = initialize_task_state();
    dependency_ready = std::move(dep_ready);
    sequence_ready = std::move(seq_ready);
}

std::pair<std::vector<int>, std::vector<int>> TaskMigrationScheduler::initialize_task_state() {
    std::vector<int> dependency_ready;
    for (const Task& task : tasks) {
        dependency_ready.push_back(task.getPredTasks().size());
    }
    std::vector<int> sequence_ready(tasks.size(), -1);

    for (const auto& sequence : sequences) {
        if (!sequence.empty()) {
            sequence_ready[sequence[0] - 1] = 0;
        }
    }

    return {dependency_ready, sequence_ready};
}

void TaskMigrationScheduler::update_task_state(Task& task) {
    if (task.getSchedulingState() != SchedulingState::KERNEL_SCHEDULED) {
        int unscheduled_pred_count = 0;
        for (const Task* pred_task : task.getPredTasks()) {
            if (pred_task->getSchedulingState() != SchedulingState::KERNEL_SCHEDULED) {
                unscheduled_pred_count++;
            }
        }
        dependency_ready[task.getId() - 1] = unscheduled_pred_count;

        for (const auto& sequence : sequences) {
            auto it = std::find(sequence.begin(), sequence.end(), task.getId());
            if (it != sequence.end()) {
                size_t idx = std::distance(sequence.begin(), it);
                
                if (idx > 0) {
                    Task& prev_task = tasks[sequence[idx - 1] - 1];
                    sequence_ready[task.getId() - 1] = (prev_task.getSchedulingState() != SchedulingState::KERNEL_SCHEDULED) ? 1 : 0;
                } else {
                    sequence_ready[task.getId() - 1] = 0;
                }
                break;
            }
        }
    }
}

void TaskMigrationScheduler::schedule_local_task(Task& task) {
    if (task.getPredTasks().empty()) {
        task.setReadyTimeLocal(0);
    } else {
        int latest_pred_time = 0;
        for (const Task* pred_task : task.getPredTasks()) {
            int pred_completion = std::max(
                pred_task->getFinishTimeLocal(),
                pred_task->getFinishTimeWirelessReceive()
            );
            latest_pred_time = std::max(latest_pred_time, pred_completion);
        }
        task.setReadyTimeLocal(latest_pred_time);
    }

    int core_index = task.getAssignment();
    vector<int> execution_times(4, -1);
    int start_time = std::max(
        RT_ls[core_index], 
        task.getReadyTimeLocal()
    );

    task.setExecutionUnitTaskStartTime(core_index, start_time);
    int finish_time = start_time + task.getCoreExecutionTimes()[core_index];
    task.setFinishTimeLocal(finish_time);
    RT_ls[core_index] = finish_time;
    task.setFinishTimeWirelessSend(-1);    // FTi^ws
    task.setFinishTimeCloud(-1);           // FTi^c
    task.setFinishTimeWirelessReceive(-1); // FTi^wr
}

void TaskMigrationScheduler::schedule_cloud_task(Task& task) {
    if (task.getPredTasks().empty()) {
        task.setReadyTimeWirelessSend(0);
    } else {
        int max_completion_time = 0;
        for (const Task* pred_task : task.getPredTasks()) {
            max_completion_time = std::max(
                max_completion_time,
                std::max(
                    pred_task->getFinishTimeLocal(),    // FTj^l
                    pred_task->getFinishTimeWirelessSend()  // FTj^ws
                )
            );
        }
        task.setReadyTimeWirelessSend(max_completion_time);
    }

    for (int i = 0; i < 4; i++) {
        task.setExecutionUnitTaskStartTime(i, -1);
    }

    int cloud_start = std::max(
        cloud_phases_ready_times[0],
        task.getReadyTimeWirelessSend()
    );
    task.setExecutionUnitTaskStartTime(3, cloud_start);

    task.setFinishTimeWirelessSend(
        task.getExecutionUnitTaskStartTime(3) + task.getCloudExecutionTimes()[0]
    );
    cloud_phases_ready_times[0] = task.getFinishTimeWirelessSend();

    int max_pred_cloud_time = 0;
    for (const Task* pred_task : task.getPredTasks()) {
        max_pred_cloud_time = std::max(
            max_pred_cloud_time,
            pred_task->getFinishTimeCloud()
        );
    }
    task.setReadyTimeCloud(std::max(
        task.getFinishTimeWirelessSend(),  // FT_ws
        max_pred_cloud_time                // max(pred_task.FT_c)
    ));

    task.setFinishTimeCloud(
        std::max(cloud_phases_ready_times[1], task.getReadyTimeCloud()) + task.getCloudExecutionTimes()[1]
    );
    cloud_phases_ready_times[1] = task.getFinishTimeCloud();

    task.setReadyTimeWirelessReceive(task.getFinishTimeCloud());
    
    task.setFinishTimeWirelessReceive(
        std::max(cloud_phases_ready_times[2], task.getReadyTimeWirelessReceive()) + task.getCloudExecutionTimes()[2]
    );

    cloud_phases_ready_times[2] = task.getFinishTimeWirelessReceive();
    task.setFinishTimeLocal(-1);
}

std::deque<Task*> TaskMigrationScheduler::initialize_queue() {
    std::deque<Task*> ready_tasks;
    
    for (Task& task : tasks) {
        if (sequence_ready[task.getId() - 1] == 0 &&
            std::all_of(task.getPredTasks().begin(), 
                       task.getPredTasks().end(),
                       [](const Task* pred) {
                           return pred->getSchedulingState() == 
                                  SchedulingState::KERNEL_SCHEDULED;
                       })) {
            ready_tasks.push_back(&task);
        }
    }
    
    return ready_tasks;
}


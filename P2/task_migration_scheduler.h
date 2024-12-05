#ifndef TASK_MIGRATION_SCHEDULER_H
#define TASK_MIGRATION_SCHEDULER_H
#include <vector>
#include <deque>
#include "task.h"

class TaskMigrationScheduler {
public:
    TaskMigrationScheduler(std::vector<Task>& tasks, std::vector<std::vector<int>> sequences);

    std::vector<Task>& tasks;                    
    std::vector<std::vector<int>> sequences;
    std::vector<int> RT_ls;
    std::vector<int> cloud_phases_ready_times;
    std::vector<int> dependency_ready;
    std::vector<int> sequence_ready;

    void update_task_state(Task& task);
    void schedule_local_task(Task& task);
    void schedule_cloud_task(Task& task);
    std::deque<Task*> initialize_queue();

private:
    std::pair<std::vector<int>, std::vector<int>> initialize_task_state();
};

#endif // TASK_MIGRATION_SCHEDULER_H
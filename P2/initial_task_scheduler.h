#ifndef INITIAL_TASK_SCHEDULER_H
#define INITIAL_TASK_SCHEDULER_H

#include <vector>
#include "task.h"

class InitialTaskScheduler {
public:
   InitialTaskScheduler(std::vector<Task>& tasks, int num_cores = 3);
   std::vector<int> getPriorityOrderedTasks() const;
   std::pair<std::vector<Task*>, std::vector<Task*>> classifyEntryTasks(const std::vector<int>& priority_order);
   std::tuple<int, int, int> identifyOptimalLocalCore(Task& task, int ready_time = 0);
   void InitialTaskScheduler::scheduleOnLocalCore(Task& task, int core, int start_time, int finish_time);
   std::tuple<int, int, int, int, int, int> InitialTaskScheduler::calculateCloudPhaseTiming(Task& task);
   void InitialTaskScheduler::scheduleOnCloud(Task& task, int send_ready, int send_finish, int cloud_ready, int cloud_finish, int receive_ready, int receive_finish);
   void InitialTaskScheduler::scheduleEntryTasks(const std::vector<Task*>& entry_tasks);
   void InitialTaskScheduler::calculateNonEntryTaskReadyTimes(Task& task);
   void InitialTaskScheduler::scheduleNonEntryTasks(const std::vector<Task*>& non_entry_tasks);

private:
   std::vector<Task>& tasks;
   int k;
   std::vector<int> core_earliest_ready;
   int ws_ready;
   int wr_ready;
   std::vector<std::vector<int>> sequences;
};

#endif // INITIAL_TASK_SCHEDULER_H
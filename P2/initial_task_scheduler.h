#ifndef INITIAL_TASK_SCHEDULER_H
#define INITIAL_TASK_SCHEDULER_H
#include <vector>
#include "task.h"
using namespace std;

class InitialTaskScheduler {
public:
   InitialTaskScheduler( vector<Task>& tasks, int num_cores = 3);

   vector<Task>& tasks;
   int k;
   vector<int> core_earliest_ready;
   int ws_ready;
   int wr_ready;
   vector< vector<int>> sequences;

   vector<int> getPriorityOrderedTasks() const;
   pair< vector<Task*>, vector<Task*>> classifyEntryTasks(const  vector<int>& priority_order);
   tuple<int, int, int> identifyOptimalLocalCore(Task& task, int ready_time = 0);
   void  scheduleOnLocalCore(Task& task, int core, int start_time, int finish_time);
   tuple<int, int, int, int, int, int> calculateCloudPhaseTiming(Task& task);
   void scheduleOnCloud(Task& task, int send_ready, int send_finish, int cloud_ready, int cloud_finish, int receive_ready, int receive_finish);
   void scheduleEntryTasks(const  vector<Task*>& entry_tasks);
   void calculateNonEntryTaskReadyTimes(Task& task);
   void scheduleNonEntryTasks(const  vector<Task*>& non_entry_tasks);
};

#endif // INITIAL_TASK_SCHEDULER_H
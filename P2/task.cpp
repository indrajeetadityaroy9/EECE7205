#include "task.h"
#include "scheduler_constants.h"

Task::Task(int taskId, const std::vector<Task*>& predTasks, const std::vector<Task*>& succTasks)
    : id(taskId)
    , pred_tasks(predTasks)
    , succ_tasks(succTasks)
    , core_execution_times(scheduler_constants::CORE_EXECUTION_TIMES.at(taskId).begin(), scheduler_constants::CORE_EXECUTION_TIMES.at(taskId).end())
    , cloud_execution_times(scheduler_constants::CLOUD_EXECUTION_TIMES.begin(), scheduler_constants::CLOUD_EXECUTION_TIMES.end())
    , FT_l(0)
    , FT_ws(0)
    , FT_c(0)
    , FT_wr(0)
    , RT_l(-1)
    , RT_ws(-1)
    , RT_c(-1)
    , RT_wr(-1)
    , priority_score(0)
    , assignment(-2)
    , is_core_task(false)
    , execution_unit_task_start_times(4, -1)  // 4 units: cloud + 3 cores
    , execution_finish_time(-1)
    , scheduling_state(SchedulingState::UNSCHEDULED)
{
}

// Basic task graph getters
int Task::getId() const { return id; }
const std::vector<Task*>& Task::getPredTasks() const { return pred_tasks; }
const std::vector<Task*>& Task::getSuccTasks() const { return succ_tasks; }

// Execution time getters
std::vector<int> Task::getCoreExecutionTimes() const { return core_execution_times; }
std::vector<int> Task::getCloudExecutionTimes() const { return cloud_execution_times; }

// Finish time getters and setters
int Task::getFinishTimeLocal() const { return FT_l; }
int Task::getFinishTimeWirelessSend() const { return FT_ws; }
int Task::getFinishTimeCloud() const { return FT_c; }
int Task::getFinishTimeWirelessReceive() const { return FT_wr; }

void Task::setFinishTimeLocal(int time) { FT_l = time; }
void Task::setFinishTimeWirelessSend(int time) { FT_ws = time; }
void Task::setFinishTimeCloud(int time) { FT_c = time; }
void Task::setFinishTimeWirelessReceive(int time) { FT_wr = time; }

// Ready time getters and setters
int Task::getReadyTimeLocal() const { return RT_l; }
int Task::getReadyTimeWirelessSend() const { return RT_ws; }
int Task::getReadyTimeCloud() const { return RT_c; }
int Task::getReadyTimeWirelessReceive() const { return RT_wr; }

void Task::setReadyTimeLocal(int time) { RT_l = time; }
void Task::setReadyTimeWirelessSend(int time) { RT_ws = time; }
void Task::setReadyTimeCloud(int time) { RT_c = time; }
void Task::setReadyTimeWirelessReceive(int time) { RT_wr = time; }

// Task scheduling parameter getters and setters
int Task::getPriorityScore() const { return priority_score; }
void Task::setPriorityScore(int score) { priority_score = score; }

int Task::getAssignment() const { return assignment; }
void Task::setAssignment(int unit) { assignment = unit; }

bool Task::isCoreTask() const { return is_core_task; }
void Task::setIsCoreTask(bool onCore) { is_core_task = onCore; }

int Task::getExecutionUnitTaskStartTime(int executionUnit) const { 
    return execution_unit_task_start_times[executionUnit]; 
}
void Task::setExecutionUnitTaskStartTime(int executionUnit, int time) { 
    execution_unit_task_start_times[executionUnit] = time; 
}

int Task::getExecutionFinishTime() const { return execution_finish_time; }
void Task::setExecutionFinishTime(int time) { execution_finish_time = time; }

SchedulingState Task::getSchedulingState() const { return scheduling_state; }
void Task::setSchedulingState(SchedulingState state) { scheduling_state = state; }

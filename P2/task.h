#ifndef TASK_H
#define TASK_H

#include <vector>
#include "scheduler_constants.h"

// Class representing a task in the mobile cloud computing environment
// Implements the task model from Section II.A of the paper
class Task {
public:
    // Constructor: Initialize task with ID and optional predecessor/successor tasks
    Task(int taskId, const std::vector<Task*>& predTasks = {}, const std::vector<Task*>& succTasks = {});

    // Basic task graph getters
    int getId() const;                              // Get task identifier
    const std::vector<Task*>& getPredTasks() const; // Get predecessor tasks (pred(vi))
    const std::vector<Task*>& getSuccTasks() const; // Get successor tasks (succ(vi))
    
    // Execution time getters
    std::vector<int> getCoreExecutionTimes() const;  // Get T_i^l_k for all cores k
    std::vector<int> getCloudExecutionTimes() const; // Get [T_i^s, T_i^c, T_i^r]
    
    // Finish time getters/setters (Section II.C)
    int getFinishTimeLocal() const;          // Get FT_i^l (local core finish time)
    int getFinishTimeWirelessSend() const;   // Get FT_i^ws (wireless send finish time)
    int getFinishTimeCloud() const;          // Get FT_i^c (cloud computation finish time)
    int getFinishTimeWirelessReceive() const;// Get FT_i^wr (wireless receive finish time)
    void setFinishTimeLocal(int time);       // Set local core finish time
    void setFinishTimeWirelessSend(int time);// Set wireless send finish time
    void setFinishTimeCloud(int time);       // Set cloud computation finish time
    void setFinishTimeWirelessReceive(int time);// Set wireless receive finish time

    // Ready time getters/setters (Section II.C)
    int getReadyTimeLocal() const;           // Get RT_i^l (local ready time)
    int getReadyTimeWirelessSend() const;    // Get RT_i^ws (wireless send ready time)
    int getReadyTimeCloud() const;           // Get RT_i^c (cloud ready time)
    int getReadyTimeWirelessReceive() const; // Get RT_i^wr (wireless receive ready time)
    void setReadyTimeLocal(int time);        // Set local ready time
    void setReadyTimeWirelessSend(int time); // Set wireless send ready time
    void setReadyTimeCloud(int time);        // Set cloud ready time
    void setReadyTimeWirelessReceive(int time);// Set wireless receive ready time

    // Task scheduling parameters (Section III)
    int getPriorityScore() const;    // Get priority(vi) = wi + max(priority(vj))
    void setPriorityScore(int score);// Set task priority score

    int getAssignment() const;       // Get execution unit assignment (0:cloud, >0:core)
    void setAssignment(int unit);    // Set execution unit assignment

    bool isCoreTask() const;         // Check if task is assigned to local core
    void setIsCoreTask(bool onCore); // Set task assignment type (local/cloud)

    // Execution timing management
    int getExecutionUnitTaskStartTime(int executionUnit) const;  // Get start time for given unit
    void setExecutionUnitTaskStartTime(int executionUnit, int time); // Set start time for unit

    int getExecutionFinishTime() const;    // Get final task completion time
    void setExecutionFinishTime(int time); // Set final task completion time

    SchedulingState getSchedulingState() const;  // Get current scheduling state
    void setSchedulingState(SchedulingState state); // Set scheduling state

private:
    // Basic task graph structure (Section II.A)
    int id;                         // Task identifier
    std::vector<Task*> pred_tasks;  // Predecessor tasks in the graph
    std::vector<Task*> succ_tasks;  // Successor tasks in the graph

    // Execution timing parameters (Section II.B)
    std::vector<int> core_execution_times;   // T_i^l_k for each core k
    std::vector<int> cloud_execution_times;  // [T_i^s, T_i^c, T_i^r]

    // Task completion timing parameters (Section II.C)
    int FT_l;    // Local core finish time
    int FT_ws;   // Wireless sending finish time
    int FT_c;    // Cloud computation finish time
    int FT_wr;   // Wireless receiving finish time

    // Ready times (Section II.C)
    int RT_l;    // Ready time for local execution
    int RT_ws;   // Ready time for wireless sending
    int RT_c;    // Ready time for cloud execution
    int RT_wr;   // Ready time for receiving results

    // Scheduling parameters (Section III)
    int priority_score;      // Task priority for scheduling
    int assignment;         // Assigned execution unit
    bool is_core_task;      // Flag for local vs cloud execution
    std::vector<int> execution_unit_task_start_times; // Start times per unit
    int execution_finish_time;  // Final completion time
    SchedulingState scheduling_state; // Current state in scheduling algorithm
};

#endif // TASK_H
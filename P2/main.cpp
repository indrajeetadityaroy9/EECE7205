#include "initial_task_scheduler.h"
#include "mcc_scheduler.h"
#include "task.h"
#include "scheduler_constants.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
using namespace std;

// Helper functions for printing task information
void printTaskGraph(vector<Task>& tasks) {
    for (Task& task : tasks) {
        cout << "Task " << task.getId() << ":\n";
        
        // Print parent task IDs
        cout << "  Parents: [";
        bool first = true;
        for (Task* pred : task.getPredTasks()) {
            if (!first) cout << ", ";
            cout << pred->getId();
            first = false;
        }
        cout << "]\n";
        
        // Print children task IDs
        cout << "  Children: [";
        first = true;
        for (Task* succ : task.getSuccTasks()) {
            if (!first) cout << ", ";
            cout << succ->getId();
            first = false;
        }
        cout << "]\n\n";
    }
}

void printFinalSequences(vector<vector<int>>& sequences) {
    cout << "\nExecution Sequences:\n";
    cout << string(40, '-') << "\n";
    
    for (size_t i = 0; i < sequences.size(); i++) {
        // Label each sequence appropriately
        string label = (i < 3) ? "Core " + to_string(i + 1) : "Cloud";
        cout << setw(12) << left << label << ": ";
        
        // Print task sequence
        cout << "[";
        for (size_t j = 0; j < sequences[i].size(); j++) {
            if (j > 0) cout << ", ";
            cout << sequences[i][j];
        }
        cout << "]\n";
    }
}

void printTaskSchedule(const vector<Task>& tasks) {
    cout << "\nTask Schedule:\n";
    cout << string(80, '-') << "\n";

    for (const Task& task : tasks) {
        cout << "Task " << setw(2) << task.getId() << " | ";

        // Determine assignment
        string assignment;
        if (task.getAssignment() >= 0 && task.getAssignment() < 3) {
            assignment = "Core " + to_string(task.getAssignment() + 1);
        } else if (task.getAssignment() == 3) {
            assignment = "Cloud";
        } else {
            assignment = "Not Scheduled";
        }

        cout << setw(12) << left << assignment << " | ";

        // Print timing information
        if (task.isCoreTask()) {
            int start = task.getExecutionUnitTaskStartTime(task.getAssignment());
            int end = start + task.getCoreExecutionTimes()[task.getAssignment()];
            cout << "Execution: " << start << " → " << end;
        } else {
            int send_start = task.getExecutionUnitTaskStartTime(3);
            int send_end = send_start + task.getCloudExecutionTimes()[0];
            int cloud_end = task.getReadyTimeCloud() + task.getCloudExecutionTimes()[1];
            int receive_end = task.getReadyTimeWirelessReceive() + task.getCloudExecutionTimes()[2];

            cout << "Send: " << send_start << " → " << send_end << " | ";
            cout << "Cloud: " << task.getReadyTimeCloud() << " → " << cloud_end << " | ";
            cout << "Receive: " << task.getReadyTimeWirelessReceive() << " → " << receive_end;
        }
        cout << "\n";
    }
}

int main() {
    vector<Task> tasks;
    for (int i = 0; i < 10; i++) {
        tasks.emplace_back(i + 1);  // Tasks are 1-indexed
    }

    tasks[8].addSuccessor(tasks[8], &tasks[9]);  // task9 -> task10
    tasks[7].addSuccessor(tasks[7], &tasks[9]);  // task8 -> task10
    tasks[6].addSuccessor(tasks[6], &tasks[9]);  // task7 -> task10
    tasks[5].addSuccessor(tasks[5], &tasks[7]);  // task6 -> task8
    tasks[4].addSuccessor(tasks[4], &tasks[8]);  // task5 -> task9
    tasks[3].addSuccessor(tasks[3], &tasks[7]);  // task4 -> task8
    tasks[3].addSuccessor(tasks[3], &tasks[8]);  // task4 -> task9
    tasks[2].addSuccessor(tasks[2], &tasks[6]);  // task3 -> task7
    tasks[1].addSuccessor(tasks[1], &tasks[7]);  // task2 -> task8
    tasks[1].addSuccessor(tasks[1], &tasks[8]);  // task2 -> task9
    tasks[0].addSuccessor(tasks[0], &tasks[1]);  // task1 -> task2
    tasks[0].addSuccessor(tasks[0], &tasks[2]);  // task1 -> task3
    tasks[0].addSuccessor(tasks[0], &tasks[3]);  // task1 -> task4
    tasks[0].addSuccessor(tasks[0], &tasks[4]);  // task1 -> task5
    tasks[0].addSuccessor(tasks[0], &tasks[5]);  // task1 -> task6
    tasks[9].addPredecessors(tasks[9], {&tasks[6], &tasks[7], &tasks[8]});  // task10 <- [task7, task8, task9]
    tasks[8].addPredecessors(tasks[8], {&tasks[1], &tasks[3], &tasks[4]});  // task9 <- [task2, task4, task5]
    tasks[7].addPredecessors(tasks[7], {&tasks[1], &tasks[3], &tasks[5]});  // task8 <- [task2, task4, task6]
    tasks[6].addPredecessors(tasks[6], {&tasks[2]});                        // task7 <- [task3]
    tasks[5].addPredecessors(tasks[5], {&tasks[0]});                        // task6 <- [task1]
    tasks[4].addPredecessors(tasks[4], {&tasks[0]});                        // task5 <- [task1]
    tasks[3].addPredecessors(tasks[3], {&tasks[0]});                        // task4 <- [task1]
    tasks[2].addPredecessors(tasks[2], {&tasks[0]});                        // task3 <- [task1]
    tasks[1].addPredecessors(tasks[1], {&tasks[0]});
    
    // Create MCCScheduler scheduler and run the algorithm
    MCCScheduler mcc(tasks);
    
    // Print initial task graph
    cout << "Initial Task Graph:\n";
    printTaskGraph(tasks);
    
    // Run scheduling phases
    mcc.primaryAssignment();
    mcc.taskPrioritizing();
    auto sequences = mcc.selectExecutionUnits();
    
    // Print results
    printFinalSequences(sequences);
    
    vector<int> core_powers = {1, 2, 4};
    double cloud_sending_power = 0.5;
    
    int total_time = mcc.totalTime();
    double total_energy = mcc.totalEnergy(core_powers, cloud_sending_power);
    
    cout << "\nINITIAL SCHEDULING APPLICATION COMPLETION TIME: " << total_time << "\n";
    cout << "INITIAL APPLICATION ENERGY CONSUMPTION: " << total_energy << "\n";
    cout << "INITIAL TASK SCHEDULE:\n";
   
    // Print the task schedule
    printTaskSchedule(tasks);

    return 0;
}
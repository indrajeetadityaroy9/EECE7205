#include "initial_task_scheduler.h"
#include "mcc_scheduler.h"
#include "task.h"
#include "scheduler_constants.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include "task_graph.h"
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
    // Initialize TaskGraph with 10 tasks (1-indexed)
    TaskGraph graph(10);
    // Add edges to build the task dependency graph
    // Add all edges from task 1 (root node) to its immediate children
    graph.addEdge(1, 2);  // task1 -> task2
    graph.addEdge(1, 3);  // task1 -> task3
    graph.addEdge(1, 4);  // task1 -> task4
    graph.addEdge(1, 5);  // task1 -> task5
    graph.addEdge(1, 6);  // task1 -> task6
    // Add middle layer connections - these tasks connect the root's children to the pre-final layer
    graph.addEdge(2, 8);  // task2 -> task8
    graph.addEdge(2, 9);  // task2 -> task9
    graph.addEdge(3, 7);  // task3 -> task7
    graph.addEdge(4, 8);  // task4 -> task8
    graph.addEdge(4, 9);  // task4 -> task9
    graph.addEdge(5, 9);  // task5 -> task9
    graph.addEdge(6, 8);  // task6 -> task8
    // Add edges to the final task (task 10)
    graph.addEdge(7, 10); // task7 -> task10
    graph.addEdge(8, 10); // task8 -> task10
    graph.addEdge(9, 10); // task9 -> task10

    // Create MCCScheduler with the task graph and run the algorithm
    MCCScheduler mcc(graph.getAllTasks());
    
    // Print the initial structure of our task graph
    cout << "Initial Task Graph:\n";
    printTaskGraph(graph.getAllTasks());
    
    // Execute the two main phases of the scheduling algorithm
    mcc.primaryAssignment();  // Determine initial cloud vs. local core assignment
    mcc.taskPrioritizing();   // Calculate task priorities for scheduling
    auto sequences = mcc.selectExecutionUnits();  // Generate final execution sequences
    
    // Print the resulting execution sequences
    printFinalSequences(sequences);
    
    // Define power consumption parameters for energy calculations
    vector<int> core_powers = {1, 2, 4};  // Power consumption for each local core
    double cloud_sending_power = 0.5;      // Power consumption for cloud communication
    
    // Calculate and display the key performance metrics
    int total_time = mcc.totalTime();
    double total_energy = mcc.totalEnergy(core_powers, cloud_sending_power);
    
    // Output the final scheduling results
    cout << "\nINITIAL SCHEDULING APPLICATION COMPLETION TIME: " << total_time << "\n";
    cout << "INITIAL APPLICATION ENERGY CONSUMPTION: " << total_energy << "\n";
    cout << "INITIAL TASK SCHEDULE:\n";
   
    // Display detailed schedule for each task
    printTaskSchedule(graph.getAllTasks());

    return 0;
}
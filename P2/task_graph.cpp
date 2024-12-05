#include "task_graph.h"
#include <stdexcept>
#include <string>

TaskGraph::TaskGraph(int size) {
    tasks.reserve(size);
    for (int i = 0; i < size; i++) {
        tasks.emplace_back(i + 1);
    }
}

void TaskGraph::validateTaskId(int id) const {
    if (id < 1 || id > static_cast<int>(tasks.size())) {
        throw std::runtime_error("Invalid task ID: " + std::to_string(id));
    }
}

void TaskGraph::addEdge(int parentId, int childId) {
    validateTaskId(parentId);
    validateTaskId(childId);
    
    Task& parent = tasks[parentId - 1];
    Task& child = tasks[childId - 1];
    
    const_cast<std::vector<Task*>&>(child.getPredTasks()).push_back(&parent);
    const_cast<std::vector<Task*>&>(parent.getSuccTasks()).push_back(&child);
}

Task& TaskGraph::getTask(int id) {
    validateTaskId(id);
    return tasks[id - 1];
}

std::vector<Task>& TaskGraph::getAllTasks() {
    return tasks;
}
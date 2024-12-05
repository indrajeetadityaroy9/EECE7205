#ifndef TASK_GRAPH_H
#define TASK_GRAPH_H

#include <vector>
#include "task.h"

class TaskGraph {
private:
    std::vector<Task> tasks;
    void validateTaskId(int id) const;

public:
    explicit TaskGraph(int size);
    void addEdge(int parentId, int childId);
    Task& getTask(int id);
    std::vector<Task>& getAllTasks();
};

#endif // TASK_GRAPH_H
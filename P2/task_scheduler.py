import matplotlib.pyplot as plt
from copy import deepcopy
import time
import sys

task_core_values = {
    1: [9, 7, 5],
    2: [8, 6, 5],
    3: [6, 5, 4],
    4: [7, 5, 3],
    5: [5, 4, 2],
    6: [7, 6, 4],
    7: [8, 5, 3],
    8: [6, 4, 2],
    9: [5, 3, 2],
    10: [7, 4, 2],
}

cloud_speed = [3, 1, 1]  # T_send, T_cloud, T_receive

class Node(object):
    def __init__(self, task_id, parents=None, children=None):
        self.task_id = task_id
        self.parents = parents or []  # Immediate predecessors
        self.children = children or []  # Immediate successors
        self.local_finish_time = 0
        self.cloud_sending_finish_time = 0
        self.cloud_finish_time = 0
        self.cloud_recieving_finish_time = 0
        self.local_ready_time = -1
        self.cloud_sending_ready_time = -1
        self.cloud_ready_time = -1
        self.priority_score = None
        self.assignment = -2  # -2 = not assigned, 0-2 = cores, 3 = cloud
        self.is_core = False
        self.start_time = [-1, -1, -1, -1]  # Start times for Core1, Core2, Core3, Cloud
        self.is_scheduled = False
        self.core_speed = task_core_values[task_id]
        self.cloud_speed = cloud_speed
        self.cloud_execution_time = sum(cloud_speed)  # Total cloud execution time

def primary_assignment(nodes, core_earliest_ready, cloud_earliest_ready):
    """
    Assign tasks to local cores or the cloud based on estimated execution times.
    """
    for node in nodes:
        # Calculate minimum local execution time (T_i^{l,min})
        t_l_min = min(node.core_speed)

        # Calculate estimated remote execution time (T_i^{re})
        t_re = sum(node.cloud_speed)

        # Assign task to cloud or local core based on execution times
        if t_re < t_l_min or (cloud_earliest_ready < min(core_earliest_ready[:3])):
            node.is_core = False  # Assign to cloud
            node.assignment = 3  # Cloud
        else:
            node.is_core = True  # Assign to local core
            node.assignment = node.core_speed.index(t_l_min)  # Fastest core

def calculate_priority(task, weights, priority_cache):
    """
    Recursively calculate priority of a task.
    """
    if task.task_id in priority_cache:
        return priority_cache[task.task_id]

    if not task.children:  # Exit task
        priority_cache[task.task_id] = weights[task.task_id - 1]
        return weights[task.task_id - 1]

    max_successor_priority = max(calculate_priority(child, weights, priority_cache) for child in task.children)
    task_priority = weights[task.task_id - 1] + max_successor_priority
    priority_cache[task.task_id] = task_priority
    return task_priority

def task_prioritizing(nodes):
    """
    Calculate priority levels for tasks using critical path lengths.
    """
    weights = []
    for node in nodes:
        if node.is_core:
            weights.append(sum(node.core_speed) / len(node.core_speed))  # Local task
        else:
            weights.append(sum(node.cloud_speed))  # Cloud task

    priority_cache = {}
    for node in nodes:
        node.priority_score = calculate_priority(node, weights, priority_cache)

def execution_unit_selection(nodes):
    """
    Schedule tasks in descending order of priority, ensuring task-precedence constraints.
    """
    k = 3  # Number of local cores
    core_seqs = [[] for _ in range(k)]  # Sequences for local cores
    cloud_seq = []  # Sequence for cloud tasks
    core_earliest_ready = [0] * (k + 1)  # Earliest ready times for cores and cloud

    # Sort tasks by priority in descending order
    nodes = sorted(nodes, key=lambda x: x.priority_score, reverse=True)

    for node in nodes:
        # Calculate the maximum finish time among all predecessors
        max_parent_finish = max(
            (max(parent.local_finish_time, parent.cloud_recieving_finish_time) for parent in node.parents),
            default=0,
        )

        if node.is_core:
            # Schedule on local cores
            local_finish_times = []
            for core_index in range(k):
                ready_time = max(max_parent_finish, core_earliest_ready[core_index])
                finish_time = ready_time + node.core_speed[core_index]
                local_finish_times.append((finish_time, core_index))

            # Select the core with the minimum finish time
            best_finish, best_core = min(local_finish_times)
            node.local_ready_time = best_finish - node.core_speed[best_core]
            node.local_finish_time = best_finish
            core_earliest_ready[best_core] = best_finish
            core_seqs[best_core].append(node.task_id)
            node.assignment = best_core
        else:
            # Schedule on the cloud
            node.cloud_sending_ready_time = max(max_parent_finish, core_earliest_ready[3])
            node.cloud_sending_finish_time = node.cloud_sending_ready_time + node.cloud_speed[0]
            node.cloud_ready_time = node.cloud_sending_finish_time
            node.cloud_finish_time = node.cloud_ready_time + node.cloud_speed[1]
            node.cloud_recieving_finish_time = node.cloud_finish_time + node.cloud_speed[2]
            core_earliest_ready[3] = node.cloud_recieving_finish_time
            cloud_seq.append(node.task_id)
            node.assignment = 3

        # Mark task as scheduled
        node.is_scheduled = True

    return core_seqs + [cloud_seq]

def total_energy(nodes, core_powers, cloud_sending_power):
    """
    Calculate total energy consumption.
    """
    total_energy = 0.0
    for node in nodes:
        if node.is_core:
            total_energy += core_powers[node.assignment] * node.core_speed[node.assignment]
        else:
            total_energy += cloud_sending_power * node.cloud_speed[0]
    return total_energy

def total_time(nodes):
    """
    Calculate total application completion time.
    """
    return max(
        max(node.local_finish_time, node.cloud_recieving_finish_time)
        for node in nodes
        if not node.children
    )

def log_task_details(nodes):
    """
    Log task execution details for debugging.
    """
    for node in nodes:
        result = {"node id": node.task_id, "assignment": node.assignment + 1}
        if node.is_core:
            result.update({
                "local start_time": node.local_ready_time,
                "local finish_time": node.local_finish_time,
            })
        else:
            result.update({
                "cloud start_time": node.cloud_ready_time,
                "cloud finish_time": node.cloud_finish_time,
                "ws start_time": node.cloud_sending_ready_time,
                "ws finish_time": node.cloud_sending_finish_time,
                "wr start_time": node.cloud_recieving_finish_time - node.cloud_speed[2],
                "wr finish_time": node.cloud_recieving_finish_time,
            })
        print(result)

if __name__ == "__main__":
    # Define task graph (nodes and dependencies)
    node10 = Node(task_id=10, parents=None, children=[])
    node9 = Node(task_id=9, parents=None, children=[node10])
    node8 = Node(task_id=8, parents=None, children=[node10])
    node7 = Node(task_id=7, parents=None, children=[node10])
    node6 = Node(task_id=6, parents=None, children=[node8])
    node5 = Node(task_id=5, parents=None, children=[node9])
    node4 = Node(task_id=4, parents=None, children=[node8, node9])
    node3 = Node(task_id=3, parents=None, children=[node7])
    node2 = Node(task_id=2, parents=None, children=[node8, node9])
    node1 = Node(task_id=1, parents=None, children=[node2, node3, node4, node5, node6])

    # Define parent relationships
    node1.parents = []
    node2.parents = [node1]
    node3.parents = [node1]
    node4.parents = [node1]
    node5.parents = [node1]
    node6.parents = [node1]
    node7.parents = [node3]
    node8.parents = [node2, node4, node6]
    node9.parents = [node2, node4, node5]
    node10.parents = [node7, node8, node9]

    node_list = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10]

    # Step One: Initial Scheduling Algorithm
    primary_assignment(node_list, core_earliest_ready=[0, 0, 0, 0], cloud_earliest_ready=0)
    task_prioritizing(node_list)
    sequences = execution_unit_selection(node_list)

    # Log task details
    log_task_details(node_list)

    # Calculate total energy and time
    total_energy_result = total_energy(node_list, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    total_time_result = total_time(node_list)

    print(f"Task Sequences: {sequences}")
    print(f"Total Energy: {total_energy_result}")
    print(f"Total Time: {total_time_result}")



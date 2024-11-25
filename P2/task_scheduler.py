import sys
from copy import deepcopy

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

class Node:
    def __init__(self, task_id, parents=None, children=None):
        self.task_id = task_id
        self.parents = parents or []
        self.children = children or []
        self.local_finish_time = 0
        self.cloud_sending_finish_time = 0
        self.cloud_finish_time = 0
        self.cloud_receiving_finish_time = 0
        self.local_ready_time = -1
        self.cloud_sending_ready_time = -1
        self.cloud_ready_time = -1
        self.priority_score = None
        self.assignment = -2  # -2=not assigned, 0-2=cores, 3=cloud
        self.is_core = False
        self.is_scheduled = False
        self.core_speed = task_core_values[task_id]
        self.cloud_speed = cloud_speed
        self.cloud_execution_time = sum(cloud_speed)

def calculate_earliest_start_time(node, core_earliest_ready, cloud_earliest_ready):
    """Calculate earliest possible start time considering predecessors"""
    if not node.parents:
        return 0, 0

    local_ready = max(
        max(parent.local_finish_time, parent.cloud_receiving_finish_time)
        for parent in node.parents
    )
    
    cloud_ready = max(
        max(parent.local_finish_time, parent.cloud_sending_finish_time)
        for parent in node.parents
    )
    
    return local_ready, cloud_ready

def primary_assignment(nodes, core_earliest_ready, cloud_earliest_ready):
    """Enhanced primary assignment considering both execution time and data transfer overhead"""
    for node in nodes:
        # Calculate earliest possible start times
        local_ready, cloud_ready = calculate_earliest_start_time(node, core_earliest_ready, cloud_earliest_ready)
        
        # Find minimum local execution time
        min_local_time = float('inf')
        best_core = -1
        for core in range(3):
            core_time = node.core_speed[core]
            start_time = max(local_ready, core_earliest_ready[core])
            total_time = start_time + core_time
            if total_time < min_local_time:
                min_local_time = total_time
                best_core = core

        # Calculate cloud execution time including transfer overhead
        cloud_start = max(cloud_ready, cloud_earliest_ready)
        cloud_total_time = cloud_start + node.cloud_execution_time

        # Make assignment decision
        if cloud_total_time < min_local_time:
            node.is_core = False
            node.assignment = 3
        else:
            node.is_core = True
            node.assignment = best_core

def calculate_priority(task, weights, priority_cache):
    """Calculate upward rank (priority) of tasks"""
    if task.task_id in priority_cache:
        return priority_cache[task.task_id]

    if not task.children:  # Exit task
        priority_cache[task.task_id] = weights[task.task_id - 1]
        return weights[task.task_id - 1]

    # Calculate maximum path to exit
    max_successor_priority = max(
        calculate_priority(child, weights, priority_cache) 
        for child in task.children
    )
    
    priority = weights[task.task_id - 1] + max_successor_priority
    priority_cache[task.task_id] = priority
    return priority

def task_prioritizing(nodes):
    """Prioritize tasks based on upward rank"""
    weights = []
    for node in nodes:
        # Average execution time as weight
        if node.is_core:
            weights.append(sum(node.core_speed) / len(node.core_speed))
        else:
            weights.append(node.cloud_execution_time)

    priority_cache = {}
    for node in nodes:
        node.priority_score = calculate_priority(node, weights, priority_cache)

def get_possible_finish_times(node, core_earliest_ready, cloud_earliest_ready):
    """Calculate possible finish times for all execution units"""
    local_ready, cloud_ready = calculate_earliest_start_time(
        node, core_earliest_ready, cloud_earliest_ready
    )
    
    finish_times = []
    # Calculate finish times for each core
    for core in range(3):
        start_time = max(local_ready, core_earliest_ready[core])
        finish_time = start_time + node.core_speed[core]
        finish_times.append((finish_time, core, True, start_time))
    
    # Calculate cloud finish time
    cloud_start = max(cloud_ready, cloud_earliest_ready)
    cloud_finish = cloud_start + node.cloud_execution_time
    finish_times.append((cloud_finish, 3, False, cloud_start))
    
    return finish_times

def execution_unit_selection(nodes):
    """Schedule tasks with improved selection logic"""
    k = 3  # Number of local cores
    core_seqs = [[] for _ in range(k)]
    cloud_seq = []
    core_earliest_ready = [0] * k
    cloud_earliest_ready = 0
    
    # Sort by priority score
    nodes = sorted(nodes, key=lambda x: x.priority_score, reverse=True)
    
    for node in nodes:
        # Get all possible finish times
        finish_times = get_possible_finish_times(
            node, core_earliest_ready, cloud_earliest_ready
        )
        
        # Select execution unit with earliest finish time
        finish_time, unit, is_core, start_time = min(finish_times)
        
        if is_core:
            node.is_core = True
            node.assignment = unit
            node.local_ready_time = start_time
            node.local_finish_time = finish_time
            core_earliest_ready[unit] = finish_time
            core_seqs[unit].append(node.task_id)
        else:
            node.is_core = False
            node.assignment = 3
            node.cloud_sending_ready_time = start_time
            node.cloud_sending_finish_time = start_time + node.cloud_speed[0]
            node.cloud_ready_time = node.cloud_sending_finish_time
            node.cloud_finish_time = node.cloud_ready_time + node.cloud_speed[1]
            node.cloud_receiving_finish_time = node.cloud_finish_time + node.cloud_speed[2]
            cloud_earliest_ready = node.cloud_receiving_finish_time
            cloud_seq.append(node.task_id)
        
        node.is_scheduled = True
    
    return core_seqs + [cloud_seq]

def total_energy(nodes, core_powers, cloud_sending_power):
    """Calculate total energy consumption"""
    total = 0.0
    for node in nodes:
        if node.is_core:
            total += core_powers[node.assignment] * node.core_speed[node.assignment]
        else:
            total += cloud_sending_power * node.cloud_speed[0]
    return total

def total_time(nodes):
    """Calculate total completion time"""
    return max(
        max(node.local_finish_time, node.cloud_receiving_finish_time)
        for node in nodes
        if not node.children
    )

def log_task_details(nodes):
    """Log execution details for each task"""
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
                "wr start_time": node.cloud_receiving_finish_time - node.cloud_speed[2],
                "wr finish_time": node.cloud_receiving_finish_time,
            })
        print(result)

if __name__ == "__main__":
    # Initialize task graph
    node10 = Node(10)
    node9 = Node(9, children=[node10])
    node8 = Node(8, children=[node10])
    node7 = Node(7, children=[node10])
    node6 = Node(6, children=[node8])
    node5 = Node(5, children=[node9])
    node4 = Node(4, children=[node8, node9])
    node3 = Node(3, children=[node7])
    node2 = Node(2, children=[node8, node9])
    node1 = Node(1, children=[node2, node3, node4, node5, node6])

    # Set parent relationships
    node10.parents = [node7, node8, node9]
    node9.parents = [node2, node4, node5]
    node8.parents = [node2, node4, node6]
    node7.parents = [node3]
    node6.parents = [node1]
    node5.parents = [node1]
    node4.parents = [node1]
    node3.parents = [node1]
    node2.parents = [node1]
    node1.parents = []

    nodes = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10]

    # Execute scheduling algorithm
    core_earliest_ready = [0, 0, 0]
    cloud_earliest_ready = 0
    
    primary_assignment(nodes, core_earliest_ready, cloud_earliest_ready)
    task_prioritizing(nodes)
    sequences = execution_unit_selection(nodes)

    print("INITIAL TIME: ", total_time(nodes))
    print("INITIAL ENERGY:", total_energy(nodes, core_powers=[1, 2, 4], cloud_sending_power=0.5))
    log_task_details(nodes)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
from copy import deepcopy

# Define task execution times on cores and cloud
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
        self.execution_unit = None  # 'core' or 'cloud'
        self.start_time = 0
        self.finish_time = 0

def calculate_earliest_start_time(node):
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

def primary_assignment(nodes):
    for node in nodes:
        local_ready, cloud_ready = calculate_earliest_start_time(node)
        min_local_time = float('inf')
        best_core = -1
        for core in range(3):
            core_time = node.core_speed[core]
            start_time = max(local_ready, 0)  # Core availability is initially 0
            total_time = start_time + core_time
            if total_time < min_local_time:
                min_local_time = total_time
                best_core = core
        cloud_start = max(cloud_ready, 0)  # Cloud availability is initially 0
        cloud_total_time = cloud_start + node.cloud_execution_time
        if cloud_total_time <= min_local_time:
            node.is_core = False
            node.assignment = 3
        else:
            node.is_core = True
            node.assignment = best_core

def calculate_priority(task, weights, priority_cache):
    if task.task_id in priority_cache:
        return priority_cache[task.task_id]
    if not task.children:
        priority_cache[task.task_id] = weights[task.task_id - 1]
        return weights[task.task_id - 1]
    max_successor_priority = max(
        calculate_priority(child, weights, priority_cache)
        for child in task.children
    )
    priority = weights[task.task_id - 1] + max_successor_priority
    priority_cache[task.task_id] = priority
    return priority

def task_prioritizing(nodes):
    weights = []
    for node in nodes:
        if node.is_core:
            weights.append(sum(node.core_speed) / len(node.core_speed))
        else:
            weights.append(node.cloud_execution_time)
    priority_cache = {}
    for node in nodes:
        node.priority_score = calculate_priority(node, weights, priority_cache)

def get_possible_finish_times(node, core_available_times, ws_channel_available_time):
    local_ready, cloud_ready = calculate_earliest_start_time(node)
    finish_times = []
    for core in range(3):
        start_time = max(local_ready, core_available_times[core])
        finish_time = start_time + node.core_speed[core]
        finish_times.append((finish_time, core, True, start_time))
    cloud_start = max(cloud_ready, ws_channel_available_time)
    cloud_finish = cloud_start + node.cloud_execution_time
    finish_times.append((cloud_finish, 3, False, cloud_start))
    return finish_times

def execution_unit_selection(nodes):
    sequences = [[] for _ in range(4)]  # 3 cores and cloud
    core_available_times = [0] * 3
    ws_channel_available_time = 0
    nodes_sorted = sorted(nodes, key=lambda x: x.priority_score, reverse=True)
    for node in nodes_sorted:
        finish_times = get_possible_finish_times(node, core_available_times, ws_channel_available_time)
        finish_time, unit, is_core, start_time = min(finish_times, key=lambda x: x[0])
        if is_core:
            node.is_core = True
            node.assignment = unit
            node.local_ready_time = start_time
            node.local_finish_time = finish_time
            core_available_times[unit] = finish_time
            sequences[unit].append(node)
        else:
            node.is_core = False
            node.assignment = 3
            node.cloud_sending_ready_time = start_time
            node.cloud_sending_finish_time = start_time + node.cloud_speed[0]
            node.cloud_ready_time = node.cloud_sending_finish_time
            node.cloud_finish_time = node.cloud_ready_time + node.cloud_speed[1]
            node.cloud_receiving_finish_time = node.cloud_finish_time + node.cloud_speed[2]
            ws_channel_available_time = node.cloud_sending_finish_time
            sequences[3].append(node)
    return sequences

def calculate_energy_consumption(node, core_powers, cloud_sending_power):
    if node.is_core:
        return core_powers[node.assignment] * node.core_speed[node.assignment]
    else:
        return cloud_sending_power * node.cloud_speed[0]

def total_energy(nodes, core_powers, cloud_sending_power):
    return sum(calculate_energy_consumption(node, core_powers, cloud_sending_power) 
               for node in nodes)

def total_time(nodes):
    return max(
        max(node.local_finish_time, node.cloud_receiving_finish_time)
        for node in nodes
        if not node.children
    )

def task_migration_algorithm(nodes, sequences, initial_time, initial_energy, T_max):
    current_time = initial_time
    current_energy = initial_energy
    improved = True
    while improved:
        improved = False
        local_tasks = [node for node in nodes if node.is_core]
        best_energy_reduction = 0
        best_task_to_migrate = None
        best_new_sequences = None
        best_new_time = None
        best_new_energy = None
        for task in local_tasks:
            original_assignment = task.assignment
            original_is_core = task.is_core
            original_times = (task.local_ready_time, task.local_finish_time)
            sequences[task.assignment].remove(task)
            task.is_core = False
            task.assignment = 3
            insert_task_into_sequence(task, sequences[3])
            temp_sequences = [seq.copy() for seq in sequences]
            success = kernel_rescheduling(nodes, temp_sequences)
            if success:
                new_energy = total_energy(nodes, core_powers=[1, 2, 4], cloud_sending_power=0.5)
                new_time = total_time(nodes)
                if new_time <= T_max:
                    energy_reduction = current_energy - new_energy
                    if energy_reduction > best_energy_reduction:
                        best_energy_reduction = energy_reduction
                        best_task_to_migrate = task
                        best_new_sequences = [seq.copy() for seq in temp_sequences]
                        best_new_time = new_time
                        best_new_energy = new_energy
            sequences[3].remove(task)
            task.is_core = original_is_core
            task.assignment = original_assignment
            task.local_ready_time, task.local_finish_time = original_times
            sequences[original_assignment].append(task)
            kernel_rescheduling(nodes, sequences)
        if best_task_to_migrate:
            improved = True
            sequences = best_new_sequences
            current_energy = best_new_energy
            current_time = best_new_time
            best_task_to_migrate.is_core = False
            best_task_to_migrate.assignment = 3
            kernel_rescheduling(nodes, sequences)
    return sequences, current_time, current_energy

def insert_task_into_sequence(task, sequence):
    predecessors = set(parent.task_id for parent in task.parents)
    for idx, seq_task in enumerate(sequence):
        if seq_task.task_id in predecessors:
            continue
        if any(child.task_id == seq_task.task_id for child in task.children):
            sequence.insert(idx, task)
            return
    sequence.append(task)

def topological_sort(nodes):
    from collections import deque
    in_degree = {node: len(node.parents) for node in nodes}
    queue = deque([node for node in nodes if in_degree[node] == 0])
    sorted_nodes = []
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for child in node.children:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return sorted_nodes

def kernel_rescheduling(nodes, sequences):
    sorted_nodes = topological_sort(nodes)
    core_available_times = {k: 0 for k in range(3)}
    cloud_sending_available_time = 0
    cloud_sending_sequence = [node for node in sorted_nodes if not node.is_core]
    
    # Set cloud times for tasks assigned to cloud
    for task in cloud_sending_sequence:
        task.cloud_sending_ready_time = cloud_sending_available_time
        task.cloud_sending_finish_time = task.cloud_sending_ready_time + task.cloud_speed[0]
        cloud_sending_available_time = task.cloud_sending_finish_time
        
        # Collect finish times from cloud and core parents
        cloud_parent_finish_times = [parent.cloud_receiving_finish_time for parent in task.parents if not parent.is_core]
        core_parent_finish_times = [parent.local_finish_time for parent in task.parents if parent.is_core]
        
        # Default to task.cloud_sending_finish_time if no parents
        cloud_parent_finish_times = cloud_parent_finish_times or [task.cloud_sending_finish_time]
        core_parent_finish_times = core_parent_finish_times or [task.cloud_sending_finish_time]
        
        task.cloud_ready_time = max(task.cloud_sending_finish_time, 
                                    max(cloud_parent_finish_times), 
                                    max(core_parent_finish_times))
        task.cloud_finish_time = task.cloud_ready_time + task.cloud_speed[1]
        task.cloud_receiving_finish_time = task.cloud_finish_time + task.cloud_speed[2]
    
    # Set local times for tasks assigned to cores
    for task in sorted_nodes:
        if task.is_core:
            ready_times = [parent.local_finish_time for parent in task.parents if parent.is_core]
            ready_times += [parent.cloud_receiving_finish_time for parent in task.parents if not parent.is_core]
            ready_time = max(ready_times) if ready_times else 0
            start_time = max(ready_time, core_available_times[task.assignment])
            task.local_ready_time = start_time
            task.local_finish_time = start_time + task.core_speed[task.assignment]
            core_available_times[task.assignment] = task.local_finish_time
    
    return True

if __name__ == "__main__":
    # Initialize task graph with correct parents and children
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

    # Initial Scheduling
    primary_assignment(nodes)
    task_prioritizing(nodes)
    sequences = execution_unit_selection(nodes)
    initial_time = total_time(nodes)
    initial_energy = total_energy(nodes, core_powers=[1, 2, 4], cloud_sending_power=0.5)

    # Prepare initial schedule details
    initial_schedule = []
    for node in nodes:
        if node.is_core:
            initial_schedule.append({
                'node id': node.task_id,
                'assignment': node.assignment + 1,  # Assuming core indices start from 1
                'local_start_time': node.local_ready_time,
                'local_finish_time': node.local_finish_time,
            })
        else:
            initial_schedule.append({
                'node id': node.task_id,
                'assignment': 4,  # Cloud assignment
                'ws_start_time': node.cloud_sending_ready_time,
                'ws_finish_time': node.cloud_sending_finish_time,
                'cloud_start_time': node.cloud_ready_time,
                'cloud_finish_time': node.cloud_finish_time,
                'wr_start_time': node.cloud_finish_time,
                'wr_finish_time': node.cloud_receiving_finish_time,
            })
    initial_schedule.sort(key=lambda x: x['node id'])

    # Print initial schedule details
    print("Initial Schedule Details:")
    for detail in initial_schedule:
        print(detail)

    # Task Migration
    T_max = initial_time * 1.5  # As per your expected final time
    sequences, final_time, final_energy = task_migration_algorithm(
        nodes, sequences, initial_time, initial_energy, T_max
    )

    # Prepare final schedule details
    final_schedule = []
    for node in nodes:
        if node.is_core:
            final_schedule.append({
                'node id': node.task_id,
                'assignment': node.assignment + 1,
                'local start_time': node.local_ready_time,
                'local finish_time': node.local_finish_time,
            })
        else:
            final_schedule.append({
                'node id': node.task_id,
                'assignment': 4,
                'ws start_time': node.cloud_sending_ready_time,
                'ws finish_time': node.cloud_sending_finish_time,
                'cloud start_time': node.cloud_ready_time,
                'cloud finish_time': node.cloud_finish_time,
                'wr start_time': node.cloud_finish_time,
                'wr finish_time': node.cloud_receiving_finish_time,
            })
    final_schedule.sort(key=lambda x: x['node id'])

    # Print final schedule details
    print(f"\nFINAL TIME: {int(final_time)}")
    print(f"FINAL ENERGY: {final_energy}")
    print("Final Schedule Details:")
    for detail in final_schedule:
        print(detail)

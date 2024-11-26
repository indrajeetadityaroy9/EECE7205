import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
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
    """
    Implements Section II.C.1 and II.C.2 for calculating ready times:
    - Equation (3): RT_i^l = max_{v_j ∈ pred(v_i)} max{FT_j^l, FT_j^wr}
    - Equation (4): RT_i^ws = max_{v_j ∈ pred(v_i)} max{FT_j^l, FT_j^ws}
    """
    if not node.parents:
        return 0, 0
    
    # Equation (3) for local ready time
    local_ready = max(
        max(parent.local_finish_time, parent.cloud_receiving_finish_time)
        for parent in node.parents
    )
    
    # Equation (4) for cloud sending ready time
    cloud_ready = max(
        max(parent.local_finish_time, parent.cloud_sending_finish_time)
        for parent in node.parents
    )
    
    return local_ready, cloud_ready

def primary_assignment(nodes, core_earliest_ready, cloud_earliest_ready):
    """
    Implements Section III.A.1 (Initial Scheduling Algorithm Primary Assignment):
    - Equation (11): T_i^{l,min} = min_{1≤k≤K} T_i^{l,k}
    - Equation (12): T_i^{re} = T_i^s + T_i^c + T_i^r
    - Assigns tasks to cloud if T_i^{re} < T_i^{l,min}
    """
    for node in nodes:
        local_ready, cloud_ready = calculate_earliest_start_time(node, core_earliest_ready, cloud_earliest_ready)
        
        # Equation (11): Find minimum local execution time
        min_local_time = float('inf')
        best_core = -1
        for core in range(3):
            core_time = node.core_speed[core]
            start_time = max(local_ready, core_earliest_ready[core])
            total_time = start_time + core_time
            if total_time < min_local_time:
                min_local_time = total_time
                best_core = core

        # Equation (12): Calculate estimated remote execution time
        cloud_start = max(cloud_ready, cloud_earliest_ready)
        cloud_total_time = cloud_start + node.cloud_execution_time

        # Assignment decision based on equations (11) and (12)
        if cloud_total_time < min_local_time:
            node.is_core = False
            node.assignment = 3
        else:
            node.is_core = True
            node.assignment = best_core

def calculate_priority(task, weights, priority_cache):
    """
    Implements Section III.A.2 (Initial Scheduling Algorithm Task prioritizing):
    - Equation (15): priority(v_i) = w_i + max_{v_j ∈ succ(v_i)} priority(v_j)
    - Equation (16): priority(v_i) = w_i for exit tasks
    """
    if task.task_id in priority_cache:
        return priority_cache[task.task_id]
    
    # Equation (16) for exit tasks
    if not task.children:
        priority_cache[task.task_id] = weights[task.task_id - 1]
        return weights[task.task_id - 1]

    # Equation (15) for non-exit tasks
    max_successor_priority = max(
        calculate_priority(child, weights, priority_cache) 
        for child in task.children
    )
    
    priority = weights[task.task_id - 1] + max_successor_priority
    priority_cache[task.task_id] = priority
    return priority

def task_prioritizing(nodes):
    """
    Implements Section III.A.2 (Initial Scheduling Algorithm Task prioritizing):
    - Equation (13): w_i = T_i^{re} for cloud tasks
    - Equation (14): w_i = avg_{1≤k≤K} T_i^{l,k} for local tasks
    """
    weights = []
    for node in nodes:
         # Equations (13) and (14) for weight calculation
        if node.is_core:
            weights.append(sum(node.core_speed) / len(node.core_speed)) # Equation (14)
        else:
            weights.append(node.cloud_execution_time) # Equation (13)

    priority_cache = {}
    for node in nodes:
        node.priority_score = calculate_priority(node, weights, priority_cache)

def get_possible_finish_times(node, core_earliest_ready, cloud_earliest_ready):
    """
    Implements Section II.C.1 and II.C.2 for calculating ready times:
    - Equation (3) for local ready time
    - Equations (4)-(6) for cloud execution timing
    """
    local_ready, cloud_ready = calculate_earliest_start_time(
        node, core_earliest_ready, cloud_earliest_ready
    )
    
    finish_times = []
    # Calculate local finish times for each core
    for core in range(3):
        start_time = max(local_ready, core_earliest_ready[core])
        finish_time = start_time + node.core_speed[core]
        finish_times.append((finish_time, core, True, start_time))
    
    # Calculate cloud finish time using equations (4)-(6)
    cloud_start = max(cloud_ready, cloud_earliest_ready)
    cloud_finish = cloud_start + node.cloud_execution_time
    finish_times.append((cloud_finish, 3, False, cloud_start))
    
    return finish_times

def execution_unit_selection(nodes):
    """
    Implements Section III.A.3 (Initial Scheduling Algorithm Execution unit selection):
    - Tasks are scheduled in descending order of priorities
    - When task v_i is selected, all its predecessors are already scheduled
      (based on priority(v_j) > priority(v_i) from equation (15))
    """
    k = 3
    core_seqs = [[] for _ in range(k)]
    cloud_seq = []
    core_earliest_ready = [0] * k
    cloud_earliest_ready = 0
    
    # Sort by priority score
    nodes = sorted(nodes, key=lambda x: x.priority_score, reverse=True)
    
    for node in nodes:
        # Get all possible finish times (both core and cloud)
        finish_times = get_possible_finish_times(
            node, core_earliest_ready, cloud_earliest_ready
        )
        
        # Select execution unit with earliest finish time
        finish_time, unit, is_core, start_time = min(finish_times)
        
        if is_core:
            # Local core scheduling:
            # 1. Ensure task-precedence (through ready_time calculation)
            # 2. Schedule on selected core
            # 3. Update core availability
            node.is_core = True
            node.assignment = unit
            node.local_ready_time = start_time
            node.local_finish_time = finish_time
            core_earliest_ready[unit] = finish_time
            core_seqs[unit].append(node.task_id)
        else:
            # Cloud scheduling:
            # 1. RT_i^ws: cloud_sending_ready_time = start_time
            # 2. FT_i^ws: cloud_sending_finish_time = RT_i^ws + T_i^s
            # 3. RT_i^c: cloud_ready_time = FT_i^ws
            # 4. FT_i^c: cloud_finish_time = RT_i^c + T_i^c
            # 5. FT_i^wr: cloud_receiving_finish_time = FT_i^c + T_i^r
            node.is_core = False
            node.assignment = 3
            node.cloud_sending_ready_time = start_time
            node.cloud_sending_finish_time = start_time + node.cloud_speed[0]
            node.cloud_ready_time = node.cloud_sending_finish_time
            node.cloud_finish_time = node.cloud_ready_time + node.cloud_speed[1]
            node.cloud_receiving_finish_time = node.cloud_finish_time + node.cloud_speed[2]
            # Update cloud channel availability
            cloud_earliest_ready = node.cloud_receiving_finish_time
            cloud_seq.append(node.task_id)
        
        node.is_scheduled = True
    
    return core_seqs + [cloud_seq]

def total_energy(nodes, core_powers, cloud_sending_power):
    """
    Implements Section II.D for calculating total energy consumption:
    - Equation (7): E_i^{l,k} = P_k * T_i^{l,k}
    - Equation (8): E_i^c = P^s * T_i^s
    - Equation (9): E^{total} = sum_{i=1}^N E_i
    """
    total = 0.0
    for node in nodes:
        if node.is_core:
            total += core_powers[node.assignment] * node.core_speed[node.assignment] # Equation (7)
        else:
            total += cloud_sending_power * node.cloud_speed[0] # Equation (8)
    return total

def total_time(nodes):
    """
    Implements Section II.D for calculating total application completion time:
    - Equation (10): T^{total} = max_{v_i ∈ exit tasks} max{FT_i^l, FT_i^wr}
    """
    return max(
        max(node.local_finish_time, node.cloud_receiving_finish_time)
        for node in nodes
        if not node.children
    )

def log_task_details(nodes):
    """
    Log execution details for each task
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
                "wr start_time": node.cloud_receiving_finish_time - node.cloud_speed[2],
                "wr finish_time": node.cloud_receiving_finish_time,
            })
        print(result)

def create_and_visualize_task_graph(nodes):
    """
    Create a NetworkX graph from Node objects and visualize it with a hierarchical layout.
    """
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.task_id)
    for node in nodes:
        for child in node.children:
            G.add_edge(node.task_id, child.task_id)
    
    plt.figure(figsize=(8, 10))
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=17)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15)
    plt.axis('off')
    return plt.gcf()

def migrate_tasks(nodes, T_max, core_powers=[1, 2, 4], cloud_sending_power=0.5):
    """
    Enhanced task migration algorithm focusing on energy reduction through cloud offloading
    while maintaining timing constraints.
    """
    best_nodes = deepcopy(nodes)
    best_energy = total_energy(nodes, core_powers, cloud_sending_power)
    best_time = total_time(nodes)
    
    # First try to migrate most energy-intensive tasks to cloud
    energy_savings = []
    for node in nodes:
        if node.is_core:
            # Calculate current energy consumption
            current_energy = core_powers[node.assignment] * node.core_speed[node.assignment]
            # Calculate potential cloud energy consumption
            cloud_energy = cloud_sending_power * cloud_speed[0]
            
            energy_savings.append((
                node,
                current_energy - cloud_energy,
                node.core_speed[node.assignment]
            ))
    
    # Sort tasks by potential energy savings
    energy_savings.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    # Try migrating tasks in order of potential energy savings
    for task, saving, _ in energy_savings:
        if saving <= 0:
            continue
            
        original_assignment = task.assignment
        original_is_core = task.is_core
        
        # Try migrating to cloud
        task.assignment = 3
        task.is_core = False
        
        # Reschedule with new assignment
        new_schedule = kernel_algorithm(nodes)
        if new_schedule is not None:
            new_time = total_time(nodes)
            new_energy = total_energy(nodes, core_powers, cloud_sending_power)
            
            if new_time <= T_max and new_energy < best_energy:
                best_nodes = deepcopy(nodes)
                best_energy = new_energy
                best_time = new_time
            else:
                # Revert changes
                task.assignment = original_assignment
                task.is_core = original_is_core
    
    # Apply best schedule found
    for i, node in enumerate(nodes):
        node.assignment = best_nodes[i].assignment
        node.is_core = best_nodes[i].is_core
        
    # Final rescheduling with best assignments
    kernel_algorithm(nodes)
    
    return nodes, best_energy, best_time

def kernel_algorithm(nodes):
    """
    Enhanced kernel algorithm with improved cloud scheduling strategy
    """
    cloud_tasks = []
    core_tasks = [[] for _ in range(3)]
    
    # Separate cloud and core tasks
    for node in nodes:
        if node.is_core:
            core_tasks[node.assignment].append(node)
        else:
            cloud_tasks.append(node)
    
    # Initialize timing
    current_time = 0
    cloud_send_time = 0
    
    # Schedule cloud tasks first
    for task in sorted(cloud_tasks, key=lambda x: len(x.parents)):
        # Calculate earliest possible start time
        start_time = cloud_send_time
        for parent in task.parents:
            if parent.is_core:
                start_time = max(start_time, parent.local_finish_time)
            else:
                start_time = max(start_time, parent.cloud_receiving_finish_time)
        
        # Set cloud execution times
        task.cloud_sending_ready_time = start_time
        task.cloud_sending_finish_time = start_time + cloud_speed[0]
        task.cloud_ready_time = task.cloud_sending_finish_time
        task.cloud_finish_time = task.cloud_ready_time + cloud_speed[1]
        task.cloud_receiving_finish_time = task.cloud_finish_time + cloud_speed[2]
        
        cloud_send_time = task.cloud_sending_finish_time
    
    # Schedule core tasks
    core_times = [0] * 3
    for core_idx, core_task_list in enumerate(core_tasks):
        current_time = 0
        for task in sorted(core_task_list, key=lambda x: len(x.parents)):
            # Calculate earliest possible start time
            start_time = current_time
            for parent in task.parents:
                if parent.is_core:
                    start_time = max(start_time, parent.local_finish_time)
                else:
                    start_time = max(start_time, parent.cloud_receiving_finish_time)
            
            # Set local execution times
            task.local_ready_time = start_time
            task.local_finish_time = start_time + task.core_speed[core_idx]
            current_time = task.local_finish_time
            core_times[core_idx] = current_time
    
    return True

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

    fig = fig = create_and_visualize_task_graph(nodes)
    plt.savefig('task_graph.png', bbox_inches='tight', dpi=300)
    plt.close()

    # Step One: Initial Scheduling Algorithm
    print("\nRunning Initial Scheduling...")
    core_earliest_ready = [0, 0, 0]
    cloud_earliest_ready = 0
    
    primary_assignment(nodes, core_earliest_ready, cloud_earliest_ready)
    task_prioritizing(nodes)
    sequences = execution_unit_selection(nodes)
    
    initial_time = total_time(nodes)
    initial_energy = total_energy(nodes, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    
    print(f"Initial Schedule Results:")
    print(f"INITIAL TIME: {initial_time}")
    print(f"INITIAL ENERGY: {initial_energy}")
    print("\nInitial Schedule Details:")
    log_task_details(nodes)

    # Step Two: Task Migration
    print("\nRunning Task Migration...")
    T_max = initial_time * 1.5  # Allow 50% increase in completion time
    
    migrated_nodes, final_energy, final_time = migrate_tasks(nodes, T_max)
    
    print(f"\nFinal Schedule Results:")
    print(f"FINAL TIME: {final_time}")
    print(f"FINAL ENERGY: {final_energy}")

    print("\nFinal Schedule Details:")
    log_task_details(migrated_nodes)
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

    def print_attributes(self):
        """Print all attributes of the Node instance in a readable format."""
        print(f"Node ID: {self.task_id}")
        print(f"Parents: {[parent.task_id for parent in self.parents]}")
        print(f"Children: {[child.task_id for child in self.children]}")
        print(f"Local Finish Time: {self.local_finish_time}")
        print(f"Cloud Sending Finish Time: {self.cloud_sending_finish_time}")
        print(f"Cloud Finish Time: {self.cloud_finish_time}")
        print(f"Cloud Receiving Finish Time: {self.cloud_receiving_finish_time}")
        print(f"Local Ready Time: {self.local_ready_time}")
        print(f"Cloud Sending Ready Time: {self.cloud_sending_ready_time}")
        print(f"Cloud Ready Time: {self.cloud_ready_time}")
        print(f"Priority Score: {self.priority_score}")
        print(f"Assignment: {self.assignment} (-2=not assigned, 0-2=cores, 3=cloud)")
        print(f"Is Core: {self.is_core}")
        print(f"Start Times: {self.start_time} (core1, core2, core3, cloud)")
        print(f"Is Scheduled: {self.is_scheduled}")
        print(f"Core Speed: {self.core_speed}")
        print(f"Cloud Speed: {self.cloud_speed}")
        print(f"Cloud Execution Time: {self.cloud_execution_time}")

def calculate_earliest_start_time(node):
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
        local_ready, cloud_ready = calculate_earliest_start_time(node)
        min_local_time = float('inf')
        best_core = -1
        for core in range(3):
            core_time = node.core_speed[core]
            start_time = max(local_ready, core_earliest_ready[core])
            total_time = start_time + core_time
            if total_time < min_local_time:
                min_local_time = total_time
                best_core = core
        cloud_start = max(cloud_ready, cloud_earliest_ready)
        cloud_total_time = cloud_start + node.cloud_execution_time
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
    """
    Implements Section III.A.2 (Initial Scheduling Algorithm Task prioritizing):
    - Equation (13): w_i = T_i^{re} for cloud tasks
    - Equation (14): w_i = avg_{1≤k≤K} T_i^{l,k} for local tasks
    """
    weights = []
    for node in nodes:
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
    local_ready, cloud_ready = calculate_earliest_start_time(node)
    
    finish_times = []
    for core in range(3):
        start_time = max(local_ready, core_earliest_ready[core])
        finish_time = start_time + node.core_speed[core]
        finish_times.append((finish_time, core, True, start_time))
    
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
    tasks_for_plotting = []

    for node in nodes:
        if node.is_core == False:
            task_details = {
                "node id": node.task_id,
                "assignment": node.assignment + 1,
                "cloud start_time": node.cloud_ready_time,
                "cloud finish_time": node.cloud_ready_time + node.cloud_speed[1],
                "ws start_time": node.cloud_sending_ready_time,
                "ws finish_time": node.cloud_sending_ready_time + node.cloud_speed[0],
                "wr start_time": node.cloud_finish_time,
                "wr finish_time": node.cloud_receiving_finish_time + node.cloud_speed[2]
            }
        else:  # Task handled locally
            task_details = {
                "node id": node.task_id,
                "assignment": node.assignment + 1,
                "local start_time": node.local_ready_time,
                "local finish_time": node.local_ready_time + node.core_speed[node.assignment]
            }

        tasks_for_plotting.append(task_details)
    
    return tasks_for_plotting

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

def migrate_tasks(nodes, sequence, T_max_constraint, max_iterations=100):
    """
    Optimizes task migration for a given set of nodes and sequence.

    Parameters:
    - nodes (list): List of node objects with task assignments and other attributes.
    - sequence (list): Initial task sequence.
    - T_max_constraint (float): Maximum time constraint for the schedule.
    - max_iterations (int): Maximum number of optimization iterations (default: 100).

    Returns:
    - Updated nodes and sequence after optimization.
    """
    iter_num = 0
    while iter_num < max_iterations:
        print("-----" * 20)
        print("iter: ", iter_num)
        
        # Calculate initial total time and energy.
        T_init = total_time(nodes)
        E_init = total_energy(nodes, core_powers, cloud_sending_power)
        print(f"initial time: {T_init}")
        print(f"initial energy: {E_init}")
        print("-----" * 20)

        # Initialize migration choices.
        migeff_ratio_choice = [[] for _ in range(len(nodes))]
        for i in range(len(nodes)):
            current_row_id = nodes[i].task_id - 1
            if nodes[i].assignment == 3:
                migeff_ratio_choice[current_row_id] = [1] * 4
            else:
                current_row_value = [0] * 4
                current_row_value[nodes[i].assignment] = 1
                migeff_ratio_choice[current_row_id] = current_row_value

        # Initialize result table for time and energy of each migration.
        result_table = [[(-1, -1) for _ in range(4)] for _ in range(len(nodes))]

        # Evaluate potential migrations.
        for n in range(len(migeff_ratio_choice)):
            nth_row = migeff_ratio_choice[n]
            for k in range(len(nth_row)):
                if nth_row[k] == 1:
                    continue

                # Simulate migration.
                seq_copy = deepcopy(sequence)
                nodes_copy = deepcopy(nodes)
                seq_copy = new_sequence(nodes_copy, n + 1, k, seq_copy)
                kernel_algorithm(nodes_copy, seq_copy)

                # Calculate total time and energy for this migration.
                current_T = total_time(nodes_copy)
                current_E = total_energy(nodes_copy, core_powers, cloud_sending_power)
                result_table[n][k] = (current_T, current_E)

        # Determine the best migration.
        n_best, k_best = -1, -1
        T_best, E_best = T_init, E_init
        eff_ratio_best = -1
        for i in range(len(result_table)):
            for j in range(len(result_table[i])):
                val = result_table[i][j]
                if val == (-1, -1) or val[0] > T_max_constraint:
                    continue

                eff_ratio = (E_best - val[1]) / abs(val[0] - T_best + 0.00005)
                if eff_ratio > eff_ratio_best:
                    eff_ratio_best = eff_ratio
                    n_best, k_best = i, j

        # If no improvement, exit loop.
        if n_best == -1 and k_best == -1:
            break

        # Apply the best migration.
        n_best += 1
        k_best += 1
        T_best, E_best = result_table[n_best - 1][k_best - 1]
        print(f"\ncurrent migration: task:{n_best}, k: {k_best}, total time: {T_best}, total energy: {E_best}")

        # Update sequence and apply kernel algorithm.
        print("\nupdate after current outer loop")
        sequence = new_sequence(nodes, n_best, k_best - 1, sequence)
        kernel_algorithm(nodes, sequence)

        # Print updated sequence.
        for s in sequence:
            print([i for i in s])

        # Recalculate current time and energy.
        T_current = total_time(nodes)
        E_current = total_energy(nodes, core_powers, cloud_sending_power)
        E_diff = E_init - E_current

        # Print post-migration stats.
        print(f"\npost migration time: {T_current}")
        print(f"post migration energy: {E_current}")

        # Increment iteration count.
        iter_num += 1

        # Exit if energy improvement is minimal.
        if E_diff <= 1:
            break

    return nodes, sequence

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

# Update validate_schedule function to handle cloud execution better
def validate_schedule(nodes):
    """
    Enhanced schedule validation with specific cloud execution checks
    """
    # Check precedence constraints
    for node in nodes:
        if node.is_core:
            start_time = node.local_ready_time
            finish_time = node.local_finish_time
        else:
            start_time = node.cloud_sending_ready_time
            finish_time = node.cloud_receiving_finish_time
            
        for parent in node.parents:
            parent_finish = parent.cloud_receiving_finish_time if not parent.is_core else parent.local_finish_time
            if start_time < parent_finish:
                return False
    
    # Check resource conflicts
    cloud_send_times = []
    cloud_receive_times = []
    core_times = [[] for _ in range(3)]
    
    for node in nodes:
        if node.is_core:
            times = core_times[node.assignment]
            times.append((node.local_ready_time, node.local_finish_time))
        else:
            cloud_send_times.append((node.cloud_sending_ready_time, node.cloud_sending_finish_time))
            cloud_receive_times.append((
                node.cloud_finish_time,
                node.cloud_receiving_finish_time
            ))
    
    # Check for overlaps in all resource usage
    for times in [*core_times, cloud_send_times, cloud_receive_times]:
        times.sort()
        for i in range(1, len(times)):
            if times[i][0] < times[i-1][1]:
                return False
    
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
    
    # Set up parent relationships
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

    # Define system parameters
    core_powers = [1, 2, 4]  # Power consumption of cores 1, 2, and 3
    cloud_sending_power = 0.5  # Power consumption for sending data to cloud

    print("\n=== Step One: Initial Scheduling ===")
    # Execute the three phases of initial scheduling
    print("\nPhase 1: Primary Assignment...")
    core_earliest_ready = [0, 0, 0]
    cloud_earliest_ready = 0
    primary_assignment(nodes, core_earliest_ready, cloud_earliest_ready)
    print("Phase 2: Task Prioritizing...")
    task_prioritizing(nodes)
    print("Phase 3: Execution Unit Selection...")
    sequence = execution_unit_selection(nodes)

    #Total time and energy at the end of initial scheduling 
    T_init_pre_kernel = total_time(nodes)
    T_init= T_init_pre_kernel
    E_init_pre_kernel = total_energy(nodes, core_powers, cloud_sending_power)
    E_init= E_init_pre_kernel
    print("INITIAL TIME: ", T_init_pre_kernel)
    print("INITIAL ENERGY:", E_init_pre_kernel)

    tasksForPlotting = log_task_details(nodes)
    
    for task in tasksForPlotting:
        print(task)




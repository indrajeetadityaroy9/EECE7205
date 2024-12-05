from copy import deepcopy
import bisect
from dataclasses import dataclass
from collections import deque
import numpy as np
from heapq import heappush, heappop

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
    10: [7, 4, 2]
}
cloud_speed = [3, 1, 1]  # T_send, T_cloud, T_receive

@dataclass
class OptimizationState:
    """Maintains the state of optimization to avoid recalculations"""
    time: float
    energy: float
    efficiency_ratio: float
    node_index: int
    target_resource: int

# define node class
class Node(object):
    def __init__(self, id, parents=None, children=None):
        self.id = id # node id
        self.parents = parents or []
        self.children = children or []
        self.core_speed = task_core_values[id]
        self.cloud_speed = cloud_speed
        self.remote_execution_time = sum(cloud_speed)
        self.local_finish_time = 0 # local finish time, inf at start
        self.wireless_sending_finish_time = 0 #wireless sending finish time
        self.cloud_finish_time = 0 #cloud finish time
        self.wireless_recieving_finish_time = 0 #wireless recieving finish time
        self.local_ready_time = -1 # local ready time
        self.wireless_sending_ready_time = -1 # cloud ready time
        self.cloud_ready_time = -1 #cloud ready time
        self.wireless_recieving_ready_time = -1 #wireless recieving ready time 
        self.priority_score = None
        self.assignment = -2 # 0 (core), 1 (core), 2 (core), 3 (cloud)
        self.is_core = False #Is the task occuring on a core or on cloud
        self.start_time = [-1, -1, -1, -1] # start time for core1, core2, core3, cloud
        self.is_scheduled = None #Has the task been schedueled

def total_time(nodes):
    return max(
        max(node.local_finish_time, node.wireless_recieving_finish_time)
        for node in nodes
        if not node.children
    )

def calculate_energy_consumption(node, core_powers, cloud_sending_power):
    if node.is_core:
        return core_powers[node.assignment] * node.core_speed[node.assignment]
    else:
        return cloud_sending_power * node.cloud_speed[0]

def total_energy(nodes, core_powers, cloud_sending_power):
    return sum(calculate_energy_consumption(node, core_powers, cloud_sending_power)  for node in nodes)

def calculate_earliest_start_time(node):
    if not node.parents:
        return 0, 0
    local_ready = max(
        max(parent.local_finish_time, parent.wireless_recieving_finish_time)
        for parent in node.parents
    )
    cloud_ready = local_ready  # Ensure both local and cloud ready times are the same
    return local_ready, cloud_ready

def primary_assignment(nodes):
    for node in nodes:
        local_ready, cloud_ready = calculate_earliest_start_time(node)
        min_local_time = float('inf')
        best_core = -1
        for core in range(3):
            core_time = node.core_speed[core]
            start_time = max(local_ready, 0)
            total_time = start_time + core_time
            if total_time < min_local_time:
                min_local_time = total_time
                best_core = core
        cloud_start = max(cloud_ready, 0)
        cloud_total_time = cloud_start + node.remote_execution_time
        if cloud_total_time < min_local_time:
            node.is_core = False
            node.assignment = 3
        else:
            node.is_core = True
            node.assignment = best_core

def calculate_priority(task, weights, priority_cache):
    if task.id in priority_cache:
        return priority_cache[task.id]
    if not task.children:
        priority_cache[task.id] = weights[task.id - 1]
        return weights[task.id - 1]
    max_successor_priority = max(
        calculate_priority(child, weights, priority_cache)
        for child in task.children
    )
    priority = weights[task.id - 1] + max_successor_priority
    priority_cache[task.id] = priority
    return priority

def task_prioritizing(nodes):
    weights = []
    for node in nodes:
        if node.is_core:
            weights.append(sum(node.core_speed) / len(node.core_speed))
        else:
            weights.append(node.remote_execution_time)
    priority_cache = {}
    for node in nodes:
        node.priority_score = calculate_priority(node, weights, priority_cache)

def get_possible_finish_times(node, core_available_times, ws_channel_available_time):
    local_ready, cloud_ready = calculate_earliest_start_time(node)
    
    # Core finish times
    start_times = np.maximum(local_ready, np.array(core_available_times))
    core_finish_times = start_times + np.array(node.core_speed[:3])
    finish_times = [
        {'finish_time': finish_time, 'unit': core, 'is_core': True, 'start_time': start_time}
        for core, (finish_time, start_time) in enumerate(zip(core_finish_times, start_times))
    ]
    
    # Cloud finish time
    ws_sending_start = max(cloud_ready, ws_channel_available_time)
    cloud_timing = {
        'ws_sending_finish': ws_sending_start + node.cloud_speed[0],
        'cloud_execution_finish': ws_sending_start + sum(node.cloud_speed[:2]),
        'ws_receiving_finish': ws_sending_start + sum(node.cloud_speed)
    }
    finish_times.append({
        'finish_time': cloud_timing['ws_receiving_finish'],
        'unit': 3,
        'is_core': False,
        'start_time': ws_sending_start,
        **cloud_timing
    })

    return finish_times

def execution_unit_selection(nodes):
    sequences = [[] for _ in range(4)]  # 3 cores and cloud
    core_available_times = [0] * 3
    ws_channel_available_time = 0

    # Sort nodes based on priority score in descending order
    nodes_sorted = sorted(nodes, key=lambda x: x.priority_score, reverse=True)

    for node in nodes_sorted:
        finish_times = get_possible_finish_times(node, core_available_times, ws_channel_available_time)

        # Select the option with the earliest finish time
        earliest_option = min(finish_times, key=lambda x: x['finish_time'])

        if earliest_option['is_core']:
            unit = earliest_option['unit']
            node.is_core = True
            node.assignment = unit
            node.local_ready_time = earliest_option['start_time']
            node.local_finish_time = earliest_option['finish_time']
            core_available_times[unit] = node.local_finish_time
            node.start_time[unit] = node.local_ready_time
            node.final_finish_time = node.local_finish_time  # Final finish time
            sequences[unit].append(node.id)
        else:
            node.is_core = False
            node.assignment = 3
            node.wireless_sending_ready_time = earliest_option['start_time']
            node.wireless_sending_finish_time = earliest_option['ws_sending_finish']
            node.cloud_ready_time = node.wireless_sending_finish_time
            node.cloud_finish_time = earliest_option['cloud_execution_finish']
            node.wireless_recieving_ready_time = node.cloud_finish_time
            node.wireless_recieving_finish_time = earliest_option['ws_receiving_finish']
            ws_channel_available_time = node.wireless_sending_finish_time
            node.final_finish_time = node.wireless_recieving_finish_time  # Final finish time
            node.start_time[3] = node.wireless_sending_ready_time
            sequences[3].append(node.id)

        node.is_scheduled = True  # Mark the node as scheduled

    return sequences

def new_sequence(nodes, targetNodeId, targetLocation, seq):
    # Step 1: Map node IDs to node objects for quick lookup.
    node_id_to_node = {node.id: node for node in nodes}
    # Step 2: Validate inputs and locate the target node.
    target_node = node_id_to_node.get(targetNodeId)
    # Step 3: Determine the ready time of the target node.
    target_node_rt = target_node.local_ready_time if target_node.is_core else target_node.wireless_sending_ready_time
    # Step 4: Remove the target node from its original sequence.
    original_assignment = target_node.assignment
    seq[original_assignment].remove(target_node.id)
    # Step 5: Prepare the new sequence for insertion.
    new_sequence_list = seq[targetLocation]
    # Precompute start times for the new sequence's nodes.
    start_times = [
        node_id_to_node[node_id].start_time[targetLocation]
        for node_id in new_sequence_list
    ]
    # Step 6: Use bisect to find the insertion index.
    insertion_index = bisect.bisect_left(start_times, target_node_rt)
    # Step 7: Insert the target node at the correct index.
    new_sequence_list.insert(insertion_index, target_node.id)
    # Step 8: Update the target node's assignment and status.
    target_node.assignment = targetLocation
    target_node.is_core = (targetLocation != 3)  # Location 3 is the cloud.
    return seq

def kernel_algorithm(nodes, sequences):
    def initialize_readiness_tracking(nodes, sequences):
    # Initialize dependency_ready with the number of parent tasks for each node
        dependency_ready = [len(node.parents) for node in nodes]
    
    # Initialize sequence_ready to -1 for all nodes
        sequence_ready = [-1] * len(nodes)
    
    # Mark first node in each sequence as ready in terms of sequence order
        for sequence in sequences:
            if sequence:
                sequence_ready[sequence[0] - 1] = 0
            else:
            # Handle empty sequences if necessary
                continue
            
        return dependency_ready, sequence_ready


    def update_node_readiness(node, nodes, sequences, dependency_ready, sequence_ready):
        if node.is_scheduled != "kernel_scheduled":
            # Update dependency readiness
            dependency_ready[node.id - 1] = sum(1 for parent in node.parents 
                                              if parent.is_scheduled != "kernel_scheduled")
            
            # Update sequence readiness
            for sequence in sequences:
                if node.id in sequence:
                    idx = sequence.index(node.id)
                    if idx > 0:
                        prev_node = nodes[sequence[idx - 1] - 1]
                        sequence_ready[node.id - 1] = 1 if prev_node.is_scheduled != "kernel_scheduled" else 0
                    else:
                        sequence_ready[node.id - 1] = 0
                    break

    def schedule_local_task(node, local_core_ready_times):
        # Calculate ready time based on parent completion
        if not node.parents:
            node.local_ready_time = 0
        else:
            parent_completion_times = (max(parent.local_finish_time, parent.wireless_recieving_finish_time) 
                                    for parent in node.parents)
            node.local_ready_time = max(parent_completion_times, default=0)
        
        # Schedule on assigned core
        core_index = node.assignment
        node.start_time = [-1] * 4
        node.start_time[core_index] = max(local_core_ready_times[core_index], node.local_ready_time)
        node.local_finish_time = node.start_time[core_index] + node.core_speed[core_index]
        
        # Update core ready time
        local_core_ready_times[core_index] = node.local_finish_time
        
        # Clear cloud-related timings
        node.wireless_sending_finish_time = node.cloud_finish_time = node.wireless_recieving_finish_time = -1

    def schedule_cloud_task(node, cloud_stage_ready_times):
        # Calculate wireless sending ready time
        if not node.parents:
            node.wireless_sending_ready_time = 0
        else:
            parent_completion_times = (max(parent.local_finish_time, parent.wireless_sending_finish_time) 
                                    for parent in node.parents)
            node.wireless_sending_ready_time = max(parent_completion_times)

        # Initialize start times
        node.start_time = [-1] * 4
        node.start_time[3] = max(cloud_stage_ready_times[0], node.wireless_sending_ready_time)

        # Schedule wireless sending
        node.wireless_sending_finish_time = node.start_time[3] + node.cloud_speed[0]
        cloud_stage_ready_times[0] = node.wireless_sending_finish_time

        # Schedule cloud processing
        node.cloud_ready_time = max(
            node.wireless_sending_finish_time,
            max((parent.cloud_finish_time for parent in node.parents), default=0)
        )
        node.cloud_finish_time = max(cloud_stage_ready_times[1], node.cloud_ready_time) + node.cloud_speed[1]
        cloud_stage_ready_times[1] = node.cloud_finish_time

        # Schedule wireless receiving
        node.wireless_recieving_ready_time = node.cloud_finish_time
        node.wireless_recieving_finish_time = (max(cloud_stage_ready_times[2], node.wireless_recieving_ready_time) 
                                             + node.cloud_speed[2])
        cloud_stage_ready_times[2] = node.wireless_recieving_finish_time
        
        # Clear local timing
        node.local_finish_time = -1

    # Initialize timing trackers
    local_core_ready_times = [0] * 3
    cloud_stage_ready_times = [0] * 3
    
    # Initialize readiness tracking
    dependency_ready, sequence_ready = initialize_readiness_tracking(nodes, sequences)
    
    # Initialize processing queue with ready nodes
    queue = deque(
        node for node in nodes 
        if sequence_ready[node.id - 1] == 0 
        and all(parent.is_scheduled == "kernel_scheduled" for parent in node.parents)
    )
    
    # Main scheduling loop
    while queue:
        current_node = queue.popleft()
        current_node.is_scheduled = "kernel_scheduled"
        
        # Schedule task based on type
        if current_node.is_core:
            schedule_local_task(current_node, local_core_ready_times)
        else:
            schedule_cloud_task(current_node, cloud_stage_ready_times)
        
        # Update readiness status for remaining nodes
        for node in nodes:
            update_node_readiness(node, nodes, sequences, dependency_ready, sequence_ready)
            
            # Add newly ready nodes to queue
            if (dependency_ready[node.id - 1] == 0 
                and sequence_ready[node.id - 1] == 0 
                and node.is_scheduled != "kernel_scheduled" 
                and node not in queue):
                queue.append(node)
    
    # Reset scheduling status
    for node in nodes:
        node.is_scheduled = None
        
    return nodes

def optimize_task_scheduling(nodes, sequence, T_init_pre_kernel, core_powers=[1, 2, 4], cloud_sending_power=0.5):
    """
    Optimized task scheduling algorithm using efficient data structures
    and memoization to reduce computational overhead.
    """
    # Use numpy arrays for faster numerical operations
    core_powers = np.array(core_powers)
    
    # Cache for storing evaluated migrations
    migration_cache = {}
    
    def get_cache_key(node_idx, target_resource):
        """Generate unique cache key for each migration scenario"""
        return (node_idx, target_resource, tuple(node.assignment for node in nodes))
    
    def evaluate_migration(nodes, seqs, node_idx, target_resource):
        """
        Evaluates migration with caching to avoid redundant calculations.
        Uses numpy for faster numerical operations.
        """
        cache_key = get_cache_key(node_idx, target_resource)
        if cache_key in migration_cache:
            return migration_cache[cache_key]
            
        seq_copy = [seq.copy() for seq in seqs]
        nodes_copy = deepcopy(nodes)
        
        seq_copy = new_sequence(nodes_copy, node_idx + 1, target_resource, seq_copy)
        kernel_algorithm(nodes_copy, seq_copy)
        
        current_T = total_time(nodes_copy)
        current_E = total_energy(nodes_copy, core_powers, cloud_sending_power)
        
        migration_cache[cache_key] = (current_T, current_E)
        return current_T, current_E

    def initialize_migration_choices(nodes):
        """Uses boolean array for efficient storage of migration choices"""
        num_nodes = len(nodes)
        choices = np.zeros((num_nodes, 4), dtype=bool)
        
        for i, node in enumerate(nodes):
            if node.assignment == 3:  # Cloud-assigned node
                choices[i, :] = True
            else:
                choices[i, node.assignment] = True
                
        return choices

    def find_best_migration(migration_results, T_init, E_init, T_max_constraint):
    
    # Step 1: Look for migrations that reduce energy without increasing time
        best_energy_reduction = 0
        best_migration = None
    
        for node_idx, resource_idx, time, energy in migration_results:
        # Skip if time constraint violated
            if time > T_max_constraint:
                continue
            
        # Calculate energy reduction
            energy_reduction = E_init - energy
        
        # Check if this migration reduces energy without increasing time
            if time <= T_init and energy_reduction > 0:
                if energy_reduction > best_energy_reduction:
                    best_energy_reduction = energy_reduction
                    best_migration = (node_idx, resource_idx, time, energy)
    
    # If we found a valid migration in Step 1, return it
        if best_migration:
            node_idx, resource_idx, time, energy = best_migration
            return OptimizationState(
            time=time,
            energy=energy,
            efficiency_ratio=best_energy_reduction,
            node_index=node_idx + 1,
            target_resource=resource_idx + 1
            )
    
    # Step 2: If no energy-reducing migrations found, look for best efficiency ratio
        candidates = []
        for node_idx, resource_idx, time, energy in migration_results:
        # Skip if time constraint violated
            if time > T_max_constraint:
                continue
            
        # Calculate efficiency ratio only if there's energy reduction
            energy_reduction = E_init - energy
            if energy_reduction > 0:
            # Calculate ratio of energy reduction to time increase
                time_increase = max(0, time - T_init)
                if time_increase == 0:
                    efficiency_ratio = float('inf')  # Prioritize no time increase
                else:
                    efficiency_ratio = energy_reduction / time_increase
            
                heappush(candidates, (-efficiency_ratio, node_idx, resource_idx, time, energy))
    
        if not candidates:
            return None
        
        neg_ratio, n_best, k_best, T_best, E_best = heappop(candidates)
        return OptimizationState(
        time=T_best,
        energy=E_best,
        efficiency_ratio=-neg_ratio,
        node_index=n_best + 1,
        target_resource=k_best + 1
        )

    # Main optimization loop

    current_energy = total_energy(nodes, core_powers, cloud_sending_power)
    
    # Continue as long as we can improve energy consumption
    energy_improved = True
    while energy_improved:
        # Store current energy as reference for this iteration
        previous_energy = current_energy
        
        # Calculate current schedule metrics
        current_time = total_time(nodes)
        T_max_constraint = T_init_pre_kernel * 1.5
        
        # Initialize possible migration choices
        migeff_ratio_choice = initialize_migration_choices(nodes)
        
        # Evaluate all possible migrations
        migration_results = []
        for node_idx in range(len(nodes)):
            for target_location in range(4):  # 0-3 for cloud and local cores
                if migeff_ratio_choice[node_idx, target_location]:
                    continue
                    
                trial_time, trial_energy = evaluate_migration(
                    nodes, sequence, node_idx, target_location)
                migration_results.append((node_idx, target_location, trial_time, trial_energy))
        
        # Find the best migration according to paper's criteria
        best_migration = find_best_migration(
            migration_results=migration_results,
            T_init=current_time,
            E_init=previous_energy,
            T_max_constraint=T_max_constraint
        )
        
        # If no valid migration exists, we're done
        if best_migration is None:
            energy_improved = False
            break
        
        # Apply the selected migration
        sequence = new_sequence(
            nodes,
            best_migration.node_index,
            best_migration.target_resource - 1,
            sequence
        )
        kernel_algorithm(nodes, sequence)
        
        # Calculate new energy and determine if we improved
        current_energy = total_energy(nodes, core_powers, cloud_sending_power)
        energy_improved = current_energy < previous_energy
        
        # Periodic cache cleanup to manage memory
        if len(migration_cache) > 1000:
            migration_cache.clear()

    return nodes, sequence

def process_nodes_for_plotting(nodes):
    assignment_mapping = {
        0: "Core 1",
        1: "Core 2",
        2: "Core 3",
        3: "Cloud",
        -2: "Not Scheduled"
    }

    tasks_for_plotting = []
    
    for node in nodes:
        assignment_value = assignment_mapping.get(node.assignment, "Unknown")

        if node.is_core:
            tasks_for_plotting.append({
                "node id": node.id,
                "assignment": assignment_value,
                "core start_time": node.start_time[node.assignment],
                "core finish_time": node.start_time[node.assignment] + node.core_speed[node.assignment]
            })
        else:
            tasks_for_plotting.append({
                "node id": node.id,
                "assignment": assignment_value,
                "wireless sending start_time": node.start_time[3],
                "wireless sending finish_time": node.start_time[3] + node.cloud_speed[0],
                "cloud start_time": node.cloud_ready_time,
                "cloud finish_time": node.cloud_ready_time + node.cloud_speed[1],
                "wireless receiving start_time": node.wireless_recieving_ready_time,
                "wireless receiving finish_time": node.wireless_recieving_ready_time + node.cloud_speed[2]
            })

    for task in tasks_for_plotting:
        print(task)

if __name__ == '__main__':
    # Define the nodes
    """
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
    """
    node10 = Node(10)
    node9 = Node(9, children=[])
    node8 = Node(8, children=[node10])
    node7 = Node(7, children=[node10])
    node6 = Node(6, children=[node9])
    node5 = Node(5, children=[node9])
    node4 = Node(4, children=[node7, node8])
    node3 = Node(3, children=[node6, node7])
    node2 = Node(2, children=[node5, node6])
    node1 = Node(1, children=[node2, node3, node4])
    node10.parents = [node7, node8]
    node9.parents = [node5, node6]
    node8.parents = [node4]
    node7.parents = [node3, node4]
    node6.parents = [node2, node3]
    node5.parents = [node2]
    node4.parents = [node1]
    node3.parents = [node1]
    node2.parents = [node1]
    node1.parents = []

    nodes = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10]

    primary_assignment(nodes)
    task_prioritizing(nodes)
    sequence = execution_unit_selection(nodes)
    T_init_pre_kernel = total_time(nodes)
    T_init= T_init_pre_kernel
    E_init_pre_kernel = total_energy(nodes, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    E_init= E_init_pre_kernel
    print("INITIAL TIME: ", T_init_pre_kernel)
    print("INITIAL ENERGY:", E_init_pre_kernel)
    print("INITIAL TASK SCHEDULE: ")
    process_nodes_for_plotting(nodes)

    nodes2, sequence = optimize_task_scheduling(nodes, sequence, T_init_pre_kernel, core_powers=[1, 2, 4], cloud_sending_power=0.5)

    print("final sequence: ")
    for s in sequence:
        print([i for i in s])

    T_final = total_time(nodes)
    E_final = total_energy(nodes, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    print("FINAL TIME: ", T_final)
    print("FINAL ENERGY:", E_final)
    print("FINAL TASK SCHEDULE: ")
    process_nodes_for_plotting(nodes2)


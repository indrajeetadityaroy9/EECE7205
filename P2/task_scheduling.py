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
    10: [7, 4, 2],
    11: [12, 3, 3],
    12: [12, 8, 4],
    13: [11, 3, 2],
    14: [12, 11, 4],
    15: [13, 4, 2],
    16: [9, 7, 3],
    17: [9, 3, 3],
    18: [13, 9, 2],
    19: [10, 5, 3],
    20: [12, 5, 4]
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

class Node(object):
    def __init__(self, id, parents=None, children=None):
        self.id = id # node id
        self.parents = parents or []
        self.children = children or []
        self.core_speed = task_core_values[id]
        self.cloud_speed = cloud_speed
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
        max(node.local_finish_time, node.wireless_recieving_finish_time) for node in nodes
        if not node.children
    )

def calculate_energy_consumption(node, core_powers, cloud_sending_power):
    if node.is_core:
        return core_powers[node.assignment] * node.core_speed[node.assignment]
    else:
        return cloud_sending_power * node.cloud_speed[0]

def total_energy(nodes, core_powers, cloud_sending_power):
    return sum(calculate_energy_consumption(node, core_powers, cloud_sending_power) for node in nodes)

def primary_assignment(nodes):
    for node in nodes:
        # Calculate minimum local execution time
        t_l_min = min(node.core_speed)
        # Calculate total remote execution time
        t_re = (node.cloud_speed[0] + node.cloud_speed[1] + node.cloud_speed[2])
        # If remote execution is faster, assign to cloud
        if t_re < t_l_min:
            node.is_core = False  # Assign to cloud
        else:
            node.is_core = True   # Assign to local core

def task_prioritizing(nodes):
    w = [0] * len(nodes)
    # Calculate computation costs (wi)
    for i, node in enumerate(nodes):
        if not node.is_core:  # Cloud task
            # Equation (13): wi = Tre_i for cloud tasks
            w[i] = (node.cloud_speed[0] +  node.cloud_speed[1] + node.cloud_speed[2])
        else:  # Local task
            # Equation (14): wi = avg(1≤k≤K) Tl,k_i for local tasks
            w[i] = sum(node.core_speed) / len(node.core_speed)

    priority_cache = {}  # For memoization

    def calculate_priority(task):
        # Check if already calculated
        if task.id in priority_cache:
            return priority_cache[task.id]
        # Base case: exit task
        # Equation (16): priority(vi) = wi for exit tasks
        if task.children == []:
            priority_cache[task.id] = w[task.id - 1]
            return w[task.id - 1]
        # Recursive case: Equation (15)
        # priority(vi) = wi + max(vj∈succ(vi)) priority(vj)
        max_successor_priority = max(calculate_priority(successor) for successor in task.children)
        task_priority = w[task.id - 1] + max_successor_priority
        priority_cache[task.id] = task_priority
        return task_priority

    # Calculate priorities for all nodes
    for task in nodes:
        calculate_priority(task)

    # Update priority scores
    for node in nodes:
        node.priority_score = priority_cache[node.id]

def execution_unit_selection(nodes):
    k = 3  # Number of cores
    
    # Initialize sequences and ready times
    sequences = [[] for _ in range(k + 1)]  # k cores + cloud
    core_earliest_ready = [0] * (k + 1)
    
    # Sort by priority (descending order as per paper)
    node_priority_list = [(node.priority_score, node.id) for node in nodes]
    node_priority_list.sort(reverse=True)  # Changed to reverse=True for descending order
    priority_order = [item[1] for item in node_priority_list]
    
    for node_id in priority_order:
        i = node_id - 1
        node = nodes[i]
        
        if not node.parents:  # Entry tasks
            min_load_core = core_earliest_ready.index(min(core_earliest_ready))
            node.local_ready_time = core_earliest_ready[min_load_core]
            node.wireless_sending_ready_time = core_earliest_ready[min_load_core]
            node.wireless_sending_finish_time = ( node.wireless_sending_ready_time + node.cloud_speed[0] )
            node.cloud_ready_time = node.wireless_sending_finish_time
            
        else:  # Tasks with parents
            # Equation (3): Local ready time
            node.local_ready_time = max(
                max(max(parent.local_finish_time, parent.wireless_recieving_finish_time) for parent in node.parents),
                0
            )
            
            # Equation (4): Wireless sending ready time
            node.wireless_sending_ready_time = max(
                max(max(parent.local_finish_time, parent.wireless_sending_finish_time) for parent in node.parents), 
                0
            )
            
            # Calculate wireless sending finish time
            node.wireless_sending_finish_time = (
                max(node.wireless_sending_ready_time, core_earliest_ready[k]) + node.cloud_speed[0]
            )
            
            # Equation (5): Cloud ready time
            node.cloud_ready_time = max(
                node.wireless_sending_finish_time,
                max(parent.cloud_finish_time for parent in node.parents)
            )
        
        # Calculate cloud execution timings
        node.wireless_recieving_ready_time = node.cloud_ready_time + node.cloud_speed[1]
        node.wireless_recieving_finish_time = (
            node.wireless_recieving_ready_time + node.cloud_speed[2]
        )
        
        if not node.is_core:  # Initially assigned to cloud
            node.ft = node.wireless_recieving_finish_time
            node.local_finish_time = 0
            core_earliest_ready[k] = node.wireless_sending_finish_time
            node.start_time[k] = node.wireless_sending_ready_time
            node.assignment = k
            node.is_scheduled = True
            sequences[k].append(node.id)
            
        else:  # Initially assigned to local core
            # Find best local core
            best_finish_time = float('inf')
            best_core = -1
            
            for core in range(k):
                ready_time = max(node.local_ready_time, core_earliest_ready[core])
                finish_time = ready_time + node.core_speed[core]
                
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_core = core
            
            # Compare with cloud execution
            if best_finish_time <= node.wireless_recieving_finish_time:
                # Execute locally
                node.local_finish_time = best_finish_time
                node.ft = best_finish_time
                node.wireless_recieving_finish_time = 0
                node.start_time[best_core] = best_finish_time - node.core_speed[best_core]
                core_earliest_ready[best_core] = best_finish_time
                node.assignment = best_core
                node.is_scheduled = True
                sequences[best_core].append(node.id)
            else:
                # Execute on cloud
                node.ft = node.wireless_recieving_finish_time
                node.local_finish_time = 0
                core_earliest_ready[k] = node.wireless_sending_finish_time
                node.start_time[k] = node.wireless_sending_ready_time
                node.assignment = k
                node.is_core = False
                node.is_scheduled = True
                sequences[k].append(node.id)
    
    return sequences


def construct_sequence(nodes, targetNodeId, targetLocation, seq):
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
    new_sequence_nodes_list = seq[targetLocation]
    # Precompute start times for the new sequence's nodes.
    start_times = [ node_id_to_node[node_id].start_time[targetLocation] for node_id in new_sequence_nodes_list ]
    # Step 6: Use bisect to find the insertion index.
    insertion_index = bisect.bisect_left(start_times, target_node_rt)
    # Step 7: Insert the target node at the correct index.
    new_sequence_nodes_list.insert(insertion_index, target_node.id)
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
            dependency_ready[node.id - 1] = sum(1 for parent in node.parents if parent.is_scheduled != "kernel_scheduled")
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
            parent_completion_times = (max(parent.local_finish_time, parent.wireless_recieving_finish_time) for parent in node.parents)
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
            parent_completion_times = (max(parent.local_finish_time, parent.wireless_sending_finish_time) for parent in node.parents)
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
        node.wireless_recieving_finish_time = (max(cloud_stage_ready_times[2], node.wireless_recieving_ready_time) + node.cloud_speed[2])
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
    core_powers = np.array(core_powers)
    # Cache for storing evaluated migrations
    migration_cache = {}
    
    def get_cache_key(node_idx, target_resource):
        """Generate unique cache key for each migration scenario"""
        return (node_idx, target_resource, tuple(node.assignment for node in nodes))
    
    def evaluate_migration(nodes, seqs, node_idx, target_resource):
        """
        Evaluates migration with caching to avoid redundant calculations.
        """
        cache_key = get_cache_key(node_idx, target_resource)
        if cache_key in migration_cache:
            return migration_cache[cache_key]
            
        seq_copy = [seq.copy() for seq in seqs]
        nodes_copy = deepcopy(nodes)
        
        seq_copy = construct_sequence(nodes_copy, node_idx + 1, target_resource, seq_copy)
        kernel_algorithm(nodes_copy, seq_copy)
        
        current_T = total_time(nodes_copy)
        current_E = total_energy(nodes_copy, core_powers, cloud_sending_power)
        
        migration_cache[cache_key] = (current_T, current_E)
        return current_T, current_E

    def initialize_migration_choices(nodes):
        """Uses boolean array for efficient storage of migration choices"""
        choices = np.zeros((len(nodes), 4), dtype=bool)
        
        for i, node in enumerate(nodes):
            if node.assignment == 3:  # Cloud-assigned node
                choices[i, :] = True
            else:
                choices[i, node.assignment] = True
                
        return choices

    def find_best_migration(migration_trials_results, T_init, E_init, T_max_constraint):
        # Step 1: Look for migrations that reduce energy without increasing time
        best_energy_reduction = 0
        best_migration = None
    
        for node_idx, resource_idx, time, energy in migration_trials_results:
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
        for node_idx, resource_idx, time, energy in migration_trials_results:
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
        migration_trials_results = []
        for node_idx in range(len(nodes)):
            for target_location in range(4):  # 0-3 for cloud and local cores
                if migeff_ratio_choice[node_idx, target_location]:
                    continue
                    
                migration_trial_time, migration_trial_energy = evaluate_migration(nodes, sequence, node_idx, target_location)
                migration_trials_results.append((node_idx, target_location, migration_trial_time, migration_trial_energy))
        
        # Find the best migration according to paper's criteria
        best_migration = find_best_migration(
            migration_trials_results=migration_trials_results,
            T_init=current_time,
            E_init=previous_energy,
            T_max_constraint=T_max_constraint
        )
        
        # If no valid migration exists, we're done
        if best_migration is None:
            energy_improved = False
            break
        
        # Apply the selected migration
        sequence = construct_sequence(
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

def print_task_schedule(nodes):
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

def check_schedule_constraints(nodes):
    """
    Validates schedule constraints considering cloud task pipelining
    
    Args:
        nodes: List of Node objects with scheduling information
    Returns:
        tuple: (is_valid, violations)
    """
    violations = []
    
    def check_sending_channel():
        """Verify wireless sending channel is used sequentially"""
        cloud_tasks = [n for n in nodes if not n.is_core]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.start_time[3])
        
        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            if current.wireless_sending_finish_time > next_task.start_time[3]:
                violations.append({
                    'type': 'Wireless Sending Channel Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} sending ends at {current.wireless_sending_finish_time} but Task {next_task.id} starts at {next_task.start_time[3]}'
                })

    def check_computing_channel():
        """Verify cloud computing is sequential"""
        cloud_tasks = [n for n in nodes if not n.is_core]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.cloud_ready_time)
        
        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            if current.cloud_finish_time > next_task.cloud_ready_time:
                violations.append({
                    'type': 'Cloud Computing Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} computing ends at {current.cloud_finish_time} but Task {next_task.id} starts at {next_task.cloud_ready_time}'
                })

    def check_receiving_channel():
        """Verify wireless receiving channel is sequential"""
        cloud_tasks = [n for n in nodes if not n.is_core]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.wireless_recieving_ready_time)
        
        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            if current.wireless_recieving_finish_time > next_task.wireless_recieving_ready_time:
                violations.append({
                    'type': 'Wireless Receiving Channel Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} receiving ends at {current.wireless_recieving_finish_time} but Task {next_task.id} starts at {next_task.wireless_recieving_ready_time}'
                })

    def check_pipelined_dependencies():
        """Verify dependencies considering pipelined execution"""
        for node in nodes:
            if not node.is_core:  # For cloud tasks
                # Check if all parents have completed necessary phases
                for parent in node.parents:
                    if parent.is_core:
                        # Core parent must complete before child starts sending
                        if parent.local_finish_time > node.start_time[3]:
                            violations.append({
                                'type': 'Core-Cloud Dependency Violation',
                                'parent': parent.id,
                                'child': node.id,
                                'detail': f'Core Task {parent.id} finishes at {parent.local_finish_time} but Cloud Task {node.id} starts sending at {node.start_time[3]}'
                            })
                    else:
                        # Cloud parent must complete sending before child starts sending
                        if parent.wireless_sending_finish_time > node.start_time[3]:
                            violations.append({
                                'type': 'Cloud Pipeline Dependency Violation',
                                'parent': parent.id,
                                'child': node.id,
                                'detail': f'Parent Task {parent.id} sending phase ends at {parent.wireless_sending_finish_time} but Task {node.id} starts sending at {node.start_time[3]}'
                            })
            else:  # For core tasks
                # All parents must complete fully before core task starts
                for parent in node.parents:
                    parent_finish = (parent.wireless_recieving_finish_time 
                                  if not parent.is_core else parent.local_finish_time)
                    if parent_finish > node.start_time[node.assignment]:
                        violations.append({
                            'type': 'Core Task Dependency Violation',
                            'parent': parent.id,
                            'child': node.id,
                            'detail': f'Parent Task {parent.id} finishes at {parent_finish} but Core Task {node.id} starts at {node.start_time[node.assignment]}'
                        })

    def check_core_execution():
        """Verify core tasks don't overlap"""
        core_tasks = [n for n in nodes if n.is_core]
        for core_id in range(3):  # Assuming 3 cores
            core_specific_tasks = [t for t in core_tasks if t.assignment == core_id]
            sorted_tasks = sorted(core_specific_tasks, key=lambda x: x.start_time[core_id])
            
            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]
                
                if current.local_finish_time > next_task.start_time[core_id]:
                    violations.append({
                        'type': f'Core {core_id} Execution Conflict',
                        'task1': current.id,
                        'task2': next_task.id,
                        'detail': f'Task {current.id} finishes at {current.local_finish_time} but Task {next_task.id} starts at {next_task.start_time[core_id]}'
                    })

    # Run all checks
    check_sending_channel()
    check_computing_channel()
    check_receiving_channel()
    check_pipelined_dependencies()
    check_core_execution()
    
    return len(violations) == 0, violations

"""
Pipeline Structure:
Each cloud task follows the three-phase execution described in Section II.B
"If task vi is offloaded onto the cloud, there are three phases in sequence:
(i) the RF sending phase,
(ii) the cloud computing phase, and
(iii) the RF receiving phase."
"The local core in the mobile device or the wireless sending channel can only process or send one task at a time, and preemption is not allowed in this framework. On the other hand, the cloud can execute a large number of tasks in parallel as long as there is no dependency among the tasks."
Task Precedence (Section II.C):


For wireless sending (RTws_i): A task can start sending only after all its predecessors have completed their relevant phases
For cloud computation (RTc_i): A task can start computing after:

It has completed its sending phase
All its cloud-executed predecessors have finished computing


For wireless receiving (RTwr_i): A task can start receiving immediately after cloud computation finishes

"""
def print_validation_report(nodes):
    """Print detailed schedule validation report"""
    is_valid, violations = check_schedule_constraints(nodes)
    
    print("\nSchedule Validation Report")
    print("=" * 50)
    
    if is_valid:
        print("Schedule is valid with all pipelining constraints satisfied!")
    else:
        print("Found constraint violations:")
        for v in violations:
            print(f"\nViolation Type: {v['type']}")
            print(f"Detail: {v['detail']}")
    
    # Print pipeline analysis
    cloud_tasks = [n for n in nodes if not n.is_core]
    if cloud_tasks:
        print("\nCloud Pipeline Analysis:")
        print("=" * 50)
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.start_time[3])
        
        for task in sorted_tasks:
            print(f"\nTask {task.id} Pipeline Stages:")
            print(f"  └─ Sending:   {task.start_time[3]:2d} -> {task.wireless_sending_finish_time:2d}")
            print(f"  └─ Computing: {task.cloud_ready_time:2d} -> {task.cloud_finish_time:2d}")
            print(f"  └─ Receiving: {task.wireless_recieving_ready_time:2d} -> {task.wireless_recieving_finish_time:2d}")
    
    # Print core execution
    core_tasks = [n for n in nodes if n.is_core]
    if core_tasks:
        print("\nCore Execution:")
        print("=" * 50)
        for core_id in range(3):
            core_specific = [t for t in core_tasks if t.assignment == core_id]
            if core_specific:
                print(f"\nCore {core_id}:")
                for task in sorted(core_specific, key=lambda x: x.start_time[core_id]):
                    print(f"  Task {task.id}: {task.start_time[core_id]:2d} -> {task.local_finish_time:2d}")


def print_task_graph(nodes):
        for node in nodes:
            children_ids = [child.id for child in node.children]
            parent_ids = [parent.id for parent in node.parents]
            print(f"Node {node.id}:")
            print(f"  Parents: {parent_ids}")
            print(f"  Children: {children_ids}")
            print()

if __name__ == '__main__':
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

    print_task_graph(nodes)
    
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
    print_task_schedule(nodes)
    print_validation_report(nodes)

    nodes2, sequence = optimize_task_scheduling(nodes, sequence, T_init_pre_kernel, core_powers=[1, 2, 4], cloud_sending_power=0.5)

    print("final sequence: ")
    for s in sequence:
        print([i for i in s])

    T_final = total_time(nodes)
    E_final = total_energy(nodes, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    print("FINAL TIME: ", T_final)
    print("FINAL ENERGY:", E_final)
    print("FINAL TASK SCHEDULE: ")
    print_task_schedule(nodes2)
    print_validation_report(nodes2)
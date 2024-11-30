from copy import deepcopy
import bisect
from dataclasses import dataclass
from collections import deque
import numpy as np
from heapq import heappush, heappop
from enum import Enum

core_execution_times = {
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
cloud_execution_times = [3, 1, 1]  # T_send, T_cloud, T_receive

class SchedulingState(Enum):
    UNSCHEDULED = 0      # Represents None - task hasn't been scheduled yet
    SCHEDULED = 1        # Represents the basic scheduled state
    KERNEL_SCHEDULED = 2 # Represents scheduling done by kernel algorithm

@dataclass
class OptimizationState:
    """Maintains the state of optimization to avoid recalculations"""
    time: float
    energy: float
    efficiency_ratio: float
    node_index: int
    target_resource: int

class Task(object):
    def __init__(self, id, parents=None, children=None):
        # Basic Task Properties
        self.id = id                # Unique task identifier vi
        self.parents = parents or [] # pred(vi) - set of immediate predecessors
        self.children = children or [] # succ(vi) - set of immediate successors
        # Execution Times
        self.core_execution_times = core_execution_times[id]       # Ti,k^l - execution time on k-th core
        self.cloud_execution_times = cloud_execution_times         # [Ti^s, Ti^c, Ti^r] - cloud timing parameters
        # Timing Parameters 
        # Finish times (FT)
        self.local_core_finish_time = 0              # FTi^l
        self.wireless_sending_finish_time = 0        # FTi^ws
        self.remote_cloud_finish_time = 0            # FTi^c
        self.wireless_recieving_finish_time = 0      # FTi^wr
        # Ready times (RT)
        self.local_core_ready_time = -1              # RTi^l
        self.wireless_sending_ready_time = -1        # RTi^ws
        self.remote_cloud_ready_time = -1            # RTi^c
        self.wireless_recieving_ready_time = -1      # RTi^wr
        # Scheduling State
        self.priority_score = None      # priority(vi) from equation (15)
        self.assignment = -2            # ki - execution location (local core or cloud)
        self.is_core_task = False      # Boolean flag for local vs cloud execution
        self.execution_unit_start_times = [-1,-1,-1,-1] # Start times for each possible execution unit
        self.is_scheduled = SchedulingState.UNSCHEDULED # Current scheduling state

def total_time(tasks):
    return max(
        max(task.local_core_finish_time, task.wireless_recieving_finish_time) for task in tasks
        if not task.children
    )

def calculate_energy_consumption(node, core_powers, cloud_sending_power):
    if node.is_core_task:
        return core_powers[node.assignment] * node.core_execution_times[node.assignment]
    else:
        return cloud_sending_power * node.cloud_execution_times[0]

def total_energy(nodes, core_powers, cloud_sending_power):
    return sum(calculate_energy_consumption(node, core_powers, cloud_sending_power) for node in nodes)

def primary_assignment(nodes):
    for node in nodes:
        # Calculate minimum local execution time
        t_l_min = min(node.core_execution_times)
        # Calculate total remote execution time
        t_re = (node.cloud_execution_times[0] + node.cloud_execution_times[1] + node.cloud_execution_times[2])
        # If remote execution is faster, assign to cloud
        if t_re < t_l_min:
            node.is_core_task = False  # Assign to cloud
        else:
            node.is_core_task = True   # Assign to local core

def task_prioritizing(nodes):
    w = [0] * len(nodes)
    # Calculate computation costs (wi)
    for i, node in enumerate(nodes):
        if not node.is_core_task:  # Cloud task
            # Equation (13): wi = Tre_i for cloud tasks
            w[i] = (node.cloud_execution_times[0] +  node.cloud_execution_times[1] + node.cloud_execution_times[2])
        else:  # Local task
            # Equation (14): wi = avg(1≤k≤K) Tl,k_i for local tasks
            w[i] = sum(node.core_execution_times) / len(node.core_execution_times)

    computed_priority_scores = {}

    def calculate_priority(task):
        # Check if already calculated
        if task.id in computed_priority_scores:
            return computed_priority_scores[task.id]
        # Base case: exit task
        # Equation (16): priority(vi) = wi for exit tasks
        if task.children == []:
            computed_priority_scores[task.id] = w[task.id - 1]
            return w[task.id - 1]
        # Recursive case: Equation (15)
        # priority(vi) = wi + max(vj∈succ(vi)) priority(vj)
        max_successor_priority = max(calculate_priority(successor) for successor in task.children)
        task_priority = w[task.id - 1] + max_successor_priority
        computed_priority_scores[task.id] = task_priority
        return task_priority

    # Calculate priorities for all nodes
    for task in nodes:
        calculate_priority(task)

    # Update priority scores
    for node in nodes:
        node.priority_score = computed_priority_scores[node.id]

def execution_unit_selection(nodes):
    k = 3  # Number of cores
    sequences = [[] for _ in range(k + 1)]  # k cores + cloud
    
    # Track resource availability
    core_earliest_ready = [0] * k
    wireless_send_ready = 0
    wireless_receive_ready = 0
    
    # Create and sort priority list
    node_priority_list = [(node.priority_score, node.id) for node in nodes]
    node_priority_list.sort(reverse=True)
    priority_order = [item[1] for item in node_priority_list]
    
    # Separate entry tasks from non-entry tasks
    entry_tasks = []
    non_entry_tasks = []
    for node_id in priority_order:
        node = nodes[node_id - 1]
        if not node.parents:
            entry_tasks.append(node)
        else:
            non_entry_tasks.append(node)
            
    # First handle entry tasks - they can execute in parallel subject to resource constraints
    cloud_entry_tasks = []
    for task in entry_tasks:
        if task.is_core_task:
            # For local core tasks, find the best core and earliest possible start time
            best_finish_time = float('inf')
            best_core = -1
            best_start_time = float('inf')
            
            for core in range(k):
                # Task must start after core's previous task finishes
                start_time = core_earliest_ready[core]
                finish_time = start_time + task.core_execution_times[core]
                
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_core = core
                    best_start_time = start_time
            
            # Schedule task on selected core
            task.local_core_finish_time = best_finish_time
            task.execution_finish_time = best_finish_time
            task.execution_unit_start_times = [-1] * 4
            task.execution_unit_start_times[best_core] = best_start_time
            core_earliest_ready[best_core] = best_finish_time
            task.assignment = best_core
            task.is_scheduled = SchedulingState.SCHEDULED
            sequences[best_core].append(task.id)
        else:
            # Collect cloud tasks for pipelined scheduling
            cloud_entry_tasks.append(task)
    
    # Schedule cloud entry tasks with proper pipeline staggering
    for task in cloud_entry_tasks:
        # Schedule sending phase
        task.wireless_sending_ready_time = wireless_send_ready
        task.wireless_sending_finish_time = task.wireless_sending_ready_time + task.cloud_execution_times[0]
        wireless_send_ready = task.wireless_sending_finish_time
        
        # Schedule cloud computation (can occur in parallel)
        task.remote_cloud_ready_time = task.wireless_sending_finish_time
        task.remote_cloud_finish_time = task.remote_cloud_ready_time + task.cloud_execution_times[1]
        
        # Schedule receiving phase
        task.wireless_recieving_ready_time = task.remote_cloud_finish_time
        task.wireless_recieving_finish_time = (
            max(wireless_receive_ready, task.wireless_recieving_ready_time) + task.cloud_execution_times[2]
        )
        wireless_receive_ready = task.wireless_recieving_finish_time
        
        # Update task parameters
        task.execution_finish_time = task.wireless_recieving_finish_time
        task.local_core_finish_time = 0
        task.execution_unit_start_times = [-1] * 4
        task.execution_unit_start_times[k] = task.wireless_sending_ready_time
        task.assignment = k
        task.is_scheduled = SchedulingState.SCHEDULED
        sequences[k].append(task.id)
    
    # Process remaining non-entry tasks
    for task in non_entry_tasks:
        # Calculate ready times based on parent task completions
        task.local_core_ready_time = max(
            max(max(parent.local_core_finish_time, parent.wireless_recieving_finish_time) 
                for parent in task.parents),
            0
        )
        
        task.wireless_sending_ready_time = max(
            max(max(parent.local_core_finish_time, parent.wireless_sending_finish_time) 
                for parent in task.parents),
            wireless_send_ready
        )
        
        # Calculate wireless sending finish time based on ready time
        task.wireless_sending_finish_time = task.wireless_sending_ready_time + task.cloud_execution_times[0]
        
        # Calculate cloud ready and finish times
        task.remote_cloud_ready_time = max(
            task.wireless_sending_finish_time,
            max(parent.remote_cloud_finish_time for parent in task.parents)
        )
        task.wireless_recieving_ready_time = task.remote_cloud_ready_time + task.cloud_execution_times[1]
        task.wireless_recieving_finish_time = (
            max(wireless_receive_ready, task.wireless_recieving_ready_time) + task.cloud_execution_times[2]
        )
        
        # Determine whether to execute locally or on cloud
        if not task.is_core_task:
            # Schedule on cloud
            task.execution_finish_time = task.wireless_recieving_finish_time
            task.local_core_finish_time = 0
            task.execution_unit_start_times = [-1] * 4
            task.execution_unit_start_times[k] = task.wireless_sending_ready_time
            task.assignment = k
            wireless_send_ready = task.wireless_sending_finish_time
            wireless_receive_ready = task.wireless_recieving_finish_time
            sequences[k].append(task.id)
        else:
            # Find best local core
            best_finish_time = float('inf')
            best_core = -1
            best_start_time = float('inf')
            
            for core in range(k):
                start_time = max(task.local_core_ready_time, core_earliest_ready[core])
                finish_time = start_time + task.core_execution_times[core]
                
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_core = core
                    best_start_time = start_time
            
            # Compare with potential cloud execution time
            if best_finish_time <= task.wireless_recieving_finish_time:
                # Execute locally
                task.local_core_finish_time = best_finish_time
                task.execution_finish_time = best_finish_time
                task.wireless_recieving_finish_time = 0
                task.execution_unit_start_times = [-1] * 4
                task.execution_unit_start_times[best_core] = best_start_time
                core_earliest_ready[best_core] = best_finish_time
                task.assignment = best_core
                sequences[best_core].append(task.id)
            else:
                # Execute on cloud
                task.execution_finish_time = task.wireless_recieving_finish_time
                task.local_core_finish_time = 0
                task.execution_unit_start_times = [-1] * 4
                task.execution_unit_start_times[k] = task.wireless_sending_ready_time
                task.assignment = k
                task.is_core_task = False
                wireless_send_ready = task.wireless_sending_finish_time
                wireless_receive_ready = task.wireless_recieving_finish_time
                sequences[k].append(task.id)
        
        task.is_scheduled = SchedulingState.SCHEDULED
    
    return sequences


def construct_sequence(nodes, targetNodeId, targetLocation, seq):
    # Step 1: Map node IDs to node objects for quick lookup.
    node_id_to_node = {node.id: node for node in nodes}
    # Step 2: Validate inputs and locate the target node.
    target_node = node_id_to_node.get(targetNodeId)
    # Step 3: Determine the ready time of the target node.
    target_node_rt = target_node.local_core_ready_time if target_node.is_core_task else target_node.wireless_sending_ready_time
    # Step 4: Remove the target node from its original sequence.
    original_assignment = target_node.assignment
    seq[original_assignment].remove(target_node.id)
    # Step 5: Prepare the new sequence for insertion.
    new_sequence_nodes_list = seq[targetLocation]
    # Precompute start times for the new sequence's nodes.
    start_times = [ node_id_to_node[node_id].execution_unit_start_times[targetLocation] for node_id in new_sequence_nodes_list ]
    # Step 6: Use bisect to find the insertion index.
    insertion_index = bisect.bisect_left(start_times, target_node_rt)
    # Step 7: Insert the target node at the correct index.
    new_sequence_nodes_list.insert(insertion_index, target_node.id)
    # Step 8: Update the target node's assignment and status.
    target_node.assignment = targetLocation
    target_node.is_core_task = (targetLocation != 3)  # Location 3 is the cloud.
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
        if node.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
            # Update dependency readiness
            dependency_ready[node.id - 1] = sum(1 for parent in node.parents if parent.is_scheduled != SchedulingState.KERNEL_SCHEDULED)
            # Update sequence readiness
            for sequence in sequences:
                if node.id in sequence:
                    idx = sequence.index(node.id)
                    if idx > 0:
                        prev_node = nodes[sequence[idx - 1] - 1]
                        sequence_ready[node.id - 1] = 1 if prev_node.is_scheduled != SchedulingState.KERNEL_SCHEDULED else 0
                    else:
                        sequence_ready[node.id - 1] = 0
                    break

    def schedule_local_task(node, local_core_ready_times):
        # Calculate ready time based on parent completion
        if not node.parents:
            node.local_core_ready_time = 0
        else:
            parent_completion_times = (max(parent.local_core_finish_time, parent.wireless_recieving_finish_time) for parent in node.parents)
            node.local_core_ready_time = max(parent_completion_times, default=0)
        
        # Schedule on assigned core
        core_index = node.assignment
        node.execution_unit_start_times = [-1] * 4
        node.execution_unit_start_times[core_index] = max(local_core_ready_times[core_index], node.local_core_ready_time)
        node.local_core_finish_time = node.execution_unit_start_times[core_index] + node.core_execution_times[core_index]
        # Update core ready time
        local_core_ready_times[core_index] = node.local_core_finish_time
        # Clear cloud-related timings
        node.wireless_sending_finish_time = node.remote_cloud_finish_time = node.wireless_recieving_finish_time = -1

    def schedule_cloud_task(node, cloud_stage_ready_times):
        # Calculate wireless sending ready time
        if not node.parents:
            node.wireless_sending_ready_time = 0
        else:
            parent_completion_times = (max(parent.local_core_finish_time, parent.wireless_sending_finish_time) for parent in node.parents)
            node.wireless_sending_ready_time = max(parent_completion_times)

        # Initialize start times
        node.execution_unit_start_times = [-1] * 4
        node.execution_unit_start_times[3] = max(cloud_stage_ready_times[0], node.wireless_sending_ready_time)

        # Schedule wireless sending
        node.wireless_sending_finish_time = node.execution_unit_start_times[3] + node.cloud_execution_times[0]
        cloud_stage_ready_times[0] = node.wireless_sending_finish_time

        # Schedule cloud processing
        node.remote_cloud_ready_time = max(
            node.wireless_sending_finish_time,
            max((parent.remote_cloud_finish_time for parent in node.parents), default=0)
        )
        node.remote_cloud_finish_time = max(cloud_stage_ready_times[1], node.remote_cloud_ready_time) + node.cloud_execution_times[1]
        cloud_stage_ready_times[1] = node.remote_cloud_finish_time

        # Schedule wireless receiving
        node.wireless_recieving_ready_time = node.remote_cloud_finish_time
        node.wireless_recieving_finish_time = (max(cloud_stage_ready_times[2], node.wireless_recieving_ready_time) + node.cloud_execution_times[2])
        cloud_stage_ready_times[2] = node.wireless_recieving_finish_time
        
        # Clear local timing
        node.local_core_finish_time = -1

    # Initialize timing trackers
    local_core_ready_times = [0] * 3
    cloud_stage_ready_times = [0] * 3
    # Initialize readiness tracking
    dependency_ready, sequence_ready = initialize_readiness_tracking(nodes, sequences)
    
    # Initialize processing queue with ready nodes
    queue = deque(
        node for node in nodes 
        if sequence_ready[node.id - 1] == 0 
        and all(parent.is_scheduled == SchedulingState.KERNEL_SCHEDULED for parent in node.parents)
    )
    
    # Main scheduling loop
    while queue:
        current_node = queue.popleft()
        current_node.is_scheduled = SchedulingState.KERNEL_SCHEDULED
        
        # Schedule task based on type
        if current_node.is_core_task:
            schedule_local_task(current_node, local_core_ready_times)
        else:
            schedule_cloud_task(current_node, cloud_stage_ready_times)
        
        # Update readiness status for remaining nodes
        for node in nodes:
            update_node_readiness(node, nodes, sequences, dependency_ready, sequence_ready)
            
            # Add newly ready nodes to queue
            if (dependency_ready[node.id - 1] == 0 
                and sequence_ready[node.id - 1] == 0 
                and node.is_scheduled != SchedulingState.KERNEL_SCHEDULED 
                and node not in queue):
                queue.append(node)
    
    # Reset scheduling status
    for node in nodes:
        node.is_scheduled = SchedulingState.UNSCHEDULED
        
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
        migration_candidates = []
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
            
                heappush(migration_candidates, (-efficiency_ratio, node_idx, resource_idx, time, energy))
    
        if not migration_candidates:
            return None
        
        neg_ratio, n_best, k_best, T_best, E_best = heappop(migration_candidates)
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

    tasks = []
    
    for node in nodes:
        assignment_value = assignment_mapping.get(node.assignment, "Unknown")

        if node.is_core_task:
            tasks.append({
                "node id": node.id,
                "assignment": assignment_value,
                "core start_time": node.execution_unit_start_times[node.assignment],
                "core finish_time": node.execution_unit_start_times[node.assignment] + node.core_execution_times[node.assignment]
            })
        else:
            tasks.append({
                "node id": node.id,
                "assignment": assignment_value,
                "wireless sending start_time": node.execution_unit_start_times[3],
                "wireless sending finish_time": node.execution_unit_start_times[3] + node.cloud_execution_times[0],
                "cloud start_time": node.remote_cloud_ready_time,
                "cloud finish_time": node.remote_cloud_ready_time + node.cloud_execution_times[1],
                "wireless receiving start_time": node.wireless_recieving_ready_time,
                "wireless receiving finish_time": node.wireless_recieving_ready_time + node.cloud_execution_times[2]
            })

    for task in tasks:
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
        cloud_tasks = [n for n in nodes if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.execution_unit_start_times[3])
        
        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            if current.wireless_sending_finish_time > next_task.execution_unit_start_times[3]:
                violations.append({
                    'type': 'Wireless Sending Channel Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} sending ends at {current.wireless_sending_finish_time} but Task {next_task.id} starts at {next_task.execution_unit_start_times[3]}'
                })

    def check_computing_channel():
        """Verify cloud computing is sequential"""
        cloud_tasks = [n for n in nodes if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.remote_cloud_ready_time)
        
        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            if current.remote_cloud_finish_time > next_task.remote_cloud_ready_time:
                violations.append({
                    'type': 'Cloud Computing Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} computing ends at {current.remote_cloud_finish_time} but Task {next_task.id} starts at {next_task.remote_cloud_ready_time}'
                })

    def check_receiving_channel():
        """Verify wireless receiving channel is sequential"""
        cloud_tasks = [n for n in nodes if not n.is_core_task]
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
            if not node.is_core_task:  # For cloud tasks
                # Check if all parents have completed necessary phases
                for parent in node.parents:
                    if parent.is_core_task:
                        # Core parent must complete before child starts sending
                        if parent.local_core_finish_time > node.execution_unit_start_times[3]:
                            violations.append({
                                'type': 'Core-Cloud Dependency Violation',
                                'parent': parent.id,
                                'child': node.id,
                                'detail': f'Core Task {parent.id} finishes at {parent.local_core_finish_time} but Cloud Task {node.id} starts sending at {node.execution_unit_start_times[3]}'
                            })
                    else:
                        # Cloud parent must complete sending before child starts sending
                        if parent.wireless_sending_finish_time > node.execution_unit_start_times[3]:
                            violations.append({
                                'type': 'Cloud Pipeline Dependency Violation',
                                'parent': parent.id,
                                'child': node.id,
                                'detail': f'Parent Task {parent.id} sending phase ends at {parent.wireless_sending_finish_time} but Task {node.id} starts sending at {node.execution_unit_start_times[3]}'
                            })
            else:  # For core tasks
                # All parents must complete fully before core task starts
                for parent in node.parents:
                    parent_finish = (parent.wireless_recieving_finish_time 
                                  if not parent.is_core_task else parent.local_core_finish_time)
                    if parent_finish > node.execution_unit_start_times[node.assignment]:
                        violations.append({
                            'type': 'Core Task Dependency Violation',
                            'parent': parent.id,
                            'child': node.id,
                            'detail': f'Parent Task {parent.id} finishes at {parent_finish} but Core Task {node.id} starts at {node.execution_unit_start_times[node.assignment]}'
                        })

    def check_core_execution():
        """Verify core tasks don't overlap"""
        core_tasks = [n for n in nodes if n.is_core_task]
        for core_id in range(3):  # Assuming 3 cores
            core_specific_tasks = [t for t in core_tasks if t.assignment == core_id]
            sorted_tasks = sorted(core_specific_tasks, key=lambda x: x.execution_unit_start_times[core_id])
            
            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]
                
                if current.local_core_finish_time > next_task.execution_unit_start_times[core_id]:
                    violations.append({
                        'type': f'Core {core_id} Execution Conflict',
                        'task1': current.id,
                        'task2': next_task.id,
                        'detail': f'Task {current.id} finishes at {current.local_core_finish_time} but Task {next_task.id} starts at {next_task.execution_unit_start_times[core_id]}'
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
    cloud_tasks = [n for n in nodes if not n.is_core_task]
    if cloud_tasks:
        print("\nCloud Pipeline Analysis:")
        print("=" * 50)
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.execution_unit_start_times[3])
        
        for task in sorted_tasks:
            print(f"\nTask {task.id} Pipeline Stages:")
            print(f"  └─ Sending:   {task.execution_unit_start_times[3]:2d} -> {task.wireless_sending_finish_time:2d}")
            print(f"  └─ Computing: {task.remote_cloud_ready_time:2d} -> {task.remote_cloud_finish_time:2d}")
            print(f"  └─ Receiving: {task.wireless_recieving_ready_time:2d} -> {task.wireless_recieving_finish_time:2d}")
    
    # Print core execution
    core_tasks = [n for n in nodes if n.is_core_task]
    if core_tasks:
        print("\nCore Execution:")
        print("=" * 50)
        for core_id in range(3):
            core_specific = [t for t in core_tasks if t.assignment == core_id]
            if core_specific:
                print(f"\nCore {core_id}:")
                for task in sorted(core_specific, key=lambda x: x.execution_unit_start_times[core_id]):
                    print(f"  Task {task.id}: {task.execution_unit_start_times[core_id]:2d} -> {task.local_core_finish_time:2d}")


def print_task_graph(nodes):
        for node in nodes:
            children_ids = [child.id for child in node.children]
            parent_ids = [parent.id for parent in node.parents]
            print(f"Node {node.id}:")
            print(f"  Parents: {parent_ids}")
            print(f"  Children: {children_ids}")
            print()

def check_mcc_constraints(nodes):
    """
    Validates and prints detailed reports for MCC-specific constraints from Section II of the paper.
    Focuses on cloud execution, wireless communication, and three-phase execution model.
    
    Args:
        nodes: List of Node objects with scheduling information
    Returns:
        tuple: (is_valid, violations)
    """
    violations = []

    def check_cloud_computation_parallelism():
        """Verifies cloud's ability to execute independent tasks in parallel"""
        cloud_tasks = [n for n in nodes if not n.is_core_task]
        for task1 in cloud_tasks:
            for task2 in cloud_tasks:
                if task1.id != task2.id:
                    if task2 not in task1.parents and task1 not in task2.parents:
                        if (task1.remote_cloud_ready_time < task2.remote_cloud_finish_time and 
                            task1.remote_cloud_finish_time > task2.remote_cloud_ready_time):
                            violations.append({
                                'type': 'Cloud Parallelism Constraint',
                                'task1': task1.id,
                                'task2': task2.id,
                                'detail': f'Independent tasks {task1.id} and {task2.id} should be allowed parallel cloud execution'
                            })

    def check_three_phase_execution():
        """Verifies the sequential RF sending → cloud computing → RF receiving phases"""
        cloud_tasks = [n for n in nodes if not n.is_core_task]
        for task in cloud_tasks:
            if not (task.wireless_sending_finish_time <= task.remote_cloud_ready_time and
                   task.remote_cloud_finish_time <= task.wireless_recieving_ready_time):
                violations.append({
                    'type': 'Three-Phase Execution Violation',
                    'task': task.id,
                    'detail': f'Task {task.id} phases must follow sending→computing→receiving order'
                })

    def check_cloud_ready_time_constraints():
        """Verifies cloud ready time calculations per equation (5)"""
        cloud_tasks = [n for n in nodes if not n.is_core_task]
        for task in cloud_tasks:
            parent_cloud_finish_times = [p.remote_cloud_finish_time for p in task.parents 
                                      if not p.is_core_task]
            max_parent_finish = max(parent_cloud_finish_times) if parent_cloud_finish_times else 0
            expected_ready_time = max(task.wireless_sending_finish_time, max_parent_finish)
            
            if task.remote_cloud_ready_time != expected_ready_time:
                violations.append({
                    'type': 'Cloud Ready Time Constraint',
                    'task': task.id,
                    'detail': f'Task {task.id} ready time violates equation (5): should be max(FTws_i, max(FTc_j))'
                })

    def check_wireless_ready_time_constraints():
        """Verifies wireless sending ready time calculations per equation (4)"""
        cloud_tasks = [n for n in nodes if not n.is_core_task]
        for task in cloud_tasks:
            parent_finish_times = []
            for parent in task.parents:
                if parent.is_core_task:
                    parent_finish_times.append(parent.local_core_finish_time)
                else:
                    parent_finish_times.append(parent.wireless_sending_finish_time)
            
            expected_ready_time = max(parent_finish_times) if parent_finish_times else 0
            
            if task.wireless_sending_ready_time < expected_ready_time:
                violations.append({
                    'type': 'Wireless Ready Time Constraint',
                    'task': task.id,
                    'detail': f'Task {task.id} must wait for parent tasks to complete before wireless sending'
                })

    # Run all checks
    check_cloud_computation_parallelism()
    check_three_phase_execution() 
    check_cloud_ready_time_constraints()
    check_wireless_ready_time_constraints()

    # Print detailed MCC constraint analysis report
    print("\nMobile Cloud Computing Constraints Analysis")
    print("=" * 50)
    
    if len(violations) == 0:
        print("All MCC-specific constraints are satisfied!")
    else:
        print("Found MCC constraint violations:")
        for v in violations:
            print(f"\nViolation Type: {v['type']}")
            print(f"Detail: {v['detail']}")

    # Print cloud execution analysis
    cloud_tasks = [n for n in nodes if not n.is_core_task]
    if cloud_tasks:
        print("\nCloud Task Execution Analysis:")
        print("=" * 50)
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.wireless_sending_ready_time)
        
        for task in sorted_tasks:
            print(f"\nTask {task.id} Three-Phase Execution:")
            print(f"  └─ RF Sending Phase:     {task.wireless_sending_ready_time:2d} -> {task.wireless_sending_finish_time:2d}")
            print(f"  └─ Cloud Computing:      {task.remote_cloud_ready_time:2d} -> {task.remote_cloud_finish_time:2d}")
            print(f"  └─ RF Receiving Phase:   {task.wireless_recieving_ready_time:2d} -> {task.wireless_recieving_finish_time:2d}")

    # Print dependency analysis
    print("\nTask Dependency Analysis:")
    print("=" * 50)
    for task in cloud_tasks:
        if task.parents:
            print(f"\nTask {task.id} Dependencies:")
            for parent in task.parents:
                if parent.is_core_task:
                    print(f"  └─ Must wait for core task {parent.id} to finish at {parent.local_core_finish_time}")
                else:
                    print(f"  └─ Must wait for cloud task {parent.id} to finish sending at {parent.wireless_sending_finish_time}")

    return len(violations) == 0, violations

if __name__ == '__main__':
    """
    node20 = Task(id=20, parents=None, children=[])
    node19 = Task(id=19, parents=None, children=[node20])
    node18 = Task(id=18, parents=None, children=[node20])
    node17 = Task(id=17, parents=None, children=[node20])
    node16 = Task(id=16, parents=None, children=[node19])
    node15 = Task(id=15, parents=None, children=[node19])
    node14 = Task(id=14, parents=None, children=[node18, node19])
    node13 = Task(id=13, parents=None, children=[node17, node18])
    node12 = Task(id=12, parents=None, children=[node17])
    node11 = Task(id=11, parents=None, children=[node15, node16])
    node10 = Task(id=10, parents=None, children=[node11,node15])
    node9 = Task(id=9, parents=None, children=[node13,node14])
    node8 = Task(id=8, parents=None, children=[node12,node13])
    node7 = Task(id=7, parents=None, children=[node12])
    node6 = Task(id=6, parents=None, children=[node10,node11])
    node5 = Task(id=5, parents=None, children=[node9,node10])
    node4 = Task(id=4, parents=None, children=[node8,node9])
    node3 = Task(id=3, parents=None, children=[node7, node8])
    node2 = Task(id=2, parents=None, children=[node7])
    node1 = Task(id=1, parents=None, children=[node7])
    node1.parents = []
    node2.parents = []
    node3.parents = []
    node4.parents = []
    node5.parents = []
    node6.parents = []
    node7.parents = [node1,node2,node3]
    node8.parents = [node3, node4]
    node9.parents = [node4,node5]
    node10.parents = [node5, node6]
    node11.parents = [node6, node10]
    node12.parents = [node7, node8]
    node13.parents = [node8, node9]
    node14.parents = [node9, node10]
    node15.parents = [node10, node11]
    node16.parents = [node11]
    node17.parents = [node12, node13]
    node18.parents = [node13, node14]
    node19.parents = [node14, node15,node16]
    node20.parents = [node17, node18,node19]

    nodes = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,node11,node12,node13,node14,node15,node16,node17,node18,node19,node20]

    node10 = Task(id=10, parents=None, children=[])
    node9 = Task(id=9, parents=None, children=[node10])
    node8 = Task(id=8, parents=None, children=[node9])
    node7 = Task(id=7, parents=None, children=[node9,node10])
    node6 = Task(id=6, parents=None, children=[node10])
    node5 = Task(id=5, parents=None, children=[node6])
    node4 = Task(id=4, parents=None, children=[node7,node8])
    node3 = Task(id=3, parents=None, children=[node7, node8])
    node2 = Task(id=2, parents=None, children=[node5,node7])
    node1 = Task(id=1, parents=None, children=[node2, node3, node4])
    node1.parents = []
    node2.parents = [node1]
    node3.parents = [node1]
    node4.parents = [node1]
    node5.parents = [node2]
    node6.parents = [node5]
    node7.parents = [node2,node3,node4]
    node8.parents = [node3, node4]
    node9.parents = [node7,node8]
    node10.parents = [node6, node7, node9]
    nodes = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10]

    node20 = Task(id=20, parents=None, children=[])
    node19 = Task(id=19, parents=None, children=[])
    node18 = Task(id=18, parents=None, children=[])
    node17 = Task(id=17, parents=None, children=[])
    node16 = Task(id=16, parents=None, children=[node19])
    node15 = Task(id=15, parents=None, children=[node19])
    node14 = Task(id=14, parents=None, children=[node18, node19])
    node13 = Task(id=13, parents=None, children=[node17, node18])
    node12 = Task(id=12, parents=None, children=[node17])
    node11 = Task(id=11, parents=None, children=[node15, node16])
    node10 = Task(id=10, parents=None, children=[node11,node15])
    node9 = Task(id=9, parents=None, children=[node13,node14])
    node8 = Task(id=8, parents=None, children=[node12,node13])
    node7 = Task(id=7, parents=None, children=[node12])
    node6 = Task(id=6, parents=None, children=[node10,node11])
    node5 = Task(id=5, parents=None, children=[node9,node10])
    node4 = Task(id=4, parents=None, children=[node8,node9])
    node3 = Task(id=3, parents=None, children=[node7, node8])
    node2 = Task(id=2, parents=None, children=[node7,node8])
    node1 = Task(id=1, parents=None, children=[node7])
    node1.parents = []
    node2.parents = []
    node3.parents = []
    node4.parents = []
    node5.parents = []
    node6.parents = []
    node7.parents = [node1,node2,node3]
    node8.parents = [node3, node4]
    node9.parents = [node4,node5]
    node10.parents = [node5, node6]
    node11.parents = [node6, node10]
    node12.parents = [node7, node8]
    node13.parents = [node8, node9]
    node14.parents = [node9, node10]
    node15.parents = [node10, node11]
    node16.parents = [node11]
    node17.parents = [node12, node13]
    node18.parents = [node13, node14]
    node19.parents = [node14, node15,node16]
    node20.parents = [node12]

    nodes = [node1, node2, node3, node4, node5, node6, node7, node8, node9, node10,node11,node12,node13,node14,node15,node16,node17,node18,node19,node20]
    """

    node10 = Task(10)
    node9 = Task(9, children=[node10])
    node8 = Task(8, children=[node10])
    node7 = Task(7, children=[node10])
    node6 = Task(6, children=[node8])
    node5 = Task(5, children=[node9])
    node4 = Task(4, children=[node8, node9])
    node3 = Task(3, children=[node7])
    node2 = Task(2, children=[node8, node9])
    node1 = Task(1, children=[node2, node3, node4, node5, node6])
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
    #check_mcc_constraints(nodes)

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
    #check_mcc_constraints(nodes2)
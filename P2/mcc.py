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
    UNSCHEDULED = 0      # Initial state
    SCHEDULED = 1        # After initial scheduling (Step 1 in paper)
    KERNEL_SCHEDULED = 2  # After kernel algorithm (Step 2 in paper)

@dataclass
class TaskMigrationState:
    time: float                # Completion time after migration
    energy: float              # Energy consumption after migration
    efficiency_ratio: float    # Measure of migration effectiveness
    task_index: int            # Which task is being migrated
    target_execution_unit: int # Where the task is being migrated to

class Task(object):
    def __init__(self, id, parents=None, children=None):
        # Basic graph structure
        self.id = id                     # Maps to vi in the paper
        self.parents = parents or []     # pred(vi) - predecessor tasks
        self.children = children or []   # succ(vi) - successor tasks
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
        self.execution_finish_time = -1
        self.is_scheduled = SchedulingState.UNSCHEDULED # Current scheduling state

def total_time(tasks):
    # Find the finish time for each exit task (tasks with no children)
    # and return the maximum among them
    return max(
        # For each exit task, take the later of:
        # 1. When it finishes on a local core
        # 2. When its results are received from the cloud
        max(task.local_core_finish_time, task.wireless_recieving_finish_time) 
        for task in tasks
        if not task.children  # Only consider exit tasks
    )

def calculate_energy_consumption(task, core_powers, cloud_sending_power):
    # For tasks running on local cores
    if task.is_core_task:
        # Energy = Power of chosen core × Execution time on that core
        # This implements equation (7): Ei,k^l = Pk × Ti,k^l
        return core_powers[task.assignment] * task.core_execution_times[task.assignment]
    # For tasks running in the cloud
    else:
        # Energy = Power for sending × Time spent sending
        # This implements equation (8): Ei^c = P^s × Ti^s
        return cloud_sending_power * task.cloud_execution_times[0]

def total_energy(tasks, core_powers, cloud_sending_power):
    # Sum up energy consumption across all tasks
    # This implements equation (9): E^total = ∑(i=1 to N) Ei
    return sum(
        calculate_energy_consumption(task, core_powers, cloud_sending_power) 
        for task in tasks
    )

def primary_assignment(tasks):
    for task in tasks:
        # Calculate minimum local execution time (T_i^l_min)
        # This implements equation (11) from the paper
        # We're finding the fastest possible local execution by taking
        # the minimum time across all available cores
        t_l_min = min(task.core_execution_times)

        # Calculate total remote execution time (T_i^re)
        # This implements equation (12) from the paper
        # Remote execution involves three phases:
        # 1. Sending data to cloud (T_i^s)
        # 2. Cloud computation (T_i^c)
        # 3. Receiving results (T_i^r)
        t_re = (task.cloud_execution_times[0] +  # Send time
                task.cloud_execution_times[1] +  # Cloud execution time
                task.cloud_execution_times[2])   # Receive time

        # Compare local vs remote execution time
        # If remote execution is faster, mark as a "cloud task"
        if t_re < t_l_min:
            task.is_core_task = False  # Will be executed in cloud
        else:
            task.is_core_task = True   # Will be executed locally

def task_prioritizing(tasks):
    w = [0] * len(tasks)
    # Calculate computation costs for each task
    for i, task in enumerate(tasks):
        if not task.is_core_task:  
            # For cloud tasks, use total remote execution time
            # Following equation (13): wi = Ti^re
            w[i] = (task.cloud_execution_times[0] +   # Send time
                   task.cloud_execution_times[1] +    # Cloud execution time
                   task.cloud_execution_times[2])     # Receive time
        else:  
            # For local tasks, use average execution time across cores
            # Following equation (14): wi = avg(1≤k≤K) Ti,k^l
            w[i] = sum(task.core_execution_times) / len(task.core_execution_times)

    computed_priority_scores = {} # Cache for storing computed priorities

    def calculate_priority(task):
        # Check if we've already calculated this task's priority
        if task.id in computed_priority_scores:
            return computed_priority_scores[task.id]
        # Base case: Exit tasks (tasks with no children)
        # Following equation (16): priority(vi) = wi for exit tasks
        if task.children == []:
            computed_priority_scores[task.id] = w[task.id - 1]
            return w[task.id - 1]
        # Recursive case: Non-exit tasks
        # Following equation (15): priority(vi) = wi + max(vj∈succ(vi)) priority(vj)
        max_successor_priority = max(calculate_priority(successor) for successor in task.children)
        task_priority = w[task.id - 1] + max_successor_priority
        computed_priority_scores[task.id] = task_priority
        return task_priority

    # Calculate priorities for all tasks
    for task in tasks:
        calculate_priority(task)

    # Update priority scores
    for task in tasks:
        task.priority_score = computed_priority_scores[task.id]

class InitialTaskScheduler:
    def __init__(self, tasks, num_cores=3):
        """Initialize the task scheduler with tasks and resources."""
        self.tasks = tasks
        self.k = num_cores  # Number of cores
        
        # Resource timing trackers
        self.core_earliest_ready = [0] * self.k  # When each local core becomes free
        self.wireless_send_ready = 0             # When we can next send to cloud
        self.wireless_receive_ready = 0          # When we can next receive from cloud
        
        # Execution sequences for each resource
        self.sequences = [[] for _ in range(self.k + 1)]  # k cores + cloud
        
    def get_priority_ordered_tasks(self):
        """Sort tasks by priority score and return ordered list of task IDs."""
        task_priority_list = [(task.priority_score, task.id) for task in self.tasks]
        task_priority_list.sort(reverse=True)
        return [item[1] for item in task_priority_list]
        
    def separate_entry_tasks(self, priority_order):
        """Separate tasks into entry tasks and non-entry tasks."""
        entry_tasks = []
        non_entry_tasks = []
        for task_id in priority_order:
            task = self.tasks[task_id - 1]
            if not task.parents:
                entry_tasks.append(task)
            else:
                non_entry_tasks.append(task)
        return entry_tasks, non_entry_tasks

    def find_best_local_core(self, task, ready_time=0):
        """Find the optimal local core for task execution."""
        best_finish_time = float('inf')
        best_core = -1
        best_start_time = float('inf')
        
        for core in range(self.k):
            start_time = max(ready_time, self.core_earliest_ready[core])
            finish_time = start_time + task.core_execution_times[core]
            
            if finish_time < best_finish_time:
                best_finish_time = finish_time
                best_core = core
                best_start_time = start_time
                
        return best_core, best_start_time, best_finish_time

    def schedule_on_local_core(self, task, core, start_time, finish_time):
        """Schedule a task on a local core."""
        task.local_core_finish_time = finish_time
        task.execution_finish_time = finish_time
        task.execution_unit_start_times = [-1] * (self.k + 1)
        task.execution_unit_start_times[core] = start_time
        self.core_earliest_ready[core] = finish_time
        task.assignment = core
        task.is_scheduled = SchedulingState.SCHEDULED
        self.sequences[core].append(task.id)

    def calculate_cloud_pipeline_timing(self, task):
        """Calculate timing for cloud execution pipeline phases."""
        # Phase 1: Sending to cloud
        send_ready = task.wireless_sending_ready_time
        send_finish = send_ready + task.cloud_execution_times[0]
        
        # Phase 2: Cloud computation
        cloud_ready = send_finish
        cloud_finish = cloud_ready + task.cloud_execution_times[1]
        
        # Phase 3: Receiving results
        receive_ready = cloud_finish
        receive_finish = (
            max(self.wireless_receive_ready, receive_ready) + 
            task.cloud_execution_times[2]
        )
        
        return send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish

    def schedule_on_cloud(self, task, send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish):
        """Schedule a task for cloud execution."""
        task.wireless_sending_ready_time = send_ready
        task.wireless_sending_finish_time = send_finish
        task.remote_cloud_ready_time = cloud_ready
        task.remote_cloud_finish_time = cloud_finish
        task.wireless_recieving_ready_time = receive_ready
        task.wireless_recieving_finish_time = receive_finish
        
        task.execution_finish_time = receive_finish
        task.local_core_finish_time = 0
        task.execution_unit_start_times = [-1] * (self.k + 1)
        task.execution_unit_start_times[self.k] = send_ready
        task.assignment = self.k
        task.is_scheduled = SchedulingState.SCHEDULED
        
        # Update resource availability
        self.wireless_send_ready = send_finish
        self.wireless_receive_ready = receive_finish
        self.sequences[self.k].append(task.id)

    def schedule_entry_tasks(self, entry_tasks):
        """Schedule tasks that have no dependencies."""
        cloud_entry_tasks = []
        
        # First schedule local core tasks
        for task in entry_tasks:
            if task.is_core_task:
                core, start_time, finish_time = self.find_best_local_core(task)
                self.schedule_on_local_core(task, core, start_time, finish_time)
            else:
                cloud_entry_tasks.append(task)
        
        # Then schedule cloud tasks with pipeline staggering
        for task in cloud_entry_tasks:
            task.wireless_sending_ready_time = self.wireless_send_ready
            timing = self.calculate_cloud_pipeline_timing(task)
            self.schedule_on_cloud(task, *timing)

    def calculate_non_entry_task_ready_times(self, task):
        """Calculate ready times for tasks with dependencies."""
        # Local core ready time
        task.local_core_ready_time = max(
            max(max(parent.local_core_finish_time, parent.wireless_recieving_finish_time) 
                for parent in task.parents),
            0
        )
        
        # Cloud sending ready time
        task.wireless_sending_ready_time = max(
            max(max(parent.local_core_finish_time, parent.wireless_sending_finish_time) 
                for parent in task.parents),
            self.wireless_send_ready
        )

    def schedule_non_entry_tasks(self, non_entry_tasks):
        """Schedule tasks that have dependencies."""
        for task in non_entry_tasks:
            self.calculate_non_entry_task_ready_times(task)
            
            if not task.is_core_task:
                # Schedule predetermined cloud tasks
                timing = self.calculate_cloud_pipeline_timing(task)
                self.schedule_on_cloud(task, *timing)
            else:
                # Find best local core
                core, start_time, finish_time = self.find_best_local_core(
                    task, task.local_core_ready_time
                )
                
                # Calculate cloud execution time for comparison
                timing = self.calculate_cloud_pipeline_timing(task)
                cloud_finish_time = timing[-1]
                
                # Choose better execution option
                if finish_time <= cloud_finish_time:
                    self.schedule_on_local_core(task, core, start_time, finish_time)
                else:
                    task.is_core_task = False
                    self.schedule_on_cloud(task, *timing)

def execution_unit_selection(tasks):
    # Create scheduler instance to handle the scheduling process
    scheduler = InitialTaskScheduler(tasks, 3)
    # Get priority-ordered tasks
    priority_orderered_tasks = scheduler.get_priority_ordered_tasks()
    # Separate entry and non-entry tasks
    entry_tasks, non_entry_tasks = scheduler.separate_entry_tasks(priority_orderered_tasks)
    # Schedule tasks in phases
    scheduler.schedule_entry_tasks(entry_tasks)
    scheduler.schedule_non_entry_tasks(non_entry_tasks)
    return scheduler.sequences

def construct_sequence(tasks, task_id, execution_unit, original_sequence):
    # Step 1: Map task IDs to task objects for quick lookup.
    task_id_to_task = {task.id: task for task in tasks}
    # Step 2: Validate inputs and locate the target task.
    target_task = task_id_to_task.get(task_id)
    # Step 3: Determine the ready time of the target task.
    target_task_rt = target_task.local_core_ready_time if target_task.is_core_task else target_task.wireless_sending_ready_time
    # Step 4: Remove the target task from its original sequence.
    original_assignment = target_task.assignment
    original_sequence[original_assignment].remove(target_task.id)
    # Step 5: Prepare the new sequence for insertion.
    new_sequence_task_list = original_sequence[execution_unit]
    # Precompute start times for the new sequence's tasks.
    start_times = [ task_id_to_task[task_id].execution_unit_start_times[execution_unit] for task_id in new_sequence_task_list ]
    # Step 6: Use bisect to find the insertion index.
    insertion_index = bisect.bisect_left(start_times, target_task_rt)
    # Step 7: Insert the target task at the correct index.
    new_sequence_task_list.insert(insertion_index, target_task.id)
    # Step 8: Update the target task's assignment and status.
    target_task.assignment = execution_unit
    target_task.is_core_task = (execution_unit != 3)  # Location 3 is the cloud.
    return original_sequence

class KernelScheduler:
    def __init__(self, tasks, sequences):
        """Initialize the kernel scheduler with tasks and their execution sequences.
        
        The kernel scheduler handles the detailed timing calculations for task execution,
        ensuring proper sequencing and dependency management.
        
        Args:
            tasks: List of computational tasks to be scheduled
            sequences: List of sequences, where each sequence represents tasks assigned 
                      to a specific execution unit (local cores or cloud)
        """
        self.tasks = tasks
        self.sequences = sequences
        
        # Initialize timing trackers for resources
        self.local_core_ready_times = [0] * 3  # Ready times for each local core
        self.cloud_stage_ready_times = [0] * 3  # Ready times for send, compute, receive
        
        # Initialize readiness tracking for dependencies and sequences
        self.dependency_ready, self.sequence_ready = self.initialize_task_state()
        
    def initialize_task_state(self):
        """Set up initial readiness states for all tasks based on dependencies and sequences.
        
        A task's readiness is determined by two factors:
        1. All its parent tasks must be completed (dependency readiness)
        2. Its predecessor in the execution sequence must be completed (sequence readiness)
        """
        # Track how many parent tasks are not yet completed for each task
        dependency_ready = [len(task.parents) for task in self.tasks]
        
        # Track whether a task is ready to execute in its sequence (-1: not in sequence, 
        # 0: ready to execute, 1: waiting for predecessor)
        sequence_ready = [-1] * len(self.tasks)
        
        # Mark the first task in each sequence as potentially ready
        for sequence in self.sequences:
            if sequence:  # Only process non-empty sequences
                sequence_ready[sequence[0] - 1] = 0
                
        return dependency_ready, sequence_ready
    
    def update_task_state(self, task):
        """Update the readiness status of a task after scheduling changes.
        
        This method recalculates both dependency and sequence readiness for a task
        based on the current state of its parents and sequence predecessors.
        
        Args:
            task: The task whose readiness needs to be updated
        """
        if task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
            # Update dependency readiness - count unscheduled parents
            self.dependency_ready[task.id - 1] = sum(
                1 for parent in task.parents 
                if parent.is_scheduled != SchedulingState.KERNEL_SCHEDULED
            )
            
            # Update sequence readiness
            for sequence in self.sequences:
                if task.id in sequence:
                    idx = sequence.index(task.id)
                    if idx > 0:
                        # Check if predecessor is scheduled
                        prev_task = self.tasks[sequence[idx - 1] - 1]
                        self.sequence_ready[task.id - 1] = (
                            1 if prev_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED 
                            else 0
                        )
                    else:
                        # First task in sequence is always sequence-ready
                        self.sequence_ready[task.id - 1] = 0
                    break
    
    def schedule_local_task(self, task):
        """Schedule a task for execution on its assigned local core.
        
        This method calculates the start and finish times for a task executing
        on a local core, considering both parent completion times and core availability.
        """
        # Calculate earliest possible start time based on parent completion
        if not task.parents:
            task.local_core_ready_time = 0
        else:
            parent_completion_times = (
                max(parent.local_core_finish_time, parent.wireless_recieving_finish_time) 
                for parent in task.parents
            )
            task.local_core_ready_time = max(parent_completion_times, default=0)
        
        # Schedule on assigned core
        core_index = task.assignment
        task.execution_unit_start_times = [-1] * 4
        task.execution_unit_start_times[core_index] = max(
            self.local_core_ready_times[core_index], 
            task.local_core_ready_time
        )
        
        # Calculate and set completion time
        task.local_core_finish_time = (
            task.execution_unit_start_times[core_index] + 
            task.core_execution_times[core_index]
        )
        
        # Update core availability
        self.local_core_ready_times[core_index] = task.local_core_finish_time
        
        # Clear cloud-related timings since this is a local task
        task.wireless_sending_finish_time = -1
        task.remote_cloud_finish_time = -1
        task.wireless_recieving_finish_time = -1
    
    def schedule_cloud_task(self, task):
        """Schedule a task for execution in the cloud.
        
        This method handles the three-phase cloud execution process:
        1. Wireless sending of data to cloud
        2. Cloud computation
        3. Wireless receiving of results
        """
        # Calculate earliest possible sending time based on parent completion
        if not task.parents:
            task.wireless_sending_ready_time = 0
        else:
            parent_completion_times = (
                max(parent.local_core_finish_time, parent.wireless_sending_finish_time) 
                for parent in task.parents
            )
            task.wireless_sending_ready_time = max(parent_completion_times)
        
        # Initialize execution unit start times
        task.execution_unit_start_times = [-1] * 4
        task.execution_unit_start_times[3] = max(
            self.cloud_stage_ready_times[0], 
            task.wireless_sending_ready_time
        )
        
        # Phase 1: Schedule wireless sending
        task.wireless_sending_finish_time = (
            task.execution_unit_start_times[3] + 
            task.cloud_execution_times[0]
        )
        self.cloud_stage_ready_times[0] = task.wireless_sending_finish_time
        
        # Phase 2: Schedule cloud processing
        task.remote_cloud_ready_time = max(
            task.wireless_sending_finish_time,
            max((parent.remote_cloud_finish_time for parent in task.parents), default=0)
        )
        task.remote_cloud_finish_time = (
            max(self.cloud_stage_ready_times[1], task.remote_cloud_ready_time) + 
            task.cloud_execution_times[1]
        )
        self.cloud_stage_ready_times[1] = task.remote_cloud_finish_time
        
        # Phase 3: Schedule wireless receiving
        task.wireless_recieving_ready_time = task.remote_cloud_finish_time
        task.wireless_recieving_finish_time = (
            max(self.cloud_stage_ready_times[2], task.wireless_recieving_ready_time) + 
            task.cloud_execution_times[2]
        )
        self.cloud_stage_ready_times[2] = task.wireless_recieving_finish_time
        
        # Clear local timing since this is a cloud task
        task.local_core_finish_time = -1
    
    def initialize_queue(self):
        """Initialize the processing queue with all ready tasks.
        
        A task is considered ready when:
        1. It has no unscheduled parents (dependency_ready == 0)
        2. It is first in its sequence or its predecessor is scheduled (sequence_ready == 0)
        """
        return deque(
            task for task in self.tasks 
            if (self.sequence_ready[task.id - 1] == 0 and
                all(parent.is_scheduled == SchedulingState.KERNEL_SCHEDULED 
                    for parent in task.parents))
        )


def kernel_algorithm(tasks, sequences):
    # Create scheduler instance to manage the scheduling process
    scheduler = KernelScheduler(tasks, sequences)
    # Initialize queue with ready tasks
    queue = scheduler.initialize_queue()
    
    # Process tasks in order of readiness
    while queue:
        current_task = queue.popleft()
        current_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED
        
        # Schedule task based on its type
        if current_task.is_core_task:
            scheduler.schedule_local_task(current_task)
        else:
            scheduler.schedule_cloud_task(current_task)
        
        # Update readiness of remaining tasks
        for task in tasks:
            scheduler.update_task_state(task)
            
            # Add newly ready tasks to queue
            if (scheduler.dependency_ready[task.id - 1] == 0 and
                scheduler.sequence_ready[task.id - 1] == 0 and
                task.is_scheduled != SchedulingState.KERNEL_SCHEDULED and
                task not in queue):
                queue.append(task)
    
    # Reset scheduling status for all tasks
    for task in tasks:
        task.is_scheduled = SchedulingState.UNSCHEDULED
    
    return tasks

def optimize_task_scheduling(tasks, sequence, T_final, core_powers=[1, 2, 4], cloud_sending_power=0.5):
    """
    Optimized task scheduling algorithm using efficient data structures
    and memoization to reduce computational overhead.
    """
    core_powers = np.array(core_powers)
    # Cache for storing evaluated migrations
    migration_cache = {}
    
    def get_cache_key(task_idx, target_execution_unit):
        """Generate unique cache key for each migration scenario"""
        return (task_idx, target_execution_unit, tuple(task.assignment for task in tasks))
    
    def evaluate_migration(tasks, seqs, task_idx, target_execution_unit):
        """
        Evaluates migration with caching to avoid redundant calculations.
        """
        cache_key = get_cache_key(task_idx, target_execution_unit)
        if cache_key in migration_cache:
            return migration_cache[cache_key]
            
        seq_copy = [seq.copy() for seq in seqs]
        tasks_copy = deepcopy(tasks)
        
        seq_copy = construct_sequence(tasks_copy, task_idx + 1, target_execution_unit, seq_copy)
        kernel_algorithm(tasks_copy, seq_copy)
        
        current_T = total_time(tasks_copy)
        current_E = total_energy(tasks_copy, core_powers, cloud_sending_power)
        
        migration_cache[cache_key] = (current_T, current_E)
        return current_T, current_E

    def initialize_migration_choices(tasks):
        """Uses boolean array for efficient storage of migration choices"""
        migration_choices = np.zeros((len(tasks), 4), dtype=bool)
        
        for i, task in enumerate(tasks):
            if task.assignment == 3:  # Cloud-assigned task
                migration_choices[i, :] = True
            else:
                migration_choices[i, task.assignment] = True
                
        return migration_choices

    def find_best_migration(migration_trials_results, T_final, E_total, T_max):
        # Step 1: Look for migrations that reduce energy without increasing time
        best_energy_reduction = 0
        best_migration = None
    
        for task_idx, resource_idx, time, energy in migration_trials_results:
            # Skip if time constraint violated
            if time > T_max:
                continue
            
            # Calculate energy reduction
            energy_reduction = E_total - energy
        
            # Check if this migration reduces energy without increasing time
            if time <= T_final and energy_reduction > 0:
                if energy_reduction > best_energy_reduction:
                    best_energy_reduction = energy_reduction
                    best_migration = (task_idx, resource_idx, time, energy)
    
        # If we found a valid migration in Step 1, return it
        if best_migration:
            task_idx, resource_idx, time, energy = best_migration
            return TaskMigrationState(
                time=time,
                energy=energy,
                efficiency_ratio=best_energy_reduction,
                task_index=task_idx + 1,
                target_execution_unit=resource_idx + 1
            )
    
        # Step 2: If no energy-reducing migrations found, look for best efficiency ratio
        migration_candidates = []
        for task_idx, resource_idx, time, energy in migration_trials_results:
            # Skip if time constraint violated
            if time > T_max:
                continue
            
            # Calculate efficiency ratio only if there's energy reduction
            energy_reduction = E_total - energy
            if energy_reduction > 0:
                # Calculate ratio of energy reduction to time increase
                time_increase = max(0, time - T_final)
                if time_increase == 0:
                    efficiency_ratio = float('inf')  # Prioritize no time increase
                else:
                    efficiency_ratio = energy_reduction / time_increase
            
                heappush(migration_candidates, (-efficiency_ratio, task_idx, resource_idx, time, energy))
    
        if not migration_candidates:
            return None
        
        neg_ratio, n_best, k_best, T_best, E_best = heappop(migration_candidates)
        return TaskMigrationState(
            time=T_best, 
            energy=E_best,
            efficiency_ratio=-neg_ratio,
            task_index=n_best + 1,
            target_execution_unit=k_best + 1
        )

    # Main optimization loop

    current_energy = total_energy(tasks, core_powers, cloud_sending_power)
    # Continue as long as we can improve energy consumption
    energy_improved = True
    while energy_improved:
        # Store current energy as reference for this iteration
        previous_energy = current_energy
        # Calculate current schedule metrics
        current_time = total_time(tasks)
        T_max = T_final * 1.5
        # Initialize possible migration choices
        migration_choice = initialize_migration_choices(tasks)
        # Evaluate all possible migrations
        migration_trials_results = []
        for task_idx in range(len(tasks)):
            for target_location in range(4):  # 0-3 for cloud and local cores
                if migration_choice[task_idx, target_location]:
                    continue
                    
                migration_trial_time, migration_trial_energy = evaluate_migration(tasks, sequence, task_idx, target_location)
                migration_trials_results.append((task_idx, target_location, migration_trial_time, migration_trial_energy))
        
        # Find the best migration according to paper's criteria
        best_migration = find_best_migration(
            migration_trials_results=migration_trials_results,
            T_final=current_time,
            E_total=previous_energy,
            T_max=T_max
        )
        
        # If no valid migration exists, we're done
        if best_migration is None:
            energy_improved = False
            break
        
        # Apply the selected migration
        sequence = construct_sequence(
            tasks,
            best_migration.task_index,
            best_migration.target_execution_unit - 1,
            sequence
        )
        kernel_algorithm(tasks, sequence)
        # Calculate new energy and determine if we improved
        current_energy = total_energy(tasks, core_powers, cloud_sending_power)
        energy_improved = current_energy < previous_energy
        
        # Periodic cache cleanup to manage memory
        if len(migration_cache) > 1000:
            migration_cache.clear()

    return tasks, sequence

def print_task_schedule(tasks):
    assignment_mapping = {
        0: "Core 1",
        1: "Core 2",
        2: "Core 3",
        3: "Cloud",
        -2: "Not Scheduled"
    }

    print_tasks = []
    
    for task in tasks:
        assignment_value = assignment_mapping.get(task.assignment, "Unknown")

        if task.is_core_task:
            print_tasks.append({
                "task id": task.id,
                "assignment": assignment_value,
                "core start_time": task.execution_unit_start_times[task.assignment],
                "core finish_time": task.execution_unit_start_times[task.assignment] + task.core_execution_times[task.assignment]
            })
        else:
            print_tasks.append({
                "task id": task.id,
                "assignment": assignment_value,
                "wireless sending start_time": task.execution_unit_start_times[3],
                "wireless sending finish_time": task.execution_unit_start_times[3] + task.cloud_execution_times[0],
                "cloud start_time": task.remote_cloud_ready_time,
                "cloud finish_time": task.remote_cloud_ready_time + task.cloud_execution_times[1],
                "wireless receiving start_time": task.wireless_recieving_ready_time,
                "wireless receiving finish_time": task.wireless_recieving_ready_time + task.cloud_execution_times[2]
            })

    for task in print_tasks:
        print(task)

def check_schedule_constraints(tasks):
    """
    Validates schedule constraints considering cloud task pipelining
    
    Args:
        tasks: List of Task objects with scheduling information
    Returns:
        tuple: (is_valid, violations)
    """
    violations = []
    
    def check_sending_channel():
        """Verify wireless sending channel is used sequentially"""
        cloud_tasks = [n for n in tasks if not n.is_core_task]
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
        cloud_tasks = [n for n in tasks if not n.is_core_task]
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
        cloud_tasks = [n for n in tasks if not n.is_core_task]
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
        for task in tasks:
            if not task.is_core_task:  # For cloud tasks
                # Check if all parents have completed necessary phases
                for parent in task.parents:
                    if parent.is_core_task:
                        # Core parent must complete before child starts sending
                        if parent.local_core_finish_time > task.execution_unit_start_times[3]:
                            violations.append({
                                'type': 'Core-Cloud Dependency Violation',
                                'parent': parent.id,
                                'child': task.id,
                                'detail': f'Core Task {parent.id} finishes at {parent.local_core_finish_time} but Cloud Task {task.id} starts sending at {task.execution_unit_start_times[3]}'
                            })
                    else:
                        # Cloud parent must complete sending before child starts sending
                        if parent.wireless_sending_finish_time > task.execution_unit_start_times[3]:
                            violations.append({
                                'type': 'Cloud Pipeline Dependency Violation',
                                'parent': parent.id,
                                'child': task.id,
                                'detail': f'Parent Task {parent.id} sending phase ends at {parent.wireless_sending_finish_time} but Task {task.id} starts sending at {task.execution_unit_start_times[3]}'
                            })
            else:  # For core tasks
                # All parents must complete fully before core task starts
                for parent in task.parents:
                    parent_finish = (parent.wireless_recieving_finish_time 
                                  if not parent.is_core_task else parent.local_core_finish_time)
                    if parent_finish > task.execution_unit_start_times[task.assignment]:
                        violations.append({
                            'type': 'Core Task Dependency Violation',
                            'parent': parent.id,
                            'child': task.id,
                            'detail': f'Parent Task {parent.id} finishes at {parent_finish} but Core Task {task.id} starts at {task.execution_unit_start_times[task.assignment]}'
                        })

    def check_core_execution():
        """Verify core tasks don't overlap"""
        core_tasks = [n for n in tasks if n.is_core_task]
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
def print_validation_report(tasks):
    """Print detailed schedule validation report"""
    is_valid, violations = check_schedule_constraints(tasks)
    
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
    cloud_tasks = [n for n in tasks if not n.is_core_task]
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
    core_tasks = [n for n in tasks if n.is_core_task]
    if core_tasks:
        print("\nCore Execution:")
        print("=" * 50)
        for core_id in range(3):
            core_specific = [t for t in core_tasks if t.assignment == core_id]
            if core_specific:
                print(f"\nCore {core_id}:")
                for task in sorted(core_specific, key=lambda x: x.execution_unit_start_times[core_id]):
                    print(f"  Task {task.id}: {task.execution_unit_start_times[core_id]:2d} -> {task.local_core_finish_time:2d}")


def print_task_graph(tasks):
        for task in tasks:
            children_ids = [child.id for child in task.children]
            parent_ids = [parent.id for parent in task.parents]
            print(f"Task {task.id}:")
            print(f"  Parents: {parent_ids}")
            print(f"  Children: {children_ids}")
            print()

if __name__ == '__main__':
    """
    task20 = Task(id=20, parents=None, children=[])
    task19 = Task(id=19, parents=None, children=[task20])
    task18 = Task(id=18, parents=None, children=[task20])
    task17 = Task(id=17, parents=None, children=[task20])
    task16 = Task(id=16, parents=None, children=[task19])
    task15 = Task(id=15, parents=None, children=[task19])
    task14 = Task(id=14, parents=None, children=[task18, task19])
    task13 = Task(id=13, parents=None, children=[task17, task18])
    task12 = Task(id=12, parents=None, children=[task17])
    task11 = Task(id=11, parents=None, children=[task15, task16])
    task10 = Task(id=10, parents=None, children=[task11,task15])
    task9 = Task(id=9, parents=None, children=[task13,task14])
    task8 = Task(id=8, parents=None, children=[task12,task13])
    task7 = Task(id=7, parents=None, children=[task12])
    task6 = Task(id=6, parents=None, children=[task10,task11])
    task5 = Task(id=5, parents=None, children=[task9,task10])
    task4 = Task(id=4, parents=None, children=[task8,task9])
    task3 = Task(id=3, parents=None, children=[task7, task8])
    task2 = Task(id=2, parents=None, children=[task7])
    task1 = Task(id=1, parents=None, children=[task7])
    task1.parents = []
    task2.parents = []
    task3.parents = []
    task4.parents = []
    task5.parents = []
    task6.parents = []
    task7.parents = [task1,task2,task3]
    task8.parents = [task3, task4]
    task9.parents = [task4,task5]
    task10.parents = [task5, task6]
    task11.parents = [task6, task10]
    task12.parents = [task7, task8]
    task13.parents = [task8, task9]
    task14.parents = [task9, task10]
    task15.parents = [task10, task11]
    task16.parents = [task11]
    task17.parents = [task12, task13]
    task18.parents = [task13, task14]
    task19.parents = [task14, task15,task16]
    task20.parents = [task17, task18,task19]

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10,task11,task12,task13,task14,task15,task16,task17,task18,task19,task20]

    task10 = Task(id=10, parents=None, children=[])
    task9 = Task(id=9, parents=None, children=[task10])
    task8 = Task(id=8, parents=None, children=[task9])
    task7 = Task(id=7, parents=None, children=[task9,task10])
    task6 = Task(id=6, parents=None, children=[task10])
    task5 = Task(id=5, parents=None, children=[task6])
    task4 = Task(id=4, parents=None, children=[task7,task8])
    task3 = Task(id=3, parents=None, children=[task7, task8])
    task2 = Task(id=2, parents=None, children=[task5,task7])
    task1 = Task(id=1, parents=None, children=[task2, task3, task4])
    task1.parents = []
    task2.parents = [task1]
    task3.parents = [task1]
    task4.parents = [task1]
    task5.parents = [task2]
    task6.parents = [task5]
    task7.parents = [task2,task3,task4]
    task8.parents = [task3, task4]
    task9.parents = [task7,task8]
    task10.parents = [task6, task7, task9]
    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10]

    task20 = Task(id=20, parents=None, children=[])
    task19 = Task(id=19, parents=None, children=[])
    task18 = Task(id=18, parents=None, children=[])
    task17 = Task(id=17, parents=None, children=[])
    task16 = Task(id=16, parents=None, children=[task19])
    task15 = Task(id=15, parents=None, children=[task19])
    task14 = Task(id=14, parents=None, children=[task18, task19])
    task13 = Task(id=13, parents=None, children=[task17, task18])
    task12 = Task(id=12, parents=None, children=[task17])
    task11 = Task(id=11, parents=None, children=[task15, task16])
    task10 = Task(id=10, parents=None, children=[task11,task15])
    task9 = Task(id=9, parents=None, children=[task13,task14])
    task8 = Task(id=8, parents=None, children=[task12,task13])
    task7 = Task(id=7, parents=None, children=[task12])
    task6 = Task(id=6, parents=None, children=[task10,task11])
    task5 = Task(id=5, parents=None, children=[task9,task10])
    task4 = Task(id=4, parents=None, children=[task8,task9])
    task3 = Task(id=3, parents=None, children=[task7, task8])
    task2 = Task(id=2, parents=None, children=[task7,task8])
    task1 = Task(id=1, parents=None, children=[task7])
    task1.parents = []
    task2.parents = []
    task3.parents = []
    task4.parents = []
    task5.parents = []
    task6.parents = []
    task7.parents = [task1,task2,task3]
    task8.parents = [task3, task4]
    task9.parents = [task4,task5]
    task10.parents = [task5, task6]
    task11.parents = [task6, task10]
    task12.parents = [task7, task8]
    task13.parents = [task8, task9]
    task14.parents = [task9, task10]
    task15.parents = [task10, task11]
    task16.parents = [task11]
    task17.parents = [task12, task13]
    task18.parents = [task13, task14]
    task19.parents = [task14, task15,task16]
    task20.parents = [task12]

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10,task11,task12,task13,task14,task15,task16,task17,task18,task19,task20]
    """

    task10 = Task(10)
    task9 = Task(9, children=[task10])
    task8 = Task(8, children=[task10])
    task7 = Task(7, children=[task10])
    task6 = Task(6, children=[task8])
    task5 = Task(5, children=[task9])
    task4 = Task(4, children=[task8, task9])
    task3 = Task(3, children=[task7])
    task2 = Task(2, children=[task8, task9])
    task1 = Task(1, children=[task2, task3, task4, task5, task6])
    task10.parents = [task7, task8, task9]
    task9.parents = [task2, task4, task5]
    task8.parents = [task2, task4, task6]
    task7.parents = [task3]
    task6.parents = [task1]
    task5.parents = [task1]
    task4.parents = [task1]
    task3.parents = [task1]
    task2.parents = [task1]
    task1.parents = []
    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10]

    print_task_graph(tasks)
    
    primary_assignment(tasks)
    task_prioritizing(tasks)
    sequence = execution_unit_selection(tasks)
    T_final = total_time(tasks)
    E_total = total_energy(tasks, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    print("INITIAL TIME: ", T_final)
    print("INITIAL ENERGY:", E_total)
    print("INITIAL TASK SCHEDULE: ")
    print_task_schedule(tasks)
    print_validation_report(tasks)
    #check_mcc_constraints(tasks)

    tasks2, sequence = optimize_task_scheduling(tasks, sequence, T_final, core_powers=[1, 2, 4], cloud_sending_power=0.5)

    print("final sequence: ")
    for s in sequence:
        print([i for i in s])

    T_final = total_time(tasks)
    E_final = total_energy(tasks, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    print("FINAL TIME: ", T_final)
    print("FINAL ENERGY:", E_final)
    print("FINAL TASK SCHEDULE: ")
    print_task_schedule(tasks2)
    print_validation_report(tasks2)
    #check_mcc_constraints(tasks2)
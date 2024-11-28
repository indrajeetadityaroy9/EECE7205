def calculate_core_benefit(task, core_id, core_powers):
    """
    Enhanced core benefit calculation that better balances workload
    across available cores.
    """
    base_execution_time = task.core_speed[core_id]
    power_consumption = core_powers[core_id]
    
    # Consider current core utilization
    core_load = sum(1 for node in nodes if node.is_core and node.assignment == core_id)
    load_factor = 1.0 / (1 + core_load)
    
    # Calculate power efficiency
    avg_power = sum(core_powers) / len(core_powers)
    power_efficiency = 1 - (power_consumption / avg_power)
    
    # Consider task characteristics
    computation_intensity = sum(task.core_speed) / len(task.core_speed)
    
    # Combine factors with emphasis on balancing
    return (power_efficiency * computation_intensity * load_factor) - \
           (base_execution_time * power_consumption)

def evaluate_parallel_potential(task, location, nodes):
    """
    Evaluates potential for parallel execution when assigning a task to a location.
    This improved implementation better accounts for:
    - Task dependencies and independence
    - Resource utilization patterns
    - Parallelization opportunities
    
    Args:
        task: Node object representing the task being evaluated
        location: Target location (0-2 for cores, 3 for cloud)
        nodes: List of all tasks in the graph
    
    Returns:
        Float value representing parallel execution potential
    """
    parallel_score = 0
    
    for other_task in nodes:
        # Skip invalid comparisons
        if other_task == task:
            continue
            
        # Skip direct dependencies
        if other_task in task.parents or other_task in task.children:
            continue
        
        # Skip indirect dependencies through common ancestors/descendants
        if any(p in other_task.parents for p in task.parents):
            continue
            
        # Calculate parallel potential based on resource location
        if location == 3:  # Cloud assignment
            if other_task.is_core:
                # Tasks can execute in parallel if one is on cloud and other on core
                # Weight the contribution based on dependency distance
                dependency_distance = len(other_task.parents) + len(other_task.children)
                weight = 1.0 / (1 + dependency_distance)
                parallel_score += weight
        else:  # Core assignment
            if not other_task.is_core:
                # Similar weighting for core assignments
                dependency_distance = len(other_task.parents) + len(other_task.children)
                weight = 1.0 / (1 + dependency_distance)
                parallel_score += weight
    
    return parallel_score

def task_migration_algorithm(nodes, T_max, core_powers, cloud_sending_power):
    """
    Enhanced task migration algorithm that properly considers the T_max constraint
    when making migration decisions. This follows the paper's two-step approach:
    1. Generate initial schedule
    2. Perform energy-reducing migrations while respecting T_max
    """
    # Start with initial scheduling using paper's equations
    core_earliest_ready = [0] * 3
    cloud_earliest_ready = 0
    
    primary_assignment(nodes, core_earliest_ready, cloud_earliest_ready)
    task_prioritizing(nodes)
    initial_sequences = execution_unit_selection(nodes)

    def evaluate_migration_impact(task, new_location, current_time):
        """
        Evaluates both energy and time impact of a potential migration.
        Returns None if migration would violate T_max constraint.
        """
        # Save current state to restore after evaluation
        original_state = save_current_state(nodes)
        
        # Try the migration
        task.assignment = new_location
        task.is_core = (new_location != 3)
        
        # Use execution_unit_selection to get valid schedule after migration
        new_sequences = execution_unit_selection(nodes)
        
        # Calculate new completion time and energy
        new_time = total_time(nodes)
        new_energy = total_energy(nodes, core_powers, cloud_sending_power)
        
        # Restore original state
        restore_state(nodes, original_state)
        
        # If migration would violate T_max, return None
        if new_time > T_max:
            return None
            
        # Calculate time slack (how close we are to T_max)
        time_slack = T_max - new_time
        
        # Calculate energy improvement
        energy_change = current_energy - new_energy
        
        return {
            'time': new_time,
            'energy': new_energy,
            'slack': time_slack,
            'energy_change': energy_change
        }

    iteration = 0
    max_iterations = 100
    
    while iteration < max_iterations:
        current_time = total_time(nodes)
        current_energy = total_energy(nodes, core_powers, cloud_sending_power)
        
        # Stop if we're already violating T_max
        if current_time > T_max:
            print(f"Warning: Current schedule exceeds T_max ({current_time} > {T_max})")
            break
            
        best_task = None
        best_location = None
        best_score = float('-inf')
        
        # Evaluate all possible migrations
        for task in nodes:
            current_location = task.assignment
            for new_location in range(4):
                if new_location != current_location:
                    # Evaluate impact of this migration
                    impact = evaluate_migration_impact(
                        task, new_location, current_time
                    )
                    
                    if impact is None:
                        continue  # Skip migrations that violate T_max
                    
                    # Score this migration based on energy savings and time slack
                    # We want to maximize energy savings while maintaining sufficient slack
                    score = impact['energy_change']
                    if impact['slack'] < 5:  # If we're getting close to T_max
                        score *= (impact['slack'] / 5)  # Reduce score to be conservative
                    
                    if score > best_score:
                        best_score = score
                        best_task = task
                        best_location = new_location
        
        if best_score <= 0:
            print("No more beneficial migrations found")
            break
            
        # Apply best migration
        print(f"Migrating Task {best_task.task_id} from "
              f"{'Cloud' if best_task.assignment == 3 else f'Core {best_task.assignment + 1}'} "
              f"to {'Cloud' if best_location == 3 else f'Core {best_location + 1}'}")
        
        best_task.assignment = best_location
        best_task.is_core = (best_location != 3)
        execution_unit_selection(nodes)
        
        iteration += 1
    
    final_time = total_time(nodes)
    if final_time > T_max:
        print(f"Warning: Final schedule exceeds T_max ({final_time} > {T_max})")
        
    return total_energy(nodes, core_powers, cloud_sending_power), final_time

def kernel_algorithm(nodes, core_powers, cloud_sending_power, T_max):
    """
    Core scheduling algorithm that determines task execution ordering and resource assignment.
    This implementation integrates the paper's prioritization method with careful resource tracking.
    
    The algorithm follows three main steps:
    1. Task prioritization using the paper's equations
    2. Resource initialization and tracking
    3. Task scheduling while maintaining timing constraints
    """
    # Initialize resource availability tracking
    core_ready_times = [0] * 3  # When each core becomes available
    cloud_phases = {
        'send': 0,     # When sending channel becomes free
        'compute': 0,  # When cloud computation can start
        'receive': 0   # When receiving channel becomes free
    }
    
    # Use paper's prioritization method instead of custom ordering
    task_prioritizing(nodes)  # Sets priority_score using equations (13)-(16)
    execution_order = sorted(nodes, key=lambda x: x.priority_score, reverse=True)
    
    for node in execution_order:
        # Calculate the earliest possible start time considering dependencies
        ready_time = calculate_ready_time(node)
        
        if node.is_core:
            finish_time = schedule_local_task(node, ready_time, core_ready_times)
        else:
            finish_time = schedule_cloud_task(node, ready_time, cloud_phases)
            
        # Verify timing consistency
        if node.is_core:
            assert node.local_finish_time > node.local_ready_time, \
                f"Invalid timing for task {node.task_id}"
        else:
            assert node.cloud_receiving_finish_time > node.cloud_sending_ready_time, \
                f"Invalid timing for task {node.task_id}"
    
    completion_time = total_time(nodes)
    current_energy = total_energy(nodes, core_powers, cloud_sending_power)
    
    return completion_time <= T_max, current_energy, completion_time

def schedule_cloud_task(node, ready_time, cloud_phases):
    """
    Schedules cloud tasks with zero gaps between wireless sending phases.
    This implementation ensures the tight sequential packing shown in Figure 4.
    """
    # The key insight is that we want to start sending immediately after
    # the previous task's sending phase, even if that means waiting longer
    # for computation and receiving phases
    
    # Find the earliest possible send start time
    send_start = max(ready_time, cloud_phases['send'])
    send_finish = send_start + node.cloud_speed[0]  # Always 3 time units
    
    # The computation phase starts after sending completes
    compute_start = send_finish
    compute_finish = compute_start + node.cloud_speed[1]  # Always 1 time unit
    
    # The receive phase follows computation
    receive_start = compute_finish
    receive_finish = receive_start + node.cloud_speed[2]  # Always 1 time unit
    
    # Update task timing
    node.cloud_sending_ready_time = send_start
    node.cloud_sending_finish_time = send_finish
    node.cloud_ready_time = compute_start
    node.cloud_finish_time = compute_finish
    node.cloud_receiving_finish_time = receive_finish
    
    # Update resource availability
    # Critical: Only update send phase timing to maintain continuous sequence
    cloud_phases['send'] = send_finish
    
    return receive_finish

def find_alternate_slot(node, ready_time, cloud_phases):
    """
    Tries to find a better scheduling slot that minimizes gaps
    between cloud task executions.
    """
    # Look for gaps in current schedule that could fit this task
    total_time = sum(node.cloud_speed)
    current_gap = cloud_phases['send'] - ready_time
    
    if current_gap >= total_time:
        # Schedule in current gap if possible
        send_start = ready_time
        send_finish = send_start + node.cloud_speed[0]
        compute_start = send_finish
        compute_finish = compute_start + node.cloud_speed[1]
        receive_start = compute_finish
        receive_finish = receive_start + node.cloud_speed[2]
        
        if receive_finish <= cloud_phases['send']:
            return schedule_in_gap(node, send_start, cloud_phases)
    
    return schedule_next_available(node, cloud_phases['send'], cloud_phases)

def schedule_in_gap(node, start_time, cloud_phases):
    """
    Schedules a cloud task in an available gap in the current schedule.
    This helps minimize idle time in cloud resource usage by utilizing available gaps.
    
    Args:
        node: The task node to be scheduled
        start_time: The potential start time for this task within the gap
        cloud_phases: Dictionary tracking cloud resource availability
        
    Returns:
        The finish time of the receive phase for this task
    """
    # Calculate timing for all three phases
    send_start = start_time
    send_finish = send_start + node.cloud_speed[0]
    
    compute_start = send_finish
    compute_finish = compute_start + node.cloud_speed[1]
    
    receive_start = compute_finish
    receive_finish = receive_start + node.cloud_speed[2]
    
    # Update task timing information
    node.cloud_sending_ready_time = send_start
    node.cloud_sending_finish_time = send_finish
    node.cloud_ready_time = compute_start
    node.cloud_finish_time = compute_finish
    node.cloud_receiving_finish_time = receive_finish
    
    # Note: We don't update cloud_phases here because this task fits in an existing gap
    # The original resource availability times remain valid
    
    return receive_finish

def schedule_next_available(node, earliest_time, cloud_phases):
    """
    Schedules a cloud task at the next available time slot when a suitable gap
    cannot be found. This ensures proper sequential scheduling when gap filling
    is not possible.
    
    Args:
        node: The task node to be scheduled
        earliest_time: The earliest time this task can be scheduled
        cloud_phases: Dictionary tracking cloud resource availability
        
    Returns:
        The finish time of the receive phase for this task
    """
    # Start at the earliest possible time
    send_start = earliest_time
    send_finish = send_start + node.cloud_speed[0]
    
    compute_start = send_finish
    compute_finish = compute_start + node.cloud_speed[1]
    
    receive_start = compute_finish
    receive_finish = receive_start + node.cloud_speed[2]
    
    # Update task timing information
    node.cloud_sending_ready_time = send_start
    node.cloud_sending_finish_time = send_finish
    node.cloud_ready_time = compute_start
    node.cloud_finish_time = compute_finish
    node.cloud_receiving_finish_time = receive_finish
    
    # Update cloud resource availability
    cloud_phases['send'] = send_finish
    cloud_phases['compute'] = compute_finish
    cloud_phases['receive'] = receive_finish
    
    return receive_finish

def schedule_local_task(node, ready_time, core_ready_times):
    """
    Schedules local tasks to maximize core utilization.
    In Figure 4, we see tasks executing back-to-back on Core 1.
    """
    core = node.assignment
    
    # Start time should be as early as possible while respecting dependencies
    # and core availability
    start_time = max(ready_time, core_ready_times[core])
    finish_time = start_time + node.core_speed[core]
    
    # Update task timing
    node.local_ready_time = start_time
    node.local_finish_time = finish_time
    core_ready_times[core] = finish_time  # Update when core becomes available
    
    # Clear cloud timing for local tasks
    node.cloud_sending_finish_time = 0
    node.cloud_finish_time = 
    node.cloud_receiving_finish_time = 0
    
    return finish_time

def calculate_ready_time(node):
    """
    Calculates the earliest possible start time for a task based on dependencies.
    Considers both local and cloud execution of parent tasks.
    """
    if not node.parents:
        return 0
    
    # Calculate the latest finish time among parent tasks
    parent_finish_times = []
    for parent in node.parents:
        if parent.is_core:
            # For local execution, use local finish time
            parent_finish_times.append(parent.local_finish_time)
        else:
            # For cloud execution, must wait for results to be received
            parent_finish_times.append(parent.cloud_receiving_finish_time)
    
    return max(parent_finish_times)

def save_current_state(nodes):
    """
    Creates a deep copy of the current scheduling state for all tasks.
    This allows trying potential migrations without affecting the original schedule.
    """
    return [
        (node.assignment, node.is_core, 
         node.local_finish_time, node.local_ready_time,
         node.cloud_sending_finish_time, node.cloud_ready_time,
         node.cloud_finish_time, node.cloud_receiving_finish_time)
        for node in nodes
    ]

def restore_state(nodes, state):
    """
    Restores a previously saved scheduling state after evaluating a potential migration.
    Ensures all timing and assignment information is properly reset.
    """
    for node, (assignment, is_core, local_ft, local_rt,
               cloud_send_ft, cloud_ready_t, cloud_ft, cloud_receive_ft) in zip(nodes, state):
        node.assignment = assignment
        node.is_core = is_core
        node.local_finish_time = local_ft
        node.local_ready_time = local_rt
        node.cloud_sending_finish_time = cloud_send_ft
        node.cloud_ready_time = cloud_ready_t
        node.cloud_finish_time = cloud_ft
        node.cloud_receiving_finish_time = cloud_receive_ft

from copy import deepcopy
import bisect
from dataclasses import dataclass
from collections import deque
from heapq import heappush, heappop
from enum import Enum
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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
    11: [10, 7, 4],
    12: [11, 8, 5],
    13: [9, 6, 3],
    14: [12, 8, 4],
    15: [10, 7, 3],
    16: [11, 7, 4],
    17: [9, 6, 3],
    18: [12, 8, 5],
    19: [10, 7, 4],
    20: [11, 8, 5]
}

cloud_execution_times = [3, 1, 1]

class SchedulingState(Enum):
    UNSCHEDULED = 0
    SCHEDULED = 1
    KERNEL_SCHEDULED = 2

@dataclass
class TaskMigrationState:
    time: float
    energy: float
    efficiency: float
    task_index: int
    target_execution_unit: int

class Task(object):
    def __init__(self, id, pred_tasks=None, succ_task=None):
        self.id = id
        self.pred_tasks = pred_tasks or []
        self.succ_task = succ_task or []
        self.core_execution_times = core_execution_times[id]
        self.cloud_execution_times = cloud_execution_times
        self.FT_l = 0 
        self.FT_ws = 0
        self.FT_c = 0  
        self.FT_wr = 0
        self.RT_l = -1
        self.RT_ws = -1
        self.RT_c = -1
        self.RT_wr = -1
        self.priority_score = None
        self.assignment = -2           
        self.is_core_task = False      
        self.execution_unit_task_start_times = [-1,-1,-1,-1] 
        self.execution_finish_time = -1
        self.is_scheduled = SchedulingState.UNSCHEDULED

def total_time(tasks):
    return max(
        max(task.FT_l, task.FT_wr)
        for task in tasks
        if not task.succ_task  
    )

def calculate_energy_consumption(task, core_powers, cloud_sending_power):
    if task.is_core_task:
        return core_powers[task.assignment] * task.core_execution_times[task.assignment]
    else:
        return cloud_sending_power * task.cloud_execution_times[0]

def total_energy(tasks, core_powers, cloud_sending_power):
    return sum(
        calculate_energy_consumption(task, core_powers, cloud_sending_power) 
        for task in tasks
    )

def primary_assignment(tasks):
    for task in tasks:
        t_l_min = min(task.core_execution_times)
        t_re = (task.cloud_execution_times[0] + task.cloud_execution_times[1] + task.cloud_execution_times[2])

        if t_re < t_l_min:
            task.is_core_task = False
        else:
            task.is_core_task = True

def task_prioritizing(tasks):
    w = [0] * len(tasks)
    for i, task in enumerate(tasks):
        if not task.is_core_task:  
            w[i] = (task.cloud_execution_times[0] +  task.cloud_execution_times[1] +  task.cloud_execution_times[2])
        else:  
            w[i] = sum(task.core_execution_times) / len(task.core_execution_times)
    computed_priority_scores = {}

    def calculate_priority(task):
        if task.id in computed_priority_scores:
            return computed_priority_scores[task.id]

        if task.succ_task == []:
            computed_priority_scores[task.id] = w[task.id - 1]
            return w[task.id - 1]

        max_successor_priority = max(calculate_priority(successor) for successor in task.succ_task)
        task_priority = w[task.id - 1] + max_successor_priority
        computed_priority_scores[task.id] = task_priority
        return task_priority

    for task in tasks:
        calculate_priority(task)

    for task in tasks:
        task.priority_score = computed_priority_scores[task.id]

class InitialTaskScheduler:
    def __init__(self, tasks, num_cores=3):
        self.tasks = tasks
        self.k = num_cores
        
        self.core_earliest_ready = [0] * self.k
        self.ws_ready = 0
        self.wr_ready = 0
        
        self.sequences = [[] for _ in range(self.k + 1)]
        
    def get_priority_ordered_tasks(self):
        task_priority_list = [(task.priority_score, task.id) for task in self.tasks]
        task_priority_list.sort(reverse=True)
        return [item[1] for item in task_priority_list]
        
    def classify_entry_tasks(self, priority_order):
        entry_tasks = []
        non_entry_tasks = []

        for id in priority_order:
            task = self.tasks[id - 1]
            
            if not task.pred_tasks:
                entry_tasks.append(task)
            else:
                non_entry_tasks.append(task)
                
        return entry_tasks, non_entry_tasks

    def identify_optimal_local_core(self, task, ready_time=0):
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
        task.FT_l = finish_time
        task.execution_finish_time = finish_time
        task.execution_unit_task_start_times = [-1] * (self.k + 1)
        task.execution_unit_task_start_times[core] = start_time
        self.core_earliest_ready[core] = finish_time
        task.assignment = core
        task.is_scheduled = SchedulingState.SCHEDULED
        self.sequences[core].append(task.id)

    def calculate_cloud_phases_timing(self, task):
        send_ready = task.RT_ws
        send_finish = send_ready + task.cloud_execution_times[0]
        cloud_ready = send_finish
        cloud_finish = cloud_ready + task.cloud_execution_times[1]
        receive_ready = cloud_finish
        receive_finish = (
            max(self.wr_ready, receive_ready) + 
            task.cloud_execution_times[2]
        )

        return send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish

    def schedule_on_cloud(self, task, send_ready, send_finish, cloud_ready, cloud_finish, receive_ready, receive_finish):
        task.RT_ws = send_ready
        task.FT_ws = send_finish
        task.RT_c = cloud_ready
        task.FT_c = cloud_finish
        task.RT_wr = receive_ready
        task.FT_wr = receive_finish
        task.execution_finish_time = receive_finish
        task.FT_l = 0
        task.execution_unit_task_start_times = [-1] * (self.k + 1)
        task.execution_unit_task_start_times[self.k] = send_ready
        task.assignment = self.k
        task.is_scheduled = SchedulingState.SCHEDULED
        self.ws_ready = send_finish
        self.wr_ready = receive_finish
        self.sequences[self.k].append(task.id)

    def schedule_entry_tasks(self, entry_tasks):
        cloud_entry_tasks = []
        for task in entry_tasks:
            if task.is_core_task:
                core, start_time, finish_time = self.identify_optimal_local_core(task)
                self.schedule_on_local_core(task, core, start_time, finish_time)
            else:
                cloud_entry_tasks.append(task)

        for task in cloud_entry_tasks:
            task.RT_ws = self.ws_ready
            timing = self.calculate_cloud_phases_timing(task)
            self.schedule_on_cloud(task, *timing)

    def calculate_non_entry_task_ready_times(self, task):
        task.RT_l = max(
            max(max(pred_task.FT_l, pred_task.FT_wr) 
                for pred_task in task.pred_tasks),
        )

        task.RT_ws = max(
            max(max(pred_task.FT_l, pred_task.FT_ws) 
                for pred_task in task.pred_tasks),
            self.ws_ready
        )

    def schedule_non_entry_tasks(self, non_entry_tasks):
        for task in non_entry_tasks:
            self.calculate_non_entry_task_ready_times(task)
            
            if not task.is_core_task:
                timing = self.calculate_cloud_phases_timing(task)
                self.schedule_on_cloud(task, *timing)
            else:
                core, start_time, finish_time = self.identify_optimal_local_core(
                    task, task.RT_l
                )
                
                timing = self.calculate_cloud_phases_timing(task)
                cloud_finish_time = timing[-1]
                
                if finish_time <= cloud_finish_time:
                    self.schedule_on_local_core(task, core, start_time, finish_time)
                else:
                    task.is_core_task = False
                    self.schedule_on_cloud(task, *timing)

def execution_unit_selection(tasks):
    scheduler = InitialTaskScheduler(tasks, 3)
    priority_orderered_tasks = scheduler.get_priority_ordered_tasks()
    entry_tasks, non_entry_tasks = scheduler.classify_entry_tasks(priority_orderered_tasks)
    scheduler.schedule_entry_tasks(entry_tasks)
    scheduler.schedule_non_entry_tasks(non_entry_tasks)
    return scheduler.sequences

def construct_sequence(tasks, id, execution_unit, original_sequence):
   id_to_task = {task.id: task for task in tasks}
   target_task = id_to_task.get(id)
   target_task_rt = target_task.RT_l if target_task.is_core_task else target_task.RT_ws
   original_assignment = target_task.assignment
   original_sequence[original_assignment].remove(target_task.id)
   new_sequence_task_list = original_sequence[execution_unit]

   start_times = [
       id_to_task[id].execution_unit_task_start_times[execution_unit] 
       for id in new_sequence_task_list
   ]

   insertion_index = bisect.bisect_left(start_times, target_task_rt)
   new_sequence_task_list.insert(insertion_index, target_task.id)
   target_task.assignment = execution_unit
   target_task.is_core_task = (execution_unit != 3)
   return original_sequence

class KernelScheduler:
    def __init__(self, tasks, sequences):
        self.tasks = tasks
        self.sequences = sequences
        self.RT_ls = [0] * 3
        self.cloud_phases_ready_times = [0] * 3
        self.dependency_ready, self.sequence_ready = self.initialize_task_state()
        
    def initialize_task_state(self):
        dependency_ready = [len(task.pred_tasks) for task in self.tasks]
        sequence_ready = [-1] * len(self.tasks)
        for sequence in self.sequences:
            if sequence:
                sequence_ready[sequence[0] - 1] = 0

        return dependency_ready, sequence_ready
    
    def update_task_state(self, task):
        if task.is_scheduled != SchedulingState.KERNEL_SCHEDULED:
            self.dependency_ready[task.id - 1] = sum(
                1 for pred_task in task.pred_tasks 
                if pred_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED
            )
            
            for sequence in self.sequences:
                if task.id in sequence:
                    idx = sequence.index(task.id)
                    if idx > 0:
                        prev_task = self.tasks[sequence[idx - 1] - 1]
                        self.sequence_ready[task.id - 1] = (
                            1 if prev_task.is_scheduled != SchedulingState.KERNEL_SCHEDULED 
                            else 0
                        )
                    else:
                        self.sequence_ready[task.id - 1] = 0
                    break
    
    def schedule_local_task(self, task):
        if not task.pred_tasks:
            task.RT_l = 0
        else:
            pred_task_completion_times = (
                max(pred_task.FT_l, pred_task.FT_wr) 
                for pred_task in task.pred_tasks
            )
            task.RT_l = max(pred_task_completion_times, default=0)

        core_index = task.assignment
        task.execution_unit_task_start_times = [-1] * 4

        task.execution_unit_task_start_times[core_index] = max(self.RT_ls[core_index],task.RT_l)

        task.FT_l = (task.execution_unit_task_start_times[core_index] + task.core_execution_times[core_index])

        self.RT_ls[core_index] = task.FT_l

        task.FT_ws = -1
        task.FT_c = -1
        task.FT_wr = -1
    
    def schedule_cloud_task(self, task):
        if not task.pred_tasks:
            task.RT_ws = 0
        else:
            pred_task_completion_times = (
                max(pred_task.FT_l, pred_task.FT_ws) 
                for pred_task in task.pred_tasks
            )
            task.RT_ws = max(pred_task_completion_times)

        task.execution_unit_task_start_times = [-1] * 4
        task.execution_unit_task_start_times[3] = max(self.cloud_phases_ready_times[0],task.RT_ws)
        task.FT_ws = (task.execution_unit_task_start_times[3] + task.cloud_execution_times[0])
        self.cloud_phases_ready_times[0] = task.FT_ws
        task.RT_c = max(task.FT_ws, max((pred_task.FT_c for pred_task in task.pred_tasks), default=0))
        task.FT_c = (max(self.cloud_phases_ready_times[1], task.RT_c) + task.cloud_execution_times[1])
        self.cloud_phases_ready_times[1] = task.FT_c
        task.RT_wr = task.FT_c
        task.FT_wr = (max(self.cloud_phases_ready_times[2], task.RT_wr) + task.cloud_execution_times[2])
        self.cloud_phases_ready_times[2] = task.FT_wr
        task.FT_l = -1
    
    def initialize_queue(self):
        return deque(
            task for task in self.tasks 
            if (
                self.sequence_ready[task.id - 1] == 0 
                and
                all(pred_task.is_scheduled == SchedulingState.KERNEL_SCHEDULED 
                    for pred_task in task.pred_tasks)
            )
        )


def kernel_algorithm(tasks, sequences):
   scheduler = KernelScheduler(tasks, sequences)
   queue = scheduler.initialize_queue()
   
   while queue:
       current_task = queue.popleft()
       current_task.is_scheduled = SchedulingState.KERNEL_SCHEDULED

       if current_task.is_core_task:
           scheduler.schedule_local_task(current_task)
       else:
           scheduler.schedule_cloud_task(current_task)
       
       for task in tasks:
           scheduler.update_task_state(task)
           
           if (scheduler.dependency_ready[task.id - 1] == 0 and
               scheduler.sequence_ready[task.id - 1] == 0 and
               task.is_scheduled != SchedulingState.KERNEL_SCHEDULED and
               task not in queue):
               queue.append(task)
   
   for task in tasks:
       task.is_scheduled = SchedulingState.UNSCHEDULED
   
   return tasks
    
def generate_cache_key(tasks, idx, target_execution_unit):
        return (idx, target_execution_unit, 
                tuple(task.assignment for task in tasks))

def evaluate_migration(tasks, seqs, idx, target_execution_unit, migration_cache, core_powers=[1, 2, 4], cloud_sending_power=0.5):
        cache_key = generate_cache_key(tasks, idx, target_execution_unit)
                    
        if cache_key in migration_cache:
            return migration_cache[cache_key]

        sequence_copy = [seq.copy() for seq in seqs]
        tasks_copy = deepcopy(tasks)

        sequence_copy = construct_sequence(
            tasks_copy, 
            idx + 1,
            target_execution_unit, 
            sequence_copy
        )

        kernel_algorithm(tasks_copy, sequence_copy)
        migration_T = total_time(tasks_copy)
        migration_E = total_energy(tasks_copy, core_powers, cloud_sending_power)
        migration_cache[cache_key] = (migration_T, migration_E)
        return migration_T, migration_E

def initialize_migration_choices(tasks):
        migration_choices = np.zeros((len(tasks), 4), dtype=bool)
        for i, task in enumerate(tasks):
            if task.assignment == 3:  
                migration_choices[i, :] = True
            else:
                migration_choices[i, task.assignment] = True
                    
        return migration_choices

def identify_optimal_migration(migration_trials_results, T_final, E_total, T_max):
        best_energy_reduction = 0
        best_migration = None

        for idx, resource_idx, time, energy in migration_trials_results:
            if time > T_max:
                continue

            energy_reduction = E_total - energy

            if time <= T_final and energy_reduction > 0:
                if energy_reduction > best_energy_reduction:
                    best_energy_reduction = energy_reduction
                    best_migration = (idx, resource_idx, time, energy)

        if best_migration:
            idx, resource_idx, time, energy = best_migration
            return TaskMigrationState(
                time=time,
                energy=energy,
                efficiency=best_energy_reduction,
                task_index=idx + 1,
                target_execution_unit=resource_idx + 1
            )

        migration_candidates = []
        for idx, resource_idx, time, energy in migration_trials_results:
            if time > T_max:
                continue
            
            energy_reduction = E_total - energy
            if energy_reduction > 0:
                time_increase = max(0, time - T_final)
                if time_increase == 0:
                    efficiency = float('inf')
                else:
                    efficiency = energy_reduction / time_increase
            
                heappush(migration_candidates, 
                        (-efficiency, idx, resource_idx, time, energy))

        if not migration_candidates:
            return None
        
        neg_ratio, n_best, k_best, T_best, E_best = heappop(migration_candidates)
        return TaskMigrationState(
            time=T_best, 
            energy=E_best,
            efficiency=-neg_ratio,
            task_index=n_best + 1,
            target_execution_unit=k_best + 1
        )

def optimize_task_scheduling(tasks, sequence, T_final, core_powers=[1, 2, 4], cloud_sending_power=0.5):
   core_powers = np.array(core_powers)
   migration_cache = {}
   current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power)

   energy_improved = True
   while energy_improved:
       previous_iteration_energy = current_iteration_energy
       current_time = total_time(tasks)
       T_max = T_final * 1.5
       migration_choices = initialize_migration_choices(tasks)

       migration_trials_results = []
       for idx in range(len(tasks)):
           for possible_execution_unit in range(4):
               if migration_choices[idx, possible_execution_unit]:
                   continue
                   
               migration_trial_time, migration_trial_energy = evaluate_migration(
                   tasks, sequence, idx, possible_execution_unit, migration_cache
               )
               migration_trials_results.append(
                   (idx, possible_execution_unit, 
                    migration_trial_time, migration_trial_energy)
               )
       
       best_migration = identify_optimal_migration(
           migration_trials_results=migration_trials_results,
           T_final=current_time,
           E_total=previous_iteration_energy,
           T_max=T_max
       )
       
       if best_migration is None:
           energy_improved = False
           break

       sequence = construct_sequence(
           tasks,
           best_migration.task_index,
           best_migration.target_execution_unit - 1,
           sequence
       )
       
       kernel_algorithm(tasks, sequence)
       current_iteration_energy = total_energy(tasks, core_powers, cloud_sending_power)
       energy_improved = current_iteration_energy < previous_iteration_energy
       
       if len(migration_cache) > 1000:
           migration_cache.clear()

   return tasks, sequence

def print_task_schedule(tasks):
    ASSIGNMENT_MAPPING = {
        0: "Core 1",
        1: "Core 2",
        2: "Core 3",
        3: "Cloud",
        -2: "Not Scheduled"
    }

    schedule_data = []
    for task in tasks:
        base_info = {
            "Task ID": task.id,
            "Assignment": ASSIGNMENT_MAPPING.get(task.assignment, "Unknown")
        }

        if task.is_core_task:
            start_time = task.execution_unit_task_start_times[task.assignment]
            schedule_data.append({
                **base_info,
                "Execution Window": f"{start_time:.2f} → "f"{start_time + task.core_execution_times[task.assignment]:.2f}"
            })
        else:
            send_start = task.execution_unit_task_start_times[3]
            send_end = send_start + task.cloud_execution_times[0]
            cloud_end = task.RT_c + task.cloud_execution_times[1]
            receive_end = task.RT_wr + task.cloud_execution_times[2]
            
            schedule_data.append({
                **base_info,
                "Send Phase": f"{send_start:.2f} → {send_end:.2f}",
                "Cloud Phase": f"{task.RT_c:.2f} → {cloud_end:.2f}",
                "Receive Phase": f"{task.RT_wr:.2f} → {receive_end:.2f}"
            })

    print("\nTask Scheduling Details:")
    print("-" * 80)
    
    for entry in schedule_data:
        print("\n", end="")
        for key, value in entry.items():
            print(f"{key:15}: {value}")
        print("-" * 40)

def check_schedule_constraints(tasks):
    violations = []
    
    def check_sending_channel():
        cloud_tasks = [n for n in tasks if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.execution_unit_task_start_times[3])
        
        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            if current.FT_ws > next_task.execution_unit_task_start_times[3]:
                violations.append({
                    'type': 'Wireless Sending Channel Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} sending ends at {current.FT_ws} but Task {next_task.id} starts at {next_task.execution_unit_task_start_times[3]}'
                })

    def check_computing_channel():
        cloud_tasks = [n for n in tasks if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.RT_c)
        
        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            if current.FT_c > next_task.RT_c:
                violations.append({
                    'type': 'Cloud Computing Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} computing ends at {current.FT_c} but Task {next_task.id} starts at {next_task.RT_c}'
                })

    def check_receiving_channel():
        cloud_tasks = [n for n in tasks if not n.is_core_task]
        sorted_tasks = sorted(cloud_tasks, key=lambda x: x.RT_wr)
        
        for i in range(len(sorted_tasks) - 1):
            current = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            if current.FT_wr > next_task.RT_wr:
                violations.append({
                    'type': 'Wireless Receiving Channel Conflict',
                    'task1': current.id,
                    'task2': next_task.id,
                    'detail': f'Task {current.id} receiving ends at {current.FT_wr} but Task {next_task.id} starts at {next_task.RT_wr}'
                })

    def check_pipelined_dependencies():
        for task in tasks:
            if not task.is_core_task:
                for pred_task in task.pred_tasks:
                    if pred_task.is_core_task:
                        if pred_task.FT_l > task.execution_unit_task_start_times[3]:
                            violations.append({
                                'type': 'Core-Cloud Dependency Violation',
                                'pred_task': pred_task.id,
                                'child': task.id,
                                'detail': f'Core Task {pred_task.id} finishes at {pred_task.FT_l} but Cloud Task {task.id} starts sending at {task.execution_unit_task_start_times[3]}'
                            })
                    else:
                        if pred_task.FT_ws > task.execution_unit_task_start_times[3]:
                            violations.append({
                                'type': 'Cloud Pipeline Dependency Violation',
                                'pred_task': pred_task.id,
                                'child': task.id,
                                'detail': f'Parent Task {pred_task.id} sending phase ends at {pred_task.FT_ws} but Task {task.id} starts sending at {task.execution_unit_task_start_times[3]}'
                            })
            else:
                for pred_task in task.pred_tasks:
                    pred_task_finish = (pred_task.FT_wr 
                                  if not pred_task.is_core_task else pred_task.FT_l)
                    if pred_task_finish > task.execution_unit_task_start_times[task.assignment]:
                        violations.append({
                            'type': 'Core Task Dependency Violation',
                            'pred_task': pred_task.id,
                            'child': task.id,
                            'detail': f'Parent Task {pred_task.id} finishes at {pred_task_finish} but Core Task {task.id} starts at {task.execution_unit_task_start_times[task.assignment]}'
                        })

    def check_core_execution():
        core_tasks = [n for n in tasks if n.is_core_task]
        for core_id in range(3):
            core_specific_tasks = [t for t in core_tasks if t.assignment == core_id]
            sorted_tasks = sorted(core_specific_tasks, key=lambda x: x.execution_unit_task_start_times[core_id])
            
            for i in range(len(sorted_tasks) - 1):
                current = sorted_tasks[i]
                next_task = sorted_tasks[i + 1]
                
                if current.FT_l > next_task.execution_unit_task_start_times[core_id]:
                    violations.append({
                        'type': f'Core {core_id} Execution Conflict',
                        'task1': current.id,
                        'task2': next_task.id,
                        'detail': f'Task {current.id} finishes at {current.FT_l} but Task {next_task.id} starts at {next_task.execution_unit_task_start_times[core_id]}'
                    })

    check_sending_channel()
    check_computing_channel()
    check_receiving_channel()
    check_pipelined_dependencies()
    check_core_execution()
    return len(violations) == 0, violations

def print_validation_report(tasks):
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


def print_task_graph(tasks):
        for task in tasks:
            succ_ids = [child.id for child in task.succ_task]
            pred_ids = [pred_task.id for pred_task in task.pred_tasks]
            print(f"Task {task.id}:")
            print(f"  Parents: {pred_ids}")
            print(f"  Children: {succ_ids}")
            print()

def print_final_sequences(sequences):
   print("\Execution Sequences:")
   print("-" * 40)

   for i, sequence in enumerate(sequences):
       if i < 3:
           label = f"Core {i+1}"
       else:
           label = "Cloud"
       task_list = [t for t in sequence]
       print(f"{label:12}: {task_list}")

def create_and_visualize_task_graph(nodes, save_path=None, formats=None, dpi=300):
    G = nx.DiGraph()
    
    for node in nodes:
        G.add_node(node.id)
    for node in nodes:
        for child in node.succ_task:
            G.add_edge(node.id, child.id)
    
    plt.figure(figsize=(8, 10))
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=17)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15)
    plt.axis('off')

    if save_path and formats:
        plt.tight_layout()
        
        for fmt in formats:
            full_path = f"{save_path}.{fmt}"
            try:
                if fmt in ['pdf', 'svg', 'eps']:
                    plt.savefig(full_path, format=fmt, bbox_inches='tight',pad_inches=0.1)
                else:
                    plt.savefig(full_path, format=fmt, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
                print(f"Successfully saved visualization as {full_path}")
            except Exception as e:
                print(f"Error saving {fmt} format: {str(e)}")
    
    return plt.gcf()

def plot_gantt(tasks, sequences, title="Schedule"):
    fig, ax = plt.subplots(figsize=(15, 8))

    task_map = {t.id: t for t in tasks}

    def add_centered_text(ax, start, duration, y_level, task_id):
        center_x = start + duration / 2
        
        renderer = ax.figure.canvas.get_renderer()
        text_obj = ax.text(0, 0, f"T{task_id}", fontsize=10, fontweight='bold')
        bbox = text_obj.get_window_extent(renderer=renderer)
        text_obj.remove()
        
        trans = ax.transData.inverted()
        text_width = trans.transform((bbox.width, 0))[0] - trans.transform((0, 0))[0]
        
        if text_width > duration * 0.9:
            ax.text(center_x, y_level + 0.3, f"T{task_id}",
                   va='bottom', ha='center',
                   color='black', fontsize=10, fontweight='bold',
                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        else:
            ax.text(center_x, y_level, f"T{task_id}",
                   va='center', ha='center',
                   color='black', fontsize=10, fontweight='bold')
            
    max_completion_time = max(
        max(t.FT_l, t.FT_wr) if t.FT_wr > 0 else t.FT_l
        for t in tasks
    )

    yticks = []
    ytick_labels = []

    colors = {
        'core': 'lightcoral',
        'sending': 'lightgreen',
        'computing': 'lightblue',
        'receiving': 'plum'
    }

    y_positions = {
        'Core 1': 5,
        'Core 2': 4,
        'Core 3': 3,
        'Cloud Sending': 2,
        'Cloud Computing': 1,
        'Cloud Receiving': 0
    }

    for core_idx in range(3):
        y_level = y_positions[f'Core {core_idx+1}']
        yticks.append(y_level)
        ytick_labels.append(f'Core {core_idx+1}')
        
        if core_idx < len(sequences):
            for task_id in sequences[core_idx]:
                task = task_map[task_id]
                if task.assignment == core_idx:
                    start_time = task.execution_unit_task_start_times[core_idx]
                    duration = task.core_execution_times[core_idx]
                    ax.barh(y_level, duration, left=start_time, height=0.4,align='center', color=colors['core'], edgecolor='black')
                    add_centered_text(ax, start_time, duration, y_level, task.id)

    cloud_phases = [
        ('Cloud Sending', 'sending', 
         lambda t: (t.execution_unit_task_start_times[3], t.cloud_execution_times[0])),
        ('Cloud Computing', 'computing', 
         lambda t: (t.RT_c, t.cloud_execution_times[1])),
        ('Cloud Receiving', 'receiving', 
         lambda t: (t.RT_wr, t.cloud_execution_times[2]))
    ]

    for phase_label, color_key, time_func in cloud_phases:
        y_level = y_positions[phase_label]
        yticks.append(y_level)
        ytick_labels.append(phase_label)
        
        if len(sequences) > 3:
            for task_id in sequences[3]:
                task = task_map[task_id]
                if not task.is_core_task:
                    start, duration = time_func(task)
                    
                    ax.barh(y_level, duration, left=start, height=0.4,align='center', color=colors[color_key], edgecolor='black')
                    add_centered_text(ax, start, duration, y_level, task.id)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Execution Unit")
    ax.set_title(title)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(0, max_completion_time + 1)
    ax.set_xticks(range(0, int(max_completion_time) + 2))

    legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='black', label=label)for label, color in colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    task10 = Task(10)
    task9 = Task(9, succ_task=[task10])
    task8 = Task(8, succ_task=[task10])
    task7 = Task(7, succ_task=[task10])
    task6 = Task(6, succ_task=[task8])
    task5 = Task(5, succ_task=[task9])
    task4 = Task(4, succ_task=[task8, task9])
    task3 = Task(3, succ_task=[task7])
    task2 = Task(2, succ_task=[task8, task9])
    task1 = Task(1, succ_task=[task2, task3, task4, task5, task6])
    task10.pred_tasks = [task7, task8, task9]
    task9.pred_tasks = [task2, task4, task5]
    task8.pred_tasks = [task2, task4, task6]
    task7.pred_tasks = [task3]
    task6.pred_tasks = [task1]
    task5.pred_tasks = [task1]
    task4.pred_tasks = [task1]
    task3.pred_tasks = [task1]
    task2.pred_tasks = [task1]
    task1.pred_tasks = []
    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10]
    
    task20 = Task(20)
    task19 = Task(19, succ_task=[])
    task18 = Task(18, succ_task=[task20])
    task17 = Task(17, succ_task=[])
    task16 = Task(16, succ_task=[task19])
    task15 = Task(15, succ_task=[task19])
    task14 = Task(14, succ_task=[task18])
    task13 = Task(13, succ_task=[task17, task18])
    task12 = Task(12, succ_task=[task17])
    task11 = Task(11, succ_task=[task15, task16])
    task10 = Task(10, succ_task=[task11,task15])
    task9 = Task(9, succ_task=[task13,task14])
    task8 = Task(8, succ_task=[task12,task13])
    task7 = Task(7, succ_task=[task12])
    task6 = Task(6, succ_task=[task10,task11])
    task5 = Task(5, succ_task=[task9,task10])
    task4 = Task(4, succ_task=[task8,task9])
    task3 = Task(3, pred_tasks=[], succ_task=[task7, task8])
    task2 = Task(2, pred_tasks=[], succ_task=[task7,task8])
    task1 = Task(1, pred_tasks=[], succ_task=[task4, task5, task6])
    task20.pred_tasks = [task18]
    task19.pred_tasks = [task15,task16]
    task18.pred_tasks = [task13, task14]
    task17.pred_tasks = [task12, task13]
    task16.pred_tasks = [task11]
    task15.pred_tasks = [task10, task11]
    task14.pred_tasks = [task9]
    task13.pred_tasks = [task8, task9]
    task12.pred_tasks = [task7, task8]
    task11.pred_tasks = [task6, task10]
    task10.pred_tasks = [task5, task6]
    task9.pred_tasks = [task4,task5]
    task8.pred_tasks = [task2,task3, task4]
    task7.pred_tasks = [task2,task3]
    task6.pred_tasks = [task1]
    task5.pred_tasks = [task1]
    task4.pred_tasks = [task1]

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10,
        task11, task12, task13, task14, task15, task16, task17, task18, task19, task20]

    task10 = Task(10)
    task9 = Task(9, succ_task=[task10])
    task8 = Task(8, succ_task=[task10])
    task7 = Task(7, succ_task=[task8, task9])
    task6 = Task(6, succ_task=[task7, task8])
    task5 = Task(5, succ_task=[task7])
    task4 = Task(4, succ_task=[task6])
    task3 = Task(3, succ_task=[task5, task6])
    task2 = Task(2, succ_task=[task4, task5])
    task1 = Task(1, succ_task=[task2, task3])
    task10.pred_tasks = [task8, task9]
    task9.pred_tasks = [task7]
    task8.pred_tasks = [task6, task7]
    task7.pred_tasks = [task5, task6]
    task6.pred_tasks = [task3, task4]
    task5.pred_tasks = [task2, task3]
    task4.pred_tasks = [task2]
    task3.pred_tasks = [task1]
    task2.pred_tasks = [task1]
    task1.pred_tasks = []

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10]

    task20 = Task(20)
    task19 = Task(19, succ_task=[task20])
    task18 = Task(18, succ_task=[task20])
    task17 = Task(17, succ_task=[task20])
    task16 = Task(16, succ_task=[task19])
    task15 = Task(15, succ_task=[task19])
    task14 = Task(14, succ_task=[task18, task19])
    task13 = Task(13, succ_task=[task17, task18])
    task12 = Task(12, succ_task=[task17])
    task11 = Task(11, succ_task=[task15, task16])
    task10 = Task(10, succ_task=[task11,task15])
    task9 = Task(9, succ_task=[task13,task14])
    task8 = Task(8, succ_task=[task12,task13])
    task7 = Task(7, succ_task=[task12])
    task6 = Task(6, succ_task=[task10,task11])
    task5 = Task(5, succ_task=[task9,task10])
    task4 = Task(4, succ_task=[task8,task9])
    task3 = Task(3, succ_task=[task7, task8])
    task2 = Task(2, succ_task=[task7,task8])
    task1 = Task(1, succ_task=[task2, task3, task4, task5, task6])
    task1.pred_tasks = []
    task2.pred_tasks = [task1]
    task3.pred_tasks = [task1]
    task4.pred_tasks = [task1] 
    task5.pred_tasks = [task1]
    task6.pred_tasks = [task1]
    task7.pred_tasks = [task2,task3]
    task8.pred_tasks = [task2,task3, task4]
    task9.pred_tasks = [task4,task5]
    task10.pred_tasks = [task5, task6]
    task11.pred_tasks = [task6, task10]
    task12.pred_tasks = [task7, task8]
    task13.pred_tasks = [task8, task9]
    task14.pred_tasks = [task9, task10]
    task15.pred_tasks = [task10, task11]
    task16.pred_tasks = [task11]
    task17.pred_tasks = [task12, task13]
    task18.pred_tasks = [task13, task14]
    task19.pred_tasks = [task14, task15,task16]
    task20.pred_tasks = [task17, task18,task19]

    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10,
        task11, task12, task13, task14, task15, task16, task17, task18, task19, task20]

    task20 = Task(20)
    task19 = Task(19, succ_task=[task20])
    task18 = Task(18, succ_task=[task20])
    task17 = Task(17, succ_task=[task20])
    task16 = Task(16, succ_task=[task19])
    task15 = Task(15, succ_task=[task19])
    task14 = Task(14, succ_task=[task18, task19])
    task13 = Task(13, succ_task=[task17, task18])
    task12 = Task(12, succ_task=[task17])
    task11 = Task(11, succ_task=[task15, task16])
    task10 = Task(10, succ_task=[task11,task15])
    task9 = Task(9, succ_task=[task13,task14])
    task8 = Task(8, succ_task=[task12,task13])
    task7 = Task(7, succ_task=[task12])
    task6 = Task(6, succ_task=[task10,task11])
    task5 = Task(5, succ_task=[task9,task10])
    task4 = Task(4, succ_task=[task8,task9])
    task3 = Task(3, succ_task=[task7, task8])
    task2 = Task(2, succ_task=[task7])
    task1 = Task(1, succ_task=[task7])
    task1.pred_tasks = []
    task2.pred_tasks = []
    task3.pred_tasks = []
    task4.pred_tasks = []
    task5.pred_tasks = []
    task6.pred_tasks = []
    task7.pred_tasks = [task1,task2,task3]
    task8.pred_tasks = [task3, task4]
    task9.pred_tasks = [task4,task5]
    task10.pred_tasks = [task5, task6]
    task11.pred_tasks = [task6, task10]
    task12.pred_tasks = [task7, task8]
    task13.pred_tasks = [task8, task9]
    task14.pred_tasks = [task9, task10]
    task15.pred_tasks = [task10, task11]
    task16.pred_tasks = [task11]
    task17.pred_tasks = [task12, task13]
    task18.pred_tasks = [task13, task14]
    task19.pred_tasks = [task14, task15, task16]
    task20.pred_tasks = [task17, task18, task19]
    tasks = [task1, task2, task3, task4, task5, task6, task7, task8, task9, task10, 
            task11, task12, task13, task14, task15, task16, task17, task18, task19, task20]

    print_task_graph(tasks)
    create_and_visualize_task_graph(tasks, 'graph', ['png'], dpi=600)

    primary_assignment(tasks)
    task_prioritizing(tasks)
    sequence = execution_unit_selection(tasks)
    print_final_sequences(sequence)
    T_final = total_time(tasks)
    E_total = total_energy(tasks, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    print("INITIAL SCHEDULING APPLICATION COMPLETION TIME: ", T_final)
    print("INITIAL APPLICATION ENERGY CONSUMPTION:", E_total)
    print("INITIAL TASK SCHEDULE: ")
    print_task_schedule(tasks)
    print_validation_report(tasks)
    plot_gantt(tasks, sequence, title="Initial Schedule")

    tasks2, sequence = optimize_task_scheduling(tasks, sequence, T_final, core_powers=[1, 2, 4], cloud_sending_power=0.5)

    print_final_sequences(sequence)

    T_final = total_time(tasks)
    E_final = total_energy(tasks, core_powers=[1, 2, 4], cloud_sending_power=0.5)
    print("FINAL SCHEDULING APPLICATION COMPLETION TIME: ", T_final)
    print("FINAL APPLICATION ENERGY CONSUMPTION:", E_final)
    print("FINAL TASK SCHEDULE: ")
    print_task_schedule(tasks2)
    print_validation_report(tasks2)
    plot_gantt(tasks, sequence, title="Final Schedule")

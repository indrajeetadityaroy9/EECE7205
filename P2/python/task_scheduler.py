from typing import List
from enum import Enum
import math

class SchedulingState(Enum):
    UNSCHEDULED = 0      # Initial state
    SCHEDULED = 1        # After initial scheduling (Step 1 in paper)
    KERNEL_SCHEDULED = 2  # After kernel algorithm (Step 2 in paper)

class Scheduler:
    def __init__(self, nodes, num_cores: int = 3):
        self.nodes = nodes
        self.k = num_cores  # Number of local cores
        self.sequences = [[] for _ in range(self.k + 1)]  # Sequences for each core and cloud
        self.core_earliest_ready = [0] * self.k  # Next available time for each core
        self.wireless_send_ready = 0  # Next available time to send to cloud
        self.wireless_receive_ready = 0  # Next available time to receive from cloud

    def execute(self):
        self.sort_tasks_by_priority()
        self.separate_entry_and_non_entry_tasks()
        self.schedule_entry_tasks()
        self.schedule_cloud_entry_tasks()
        self.process_non_entry_tasks()
        return self.sequences

    def sort_tasks_by_priority(self):
        # Sort tasks by priority score in descending order
        self.node_priority_list = sorted(
            ((node.priority_score, node.id) for node in self.nodes),
            reverse=True
        )
        self.priority_order = [item[1] for item in self.node_priority_list]

    def separate_entry_and_non_entry_tasks(self):
        # Separate tasks into entry (no parents) and non-entry tasks
        self.entry_tasks = []
        self.non_entry_tasks = []
        for node_id in self.priority_order:
            node = self.nodes[node_id - 1]  # Adjust for 0-based index
            if not node.parents:
                self.entry_tasks.append(node)
            else:
                self.non_entry_tasks.append(node)

    def schedule_entry_tasks(self):
        # Schedule entry tasks
        self.cloud_entry_tasks = []
        for task in self.entry_tasks:
            if task.is_core_task:
                self.schedule_task_on_core(task)
            else:
                self.cloud_entry_tasks.append(task)

    def schedule_cloud_entry_tasks(self):
        # Schedule cloud entry tasks with pipeline staggering
        for task in self.cloud_entry_tasks:
            self.schedule_task_on_cloud(task)

    def process_non_entry_tasks(self):
        # Process non-entry tasks
        for task in self.non_entry_tasks:
            self.calculate_ready_times(task)
            self.calculate_cloud_pipeline_timing(task)
            if not task.is_core_task:
                # Pre-determined cloud tasks
                self.schedule_task_on_cloud(task)
            else:
                # Decide between local core and cloud
                local_finish_time = self.estimate_local_finish_time(task)
                cloud_finish_time = task.wireless_recieving_finish_time
                if local_finish_time <= cloud_finish_time:
                    self.schedule_task_on_core(task, local_finish_time)
                else:
                    self.schedule_task_on_cloud(task)

    def schedule_task_on_core(self, task, finish_time: float = None):
        # Find the best core and schedule the task
        best_finish_time = finish_time if finish_time else math.inf
        best_core = -1
        best_start_time = 0

        for core in range(self.k):
            start_time = max(task.local_core_ready_time, self.core_earliest_ready[core])
            core_finish_time = start_time + task.core_execution_times[core]

            if core_finish_time < best_finish_time:
                best_finish_time = core_finish_time
                best_core = core
                best_start_time = start_time

        # Update task scheduling information
        task.local_core_finish_time = best_finish_time
        task.execution_finish_time = best_finish_time
        task.execution_unit_start_times = [-1] * 4
        task.execution_unit_start_times[best_core] = best_start_time
        self.core_earliest_ready[best_core] = best_finish_time
        task.assignment = best_core
        task.is_scheduled = SchedulingState.SCHEDULED
        self.sequences[best_core].append(task.id)

    def schedule_task_on_cloud(self, task):
        # Phase 1: Sending to cloud
        task.wireless_sending_ready_time = max(
            self.wireless_send_ready,
            max(
                (parent.local_core_finish_time for parent in task.parents),
                default=0
            )
        )
        task.wireless_sending_finish_time = (
            task.wireless_sending_ready_time + task.cloud_execution_times[0]
        )
        self.wireless_send_ready = task.wireless_sending_finish_time

        # Phase 2: Cloud computation
        task.remote_cloud_ready_time = task.wireless_sending_finish_time
        task.remote_cloud_finish_time = (
            task.remote_cloud_ready_time + task.cloud_execution_times[1]
        )

        # Phase 3: Receiving results
        task.wireless_recieving_ready_time = task.remote_cloud_finish_time
        task.wireless_recieving_finish_time = (
            max(self.wireless_receive_ready, task.wireless_recieving_ready_time)
            + task.cloud_execution_times[2]
        )
        self.wireless_receive_ready = task.wireless_recieving_finish_time

        # Update task parameters
        task.execution_finish_time = task.wireless_recieving_finish_time
        task.local_core_finish_time = 0
        task.execution_unit_start_times = [-1] * 4
        task.execution_unit_start_times[self.k] = task.wireless_sending_ready_time
        task.assignment = self.k
        task.is_scheduled = SchedulingState.SCHEDULED
        self.sequences[self.k].append(task.id)

    def calculate_ready_times(self, task):
        # Calculate ready times based on parents
        task.local_core_ready_time = max(
            (max(parent.local_core_finish_time, parent.wireless_recieving_finish_time)
             for parent in task.parents),
            default=0
        )

    def calculate_cloud_pipeline_timing(self, task):
        # Calculate cloud pipeline timing
        task.wireless_sending_ready_time = max(
            self.wireless_send_ready,
            max(
                (parent.local_core_finish_time for parent in task.parents),
                default=0
            )
        )
        task.wireless_sending_finish_time = (
            task.wireless_sending_ready_time + task.cloud_execution_times[0]
        )

        task.remote_cloud_ready_time = task.wireless_sending_finish_time
        task.remote_cloud_finish_time = (
            task.remote_cloud_ready_time + task.cloud_execution_times[1]
        )

        task.wireless_recieving_ready_time = task.remote_cloud_finish_time
        task.wireless_recieving_finish_time = (
            max(self.wireless_receive_ready, task.wireless_recieving_ready_time)
            + task.cloud_execution_times[2]
        )

    def estimate_local_finish_time(self, task) -> float:
        # Estimate the earliest finish time if executed on any core
        best_finish_time = math.inf
        for core in range(self.k):
            start_time = max(task.local_core_ready_time, self.core_earliest_ready[core])
            finish_time = start_time + task.core_execution_times[core]
            if finish_time < best_finish_time:
                best_finish_time = finish_time
        return best_finish_time

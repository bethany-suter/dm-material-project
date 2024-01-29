"""
mpi/base.py

This module offers a framework for distributed task execution using MPI
(Message Passing Interface) in Python, aimed at facilitating parallel
computing across multiple processes in high-performance computing settings.

It features the DistributedTask class, establishing MPI communication and
orchestrating root (coordinator) and worker roles in distributed computation.
The class includes methods for running tasks, task distribution, worker
process communication, and termination of processes upon completion or
exception.

Key Components:
- DistributedTask: Manages the root process and worker processes, handling
  task distribution and result collection.

Usage:
- Subclass DistributedTask and implement _run_root, _func, and _run_worker
  methods for specific task behavior.
- Instantiate the subclass and call run to initiate distributed computation.
- The root process distributes tasks to workers and gathers results. Workers
  execute tasks and return results to the root.

Dependencies:
- mpi4py: Python wrapper for MPI, used for process communication.

Note:
The script requires an MPI environment and is designed for high-performance
computing environments where parallel processing can significantly optimize
computation times.
"""


import logging
from mpi4py import MPI

# Constants for MPI communication
ROOT_RANK = 0
EXCEPTION_SIGNAL = "exception occurred"
TERMINATE_SIGNAL = "terminate worker"

# Configuring logging for the module
logging.basicConfig()
LOGGER = logging.getLogger('MPI')
LOGGER.setLevel(logging.DEBUG)


class DistributedTask(object):
    """
    A class for managing distributed tasks using MPI.

    This class sets up an MPI environment, distributing tasks among multiple
    processes and handling communication between them. It defines the base
    structure for root and worker roles in the distributed computation.

    Methods to be implemented in subclasses:
    - _run_root: Define the task distribution and result collection at the root.
    - _func: Define the actual computation/task to be performed by workers.
    - _run_worker: Define how workers receive and process tasks.
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the DistributedTask with MPI communication setup.
        """
        self.comm = MPI.COMM_WORLD  # MPI communicator
        self.rank = self.comm.Get_rank()  # Rank of the current process
        self.nprocs = self.comm.Get_size()  # Total number of processes

        # Determine if the current process is the root
        self.root = self.rank == ROOT_RANK

    def run(self, *args, **kwargs):
        """
        Execute the distributed task.

        This method distinguishes between root and worker roles, and initiates
        their respective methods.

        Returns:
            Result of the task if called from the root process.
        """
        if self.root:
            # Execute root-specific method and signal completion
            result = self._run_root(*args, **kwargs)
            self._done()
            return result
        else:
            try:
                # Execute worker-specific method
                self._run_worker(*args, **kwargs)
            except BaseException:
                # Send exception signal to root and re-raise exception
                self.comm.send((self.rank, EXCEPTION_SIGNAL), dest=ROOT_RANK, tag=0)
                raise

    def _run_root(self, *args, **kwargs):
        """
        Root process logic to distribute tasks and collect results.

        This method should be implemented in a subclass.
        """
        raise NotImplementedError

    def _scatter(self, tasks, results):
        """
        Distribute tasks among workers and collect the results.

        Args:
            tasks: A list of tasks to be distributed.
            results: A list to store the results.

        Returns:
            A list of results corresponding to the tasks.
        """
        LOGGER.error(
            "ROOT: Will distribute %d tasks among %d workers" % (
                len(tasks), self.nprocs - 1
            )
        )
        # Initial setup for task distribution
        undispatched_task_ids = list(range(len(tasks)))
        unfinished_task_ids = list(range(len(tasks)))
        # `results` is really a buffer. Pop everything and repopulate
        while len(results) > 0:
            results.pop(0)
        while len(results) < len(tasks):
            results.append(None)
        # Distribute initial tasks and loop for receiving and redistributing
        for worker_rank in range(self.nprocs):
            if worker_rank == ROOT_RANK:
                continue
            try:
                next_task_id = undispatched_task_ids.pop(0)
                next_task = tasks[next_task_id]
            except IndexError:
                break
            self.comm.send(
                (next_task_id, next_task),
                dest=worker_rank,
                tag=0
            )
        # Receive output and redistribute
        while len(unfinished_task_ids):
            LOGGER.error("ROOT: Waiting...")
            worker_rank, message = self.comm.recv(tag=0)
            if message == EXCEPTION_SIGNAL:
                self._done()
                raise RuntimeError
            result_id, result = message
            LOGGER.error(
                "ROOT: Worker %d finished task %d" % (
                    worker_rank, result_id
                )
            )
            unfinished_task_ids.remove(result_id)
            results[result_id] = result
            # Dispatch the next task
            if len(undispatched_task_ids):
                LOGGER.error("ROOT: There are still %d tasks to distribute" %
                             len(undispatched_task_ids))
                next_task_id = undispatched_task_ids.pop(0)
                next_task = tasks[next_task_id]
                LOGGER.error(
                    "ROOT: Will give this worker task %d" % next_task_id
                )
                self.comm.send(
                    (next_task_id, next_task),
                    dest=worker_rank,
                    tag=0
                )
        LOGGER.error("All tasks completed")
        return results

    def _run_worker(self, *args, **kwargs):
        """
        Worker process logic to receive, process tasks, and send results.

        Continuously listens for tasks, processes them using _func, and sends
        results back to the root process.
        """
        while True:
            # Wait for a task from the root process
            LOGGER.error("WORKER %d: waiting for task" % self.rank)
            message = self.comm.recv(source=ROOT_RANK)

            # Check for termination signal
            if message == TERMINATE_SIGNAL:
                LOGGER.error("WORKER %d: received termination signal" % self.rank)
                break
            else:
                task_id, task = message

            # Process the received task
            LOGGER.error("WORKER %d: received task %d" % (self.rank, task_id))
            result = self._func(task, *args, **kwargs)

            # Send the result back to the root
            LOGGER.error(
                "WORKER %d: finished task %d, sending results to root" % (
                    self.rank, task_id
                )
            )
            self.comm.send((self.rank, (task_id, result)), dest=ROOT_RANK, tag=0)

    def _func(self, *args, **kwargs):
        """
        The main computation function for worker processes.

        This method should be implemented in a subclass to define the specific
        task that each worker will perform.

        Raises:
            NotImplementedError: If not overridden in a subclass.
        """
        raise NotImplementedError

    def _done(self):
        """
        Send termination signal to worker processes.

        This method is called by the root process to signal all worker processes
        to terminate.
        """
        LOGGER.error("ROOT: Broadcasting termination signal")
        for worker_rank in range(1, self.nprocs):
            self.comm.send(TERMINATE_SIGNAL, dest=worker_rank, tag=0)

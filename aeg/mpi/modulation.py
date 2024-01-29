"""
mpi/modulation.py

This module extends the distributed task framework provided in 'mpi/base.py' to 
specifically handle the computation of modulation effects in dark matter 
detection experiments. It uses MPI for parallel processing across multiple nodes 
or processors, efficiently handling large-scale computations that are common in 
particle physics simulations.

The module's primary class, ModulationEnsemble, inherits from DistributedTask 
and implements methods to distribute and collect results of modulation 
calculations. It is designed to explore a range of parameters, such as mass ratios 
and velocity ratios, to study their effects on modulation signals.

The script can be run directly with command-line arguments specifying various 
parameters like output file, number of points, and ranges for mass and velocity 
ratios. It uses argparse to parse these command-line arguments.

Key Features:
- ModulationEnsemble: A class that manages distributed tasks for computing 
  modulation effects in dark matter experiments.
- Command-line interface: Allows users to specify parameters for the modulation 
  calculations and output.

Usage:
- Run as a script with required command-line arguments to perform modulation 
  studies.
- The script outputs results in either JSON or NPZ format, depending on the 
  provided filename extension.

Dependencies:
- MPI and mpi4py for distributed computing.
- numpy for numerical operations.
- rate module for core rate calculation functionalities.

Note:
This script is intended for use in computational physics research, particularly 
in the field of dark matter detection. It requires an MPI-enabled environment for 
execution.
"""


import json
import argparse

import numpy as np

from .base import DistributedTask, LOGGER
from .. import rate


class ModulationEnsemble(DistributedTask):
    """
    A class derived from DistributedTask to handle the distributed computation 
    of modulation effects in dark matter experiments.

    This class manages the setup and execution of a grid of simulation tasks, 
    each corresponding to different parameters in dark matter detection 
    simulations. It allows for exploring a range of values in parameters such as 
    mass ratios, velocity ratios, and mediator masses. The class is designed to 
    efficiently distribute these tasks across multiple MPI processes and collect 
    the results.

    The modulation ensemble operates by creating a meshgrid of parameter values 
    and distributing corresponding tasks to worker nodes. It supports output in 
    JSON or NPZ format, storing results from the distributed computation.

    Key Methods:
    - _run_root: Orchestrates the distribution of tasks and collection of results 
      from worker processes.
    - _func: Defines the task to be executed by each worker, including the 
      calculation of modulation rates or differences based on the provided 
      parameters.

    Usage:
    - The class should be instantiated and the run method called with appropriate 
      parameters. The run method requires a dictionary of parameters including 
      ranges for mass ratio, velocity ratio, and other simulation settings.
    """

    def _run_root(self, *args, **kwargs):
        """
        Executes the root process logic for the ModulationEnsemble class.

        This method is responsible for initializing and managing the distribution 
        of tasks in the modulation ensemble. It sets up the range of parameter 
        values to explore, generates tasks based on these parameters, and 
        distributes them across the available worker processes. It then collects 
        the results from these tasks, handles exceptions, and saves the results 
        to a specified output file.

        The method supports outputting results in JSON or NPZ format. It ensures 
        that the results are saved in an organized and accessible manner, 
        facilitating further analysis.

        Args:
            *args: Variable length argument list, accommodating additional parameters.
            **kwargs: Keyword arguments containing simulation parameters and 
                      settings such as mass ratios, velocity ratios, mediator masses, 
                      and output file specifications.

        Raises:
            ValueError: If the output file format is unrecognized.
            RuntimeError: If an exception is raised in any of the worker processes.

        Returns:
            None: This method does not return a value but saves the results to a file.
        """
        # Validate required keys and set attributes from kwargs
        required_keys = (
            'outfile', 'n_x', 'n_y', 'n_points_init', 'n_points_max', 'tol',
            'm_med', 'particle_hole'
        )
        for key in required_keys:
            setattr(self, key, kwargs[key])

        # Check and ensure output file format is either JSON or NPZ
        if not self.outfile.endswith('.json') \
                and not self.outfile.endswith('.npz'):
            raise ValueError("Unrecognized output format")

        # Set additional parameters, with defaults for optional ones
        self.mass_ratio = kwargs.get('mass_ratio')
        w_min_eV = kwargs.get('w_min', 1e-2)
        w_max_eV = kwargs.get('w_max', 10.)
        self.v_ratio = kwargs.get('v_ratio')

        # Set up the x-axis values (velocity ratio or time)
        if self.v_ratio is None:
            v_ratio_max = kwargs.get('v_ratio_min', 0.08)
            v_ratio_min = kwargs.get('v_ratio_max', 2.)
            x_vals = np.geomspace(v_ratio_min, v_ratio_max, self.n_x)
            x_key = 'v0_over_vF'
        else:
            self.v0_over_vF = self.v_ratio
            x_vals = np.linspace(0, 1, self.n_x)
            x_key = 't'

        # Set up the y-axis values (mass ratio or DM mass)
        if self.mass_ratio is None:
            y_max = kwargs.get('mass_ratio_max', 100.)
            y_min = kwargs.get('mass_ratio_min', 1.)
            y_key = 'mass_ratio'
        else:
            y_max = kwargs.get('mass_ratio_max', 1e-2)
            y_min = kwargs.get('mass_ratio_min', 1e2)
            y_key = 'mX_eV'
        y_vals = np.geomspace(y_min, y_max, self.n_y)

        # Create a meshgrid for the parameter space and tasks array
        X, Y = np.meshgrid(x_vals, y_vals)
        Z_id = np.arange(X.size).reshape(X.shape)
        tasks = []

        # Populate the tasks list with parameters for each point in the grid
        for j, row in enumerate(Y):
            for i, (col, x) in enumerate(zip(row, x_vals)):
                task = dict(
                    task_id=int(Z_id[j, i]),
                    w_min_eV=w_min_eV,
                    w_max_eV=w_max_eV,
                    m_med=self.m_med,
                    n_points_init=self.n_points_init,
                    n_points_max=self.n_points_max,
                    tol=self.tol,
                    ph_only=self.particle_hole
                )
                task[x_key] = X[j, i]
                task[y_key] = Y[j, i]
                # Set mass ratio or velocity ratio if not None
                if self.mass_ratio is not None:
                    task['mass_ratio'] = self.mass_ratio
                if self.v_ratio is not None:
                    task['v0_over_vF'] = self.v0_over_vF
                tasks.append(task)

        # Initialize a results buffer and scatter tasks to workers
        results = []
        self._scatter(tasks, results)

        # Save results to disk in the specified format
        if self.outfile.endswith('.json'):
            # Save as JSON
            with open(self.outfile, 'w') as fh:
                json.dump(results, fh, indent=4)
        elif self.outfile.endswith('.npz'):
            # Save as NPZ, organizing data for each task
            Z = np.zeros((X.shape[0], X.shape[1], 2))
            n_points = np.zeros_like(X, dtype=int)
            success = np.zeros_like(X, dtype=bool)
            for task in results:
                index = np.where(Z_id == task['task_id'])
                Z[index] = task['result']
                n_points[index] = task['n_points']
                success[index] = task['success']
            np.savez(self.outfile, X=X, Y=Y, Z=Z,
                     n_points=n_points, success=success)

    def _func(self, task, *args, **kwargs):
        """
        Perform the core computation for a given task in the worker process.

        This method is called by worker nodes to execute the assigned task. It 
        involves calculating modulation rates or differences based on the provided 
        task parameters, including minimum and maximum energy window (w_min, w_max), 
        velocity ratios, mass ratios, and mediator masses.

        Args:
            task (dict): A dictionary containing parameters for the task.
            *args: Variable length argument list.
            **kwargs: Keyword arguments passed to the rate calculation functions.

        Returns:
            dict: The task dictionary updated with results and other relevant information.
        """
        # Initialize Scaler with velocity ratio from the task
        s = rate.Scaler(v0_over_vF=task['v0_over_vF'])

        # Convert energy window from eV to internal units
        w_min = task['w_min_eV']*s.eV
        w_max = task['w_max_eV']*s.eV

        # Set up tolerance and mass ratio
        tol = task['tol']
        mass_ratio = task['mass_ratio']
        masses = rate.mass_vector(mass_ratio)

        # Extract dark matter mass, defaulting to 1 if not provided
        try:
            mX = task['mX_eV']  # *s.eV  # Oops, ignore the eV
        except KeyError:
            mX = 1.

        # Initial setup for adaptive point sampling
        step = 5.
        n_points = task['n_points_init']
        n_points_max = task['n_points_max']
        above_tol = True

        # Loop for adaptive point sampling in Monte Carlo integration
        while above_tol and n_points <= n_points_max:
            LOGGER.error(
                "WORKER %d: above tolerance. Attempting with %d points." % (
                    self.rank, n_points
                )
            )

            # Set up kwargs for rate calculation and perform the calculation
            kwargs = dict(w_min=w_min, w_max=w_max, ph_only=task['ph_only'],
                          m=masses, mX=mX, m_med=task['m_med'],
                          n_points=n_points, n_iter=10, spherical=True)
            if 't' in task:
                dr, dr_err = rate.rate(s, **kwargs, t=task['t'])
            else:
                dr, dr_err = rate.delta_r(s, **kwargs)

            # Check if results are within the specified tolerance
            frac_err = np.abs(dr_err / dr)
            above_tol = (not np.isfinite(dr)) or (frac_err >= tol)
            n_points = int(n_points * step)

        # Update the task dictionary with results and return
        task['result'] = (dr, dr_err)
        task['n_points'] = n_points / step
        task['success'] = not above_tol
        return task


if __name__ == '__main__':
    # This section executes when the script is run directly from the command line.
    # It sets up the argument parser and extracts the necessary parameters to run 
    # the ModulationEnsemble distributed tasks.

    # Initialize an argument parser for command line inputs
    parser = argparse.ArgumentParser()

    # Add arguments to the parser for various parameters needed in the simulation
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--n-x', type=int, required=True)
    parser.add_argument('--n-y', type=int, required=True)
    parser.add_argument('--n-points-init', type=int, required=True)
    parser.add_argument('--n-points-max', type=int, required=True)
    parser.add_argument('--v-ratio', type=float, required=False)
    parser.add_argument('--v-ratio-min', type=float, required=False)
    parser.add_argument('--v-ratio-max', type=float, required=False)
    parser.add_argument('--mass-ratio-min', type=float, required=False)
    parser.add_argument('--mass-ratio-max', type=float, required=False)
    parser.add_argument('--tol', type=float, required=True)
    parser.add_argument('--m-med', type=float, required=True)
    parser.add_argument('--mass-ratio', type=float, required=False)
    parser.add_argument('--particle-hole', action='store_true')

    # Parse the command line arguments
    args = parser.parse_args()

    # Convert parsed arguments into a dictionary
    kwargs = vars(args)

    # Remove any arguments that were not provided (i.e., are None)
    for key in list(kwargs.keys()):  # We modify inflight
        if kwargs[key] is None:
            del kwargs[key]

    # Instantiate the ModulationEnsemble class
    ensemble = ModulationEnsemble()

    # Run the ensemble with the provided arguments
    ensemble.run(**kwargs)

MPI4PY Estimation
======================================

This guide explains how to run `estimation_example.py` script to fit statistical models in parallel using `mpi4py` and the DMLearn package. The file is found in `dml/mpi4py/examples`.

Overview
--------

The script demonstrates:

- Simulating data on the master MPI node (rank 0)
- Defining composite and mixture distributions
- Fitting a mixture model using DMLearn estimators
- Saving the fitted model to disk

Usage
-----

To launch the estimation script with 4 MPI processes, run:

.. code-block:: bash

   mpiexec -n 4 python3 estimation_example.py

You can adjust the number of processes (`-n 4`) as needed.

What the Script Does
--------------------

1. **Data Simulation (on master node):**
   - Creates a two state composite mixture of Gaussian and Categorical distributions.
   - Samples 1000 data points from the MixtureSampler object.
   - The data is simulated only on the master node (rank 0).

2. **Estimator Setup:**
   - Defines a composite estimator with Gaussian and categorical estimators.
   - Wraps the composite estimator in a two state mixture estimator.

3. **Parallel Model Fitting:**
   - Uses `optimize_mpi` to fit the model in parallel across MPI processes.
   - The fitted model resides on the master node (rank 0).

4. **Model Saving:**
   - The fitted model is pickled and saved as `mpi4py_model_fit.pkl` on the master node.

Script Output
-------------

- Each process prints whether it has the fitted model.
- The master node prints confirmation of the saved file.

Example Output:

.. code-block:: text

   Rank 0: Model is None == False
   Rank 1: Model is None == True
   Rank 2: Model is None == True
   Rank 3: Model is None == True
   Wrote file ./mpi4py_model_fit.pkl

Notes
-----

- Only the master node (rank 0) will have the fitted model and write the output file.
- You can modify the script to read your own data instead of simulating it.

References
----------

- `mpi4py` documentation: https://mpi4py.readthedocs.io/
- DMLearn package documentation (replace with your package's URL)




MPI4PY Estimation Example
======================================

This guide explains how to run `estimation_example.py` to fit statistical models in parallel using `mpi4py` and the DMLearn package. Each step below includes the relevant Python code snippet. The file is found in `dml/mpi4py/examples`

Running the Script
------------------

To launch the script with 4 MPI processes:

.. code-block:: bash

   mpiexec -n 4 python3 estimation_example.py

Step 1: Import Libraries and Set Up MPI
---------------------------------------

.. code-block:: python

   import os
   os.environ['NUMBA_DISABLE_JIT'] =  '1'

   from mpi4py import MPI
   from numpy.random import RandomState
   import pickle
   from dml.stats import *
   from dml.mpi4py.stats import *
   from dml.mpi4py.utils.estimation import optimize_mpi
   from dml.mpi4py.utils.optsutil import pickle_on_master

   comm = MPI.COMM_WORLD
   world_rank = comm.Get_rank()
   world_size = comm.Get_size()

Step 2: Simulate Data on the Master Node
----------------------------------------
Note that the data simulation is only performed on the master node (rank 0). Other nodes will receive `None` for data. We use DMLearn's DistributionSampler object to sample from the two state composite mixture distribution.

.. code-block:: python

   if world_rank == 0: 
       d00 = GaussianDistribution(mu=0.0, sigma2=1.0)
       d01 = CategoricalDistribution({'a': 0.3, 'b': 0.7})
       d0 = CompositeDistribution([d00, d01])

       d10 = GaussianDistribution(mu=3.0, sigma2=1.0)
       d11 = CategoricalDistribution({'a': 0.7, 'b': 0.3})
       d1 = CompositeDistribution([d10, d11])

       dist = MixtureDistribution([d0, d1], w=[0.25, 0.75])

       data = dist.sampler(seed=1).sample(1000)
   else:
       data = None

Step 3: Define the Estimator
----------------------------
The estimtor can be defined on all the nodes. This object is lightweight and is later broadcasted to all nodes. Here we define a composite estimator that combines Gaussian and Categorical estimators, and then wrap it in a mixture estimator.

.. code-block:: python

   e0 = CompositeEstimator([GaussianEstimator(), CategoricalEstimator()])
   est = MixtureEstimator([e0]*2)

Step 4: Fit the Model in Parallel
---------------------------------
The data is passed to the `optimize_mpi` function along with the estimator and a random number generator. This function will handle the dissemination of data and parrallel fitting of the model across all MPI processes.

.. code-block:: python

   rng = RandomState(1)
   fit = optimize_mpi(data=data, estimator=est, rng=rng)

Step 5: Check Model Presence on Each Node
-----------------------------------------
The snippets below are included to demonstrate to the user that only the master node will have the fitted model. Each node prints whether it has the fitted model or not.

.. code-block:: python

   print(f"Rank {world_rank}: Model is None == {fit is None}")

Step 6: Save the Model on the Master Node
-----------------------------------------
The fitted model is pickled and saved to a file only on the master node (rank 0) using `pickle_on_master`. This ensures that the model is not duplicated across all nodes, which would be inefficient.

.. code-block:: python

   pickle_on_master(fit, "mpi4py_model_fit.pkl")

   if world_rank == 0:
       print(f"Wrote file ./mpi4py_model_fit.pkl")

Full Script Example
-------------------

Here is the complete script for reference:

.. code-block:: python

   import os
   os.environ['NUMBA_DISABLE_JIT'] =  '1'

   from mpi4py import MPI
   from numpy.random import RandomState
   import pickle
   from dml.stats import *
   from dml.mpi4py.stats import *
   from dml.mpi4py.utils.estimation import optimize_mpi
   from dml.mpi4py.utils.optsutil import pickle_on_master

   comm = MPI.COMM_WORLD
   world_rank = comm.Get_rank()
   world_size = comm.Get_size()

   if __name__ == "__main__":
       if world_rank == 0: 
           d00 = GaussianDistribution(mu=0.0, sigma2=1.0)
           d01 = CategoricalDistribution({'a': 0.3, 'b': 0.7})
           d0 = CompositeDistribution([d00, d01])

           d10 = GaussianDistribution(mu=3.0, sigma2=1.0)
           d11 = CategoricalDistribution({'a': 0.7, 'b': 0.3})
           d1 = CompositeDistribution([d10, d11])

           dist = MixtureDistribution([d0, d1], w=[0.25, 0.75])

           data = dist.sampler(seed=1).sample(1000)
       else:
           data = None

       e0 = CompositeEstimator([GaussianEstimator(), CategoricalEstimator()])
       est = MixtureEstimator([e0]*2)

       rng = RandomState(1)
       fit = optimize_mpi(data=data, estimator=est, rng=rng)

       print(f"Rank {world_rank}: Model is None == {fit is None}")

       pickle_on_master(fit, "mpi4py_model_fit.pkl")

       if world_rank == 0:
           print(f"Wrote file ./mpi4py_model_fit.pkl")

Notes
-----

- Only the master node (rank 0) will have the fitted model and write the output file.
- You can modify the script to read your own data instead of simulating it.

References
----------

- `mpi4py` documentation: https://mpi4py.readthedocs.io/
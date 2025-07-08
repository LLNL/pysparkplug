pysparkplug - A package for distributed heterogeneous density estimation. With only a few lines of code you can specify and fit complex models on variable-length heterogenous data.

--------------------------------------------------------------------------------

## Installation

User installation with pip
```
> pip install --user /path/to/package
```


## Building with mpi4py and umap-learn

```
> cd /path/to/package
> pip install --user .[optional]
```

## Stats Examples
Examples using `stats` distributions that run locally are located in ./pysp/examples/

```
> export PYHONPATH=$PYTHONPATH:/path./to/pysparkplug
> PYTHONPATH=/path/to/package/ python ./pysp/examples/stats_examples/mixture_example.py
```

## Running with Spark
Examples that run with Apache Spark are located in./pysp/examples_spark/

First build a wheel
```
> cd /path/to/pysparkplug
> pip install setuptools wheel
> python setup.py bdist_wheel
```

Run the example with below
```
> /path/to/spark/bin/spark-submit --master local[*] --py-files /path/to/package/dist/pysparkplug-0.1.8.4-py3-none-any.whl ./pysp/examples_spark/mixture_example.py
```

## Running with MPI4PY
Examples that run with mpi4py are located in ./pysp/mpi4py/examples/

Below will run the example ./pysp/mpi4py/examples/estimation_example.py with 4 cores.
```
> export PYHONPATH=$PYTHONPATH:/path./to/pysparkplug
> mpiexec -n 4 python /path/to/package/pysp/mpi4py/examples/estimation_example.py
```

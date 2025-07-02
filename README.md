pysparkplug - A package for distributed heterogeneous density estimation. With only a few lines of code you can specify and fit complex models on variable-length heterogenous data.

--------------------------------------------------------------------------------

## Installation

User installation with pip
```
> pip install --user /path/to/package
```


## Building an egg (for distributed estimation)

To make an egg, go to the project folder (where this README is located) and execute:

> python setup.py bdist_egg



## Examples

Examples that run locally are located in ./pysp/examples/

```
> PYTHONPATH=/path/to/package/ python ./pysp/examples/mixture_example.py
```


Examples that run with Apache Spark are located in./pysp/examples_spark/

```
/path/to/spark/bin/spark-submit --master local[*] --py-files /path/to/package/dist/pysparkplug-0.1.8.4-py3.7.egg ./pysp/examples_spark/mixture_example.py 
```


## Tutorial notebooks

The package comes with many Jupiter notebooks

 * ./notebooks/distributions_and_combinators.ipynb (a good starting point)
 * ./notebooks/graphical_models.ipynb (introduction to graphical models)
 * ./notebooks/estimation_using_spark.ipynb (example notebooks using Apache Spark for distributed estimation)

Geometric
=========

Data Type: int

The geometric distribution is used to model the number of trials until the first success. 
The probability mass function is given by 

.. math::
   f(x | n, p) = p * (1-p)^{n-1}, \: 1 \leq x. 

For more info see `Geometric Distribution <https://en.wikipedia.org/wiki/Geometric_distribution_>`__.

GeometricDistribution
-----------------------

.. autoclass:: dml.stats.geometric.GeometricDistribution
   :members:
   :special-members: __init__

GeometricEstimator
-------------------

.. autoclass:: dml.stats.geometric.GeometricEstimator
   :members:
   :special-members: __init__

GeometricSampler
----------------

.. autoclass:: dml.stats.geometric.GeometricSampler
   :members:


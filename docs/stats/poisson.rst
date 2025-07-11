Poisson
=========

Data Type: int

The Poisson distribution is used to model counts.  The probability mass function is given by 

.. math::

   f(x | \lambda) = \frac{\lambda^{x}e^{-x}}{x!}, \; 0 \leq x. 

For more info see `Poisson Distribution <https://en.wikipedia.org/wiki/Poisson_distribution_>`__.


PoissonDistribution
-----------------------

.. autoclass:: pysp.stats.poisson.PoissonDistribution
   :members:
   :special-members: __init__

PoissonEstimator
-------------------

.. autoclass:: pysp.stats.poisson.PoissonEstimator
   :members:
   :special-members: __init__

PoissonSampler
----------------

.. autoclass:: pysp.stats.poisson.PoissonSampler
   :members:


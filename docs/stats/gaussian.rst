Gaussian (Normal)
=================

Data Type: float 

The Gaussian distribution is a symmetric bell-shaped distribution on the real line.  The probability density function is given by 

.. math::

   f\left(x | \mu, \sigma^2 \right) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}, \; x \in \mathbb{R}. 

For more info see `Gaussian Distribution <https://en.wikipedia.org/wiki/Normal_distribution_>`__.


GaussianDistribution
-----------------------

.. autoclass:: dml.stats.gaussian.GaussianDistribution
   :members:
   :special-members: __init__

GaussianEstimator
-------------------

.. autoclass:: dml.stats.gaussian.GaussianEstimator
   :members:
   :special-members: __init__

GaussianSampler
----------------

.. autoclass:: dml.stats.gaussian.GaussianSampler
   :members:


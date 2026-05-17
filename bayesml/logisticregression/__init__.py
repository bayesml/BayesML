# Document Author
# Yuji Iikubo <yuji-iikubo.8@fuji.waseda.jp>
r"""
The logistic regression model with the Gaussian prior distribution.

The stochastic data generative model is as follows:

* :math:`d \in \mathbb N`: a dimension
* :math:`\boldsymbol{x} \in \mathbb{R}^d`: an explanatory variable. If you consider an intercept term, it should be included as one of the elements of :math:`\boldsymbol{x}`.
* :math:`y\in\{ 0, 1\}`: an objective variable
* :math:`\boldsymbol{w}\in\mathbb{R}^{d}`: a parameter

.. math::
    p(y|\boldsymbol{x},\boldsymbol{w}) = \sigma( \boldsymbol{w}^\top \boldsymbol{x} )^y \left\{ 1 - \sigma( \boldsymbol{w}^\top \boldsymbol{x} ) \right\}^{1 - y},

where :math:`\sigma(\cdot)` is defined as follows (called a sigmoid function):

.. math::
    \sigma(a) = \frac{1}{1+\exp(-a)}.

The prior distribution is as follows:

* :math:`\boldsymbol{\mu}_0 \in \mathbb{R}^d`: a hyperparameter
* :math:`\boldsymbol{\Lambda}_0 \in \mathbb{R}^{d\times d}`: a hyperparameter (a positive definite matrix)

.. math::
    p(\boldsymbol{w}) &= \mathcal{N}(\boldsymbol{w}|\boldsymbol{\mu}_0, \boldsymbol{\Lambda}_0^{-1})\\
    &= \frac{|\boldsymbol{\Lambda}_0|^{1/2}}{(2 \pi)^{d/2}} \exp \left\{ -\frac{1}{2} (\boldsymbol{w} - \boldsymbol{\mu}_0)^\top \boldsymbol{\Lambda}_0 (\boldsymbol{w} - \boldsymbol{\mu}_0) \right\}.

The apporoximate posterior distribution in the :math:`t`-th iteration of a variational Bayesian method is as follows:

* :math:`n \in \mathbb N`: a sample size
* :math:`\boldsymbol{x}^n = (\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n) \in \mathbb{R}^{n \times d}`
* :math:`\boldsymbol{y}^n = (y_1, y_2, \dots , y_n) \in \{0,1\}^n`
* :math:`\boldsymbol{\mu}_n^{(t)}\in \mathbb{R}^d`: a hyperparameter
* :math:`\boldsymbol{\Lambda}_n^{(t)} \in \mathbb{R}^{d\times d}`: a hyperparameter (a positive definite matrix)

.. math::
    q(\boldsymbol{w}) &= \mathcal{N}(\boldsymbol{w}|\boldsymbol{\mu}_n^{(t)}, (\boldsymbol{\Lambda}_n^{(t)})^{-1})\\
    &= \frac{|\boldsymbol{\Lambda}_n^{(t)}|^{1/2}}{(2 \pi)^{d/2}} \exp \left\{ -\frac{1}{2} (\boldsymbol{w} - \boldsymbol{\mu}_n^{(t)})^\top \boldsymbol{\Lambda}_n^{(t)} (\boldsymbol{w} - \boldsymbol{\mu}_n^{(t)}) \right\},

where the updating rules of the hyperparameters are as follows:

* :math:`\boldsymbol{\xi}^{(t)} = (\xi_{1}^{(t)}, \xi_{2}^{(t)}, \dots, \xi_{n}^{(t)}) \in \mathbb{R}_{\geq 0}^n`: a variational parameter

.. math::
    \boldsymbol{\Lambda}_n^{(t+1)} &= \boldsymbol{\Lambda}_0 + 2 \sum_{i=1}^{n} \lambda(\xi_i^{(t)}) \boldsymbol{x}_i \boldsymbol{x}_i^\top,\\
    \boldsymbol{\mu}_n^{(t+1)} &= \left(\boldsymbol{\Lambda}_n^{(t+1)}\right)^{-1} \left(\boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 + \sum_{i=1}^{n} (y_i - 1/2) \boldsymbol{x}_{i} \right),\\
    \xi_i^{(t+1)} &= \left[ \boldsymbol{x}_{i}^\top \left\{ \left(\boldsymbol{\Lambda}_n^{(t+1)} \right)^{-1} + \boldsymbol{\mu}_n^{(t+1)} \boldsymbol{\mu}_n^{(t+1)\top} \right\} \boldsymbol{x}_{i} \right]^{1/2}, 

where :math:`\lambda(\cdot)` is defined as follows:

.. math::
    \lambda(\xi) = \frac{1}{2\xi} \left\{ \sigma(\xi) - \frac{1}{2} \right\}.

The approximate predictive distribution is as follows:

* :math:`\boldsymbol{x}_{n+1}\in \mathbb{R}^d`: a new data point
* :math:`y_{n+1}\in \{ 0, 1\}`: a new objective variable

.. math::
    p(y_{n+1} | \boldsymbol{x}^n, \boldsymbol{y}^n, \boldsymbol{x}_{n+1} ) = \sigma \left( \kappa(\sigma_\mathrm{p}^2) \mu_\mathrm{p} \right)^y \left\{ 1 - \sigma \left( \kappa(\sigma_\mathrm{p}^2) \mu_\mathrm{p} \right) \right\}^{1 - y},

where :math:`\sigma_\mathrm{p}^2`, :math:`\mu_\mathrm{p}` are obtained from the hyperparameters of the approximate posterior distribution as follows:

.. math::
    \sigma_\mathrm{p}^2 &= \boldsymbol{x}_{n+1}^\top (\boldsymbol{\Lambda}_n^{(t)})^{-1} \boldsymbol{x}_{n+1} , \\
    \mu_\mathrm{p} &= \boldsymbol{x}_{n+1}^\top \boldsymbol{\mu}_n^{(t)}, 

and :math:`\kappa(\cdot)` is defined as 

.. math::
    \kappa(\sigma^2) = (1 + \pi \sigma^2 / 8)^{-1/2}.
"""

from ._logisticregression import GenModel
from ._logisticregression import LearnModel

__all__ = ["GenModel","LearnModel"]
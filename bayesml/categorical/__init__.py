# Document Author
# Kohei Horinouchi <horinochi_18@toki.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Koki Kazama <kazama@info.shonan-it.ac.jp>
r"""
The categorical distribution with the dirichlet prior distribution

The stochastic data generative model is as follows:

* :math:`d\in \mathbb{Z}`: a dimension (:math:`d \geq 2`)
* :math:`\boldsymbol{x} \in \{ 0, 1\}^d`: a data point, (a one-hot vector, i.e., :math:`\sum_{k=1}^d x_k=1`)
* :math:`\boldsymbol{\theta} \in [0, 1]^d`: a parameter, (:math:`\sum_{k=1}^d \theta_k=1`)

.. math:: 
    p(\boldsymbol{x} | \boldsymbol{\theta}) = \mathrm{Cat}(\boldsymbol{x}|\boldsymbol{\theta}) = \prod_{k=1}^d \theta_k^{x_k},

.. math::
    \mathbb{E}[\boldsymbol{x} | \boldsymbol{\theta}] &= \boldsymbol{\theta}, \\
    \mathbb{V}[x_k | \boldsymbol{\theta}] &= \theta_k (1 - \theta_k), \\
    \mathrm{Cov}[x_k, x_{k'} | \boldsymbol{\theta}] &= -\theta_k \theta_{k'}.


The prior distribution is as follows:

* :math:`\boldsymbol{\alpha}_0 \in \mathbb{R}_{>0}^d`: a hyperparameter
* :math:`\Gamma (\cdot)`: the gamma function
* :math:`\tilde{\alpha}_0 = \sum_{k=1}^d \alpha_{0,k}`
* :math:`C(\boldsymbol{\alpha}_0)=\frac{\Gamma(\tilde{\alpha}_0)}{\Gamma(\alpha_{0,1})\cdots\Gamma(\alpha_{0,d})}`

.. math::
    p(\boldsymbol{\theta}) = \mathrm{Dir}(\boldsymbol{\theta}|\boldsymbol{\alpha}_0) = C(\boldsymbol{\alpha}_0)\prod_{k=1}^d\theta_k^{\alpha_{0,k}-1},

.. math::
    \mathbb{E}[\boldsymbol{\theta}] &= \frac{\boldsymbol{\alpha}_0}{\tilde{\alpha}_0}, \\
    \mathbb{V}[\theta_k] &= \frac{1}{\tilde{\alpha}_0 + 1} \frac{\alpha_{0,k}}{\tilde{\alpha}_0} \left(1 - \frac{\alpha_{0,k}}{\tilde{\alpha}_0} \right), \\
    \mathrm{Cov}[\theta_k, \theta_{k'}] &= - \frac{1}{\tilde{\alpha}_0 + 1} \frac{\alpha_{0,k}}{\tilde{\alpha}_0} \frac{\alpha_{0,k'}}{\tilde{\alpha}_0}.

The posterior distribution is as follows:

* :math:`\boldsymbol{x}^n = (\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n) \in \{ 0, 1\}^{d\times n}`: given data
* :math:`\boldsymbol{\alpha}_n \in \mathbb{R}_{>0}^d`: a hyperparameter
* :math:`\tilde{\alpha}_n = \sum_{k=1}^d \alpha_{n,k}`
* :math:`C(\boldsymbol{\alpha}_n)=\frac{\Gamma(\tilde{\alpha}_n)}{\Gamma(\alpha_{n,1})\cdots\Gamma(\alpha_{n,d})}`

.. math::
    p(\boldsymbol{\theta} | \boldsymbol{x}^n) = \mathrm{Dir}(\boldsymbol{\theta}|\boldsymbol{\alpha}_n) = C(\boldsymbol{\alpha}_n)\prod_{k=1}^d\theta_k^{\alpha_{n,k}-1},

.. math::
    \mathbb{E}[\boldsymbol{\theta} | \boldsymbol{x}^n] &= \frac{\boldsymbol{\alpha}_n}{\tilde{\alpha}_n}, \\
    \mathbb{V}[\theta_k | \boldsymbol{x}^n] &= \frac{1}{\tilde{\alpha}_n + 1} \frac{\alpha_{n,k}}{\tilde{\alpha}_n} \left(1 - \frac{\alpha_{n,k}}{\tilde{\alpha}_n} \right), \\
    \mathrm{Cov}[\theta_k, \theta_{k'} | \boldsymbol{x}^n] &= - \frac{1}{\tilde{\alpha}_n + 1} \frac{\alpha_{n,k}}{\tilde{\alpha}_n} \frac{\alpha_{n,k'}}{\tilde{\alpha}_n},

where the updating rule of the hyperparameters is as follows.

.. math::
    \alpha_{n,k} = \alpha_{0,k} + \sum_{i=1}^n x_{i,k}, \quad (k \in \{ 1, 2, \dots , d \}).

The predictive distribution is as follows:

* :math:`\boldsymbol{x}_{n+1} \in \{ 0, 1\}^d`: a new data point
* :math:`\boldsymbol{\theta}_\mathrm{p} \in [0, 1]^d`: the hyperparameter of the posterior (:math:`\sum_{k=1}^d \theta_{\mathrm{p},k} = 1`)

.. math::
    p(\boldsymbol{x}_{n+1} | \boldsymbol{x}^n) = \mathrm{Cat}(\boldsymbol{x}_{n+1}|\boldsymbol{\theta}_\mathrm{p}) = \prod_{k=1}^d \theta_{\mathrm{p},k}^{x_{n+1,k}},

.. math::
    \mathbb{E}[\boldsymbol{x}_{n+1} | \boldsymbol{x}^n] &= \boldsymbol{\theta}_\mathrm{p}, \\
    \mathbb{V}[x_{n+1,k} | \boldsymbol{x}^n] &= \theta_{\mathrm{p},k} (1 - \theta_{\mathrm{p},k}), \\
    \mathrm{Cov}[x_{n+1,k}, x_{n+1,k'} | \boldsymbol{x}^n] &= -\theta_{\mathrm{p},k} \theta_{\mathrm{p},k'},

where the parameters are obtained from the hyperparameters of the posterior distribution as follows:

.. math::
    \theta_{\mathrm{p},k} = \frac{\alpha_{n,k}}{\sum_{k=1}^d \alpha_{n,k}}, \quad (k \in \{ 1, 2, \dots , d \}).
"""

from ._categorical import GenModel
from ._categorical import LearnModel

__all__ = ["GenModel", "LearnModel"]

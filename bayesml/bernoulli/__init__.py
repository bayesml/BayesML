# Document Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Koki Kazama <kazama@info.shonan-it.ac.jp>
r"""
The Bernoulli distribution with the beta prior distribution.

The stochastic data generative model is as follows:

* :math:`x \in \{ 0, 1\}`: a data point
* :math:`\theta \in [0, 1]`: a parameter

.. math::
    p(x | \theta) = \mathrm{Bern}(x|\theta) = \theta^x (1-\theta)^{1-x}.

.. math::
    \mathbb{E}[x | \theta] &= \theta, \\
    \mathbb{V}[x | \theta] &= \theta (1 - \theta).

The prior distribution is as follows:

* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`B(\cdot,\cdot): \mathbb{R}_{>0} \times \mathbb{R}_{>0} \to \mathbb{R}_{>0}`: the Beta function

.. math::
    p(\theta) = \mathrm{Beta}(\theta|\alpha_0,\beta_0) = \frac{1}{B(\alpha_0, \beta_0)} \theta^{\alpha_0 - 1} (1-\theta)^{\beta_0 - 1}.

.. math::
    \mathbb{E}[\theta] &= \frac{\alpha_0}{\alpha_0 + \beta_0}, \\
    \mathbb{V}[\theta] &= \frac{\alpha_0 \beta_0}{(\alpha_0 + \beta_0)^2 (\alpha_0 + \beta_0 + 1)}.

The posterior distribution is as follows:

* :math:`x^n = (x_1, x_2, \dots , x_n) \in \{ 0, 1\}^n`: given data
* :math:`\alpha_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n \in \mathbb{R}_{>0}`: a hyperparameter

.. math::
    p(\theta | x^n) = \mathrm{Beta}(\theta|\alpha_n,\beta_n) = \frac{1}{B(\alpha_n, \beta_n)} \theta^{\alpha_n - 1} (1-\theta)^{\beta_n - 1},

.. math::
    \mathbb{E}[\theta | x^n] &= \frac{\alpha_n}{\alpha_n + \beta_n}, \\
    \mathbb{V}[\theta | x^n] &= \frac{\alpha_n \beta_n}{(\alpha_n + \beta_n)^2 (\alpha_n + \beta_n + 1)},

where the updating rule of the hyperparameters is

.. math::
    \alpha_n = \alpha_0 + \sum_{i=1}^n I \{ x_i = 1 \},\\
    \beta_n = \beta_0 + \sum_{i=1}^n I \{ x_i = 0 \}.

The predictive distribution is as follows:

* :math:`x_{n+1} \in \{ 0, 1\}`: a new data point
* :math:`\alpha_\mathrm{p} \in \mathbb{R}_{>0}`: a parameter
* :math:`\beta_\mathrm{p} \in \mathbb{R}_{>0}`: a parameter
* :math:`\theta_\mathrm{p} \in [0,1]`: a parameter

.. math::
    p(x_{n+1} | x^n) = \mathrm{Bern}(x_{n+1}|\theta_\mathrm{p}) =\theta_\mathrm{p}^{x_{n+1}}(1-\theta_\mathrm{p})^{1-x_{n+1}},

.. math::
    \mathbb{E}[x_{n+1} | x^n] &= \theta_\mathrm{p}, \\
    \mathbb{V}[x_{n+1} | x^n] &= \theta_\mathrm{p} (1 - \theta_\mathrm{p}),

where the parameters are obtained from the hyperparameters of the posterior distribution as follows.

.. math::
    \theta_\mathrm{p} = \frac{\alpha_n}{\alpha_n + \beta_n}.
"""

from ._bernoulli import GenModel
from ._bernoulli import LearnModel

__all__ = ["GenModel","LearnModel"]

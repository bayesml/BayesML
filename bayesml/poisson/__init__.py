# Document Author
# Ryota Maniwa <r_maniwa115@fuji.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Koki Kazama <kazama@info.shonan-it.ac.jp>
r"""
The Poisson distribution with the gamma prior distribution.

The stochastic data generative model is as follows:

* :math:`x \in \mathbb{N}`: a data point
* :math:`\lambda \in \mathbb{R}_{>0}`: a parameter

.. math::
    p(x | \lambda) = \mathrm{Po}(x|\lambda) = \frac{ \lambda^{x} }{x!}\exp \{ -\lambda \}.

The prior distribution is as follows:

* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\Gamma(\cdot): \mathbb{R}_{>0} \to \mathbb{R}_{>0}`: the Gamma function

.. math::
    p(\lambda) = \mathrm{Gam}(\lambda|\alpha_0,\beta_0) = \frac{\beta_0^{\alpha_0}}{\Gamma (\alpha_0)} \lambda^{\alpha_0 - 1} \exp \{ -\beta_0 \lambda \},

.. math::
    \mathbb{E}[\lambda] &= \frac{\alpha_0}{\beta_0}, \\
    \mathbb{V}[\lambda] &= \frac{\alpha_0}{\beta_0^2}.

The posterior distribution is as follows:

* :math:`x^n = (x_1, x_2, \dots , x_n) \in \mathbb{N}^n`: given data
* :math:`\alpha_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n \in \mathbb{R}_{>0}`: a hyperparameter

.. math::
    p(\lambda | x^n) = \mathrm{Gam}(\lambda|\alpha_n,\beta_n) = \frac{\beta_n^{\alpha_n}}{\Gamma (\alpha_n)} \lambda^{\alpha_n - 1} \exp \{ -\beta_n \lambda \},

.. math::
    \mathbb{E}[\lambda | x^n] &= \frac{\alpha_n}{\beta_n}, \\
    \mathbb{V}[\lambda | x^n] &= \frac{\alpha_n}{\beta_n^2},

where the updating rule of the hyperparameters is

.. math::
    \alpha_n &= \alpha_0 + \sum_{i=1}^n x_i,\\
    \beta_n &= \beta_0 + n.

The predictive distribution is as follows:

* :math:`x_{n+1} \in \mathbb{N}`: a new data point
* :math:`\theta_\mathrm{p} \in \mathbb{R}_{>0}, 0<\theta_\mathrm{p}<1`: the hyperparameter of the posterior
* :math:`r_\mathrm{p} \in \mathbb{N}`: the hyperparameter of the posterior

.. math::
    \mathrm{NB}(x_{n+1}|r_\mathrm{p},\theta_\mathrm{p}) = \frac{\Gamma(x_{n+1}+r_\mathrm{p})}{\Gamma(x_{n+1}+1)\Gamma(r_\mathrm{p})}\theta_\mathrm{p}^{x_{n+1}}(1-\theta_\mathrm{p})^{r_\mathrm{p}},

.. math::
    \mathbb{E}[x_{n+1} | x^n] &= \frac{r_\mathrm{p} \theta_\mathrm{p}}{1 - \theta_\mathrm{p}}, \\
    \mathbb{V}[x_{n+1} | x^n] &= \frac{r_\mathrm{p} \theta_\mathrm{p}}{(1 - \theta_\mathrm{p})^2},

where the parameters are obtained from the hyperparameters of the posterior distribution as follows:

.. math::
    &r_\mathrm{p}=\alpha_n, \\
    &\theta_\mathrm{p} = \frac{1}{\beta_n + 1}.
"""

from ._poisson import GenModel
from ._poisson import LearnModel

__all__ = ["GenModel","LearnModel"]

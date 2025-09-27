# Document Author
# Keito Tajima <wool812@akane.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Koki Kazama <kazama@info.shonan-it.ac.jp>
r"""
The multivariate normal distribution with normal-wishart prior distribution.

The stochastic data generative model is as follows:

* :math:`D \in \mathbb{N}`: a dimension of data
* :math:`\boldsymbol{x} \in \mathbb{R}^D`: a data point
* :math:`\boldsymbol{\mu} \in \mathbb{R}^D`: a parameter
* :math:`\boldsymbol{\Lambda} \in \mathbb{R}^{D\times D}` : a parameter (a positive definite matrix)
* :math:`| \boldsymbol{\Lambda} | \in \mathbb{R}`: the determinant of :math:`\boldsymbol{\Lambda}`

.. math::
    p(\boldsymbol{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}) &= \mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu},\boldsymbol{\Lambda}^{-1}) \\
    &= \frac{| \boldsymbol{\Lambda} |^{1/2}}{(2\pi)^{D/2}} \exp \left\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^\top \boldsymbol{\Lambda} (\boldsymbol{x}-\boldsymbol{\mu}) \right\},

.. math::
    \mathbb{E} [\boldsymbol{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}] &= \boldsymbol{\mu}, \\
    \mathrm{Cov} [\boldsymbol{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}] &= \boldsymbol{\Lambda}^{-1}.

The prior distribution is as follows:

* :math:`\boldsymbol{m}_0 \in \mathbb{R}^{D}`: a hyperparameter
* :math:`\kappa_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\nu_0 \in \mathbb{R}`: a hyperparameter (:math:`\nu_0 > D-1`)
* :math:`\boldsymbol{W}_0 \in \mathbb{R}^{D\times D}`: a hyperparameter (a positive definite matrix)
* :math:`\mathrm{Tr} \{ \cdot \}`: a trace of a matrix
* :math:`\Gamma (\cdot)`: the gamma function

.. math::
    p(\boldsymbol{\mu},\boldsymbol{\Lambda}) &= \mathcal{N}(\boldsymbol{\mu}|\boldsymbol{m}_0,(\kappa_0 \boldsymbol{\Lambda})^{-1})\mathcal{W}(\boldsymbol{\Lambda}|\boldsymbol{W}_0, \nu_0) \\
    &= \left( \frac{\kappa_0}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}|^{1/2} \exp \left\{ -\frac{\kappa_0}{2}(\boldsymbol{\mu}-\boldsymbol{m}_0)^\top \boldsymbol{\Lambda} (\boldsymbol{\mu}-\boldsymbol{m}_0) \right\} \\
    &\qquad \times B(\boldsymbol{W}_0, \nu_0) | \boldsymbol{\Lambda} |^{(\nu_0 - D - 1) / 2} \exp \left\{ -\frac{1}{2} \mathrm{Tr} \{ \boldsymbol{W}_0^{-1} \boldsymbol{\Lambda} \} \right\},\\

.. math::
    \mathbb{E}[\boldsymbol{\mu}] &= \boldsymbol{m}_0 & (\nu_n > D), \\
    \mathrm{Cov}[\boldsymbol{\mu}] &= \frac{1}{\kappa_0 (\nu_0 - D - 1)} \boldsymbol{W}_0^{-1} & (\nu_n > D + 1), \\
    \mathbb{E}[\boldsymbol{\Lambda}] &= \nu_0 \boldsymbol{W}_0,

where :math:`B(\boldsymbol{W}_0, \nu_0)` is defined as follows:

.. math::
    B(\boldsymbol{W}_0, \nu_0) = | \boldsymbol{W}_0 |^{-\nu_0 / 2} \left( 2^{\nu_0 D / 2} \pi^{D(D-1)/4} \prod_{i=1}^D \Gamma \left( \frac{\nu_0 + 1 - i}{2} \right) \right)^{-1}.

The posterior distribution is as follows:

* :math:`\boldsymbol{x}^n = (\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n) \in \mathbb{R}^{D\times n}`: given data
* :math:`\boldsymbol{m}_n \in \mathbb{R}^{D}`: a hyperparameter
* :math:`\kappa_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\nu_n \in \mathbb{R}`: a hyperparameter :math:`(\nu_n > D-1)`
* :math:`\boldsymbol{W}_n \in \mathbb{R}^{D\times D}`: a hyperparameter (a positive definite matrix)

.. math::
    p(\boldsymbol{\mu},\boldsymbol{\Lambda} | \boldsymbol{x}^n) &= \mathcal{N}(\boldsymbol{\mu}|\boldsymbol{m}_n,(\kappa_n \boldsymbol{\Lambda})^{-1})\mathcal{W}(\boldsymbol{\Lambda}|\boldsymbol{W}_n, \nu_n) \\
    &= \left( \frac{\kappa_n}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}|^{1/2} \exp \left\{ -\frac{\kappa_n}{2}(\boldsymbol{\mu}-\boldsymbol{m}_n)^\top \boldsymbol{\Lambda} (\boldsymbol{\mu}-\boldsymbol{m}_n) \right\} \\
    &\qquad \times B(\boldsymbol{W}_n, \nu_n) | \boldsymbol{\Lambda} |^{(\nu_n - D - 1) / 2} \exp \left\{ -\frac{1}{2} \mathrm{Tr} \{ \boldsymbol{W}_n^{-1} \boldsymbol{\Lambda} \} \right\},

.. math::
    \mathbb{E}[\boldsymbol{\mu} | \boldsymbol{x}^n] &= \boldsymbol{m}_n & (\nu_n > D), \\
    \mathrm{Cov}[\boldsymbol{\mu} | \boldsymbol{x}^n] &= \frac{1}{\kappa_n (\nu_n - D - 1)} \boldsymbol{W}_n^{-1} & (\nu_n > D + 1), \\
    \mathbb{E}[\boldsymbol{\Lambda} | \boldsymbol{x}^n] &= \nu_n \boldsymbol{W}_n,

where the updating rule of the hyperparameters is

.. math::
    \bar{\boldsymbol{x}} &= \frac{1}{n} \sum_{i=1}^n \boldsymbol{x}_i, \\
    \boldsymbol{m}_n &= \frac{\kappa_0\boldsymbol{\mu}_0+n\bar{\boldsymbol{x}}}{\kappa_0+n}, \\
    \kappa_n &= \kappa_0 + n, \\
    \boldsymbol{W}_n^{-1} &= \boldsymbol{W}_0^{-1} + \sum_{i=1}^{n}(\boldsymbol{x}_i-\bar{\boldsymbol{x}})(\boldsymbol{x}_i-\bar{\boldsymbol{x}})^\top + \frac{\kappa_0 n}{\kappa_0+n}(\bar{\boldsymbol{x}}-\boldsymbol{\mu}_0)(\bar{\boldsymbol{x}}-\boldsymbol{\mu}_0)^\top, \\
    \nu_n &= \nu_0 + n.\\

The predictive distribution is as follows:

* :math:`\boldsymbol{x}_{n+1} \in \mathbb{R}^D`: a new data point
* :math:`\boldsymbol{\mu}_\mathrm{p} \in \mathbb{R}^D`: the hyperparameter of the predictive distribution
* :math:`\boldsymbol{\Lambda}_\mathrm{p} \in \mathbb{R}^{D \times D}`: the hyperparameter of the predictive distribution (a positive definite matrix)
* :math:`\nu_\mathrm{p} \in \mathbb{R}_{>0}`: the hyperparameter of the predictive distribution

.. math::
    &p(x_{n+1}|x^n) \\
    &= \mathrm{St}(x_{n+1}|\boldsymbol{\mu}_\mathrm{p},\boldsymbol{\Lambda}_\mathrm{p}, \nu_\mathrm{p}) \\
    &= \frac{\Gamma (\nu_\mathrm{p} / 2 + D / 2)}{\Gamma (\nu_\mathrm{p} / 2)} \frac{|\boldsymbol{\Lambda}_\mathrm{p}|^{1/2}}{(\nu_\mathrm{p} \pi)^{D/2}} \left( 1 + \frac{1}{\nu_\mathrm{p}} (\boldsymbol{x}_{n+1} - \boldsymbol{\mu}_\mathrm{p})^\top \boldsymbol{\Lambda}_\mathrm{p} (\boldsymbol{x}_{n+1} - \boldsymbol{\mu}_\mathrm{p}) \right)^{-\nu_\mathrm{p}/2 - D/2},

.. math::
    \mathbb{E}[\boldsymbol{x}_{n+1} | \boldsymbol{x}^n] &= \boldsymbol{\mu}_\mathrm{p} & (\nu_\mathrm{p} > 1), \\
    \mathrm{Cov}[\boldsymbol{x}_{n+1} | \boldsymbol{x}^n] &= \frac{\nu_\mathrm{p}}{\nu_\mathrm{p}-2} \boldsymbol{\Lambda}_\mathrm{p}^{-1} & (\nu_\mathrm{p} > 2),

where the parameters are obtained from the hyperparameters of the posterior distribution as follows:

.. math::
    \boldsymbol{\mu}_\mathrm{p} &= \boldsymbol{m}_n, \\
    \boldsymbol{\Lambda}_\mathrm{p} &= \frac{\kappa_n (\nu_n - D + 1)}{\kappa_n + 1} \boldsymbol{W}_n, \\
    \nu_\mathrm{p} &= \nu_n - D + 1.
"""

from ._multivariatenormal import GenModel
from ._multivariatenormal import LearnModel

__all__ = ["GenModel","LearnModel"]
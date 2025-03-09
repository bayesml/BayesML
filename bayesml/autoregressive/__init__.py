# Document Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Koki Kazama <kazama@info.shonan-it.ac.jp>
r"""
The linear autoregressive model with the normal-gamma prior distribution.

The stochastic data generative model is as follows:

* :math:`d \in \mathbb{N}`: the degree of the model
* :math:`n \in \mathbb{N}`: time index
* :math:`x_n \in \mathbb{R}`: a data point at :math:`n`
* :math:`\boldsymbol{x}'_n := [1, x_{n-d}, x_{n-d+1}, \dots , x_{n-1}]^\top \in \mathbb{R}^{d+1}`. Here we assume :math:`x_n` for :math:`n < 1` is given as a initial value.
* :math:`\boldsymbol{\theta} \in \mathbb{R}^{d+1}`: a regression coefficient parameter
* :math:`\tau \in \mathbb{R}_{>0}`: a precision parameter of noise

.. math::
    p(x_n | \boldsymbol{x}'_{n-1}, \boldsymbol{\theta}, \tau) &= \mathcal{N}(x_n|\boldsymbol{\theta}^\top \boldsymbol{x}'_{n-1}, \tau^{-1}) \\
    &= \sqrt{\frac{\tau}{2 \pi}} \exp \left\{ -\frac{\tau}{2} (x_n - \boldsymbol{\theta}^\top \boldsymbol{x}'_{n-1})^2 \right\}.

.. math::
    &\mathbb{E}[ x_n | \boldsymbol{x}'_{n-1},\boldsymbol{\theta},\tau] = \boldsymbol{\theta}^{\top} \boldsymbol{x}'_{n-1}, \\
    &\mathbb{V}[ x_n | \boldsymbol{x}'_{n-1},\boldsymbol{\theta},\tau ] = \tau^{-1}.


The prior distribution is as follows:

* :math:`\boldsymbol{\mu}_0 \in \mathbb{R}^{d+1}`: a hyperparameter for :math:`\boldsymbol{\theta}`
* :math:`\boldsymbol{\Lambda}_0 \in \mathbb{R}^{(d+1) \times (d+1)}`: a hyperparameter for :math:`\boldsymbol{\theta}` (a positive definite matrix)
* :math:`| \boldsymbol{\Lambda}_0 | \in \mathbb{R}`: the determinant of :math:`\boldsymbol{\Lambda}_0`
* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter for :math:`\tau`
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter for :math:`\tau`
* :math:`\Gamma(\cdot): \mathbb{R}_{>0} \to \mathbb{R}`: the Gamma function

.. math::
    p(\boldsymbol{\theta}, \tau) &= \mathcal{N}(\boldsymbol{\theta}|\boldsymbol{\mu}_0, (\tau \boldsymbol{\Lambda}_0)^{-1}) \mathrm{Gam}(\tau|\alpha_0,\beta_0)\\
    &= \frac{|\tau \boldsymbol{\Lambda}_0|^{1/2}}{(2 \pi)^{(d+1)/2}} 
    \exp \left\{ -\frac{\tau}{2} (\boldsymbol{\theta} - \boldsymbol{\mu}_0)^\top 
    \boldsymbol{\Lambda}_0 (\boldsymbol{\theta} - \boldsymbol{\mu}_0) \right\}
    \frac{\beta_0^{\alpha_0}}{\Gamma (\alpha_0)} \tau^{\alpha_0 - 1} \exp \{ -\beta_0 \tau \} .

.. math::
    \mathbb{E}[\boldsymbol{\theta}] &= \boldsymbol{\mu}_0 & \left( \alpha_0 > \frac{1}{2} \right), \\
    \mathrm{Cov}[\boldsymbol{\theta}] &= \frac{\beta_0}{\alpha_0 - 1} \boldsymbol{\Lambda}_0^{-1} & (\alpha_0 > 1), \\
    \mathbb{E}[\tau] &= \frac{\alpha_0}{\beta_0}, \\
    \mathbb{V}[\tau] &= \frac{\alpha_0}{\beta_0^2}.

The posterior distribution is as follows:

* :math:`x^n := [x_1, x_2, \dots , x_n]^\top \in \mathbb{R}^n`: given data
* :math:`\boldsymbol{X}_n = [\boldsymbol{x}'_1, \boldsymbol{x}'_2, \dots , \boldsymbol{x}'_n]^\top \in \mathbb{R}^{n \times (d+1)}`
* :math:`\boldsymbol{\mu}_n \in \mathbb{R}^{d+1}`: a hyperparameter for :math:`\boldsymbol{\theta}`
* :math:`\boldsymbol{\Lambda}_n \in \mathbb{R}^{(d+1) \times (d+1)}`: a hyperparameter for :math:`\boldsymbol{\theta}` (a positive definite matrix)
* :math:`\alpha_n \in \mathbb{R}_{>0}`: a hyperparameter for :math:`\tau`
* :math:`\beta_n \in \mathbb{R}_{>0}`: a hyperparameter for :math:`\tau`

.. math::
    p(\boldsymbol{\theta}, \tau | x^n) &= \mathcal{N}(\boldsymbol{\theta}|\boldsymbol{\mu}_n, (\tau \boldsymbol{\Lambda}_n)^{-1}) \mathrm{Gam}(\tau|\alpha_n,\beta_n)\\
    &= \frac{|\boldsymbol{\tau \Lambda}_n|^{1/2}}{(2 \pi)^{(d+1)/2}}
    \exp \left\{ -\frac{\tau}{2} (\boldsymbol{\theta} - \boldsymbol{\mu}_n)^\top 
    \boldsymbol{\Lambda}_n (\boldsymbol{\theta} - \boldsymbol{\mu}_n) \right\}
    \frac{\beta_n^{\alpha_n}}{\Gamma (\alpha_n)} \tau^{\alpha_n - 1} \exp \{ -\beta_n \tau \} .

.. math::
    \mathbb{E}[\boldsymbol{\theta} | x^n] &= \boldsymbol{\mu}_n & \left( \alpha_n > \frac{1}{2} \right), \\
    \mathrm{Cov}[\boldsymbol{\theta} | x^n] &= \frac{\beta_n}{\alpha_n - 1} \boldsymbol{\Lambda}_n^{-1} & (\alpha_n > 1), \\
    \mathbb{E}[\tau | x^n] &= \frac{\alpha_n}{\beta_n}, \\
    \mathbb{V}[\tau | x^n] &= \frac{\alpha_n}{\beta_n^2},

where the updating rules of the hyperparameters are

.. math::
    \boldsymbol{\Lambda}_n &= \boldsymbol{\Lambda}_0 + \boldsymbol{X}_n^\top \boldsymbol{X}_n,\\
    \boldsymbol{\mu}_n &= \boldsymbol{\Lambda}_n^{-1} (\boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 + \boldsymbol{X}_n^\top x^n),\\
    \alpha_n &= \alpha_0 + \frac{n}{2},\\
    \beta_n &= \beta_0 + \frac{1}{2} \left( -\boldsymbol{\mu}_n^\top \boldsymbol{\Lambda}_n \boldsymbol{\mu}_n 
    + (x^n)^\top x^n + \boldsymbol{\mu}_0^\top \boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 \right).

The predictive distribution is as follows:

* :math:`x_{n+1} \in \mathbb{R}`: a new data point
* :math:`m_\mathrm{p} \in \mathbb{R}`: a parameter
* :math:`\lambda_\mathrm{p} \in \mathbb{R}_{>0}`: a parameter
* :math:`\nu_\mathrm{p} \in \mathbb{R}_{>0}`: a parameter

.. math::
    \mathrm{St}(x_{n+1}|m_\mathrm{p}, \lambda_\mathrm{p}, \nu_\mathrm{p})
    = \frac{\Gamma (\nu_\mathrm{p}/2 + 1/2)}{\Gamma (\nu_\mathrm{p}/2)}
    \left( \frac{\lambda_\mathrm{p}}{\pi \nu_\mathrm{p}} \right)^{1/2}
    \left[ 1 + \frac{\lambda_\mathrm{p}(x_{n+1}-m_\mathrm{p})^2}{\nu_\mathrm{p}} \right]^{-\nu_\mathrm{p}/2 - 1/2}.

.. math::
    \mathbb{E}[x_{n+1} | x^n] &= m_\mathrm{p} & (\nu_\mathrm{p} > 1), \\
    \mathbb{V}[x_{n+1} | x^n] &= \frac{1}{\lambda_\mathrm{p}} \frac{\nu_\mathrm{p}}{\nu_\mathrm{p}-2} & (\nu_\mathrm{p} > 2),

where the parameters are obtained from the hyperparameters of the posterior distribution as follows.

.. math::
    m_\mathrm{p} &= \boldsymbol{\mu}_n^\top \boldsymbol{x}'_n,\\
    \lambda_\mathrm{p} &= \frac{\alpha_n}{\beta_n} (1 + (\boldsymbol{x}'_n)^\top \boldsymbol{\Lambda}_n^{-1} \boldsymbol{x}'_n)^{-1},\\
    \nu_\mathrm{p} &= 2 \alpha_n.
"""

from ._autoregressive import GenModel
from ._autoregressive import LearnModel

__all__ = ["GenModel","LearnModel"]

# Document Author
# Taisuke Ishiwatari <taisuke.ishiwatari@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Koki Kazama <kazama@info.shonan-it.ac.jp>
r"""
The  Baysian Linear Regression.

The stochastic data generative model is as follows:

* :math:`d \in \mathbb N`: a dimension
* :math:`\boldsymbol{x} = [x_1, x_2, \dots , x_d] \in \mathbb{R}^d`: an explanatory variable. If you consider an intercept term, it should be included as one of the elements of :math:`\boldsymbol{x}`.
* :math:`y\in\mathbb{R}`: an objective variable
* :math:`\tau \in\mathbb{R}_{>0}`: a parameter
* :math:`\boldsymbol{\theta}\in\mathbb{R}^{d}`: a parameter

.. math::
    p(y|\boldsymbol{x},\boldsymbol{\theta},\tau) &= \mathcal N (y| \boldsymbol{\theta}^{\top} \boldsymbol{x},\tau^{-1}) \\
    &= \sqrt{\frac{\tau}{2 \pi}} \exp \left\{ -\frac{\tau}{2} (y - \boldsymbol{\theta}^\top \boldsymbol{x})^2 \right\}.


.. math::
    &\mathbb{E}[ y | \boldsymbol{x},\boldsymbol{\theta},\tau] = \boldsymbol{\theta}^{\top} \boldsymbol{x}, \\
    &\mathbb{V}[ y | \boldsymbol{x},\boldsymbol{\theta},\tau ] = \tau^{-1}.

The prior distribution is as follows:

* :math:`\boldsymbol{\mu_0} \in \mathbb{R}^d`: a hyperparameter
* :math:`\boldsymbol{\Lambda_0} \in \mathbb{R}^{d\times d}`: a hyperparameter (a positive definite matrix)
* :math:`\alpha_0\in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0\in \mathbb{R}_{>0}`: a hyperparameter

.. math::
    p(\boldsymbol{\theta}, \tau) &= \mathcal{N}(\boldsymbol{\theta}|\boldsymbol{\mu}_0, (\tau \boldsymbol{\Lambda}_0)^{-1}) \mathrm{Gam}(\tau|\alpha_0,\beta_0)\\
    &= \frac{|\tau \boldsymbol{\Lambda}_0|^{1/2}}{(2 \pi)^{d/2}} \exp \left\{ -\frac{\tau}{2} (\boldsymbol{\theta} - \boldsymbol{\mu}_0)^\top \boldsymbol{\Lambda}_0 (\boldsymbol{\theta} - \boldsymbol{\mu}_0) \right\} \frac{\beta_0^{\alpha_0}}{\Gamma (\alpha_0)} \tau^{\alpha_0 - 1} \exp \{ -\beta_0 \tau \} .

.. math::
    \mathbb{E}[\boldsymbol{\theta}] &= \boldsymbol{\mu}_0 & \left( \alpha_0 > \frac{1}{2} \right), \\
    \mathrm{Cov}[\boldsymbol{\theta}] &= \frac{\beta_0}{\alpha_0 - 1} \boldsymbol{\Lambda}_0^{-1} & (\alpha_0 > 1), \\
    \mathbb{E}[\tau] &= \frac{\alpha_0}{\beta_0}, \\
    \mathbb{V}[\tau] &= \frac{\alpha_0}{\beta_0^2}.


The posterior distribution is as follows:

* :math:`n \in \mathbb N`: a sample size
* :math:`\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n]^\top \in \mathbb{R}^{n \times d}`
* :math:`\boldsymbol{y} = [y_1, y_2, \dots , y_n]^\top \in \mathbb{R}^n`
* :math:`\boldsymbol{\mu}_n\in \mathbb{R}^d`: a hyperparameter
* :math:`\boldsymbol{\Lambda_n} \in \mathbb{R}^{d\times d}`: a hyperparameter (a positive definite matrix)
* :math:`\alpha_n\in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n\in \mathbb{R}_{>0}`: a hyperparameter

.. math::
    p(\boldsymbol{\theta}, \tau | \boldsymbol{X}, \boldsymbol{y}) &= \mathcal{N}(\boldsymbol{\theta}|\boldsymbol{\mu}_n, (\tau \boldsymbol{\Lambda}_n)^{-1}) \mathrm{Gam}(\tau|\alpha_n,\beta_n)\\
    &= \frac{|\tau \boldsymbol{\Lambda}_n|^{1/2}}{(2 \pi)^{d/2}} \exp \left\{ -\frac{\tau}{2} (\boldsymbol{\theta} - \boldsymbol{\mu}_n)^\top \boldsymbol{\Lambda}_n (\boldsymbol{\theta} - \boldsymbol{\mu}_n) \right\} \frac{\beta_n^{\alpha_n}}{\Gamma (\alpha_n)} \tau^{\alpha_n - 1} \exp \{ -\beta_n \tau \} .

.. math::
    \mathbb{E}[\boldsymbol{\theta} | \boldsymbol{X}, \boldsymbol{y}] &= \boldsymbol{\mu}_n & \left( \alpha_n > \frac{1}{2} \right), \\
    \mathrm{Cov}[\boldsymbol{\theta} | \boldsymbol{X}, \boldsymbol{y}] &= \frac{\beta_n}{\alpha_n - 1} \boldsymbol{\Lambda}_n^{-1} & (\alpha_n > 1), \\
    \mathbb{E}[\tau | \boldsymbol{X}, \boldsymbol{y}] &= \frac{\alpha_n}{\beta_n}, \\
    \mathbb{V}[\tau | \boldsymbol{X}, \boldsymbol{y}] &= \frac{\alpha_n}{\beta_n^2},

where the updating rules of the hyperparameters are

.. math::
    \boldsymbol{\Lambda}_n &= \boldsymbol{\Lambda}_0 + \boldsymbol{X}^\top \boldsymbol{X},\\
    \boldsymbol{\mu}_n &= \boldsymbol{\Lambda}_n^{-1} (\boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 + \boldsymbol{X}^\top \boldsymbol{y}),\\
    \alpha_n &= \alpha_0 + \frac{n}{2},\\
    \beta_n &= \beta_0 + \frac{1}{2} \left( -\boldsymbol{\mu}_n^\top \boldsymbol{\Lambda}_n \boldsymbol{\mu}_n + \boldsymbol{y}^\top \boldsymbol{y} + \boldsymbol{\mu}_0^\top \boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 \right).

The predictive distribution is as follows:

* :math:`\boldsymbol{x}_{n+1}\in \mathbb{R}^d`: a new data point
* :math:`y_{n+1}\in \mathbb{R}`: a new objective variable
* :math:`m_\mathrm{p}\in \mathbb{R}`: a parameter
* :math:`\lambda_\mathrm{p}\in \mathbb{R}`: a parameter
* :math:`\nu_\mathrm{p}\in \mathbb{R}`: a parameter

.. math::
    p(y_{n+1} | \boldsymbol{X}, \boldsymbol{y}, \boldsymbol{x}_{n+1} ) &= \mathrm{St}\left(y_{n+1} \mid m_\mathrm{p}, \lambda_\mathrm{p}, \nu_\mathrm{p}\right) \\
    &= \frac{\Gamma (\nu_\mathrm{p} / 2 + 1/2 )}{\Gamma (\nu_\mathrm{p} / 2)} \left( \frac{\lambda_\mathrm{p}}{\pi \nu_\mathrm{p}} \right)^{1/2} \left( 1 + \frac{\lambda_\mathrm{p} (y_{n+1} - m_\mathrm{p})^2}{\nu_\mathrm{p}} \right)^{-\nu_\mathrm{p}/2 - 1/2},

.. math::
    \mathbb{E}[y_{n+1} | \boldsymbol{X}, \boldsymbol{y}, \boldsymbol{x}_{n+1}] &= m_\mathrm{p} & (\nu_\mathrm{p} > 1), \\
    \mathbb{V}[y_{n+1} | \boldsymbol{X}, \boldsymbol{y}, \boldsymbol{x}_{n+1}] &= \frac{1}{\lambda_\mathrm{p}} \frac{\nu_\mathrm{p}}{\nu_\mathrm{p}-2} & (\nu_\mathrm{p} > 2),

where the parameters are obtained from the hyperparameters of the posterior distribution as follows.

.. math::
    m_\mathrm{p} &= \boldsymbol{x}_{n+1}^{\top} \boldsymbol{\mu}_{n}, \\
    \lambda_\mathrm{p} &= \frac{\alpha_{n}}{\beta_{n}}\left(1+\boldsymbol{x}_{n+1}^{\top} \boldsymbol{\Lambda}_{n} \boldsymbol{x}_{n+1}\right)^{-1}, \\
    \nu_\mathrm{p} &= 2 \alpha_{n}.
"""

from ._linearregression import GenModel
from ._linearregression import LearnModel

__all__ = ["GenModel","LearnModel"]

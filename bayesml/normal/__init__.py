# Document Author
# Noboru Namegaya <n.noboru20180403@toki.waseda.jp>
# Koshi Shimada <shimada.koshi.re@gmail.com>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Koki Kazama <kazama@info.shonan-it.ac.jp>
r"""
The normal distribution with the normaml-gamma prior distribution.

The stochastic data generative model is as follows:

* :math:`x \in \mathbb{R}`: a data point
* :math:`\mu \in \mathbb{R}`: a mean parameter
* :math:`\tau \in \mathbb{R}_{>0}`: a precision parameter

.. math::
    p(x | \mu, \tau) &= \mathcal{N}(x|\mu,\tau^{-1})\\
    &= \sqrt{\frac{\tau}{2\pi}} \exp \left\{ -\frac{\tau}{2}(x-\mu)^2 \right\},


.. math::
    &\mathbb{E}[x | \mu, \tau] = \mu, \\
    &\mathbb{V}[x | \mu, \tau] = \tau^{-1}.


The prior distribution is as follows:

* :math:`m_0 \in \mathbb{R}`: a hyperparameter
* :math:`\kappa_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\Gamma ( \cdot )`: the gamma function

.. math::
    p(\mu, \tau) &= \mathcal{N}(\mu|m_0,(\kappa_0 \tau)^{-1}) \mathrm{Gam}(\tau | \alpha_0, \beta_0) \\
    &= \sqrt{\frac{\kappa_0 \tau}{2\pi}} \exp \left\{ -\frac{\kappa_0 \tau}{2}(\mu-m_0)^2 \right\} \frac{\beta_0^{\alpha_0}}{\Gamma (\alpha_0)} \tau^{\alpha_0 - 1} \exp \{ -\beta_0 \tau \},

.. math::
    \mathbb{E}[\mu] &= m_0 & \left( \alpha_0 > \frac{1}{2} \right), \\
    \mathbb{V}[\mu] &= \frac{\beta_0}{\kappa_0 (\alpha_0 - 1)} & (\alpha_0 > 1), \\
    \mathbb{E}[\tau] &= \frac{\alpha_0}{\beta_0}, \\
    \mathbb{V}[\tau] &= \frac{\alpha_0}{\beta_0^2}.

The posterior distribution is as follows:

* :math:`x^n = (x_1, x_2, \dots , x_n) \in \mathbb{R}^n`: given data
* :math:`m_n \in \mathbb{R}`: a hyperparameter
* :math:`\kappa_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\alpha_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\Gamma ( \cdot )`: the gamma function

.. math::
    p(\mu, \tau | x^n) &= \mathcal{N}(\mu|m_n,(\kappa_n \tau)^{-1}) \mathrm{Gam}(\tau | \alpha_n, \beta_n) \\
    &= \sqrt{\frac{\kappa_n \tau}{2\pi}} \exp \left\{ -\frac{\kappa_n \tau}{2}(\mu-m_n)^2 \right\} \frac{\beta_n^{\alpha_n}}{\Gamma (\alpha_n)} \tau^{\alpha_n - 1} \exp \{ -\beta_n \tau \},

.. math::
    \mathbb{E}[\mu | x^n] &= m_n & \left( \alpha_n > \frac{1}{2} \right), \\
    \mathbb{V}[\mu | x^n] &= \frac{\beta_n}{\kappa_n (\alpha_n - 1)} & (\alpha_n > 1), \\
    \mathbb{E}[\tau | x^n] &= \frac{\alpha_n}{\beta_n}, \\
    \mathbb{V}[\tau | x^n] &= \frac{\alpha_n}{\beta_n^2},

where the updating rule of the hyperparameters is

.. math::
    \bar{x} &= \frac{1}{n} \sum_{i=1}^n x_i, \\
    m_n &= \frac{\kappa_0 m_0 + n \bar{x}}{\kappa_0 + n}, \\
    \kappa_n &= \kappa_0 + n, \\
    \alpha_n &= \alpha_0 + \frac{n}{2}, \\
    \beta_n &=  \beta_0 + \frac{1}{2} \left( \sum_{i=1}^n (x_i - \bar{x})^2 + \frac{\kappa_0 n}{\kappa_0 + n} (\bar{x} - m_0)^2 \right).

The predictive distribution is as follows:

* :math:`x_{n+1} \in \mathbb{R}`: a new data point
* :math:`\mu_\mathrm{p} \in \mathbb{R}`: the hyperparameter of the predictive distribution
* :math:`\lambda_\mathrm{p} \in \mathbb{R}_{>0}`: the hyperparameter of the predictive distribution
* :math:`\nu_\mathrm{p} \in \mathbb{R}_{>0}`: the hyperparameter of the predictive distribution

.. math::
    p(x_{n+1} | x^{n} ) &= \mathrm{St}(x_{n+1} | \mu_\mathrm{p}, \lambda_\mathrm{p}, \nu_\mathrm{p}) \\
    &= \frac{\Gamma (\nu_\mathrm{p} / 2 + 1/2 )}{\Gamma (\nu_\mathrm{p} / 2)} \left( \frac{\lambda_\mathrm{p}}{\pi \nu_\mathrm{p}} \right)^{1/2} \left( 1 + \frac{\lambda_\mathrm{p} (x_{n+1} - \mu_\mathrm{p})^2}{\nu_\mathrm{p}} \right)^{-\nu_\mathrm{p}/2 - 1/2},

.. math::
    \mathbb{E}[x_{n+1} | x^n] &= \mu_\mathrm{p} & (\nu_\mathrm{p} > 1), \\
    \mathbb{V}[x_{n+1} | x^n] &= \frac{1}{\lambda_\mathrm{p}} \frac{\nu_\mathrm{p}}{\nu_\mathrm{p}-2} & (\nu_\mathrm{p} > 2),


where the parameters are obtained from the hyperparameters of the posterior distribution as follows:

.. math::
    \mu_\mathrm{p} &= \mu_n, \\
    \lambda_\mathrm{p} &= \frac{\kappa_n}{\kappa_n + 1} \frac{\alpha_n}{\beta_n}, \\
    \nu_\mathrm{p} &= 2 \alpha_n.
"""
from ._normal import GenModel
from ._normal import LearnModel

__all__ = ["GenModel","LearnModel"]

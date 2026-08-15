# Document Author
# Kohei Horinouchi <horinouchi@aoni.waseda.jp>
r"""
This module provides the sparse linear regression model with the Laplace prior distribution (represented as a Gaussian scale mixture).

.. image:: ./images/sparselinearregression_example.png

Stochastic Data Generative Model
--------------------------------

The stochastic data generative model is as follows:

* :math:`d \in \mathbb{N}`: a dimension of explanatory variables
* :math:`\boldsymbol{x} \in \mathbb{R}^d`: an explanatory variable. If you consider an intercept term, it should be included as one of the elements of :math:`\boldsymbol{x}`.
* :math:`y \in \mathbb{R}`: an objective variable
* :math:`\boldsymbol{\theta} \in \mathbb{R}^d`: a regression coefficient vector
* :math:`\tau \in \mathbb{R}_{>0}`: a precision parameter
* :math:`\boldsymbol{v} = (v_1, \ldots, v_d)^\top \in \mathbb{R}_{>0}^d`: latent scale variables

.. math::
    p(y | \boldsymbol{x}, \boldsymbol{\theta}, \tau)
    &= \mathcal{N}(y | \boldsymbol{x}^{\top}\boldsymbol{\theta}, \tau^{-1}) \\
    &= \sqrt{\frac{\tau}{2\pi}}
    \exp\left\{
        -\frac{\tau}{2}(y-\boldsymbol{x}^{\top}\boldsymbol{\theta})^2
    \right\}.

.. math::
    \mathbb{E}[y | \boldsymbol{x}, \boldsymbol{\theta}, \tau] &= \boldsymbol{x}^{\top}\boldsymbol{\theta}, \\
    \mathbb{V}[y | \boldsymbol{x}, \boldsymbol{\theta}, \tau] &= \tau^{-1}.

Prior Distribution
------------------

The prior distribution is as follows:

* :math:`\boldsymbol{\lambda}_0 = (\lambda_{0,1}, \ldots, \lambda_{0,d})^\top \in \mathbb{R}_{>0}^d`: a hyperparameter vector (Laplace regularization strength for each dimension)
* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\Gamma(\cdot)`: the gamma function

.. math::
    p(\boldsymbol{\theta} | \tau, \boldsymbol{v})
    &= \mathcal{N}(\boldsymbol{\theta} | \boldsymbol{0}, \tau^{-1} \mathrm{diag}(\boldsymbol{v})) \\
    &= \left(\frac{\tau}{2\pi}\right)^{d/2}
    \left(\prod_{j=1}^d v_j\right)^{-1/2}
    \exp\left\{
        -\frac{\tau}{2}\boldsymbol{\theta}^\top
        \mathrm{diag}(\boldsymbol{v})^{-1}
        \boldsymbol{\theta}
    \right\}, \\
    p(\boldsymbol{v} | \boldsymbol{\lambda}_0)
    &= \prod_{j=1}^d \mathrm{Exp}\left(v_j \middle| \frac{\lambda_{0,j}^2}{2}\right)
    = \prod_{j=1}^d \frac{\lambda_{0,j}^2}{2}\exp\left(-\frac{\lambda_{0,j}^2}{2}v_j\right), \\
    p(\tau | \alpha_0, \beta_0)
    &= \mathrm{Gam}(\tau | \alpha_0, \beta_0)
    = \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)}\tau^{\alpha_0-1}\exp(-\beta_0\tau).

Marginalizing out each :math:`v_j` yields the Laplace prior on :math:`\theta_j`:

.. math::
    p(\theta_j | \tau, \lambda_{0,j})
    &= \int_0^\infty p(\theta_j | \tau, v_j) p(v_j | \lambda_{0,j})\, dv_j \\
    &= \frac{\lambda_{0,j} \sqrt{\tau}}{2}
    \exp\left(-\lambda_{0,j} \sqrt{\tau}\, |\theta_j|\right).

Posterior Distribution
----------------------

The approximate posterior distribution in the :math:`t`-th iteration of a variational Bayesian method is as follows:

* :math:`\boldsymbol{X} = (\boldsymbol{x}_1, \ldots, \boldsymbol{x}_n)^\top \in \mathbb{R}^{n \times d}`: given explanatory variables
* :math:`\boldsymbol{y} = (y_1, \ldots, y_n)^\top \in \mathbb{R}^n`: given objective variables
* :math:`\boldsymbol{\mu}_n^{(t)} \in \mathbb{R}^d`: a hyperparameter
* :math:`\boldsymbol{\Lambda}_n^{(t)} \in \mathbb{R}^{d \times d}`: a hyperparameter (a positive definite matrix)
* :math:`\alpha_n^{(t)} \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n^{(t)} \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`a_{n,j}^{(t)} \in \mathbb{R}_{>0}`: a hyperparameter for :math:`q(v_j)`
* :math:`b_{n,j}^{(t)} \in \mathbb{R}_{>0}`: a hyperparameter for :math:`q(v_j)`
* :math:`K_{\nu}(\cdot)`: the modified Bessel function of the second kind

.. math::
    &q(\boldsymbol{\theta}, \tau, \boldsymbol{v}) \nonumber \\
    &= \mathcal{N}\left(
        \boldsymbol{\theta} \middle|
        \boldsymbol{\mu}_n^{(t)},
        (\tau \boldsymbol{\Lambda}_n^{(t)})^{-1}
    \right)
    \mathrm{Gam}(\tau | \alpha_n^{(t)}, \beta_n^{(t)})  \prod_{j=1}^d \mathrm{GIG}\left(
        v_j \middle| 1/2,\, a_{n,j}^{(t)},\, b_{n,j}^{(t)}
    \right) \\
    &= \frac{|\tau \boldsymbol{\Lambda}_n^{(t)}|^{1/2}}{(2\pi)^{d/2}} \exp \left\{ -\frac{\tau}{2}(\boldsymbol{\theta} -\boldsymbol{\mu}_n^{(t)})^\top \boldsymbol{\Lambda}_n^{(t)} (\boldsymbol{\theta} - \boldsymbol{\mu}_n^{(t)}) \right\} \\
    &\qquad \times \frac{(\beta_n^{(t)})^{\alpha_n^{(t)}}}{\Gamma(\alpha_n^{(t)})}\tau^{\alpha_n^{(t)}-1}\exp\{-\beta_n^{(t)}\tau\} \nonumber \\
    &\qquad \times \prod_{j=1}^d \frac{(a_{n,j}^{(t)}/b_{n,j}^{(t)})^{1/4}}{2K_{1/2}\left(\sqrt{a_{n,j}^{(t)} b_{n,j}^{(t)}}\right)} v_j^{-1/2} \exp\left\{-\frac{1}{2}(a_{n,j}^{(t)} v_j + b_{n,j}^{(t)} v_j^{-1})\right\}

where :math:`\mathrm{GIG}(x | p, a, b) = \frac{(a/b)^{p/2}}{2K_p(\sqrt{ab})} x^{p-1} \exp\left\{-\frac{1}{2}(ax + bx^{-1})\right\}`.

The updating rule of the hyperparameters is as follows.

.. math::
    \boldsymbol{\Lambda}_n^{(t+1)}
    &= \boldsymbol{X}^\top \boldsymbol{X}
    + \mathrm{diag}\left(
        \sqrt{\frac{a_{n,1}^{(t)}}{b_{n,1}^{(t)}}},
        \ldots,
        \sqrt{\frac{a_{n,d}^{(t)}}{b_{n,d}^{(t)}}}
    \right), \\
    \boldsymbol{\mu}_n^{(t+1)}
    &= (\boldsymbol{\Lambda}_n^{(t+1)})^{-1} \boldsymbol{X}^\top \boldsymbol{y}, \\
    \alpha_n^{(t+1)}
    &= \alpha_0 + \frac{n}{2}, \\
    \beta_n^{(t+1)}
    &= \beta_0 + \frac{1}{2}
    \left(
        \boldsymbol{y}^\top \boldsymbol{y}
        - (\boldsymbol{\mu}_n^{(t+1)})^\top \boldsymbol{\Lambda}_n^{(t+1)} \boldsymbol{\mu}_n^{(t+1)}
    \right), \\
    a_{n,j}^{(t+1)}
    &= \lambda_{0,j}^2, \\
    b_{n,j}^{(t+1)}
    &= \frac{\alpha_n^{(t+1)}}{\beta_n^{(t+1)}} (\mu_{n,j}^{(t+1)})^2
    + \left((\boldsymbol{\Lambda}_n^{(t+1)})^{-1}\right)_{jj}.

Accordingly,

.. math::
    \mathbb{E}[\boldsymbol{\theta} | \boldsymbol{X}, \boldsymbol{y}]
    &\approx \boldsymbol{\mu}_n, \\
    \mathbb{E}[\tau | \boldsymbol{X}, \boldsymbol{y}]
    &\approx \frac{\alpha_n}{\beta_n}.

Predictive Distribution
-----------------------

The approximate predictive distribution is as follows:

* :math:`\boldsymbol{x}_{n+1} \in \mathbb{R}^d`: a new explanatory variable
* :math:`y_{n+1} \in \mathbb{R}`: a new objective variable
* :math:`m_{\mathrm{p}} \in \mathbb{R}`: a parameter of the predictive distribution
* :math:`\lambda_{\mathrm{p}} \in \mathbb{R}_{>0}`: a parameter of the predictive distribution
* :math:`\nu_{\mathrm{p}} \in \mathbb{R}_{>0}`: a parameter of the predictive distribution

.. math::
    &p(y_{n+1} | \boldsymbol{X}, \boldsymbol{y}, \boldsymbol{x}_{n+1}) \nonumber \\
    &\approx \mathrm{St}\left(y_{n+1} | m_{\mathrm{p}}, \lambda_{\mathrm{p}}, \nu_{\mathrm{p}}\right) \nonumber \\
    &= \frac{\Gamma (\nu_\mathrm{p} / 2 + 1/2 )}{\Gamma (\nu_\mathrm{p} / 2)} \left( \frac{\lambda_\mathrm{p}}{\pi \nu_\mathrm{p}} \right)^{1/2} \left( 1 + \frac{\lambda_\mathrm{p} (y_{n+1} - m_\mathrm{p})^2}{\nu_\mathrm{p}} \right)^{-\nu_\mathrm{p}/2 - 1/2},

.. math::
    \mathbb{E}[y_{n+1} | \boldsymbol{X}, \boldsymbol{y}, \boldsymbol{x}_{n+1}] &\approx m_{\mathrm{p}} \quad (\nu_{\mathrm{p}} > 1), \\
    \mathbb{V}[y_{n+1} | \boldsymbol{X}, \boldsymbol{y}, \boldsymbol{x}_{n+1}] &\approx \frac{1}{\lambda_{\mathrm{p}}} \frac{\nu_{\mathrm{p}}}{\nu_{\mathrm{p}} - 2} \quad (\nu_{\mathrm{p}} > 2),

where the parameters are obtained from the hyperparameters of the posterior distribution as follows:

.. math::
    m_{\mathrm{p}} &= \boldsymbol{x}_{n+1}^{\top} \boldsymbol{\mu}_n, \\
    \lambda_{\mathrm{p}} &= \frac{\alpha_n}{\beta_n}\left(1 + \boldsymbol{x}_{n+1}^{\top} \boldsymbol{\Lambda}_n^{-1} \boldsymbol{x}_{n+1}\right)^{-1}, \\
    \nu_{\mathrm{p}} &= 2\alpha_n.

Star Us on GitHub
-----------------

.. include:: _star.rst

Class and Methods
-----------------
"""

from ._sparselinearregression import GenModel
from ._sparselinearregression import LearnModel

__all__ = ["GenModel", "LearnModel"]

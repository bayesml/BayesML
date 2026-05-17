# Document Author
# Haruka Murayama <h-murayama@ruri.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
r'''
The mixture of linear regression model with the Gauss-Gamma prior distribution and the Dirichlet prior distribution.

The stochastic data generative model is as follows:

* :math:`K \in \mathbb{N}`: number of latent classes
* :math:`\boldsymbol{z} \in \{ 0, 1 \}^K`: a one-hot vector representing the latent class (latent variable)
* :math:`\boldsymbol{\pi} \in [0, 1]^K`: a parameter for latent classes, (:math:`\sum_{k=1}^K \pi_k=1`)
* :math:`D \in \mathbb{N}`: a dimension of data
* :math:`y\in\mathbb{R}`: an objective variable
* :math:`\boldsymbol{x} \in \mathbb{R}^D`: a data point
* :math:`\boldsymbol{\theta}_k\in\mathbb{R}^{D}`: a parameter
* :math:`\boldsymbol{\theta} = \{ \boldsymbol{\theta}_k \}_{k=1}^K`
* :math:`\tau_k \in \mathbb{R}_{>0}` : a parameter 
* :math:`\boldsymbol{\tau} = \{ \tau_k \}_{k=1}^K`

.. math::
    p(\boldsymbol{z} | \boldsymbol{\pi}) &= \mathrm{Cat}(\boldsymbol{z}|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_k},\\
    p(y | \boldsymbol{x}, \boldsymbol{\theta}, \boldsymbol{\tau}, \boldsymbol{z}) &= \prod_{k=1}^K \mathcal{N}(y | \boldsymbol{\theta}^\top_k \boldsymbol{x},\tau_k^{-1})^{z_k} \\
    &= \prod_{k=1}^K \left( \sqrt{\frac{\tau_k}{2\pi}} \exp \left\{ -\frac{\tau_k}{2}(y - \boldsymbol{\theta}^\top_k\boldsymbol{x})^2 \right\} \right)^{z_k}.

The prior distribution is as follows:

* :math:`\boldsymbol{\mu}_0 \in \mathbb{R}^{D}`: a hyperparameter
* :math:`\boldsymbol{\Lambda}_0 \in \mathbb{R}^{D\times D}`: a hyperparameter (a positive definite matrix)
* :math:`\alpha_0 \in \mathbb{R}_{> 0}`: a hyperparameter
* :math:`\beta_0\in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\boldsymbol{\gamma}_0 \in \mathbb{R}_{>0}^K`: a hyper parameter
* :math:`\Gamma (\cdot)`: the gamma function

.. math::
    p(\boldsymbol{\theta},\boldsymbol{\tau},\boldsymbol{\pi}) &= \left\{ \prod_{k=1}^K \mathcal{N}(\boldsymbol{\theta}_k|\boldsymbol{\mu}_0,(\tau_k \boldsymbol{\Lambda}_0)^{-1})\mathrm{Gam}(\tau_k|\alpha_0, \beta_0) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\gamma}_0) \\
    &= \Biggl[ \prod_{k=1}^K  \frac{|\tau_k \boldsymbol{\Lambda}_0|^{1/2}}{(2\pi)^{d/2}} \exp \left\{ -\frac{\tau_k}{2}(\boldsymbol{\theta}_k -\boldsymbol{\mu}_0)^\top \boldsymbol{\Lambda}_0 (\boldsymbol{\theta}_k - \boldsymbol{\mu}_0) \right\} \\
    &\qquad \times \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)}\tau_k^{\alpha_0-1}\exp\{-\beta_0\tau_k\} \Biggl] C(\boldsymbol{\gamma}_0)\prod_{k=1}^K \pi_k^{\gamma_{0,k}-1},\\

where :math:`C(\boldsymbol{\gamma}_0)` are defined as follows:

.. math::
    C(\boldsymbol{\gamma}_0) = \frac{\Gamma(\sum_{k=1}^K \gamma_{0,k})}{\Gamma(\gamma_{0,1})\cdots\Gamma(\gamma_{0,K})}.

The apporoximate posterior distribution in the :math:`t`-th iteration of a variational Bayesian method is as follows:

* :math:`\boldsymbol{X} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n]^\top \in \mathbb{R}^{n \times D}`: given explanatory variables
* :math:`\boldsymbol{z}^n = (\boldsymbol{z}_1, \boldsymbol{z}_2, \dots , \boldsymbol{z}_n) \in \{ 0, 1 \}^{K \times n}`: latent classes of given data
* :math:`\boldsymbol{r}_i^{(t)} = (r_{i,1}^{(t)}, r_{i,2}^{(t)}, \dots , r_{i,K}^{(t)}) \in [0,1]^K`: a parameter for :math:`i`-th latent class (:math:`\sum_{k=1}^K r_{i,k}^{(t)} = 1`)
* :math:`\boldsymbol{y} = [y_1, y_2, \dots , y_n]^\top \in \mathbb{R}^n`: given objective variables
* :math:`\boldsymbol{\mu}_{n,k}^{(t)} \in \mathbb{R}^{D}`: a hyperparameter
* :math:`\boldsymbol{\Lambda}_{n,k}^{(t)} \in \mathbb{R}^{D\times D}`: a hyperparameter (a positive definite matrix)
* :math:`\alpha_{n,k}^{(t)} \in \mathbb{R}_{> 0}`: a hyperparameter
* :math:`\beta_{n,k}^{(t)} \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\boldsymbol{\gamma}_n^{(t)} \in \mathbb{R}_{>0}^K`: a hyper parameter
* :math:`\psi (\cdot)`: the digamma function

.. math::
    &q(\boldsymbol{z}^n, \boldsymbol{\theta},\boldsymbol{\tau},\boldsymbol{\pi}) \nonumber \\
    &= \left\{ \prod_{i=1}^n \mathrm{Cat}(\boldsymbol{z}_i|\boldsymbol{r}_i^{(t)}) \right\} \left\{ \prod_{k=1}^K \mathcal{N}(\boldsymbol{\theta}_k|\boldsymbol{\mu}_{n,k}^{(t)},(\tau_k \boldsymbol{\Lambda}_{n,k}^{(t)})^{-1})\mathrm{Gam}(\tau_k|\alpha_{n,k}^{(t)}, \beta_{n,k}^{(t)}) \right\} \\
    &\qquad \times \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\gamma}_{n,k}^{(t)}) \\
    &= \Biggl[ \prod_{i=1}^n \prod_{k=1}^K (r_{i,k}^{(t)})^{z_{i,k}} \Biggr] \Biggl[ \prod_{k=1}^K  \frac{|\tau_k \boldsymbol{\Lambda}_{n,k}^{(t)}|^{1/2}}{(2\pi)^{d/2}} \exp \left\{ -\frac{\tau_k}{2}(\boldsymbol{\theta}_k -\boldsymbol{\mu}_{n,k}^{(t)})^\top \boldsymbol{\Lambda}_{n,k}^{(t)} (\boldsymbol{\theta}_k - \boldsymbol{\mu}_{n,k}^{(t)}) \right\} \\
    &\qquad \times \frac{(\beta_{n,k}^{(t)})^{\alpha_{n,k}^{(t)}}}{\Gamma(\alpha_{n,k}^{(t)})}\tau_k^{\alpha_{n,k}^{(t)}-1}\exp\{-\beta_{n,k}^{(t)}\tau_k\} \Biggl] C(\boldsymbol{\gamma}_n)\prod_{k=1}^K \pi_k^{\gamma_{n,k}-1},\\

where the updating rules of the hyperparameters are as follows:

.. math::
    N_{k}^{(t)} &= \sum_{i=1}^{n} r_{i,k}^{(t)}, \\
    \boldsymbol{R}_k^{(t)} &= \mathrm{diag} (r_{1,k}^{(t)}, r_{2,k}^{(t)}, \dots , r_{n,k}^{(t)}), \\
    \boldsymbol{\Lambda}_{n,k}^{(t+1)} &= \boldsymbol{\Lambda}_0 + \boldsymbol{X}^\top \boldsymbol{R}_k^{(t)} \boldsymbol{X}, \\
    \boldsymbol{\mu}_{n,k}^{(t+1)} &= \left( \boldsymbol{\Lambda}_{n,k}^{(t+1)} \right)^{-1} \left( \boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 + \boldsymbol{X}^\top \boldsymbol{R}_k^{(t)} \boldsymbol{y} \right), \\
    a_{n,k}^{(t+1)} &= a_0 + \frac{1}{2} N_k^{(t)}, \\
    b_{n,k}^{(t+1)} &= b_0 + \frac{1}{2} \left( -(\boldsymbol{\mu}_{n,k}^{(t+1)})^\top \boldsymbol{\Lambda}_{n,k}^{(t+1)} \boldsymbol{\mu}_{n,k}^{(t+1)} + \boldsymbol{y}^\top \boldsymbol{R}_k^{(t)} \boldsymbol{y} + \boldsymbol{\mu}_0^\top \boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 \right), \\
    \gamma_{n,k}^{(t+1)} &= \gamma_0 + N_{k}^{(t)}, \\
    \ln \rho_{i,k}^{(t)} &= \psi (\gamma_{n,k}^{(t+1)}) - \psi \left( {\textstyle \sum_{k=1}^K \gamma_{n,k}^{(t+1)}} \right) \nonumber \\
    &\qquad - \frac{1}{2} \ln (2 \pi) - \frac{1}{2} \left( \psi (\alpha_{n,k}^{(t+1)}) - \ln \beta_{n,k}^{(t+1)} \right) \nonumber \\
    &\qquad -\frac{1}{2} \left( \frac{\alpha_{n,k}^{(t+1)}}{\beta_{n,k}^{(t+1)}} \left(y_i - (\boldsymbol{\mu}_{n,k}^{(t+1)})^\top \boldsymbol{x}_i \right)^2 + \boldsymbol{x}_i^\top \boldsymbol{\Lambda}_{n,k}^{(t+1)} \boldsymbol{x}_i \right), \\
    r_{i,k}^{(t+1)} &= \frac{\rho_{i,k}^{(t+1)}}{\sum_{k=1}^K \rho_{i,k}^{(t+1)}}.

The predictive distribution is as follows:

* :math:`\boldsymbol{x}_{n+1}\in \mathbb{R}^D`: a new data point
* :math:`y_{n+1}\in \mathbb{R}`: a new objective variable
* :math:`m_{\mathrm{p},k}\in \mathbb{R}`: a parameter
* :math:`\lambda_{\mathrm{p},k}\in \mathbb{R}_{>0}`: a parameter
* :math:`\nu_{\mathrm{p},k}\in \mathbb{R}_{>0}`: a parameter

.. math::
    &p(y_{n+1} | \boldsymbol{X}, \boldsymbol{y}, \boldsymbol{x}_{n+1} ) \nonumber \\
    &= \frac{1}{\sum_{k+1}^K \gamma_{n,k}^{(t)}} \sum_{k=1}^K \gamma_{n,k}^{(t)} \mathrm{St}\left(y_{n+1} \mid m_{\mathrm{p},k}, \lambda_{\mathrm{p},k}, \nu_{\mathrm{p},k}\right) \\
    &= \frac{1}{\sum_{k+1}^K \gamma_{n,k}^{(t)}} \sum_{k=1}^K \gamma_{n,k}^{(t)} \frac{\Gamma (\nu_{\mathrm{p},k} / 2 + 1/2 )}{\Gamma (\nu_{\mathrm{p},k} / 2)} \left( \frac{\lambda_{\mathrm{p},k}}{\pi \nu_{\mathrm{p},k}} \right)^{1/2} \left( 1 + \frac{\lambda_{\mathrm{p},k} (y_{n+1} - m_{\mathrm{p},k})^2}{\nu_{\mathrm{p},k}} \right)^{-\nu_{\mathrm{p},k}/2 - 1/2},

where the parameters are obtained from the hyperparameters of the posterior distribution as follows.

.. math::
    m_{\mathrm{p},k} &= \boldsymbol{x}_{n+1}^{\top} \boldsymbol{\mu}_{n,k}^{(t)}, \\
    \lambda_{\mathrm{p},k} &= \frac{\alpha_{n,k}^{(t)}}{\beta_{n,k}^{(t)}}\left(1+\boldsymbol{x}_{n+1}^{\top} \boldsymbol{\Lambda}_{n,k}^{(t)} \boldsymbol{x}_{n+1}\right)^{-1}, \\
    \nu_{\mathrm{p},k} &= 2 \alpha_{n,k}^{(t)}.
'''
from ._linearregressionmixture import GenModel
from ._linearregressionmixture import LearnModel

__all__ = ["GenModel","LearnModel"]
# Document Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
r"""
The Gaussian mixture model with the Gauss-Wishart prior distribution and the Dirichlet prior distribution.

The stochastic data generative model is as follows:

* :math:`K \in \mathbb{N}`: number of latent classes
* :math:`\boldsymbol{z} \in \{ 0, 1 \}^K`: a one-hot vector representing the latent class (latent variable)
* :math:`\boldsymbol{\pi} \in [0, 1]^K`: a parameter for latent classes, (:math:`\sum_{k=1}^K \pi_k=1`)
* :math:`D \in \mathbb{N}`: a dimension of data
* :math:`\boldsymbol{x} \in \mathbb{R}^D`: a data point
* :math:`\boldsymbol{\mu}_k \in \mathbb{R}^D`: a parameter
* :math:`\boldsymbol{\mu} = \{ \boldsymbol{\mu}_k \}_{k=1}^K`
* :math:`\boldsymbol{\Lambda}_k \in \mathbb{R}^{D\times D}` : a parameter (a positive definite matrix)
* :math:`\boldsymbol{\Lambda} = \{ \boldsymbol{\Lambda}_k \}_{k=1}^K`
* :math:`| \boldsymbol{\Lambda}_k | \in \mathbb{R}`: the determinant of :math:`\boldsymbol{\Lambda}_k`

.. math::
    p(\boldsymbol{z} | \boldsymbol{\pi}) &= \mathrm{Cat}(\boldsymbol{z}|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_k},\\
    p(\boldsymbol{x} | \boldsymbol{\mu}, \boldsymbol{\Lambda}, \boldsymbol{z}) &= \prod_{k=1}^K \mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k^{-1})^{z_k} \\
    &= \prod_{k=1}^K \left( \frac{| \boldsymbol{\Lambda}_k |^{1/2}}{(2\pi)^{D/2}} \exp \left\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu}_k)^\top \boldsymbol{\Lambda}_k (\boldsymbol{x}-\boldsymbol{\mu}_k) \right\} \right)^{z_k}.

The prior distribution is as follows:

* :math:`\boldsymbol{m}_0 \in \mathbb{R}^{D}`: a hyperparameter
* :math:`\kappa_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\nu_0 \in \mathbb{R}`: a hyperparameter (:math:`\nu_0 > D-1`)
* :math:`\boldsymbol{W}_0 \in \mathbb{R}^{D\times D}`: a hyperparameter (a positive definite matrix)
* :math:`\boldsymbol{\alpha}_0 \in \mathbb{R}_{> 0}^K`: a hyperparameter
* :math:`\mathrm{Tr} \{ \cdot \}`: a trace of a matrix
* :math:`\Gamma (\cdot)`: the gamma function

.. math::
    p(\boldsymbol{\mu},\boldsymbol{\Lambda},\boldsymbol{\pi}) &= \left\{ \prod_{k=1}^K \mathcal{N}(\boldsymbol{\mu}_k|\boldsymbol{m}_0,(\kappa_0 \boldsymbol{\Lambda}_k)^{-1})\mathcal{W}(\boldsymbol{\Lambda}_k|\boldsymbol{W}_0, \nu_0) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}_0) \\
    &= \Biggl[\, \prod_{k=1}^K \left( \frac{\kappa_0}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}_k|^{1/2} \exp \left\{ -\frac{\kappa_0}{2}(\boldsymbol{\mu}_k -\boldsymbol{m}_0)^\top \boldsymbol{\Lambda}_k (\boldsymbol{\mu}_k - \boldsymbol{m}_0) \right\} \notag \\
    &\qquad \times B(\boldsymbol{W}_0, \nu_0) | \boldsymbol{\Lambda}_k |^{(\nu_0 - D - 1) / 2} \exp \left\{ -\frac{1}{2} \mathrm{Tr} \{ \boldsymbol{W}_0^{-1} \boldsymbol{\Lambda}_k \} \right\} \Biggr] \notag \\
    &\qquad \times C(\boldsymbol{\alpha}_0)\prod_{k=1}^K \pi_k^{\alpha_{0,k}-1},\\

where :math:`B(\boldsymbol{W}_0, \nu_0)` and :math:`C(\boldsymbol{\alpha}_0)` are defined as follows:

.. math::
    B(\boldsymbol{W}_0, \nu_0) &= | \boldsymbol{W}_0 |^{-\nu_0 / 2} \left( 2^{\nu_0 D / 2} \pi^{D(D-1)/4} \prod_{i=1}^D \Gamma \left( \frac{\nu_0 + 1 - i}{2} \right) \right)^{-1}, \\
    C(\boldsymbol{\alpha}_0) &= \frac{\Gamma(\sum_{k=1}^K \alpha_{0,k})}{\Gamma(\alpha_{0,1})\cdots\Gamma(\alpha_{0,K})}.

The apporoximate posterior distribution in the :math:`t`-th iteration of a variational Bayesian method is as follows:

* :math:`\boldsymbol{x}^n = (\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n) \in \mathbb{R}^{D \times n}`: given data
* :math:`\boldsymbol{z}^n = (\boldsymbol{z}_1, \boldsymbol{z}_2, \dots , \boldsymbol{z}_n) \in \{ 0, 1 \}^{K \times n}`: latent classes of given data
* :math:`\boldsymbol{r}_i^{(t)} = (r_{i,1}^{(t)}, r_{i,2}^{(t)}, \dots , r_{i,K}^{(t)}) \in [0, 1]^K`: a parameter for the :math:`i`-th latent class. (:math:`\sum_{k=1}^K r_{i, k}^{(t)} = 1`)
* :math:`\boldsymbol{m}_{n,k}^{(t)} \in \mathbb{R}^{D}`: a hyperparameter
* :math:`\kappa_{n,k}^{(t)} \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\nu_{n,k}^{(t)} \in \mathbb{R}`: a hyperparameter :math:`(\nu_n > D-1)`
* :math:`\boldsymbol{W}_{n,k}^{(t)} \in \mathbb{R}^{D\times D}`: a hyperparameter (a positive definite matrix)
* :math:`\boldsymbol{\alpha}_n^{(t)} \in \mathbb{R}_{> 0}^K`: a hyperparameter

.. math::
    q(\boldsymbol{z}^n, \boldsymbol{\mu},\boldsymbol{\Lambda},\boldsymbol{\pi}) &= \left\{ \prod_{i=1}^n \mathrm{Cat} (\boldsymbol{z}_i | \boldsymbol{r}_i^{(t)}) \right\} \left\{ \prod_{k=1}^K \mathcal{N}(\boldsymbol{\mu}_k|\boldsymbol{m}_{n,k}^{(t)},(\kappa_{n,k}^{(t)} \boldsymbol{\Lambda}_k)^{-1})\mathcal{W}(\boldsymbol{\Lambda}_k|\boldsymbol{W}_{n,k}^{(t)}, \nu_{n,k}^{(t)}) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\alpha}_n^{(t)}) \\
    &= \Biggl[\, \prod_{i=1}^n \prod_{k=1}^K (r_{i,k}^{(t)})^{z_{i,k}} \Biggr] \Biggl[\, \prod_{k=1}^K \left( \frac{\kappa_{n,k}^{(t)}}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}_k|^{1/2} \exp \left\{ -\frac{\kappa_{n,k}^{(t)}}{2}(\boldsymbol{\mu}_k -\boldsymbol{m}_{n,k}^{(t)})^\top \boldsymbol{\Lambda}_k (\boldsymbol{\mu}_k - \boldsymbol{m}_{n,k}^{(t)}) \right\} \\
    &\qquad \times B(\boldsymbol{W}_{n,k}^{(t)}, \nu_{n,k}^{(t)}) | \boldsymbol{\Lambda}_k |^{(\nu_{n,k}^{(t)} - D - 1) / 2} \exp \left\{ -\frac{1}{2} \mathrm{Tr} \{ ( \boldsymbol{W}_{n,k}^{(t)} )^{-1} \boldsymbol{\Lambda}_k \} \right\} \Biggr] \\
    &\qquad \times C(\boldsymbol{\alpha}_n^{(t)})\prod_{k=1}^K \pi_k^{\alpha_{n,k}^{(t)}-1},\\

where the updating rule of the hyperparameters is as follows.

.. math::
    N_k^{(t)} &= \sum_{i=1}^n r_{i,k}^{(t)}, \\
    \bar{\boldsymbol{x}}_k^{(t)} &= \frac{1}{N_k^{(t)}} \sum_{i=1}^n r_{i,k}^{(t)} \boldsymbol{x}_i, \\
    \boldsymbol{m}_{n,k}^{(t+1)} &= \frac{\kappa_0\boldsymbol{m}_0 + N_k^{(t)} \bar{\boldsymbol{x}}_k^{(t)}}{\kappa_0 + N_k^{(t)}}, \\
    \kappa_{n,k}^{(t+1)} &= \kappa_0 + N_k^{(t)}, \\
    (\boldsymbol{W}_{n,k}^{(t+1)})^{-1} &= \boldsymbol{W}_0^{-1} + \sum_{i=1}^{n} r_{i,k}^{(t)} (\boldsymbol{x}_i-\bar{\boldsymbol{x}}_k^{(t)})(\boldsymbol{x}_i-\bar{\boldsymbol{x}}_k^{(t)})^\top + \frac{\kappa_0 N_k^{(t)}}{\kappa_0 + N_k^{(t)}}(\bar{\boldsymbol{x}}_k^{(t)}-\boldsymbol{m}_0)(\bar{\boldsymbol{x}}_k^{(t)}-\boldsymbol{m}_0)^\top, \\
    \nu_{n,k}^{(t+1)} &= \nu_0 + N_k^{(t)},\\
    \alpha_{n,k}^{(t+1)} &= \alpha_{0,k} + N_k^{(t)}, \\
    \ln \rho_{i,k}^{(t+1)} &= \psi (\alpha_{n,k}^{(t+1)}) - \psi ( {\textstyle \sum_{k=1}^K \alpha_{n,k}^{(t+1)}} ) \notag \\
    &\qquad + \frac{1}{2} \Biggl[\, \sum_{d=1}^D \psi \left( \frac{\nu_{n,k}^{(t+1)} + 1 - d}{2} \right) + D \ln 2 + \ln | \boldsymbol{W}_{n,k}^{(t+1)} | \notag \\
    &\qquad - D \ln (2 \pi ) - \frac{D}{\kappa_{n,k}^{(t+1)}} - \nu_{n,k}^{(t+1)} (\boldsymbol{x}_i - \boldsymbol{m}_{n,k}^{(t+1)})^\top \boldsymbol{W}_{n,k}^{(t+1)} (\boldsymbol{x}_i - \boldsymbol{m}_{n,k}^{(t+1)}) \Biggr], \\
    r_{i,k}^{(t+1)} &= \frac{\rho_{i,k}^{(t+1)}}{\sum_{j=1}^K \rho_{i,j}^{(t+1)}}.

The approximate predictive distribution is as follows:

* :math:`\boldsymbol{x}_{n+1} \in \mathbb{R}^D`: a new data point
* :math:`\boldsymbol{\mu}_{\mathrm{p},k} \in \mathbb{R}^D`: the parameter of the predictive distribution
* :math:`\boldsymbol{\Lambda}_{\mathrm{p},k} \in \mathbb{R}^{D \times D}`: the parameter of the predictive distribution (a positive definite matrix)
* :math:`\nu_{\mathrm{p},k} \in \mathbb{R}_{>0}`: the parameter of the predictive distribution

.. math::
    &p(x_{n+1}|x^n) \\
    &= \frac{1}{\sum_{k=1}^K \alpha_{n,k}^{(t)}} \sum_{k=1}^K \alpha_{n,k}^{(t)} \mathrm{St}(x_{n+1}|\boldsymbol{\mu}_{\mathrm{p},k},\boldsymbol{\Lambda}_{\mathrm{p},k}, \nu_{\mathrm{p},k}) \\
    &= \frac{1}{\sum_{k=1}^K \alpha_{n,k}^{(t)}} \sum_{k=1}^K \alpha_{n,k}^{(t)} \Biggl[ \frac{\Gamma (\nu_{\mathrm{p},k} / 2 + D / 2)}{\Gamma (\nu_{\mathrm{p},k} / 2)} \frac{|\boldsymbol{\Lambda}_{\mathrm{p},k}|^{1/2}}{(\nu_{\mathrm{p},k} \pi)^{D/2}} \notag \\
    &\qquad \qquad \qquad \qquad \qquad \times \left( 1 + \frac{1}{\nu_{\mathrm{p},k}} (\boldsymbol{x}_{n+1} - \boldsymbol{\mu}_{\mathrm{p},k})^\top \boldsymbol{\Lambda}_{\mathrm{p},k} (\boldsymbol{x}_{n+1} - \boldsymbol{\mu}_{\mathrm{p},k}) \right)^{-\nu_{\mathrm{p},k}/2 - D/2} \Biggr],

where the parameters are obtained from the hyperparameters of the posterior distribution as follows:

.. math::
    \boldsymbol{\mu}_{\mathrm{p},k} &= \boldsymbol{m}_{n,k}^{(t)}, \\
    \nu_{\mathrm{p},k} &= \nu_{n,k}^{(t)} - D + 1,\\
    \boldsymbol{\Lambda}_{\mathrm{p},k} &= \frac{\kappa_{n,k}^{(t)} \nu_{\mathrm{p},k}}{\kappa_{n,k}^{(t)} + 1} \boldsymbol{W}_{n,k}^{(t)}.
"""

from ._gaussianmixture import GenModel
from ._gaussianmixture import LearnModel

__all__ = ["GenModel","LearnModel"]
# Document Author
# Ryohei Oka <o.ryohei07@gmail.com>
r"""
The hidden Markov model with the Gauss-Wishart prior distribution and the Dirichlet prior distribution.

The stochastic data generative model is as follows:

* :math:`K \in \mathbb{N}`: number of latent classes
* :math:`\boldsymbol{z} \in \{ 0, 1 \}^K`: a one-hot vector representing the latent class (latent variable)
* :math:`\boldsymbol{\pi} \in [0, 1]^K`: a parameter for latent classes, (:math:`\sum_{k=1}^K \pi_k=1`)
* :math:`a_{j,k} \in [0,1]` : transition probability to latent state k under latent state j
* :math:`\boldsymbol{a}_j = [a_{j,1}, a_{j,2}, \dots , a_{j,K}]\in [0,1]^K`, a vector of the transition probability (:math:`\sum_{k=1}^K a_{j,k}=1`)
* :math:`\boldsymbol{A}=(a_{j,k})_{1\leq j,k\leq K} \in [0, 1]^{K\times K}`: a matrix of the transition probability
* :math:`D \in \mathbb{N}`: a dimension of data
* :math:`\boldsymbol{x} \in \mathbb{R}^D`: a data point
* :math:`\boldsymbol{\mu}_k \in \mathbb{R}^D`: a parameter
* :math:`\boldsymbol{\mu} = \{ \boldsymbol{\mu}_k \}_{k=1}^K`
* :math:`\boldsymbol{\Lambda}_k \in \mathbb{R}^{D\times D}` : a parameter (a positive definite matrix)
* :math:`\boldsymbol{\Lambda} = \{ \boldsymbol{\Lambda}_k \}_{k=1}^K`
* :math:`| \boldsymbol{\Lambda}_k | \in \mathbb{R}`: the determinant of :math:`\boldsymbol{\Lambda}_k`

.. math::
    p(\boldsymbol{z}_{1} | \boldsymbol{\pi}) &= \mathrm{Cat}(\boldsymbol{z}_{1}|\boldsymbol{\pi}) = \prod_{k=1}^K \pi_k^{z_{1,k}},\\
    p(\boldsymbol{z}_{n} |\boldsymbol{z}_{n-1} ,\boldsymbol{A}) &= \prod_{k=1}^K \prod_{j=1}^K a_{j,k}^{z_{n-1,j}z_{n,k}},\\
    p(\boldsymbol{x}_{n} | \boldsymbol{\mu}, \boldsymbol{\Lambda}, \boldsymbol{z}_{n}) &= \prod_{k=1}^K \mathcal{N}(\boldsymbol{x}|\boldsymbol{\mu}_k,\boldsymbol{\Lambda}_k^{-1})^{z_{n,k}} \\
    &= \prod_{k=1}^K \left( \frac{| \boldsymbol{\Lambda}_{k} |^{1/2}}{(2\pi)^{D/2}} \exp \left\{ -\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu}_{k})^\top \boldsymbol{\Lambda}_{k} (\boldsymbol{x}-\boldsymbol{\mu}_{k}) \right\} \right)^{z_{n,k}},

The prior distribution is as follows:

* :math:`\boldsymbol{m}_0 \in \mathbb{R}^{D}`: a hyperparameter
* :math:`\kappa_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\nu_0 \in \mathbb{R}`: a hyperparameter (:math:`\nu_0 > D-1`)
* :math:`\boldsymbol{W}_0 \in \mathbb{R}^{D\times D}`: a hyperparameter (a positive definite matrix)
* :math:`\boldsymbol{\eta}_0 \in \mathbb{R}_{> 0}^K`: a hyperparameter
* :math:`\boldsymbol{\zeta}_{0,j} \in \mathbb{R}_{> 0}^K`: a hyperparameter
* :math:`\mathrm{Tr} \{ \cdot \}`: a trace of a matrix
* :math:`\Gamma (\cdot)`: the gamma function

.. math::
    p(\boldsymbol{\mu},\boldsymbol{\Lambda},\boldsymbol{\pi},\boldsymbol{A}) &= \left\{ \prod_{k=1}^K \mathcal{N}(\boldsymbol{\mu}_k|\boldsymbol{m}_0,(\kappa_0 \boldsymbol{\Lambda}_k)^{-1})\mathcal{W}(\boldsymbol{\Lambda}_k|\boldsymbol{W}_0, \nu_0) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\eta}_0) \prod_{j=1}^{K}\mathrm{Dir}(\boldsymbol{a}_{j}|\boldsymbol{\zeta}_{0,j}), \\
    &= \Biggl[ \prod_{k=1}^K \left( \frac{\kappa_0}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}_k|^{1/2} \exp \left\{ -\frac{\kappa_0}{2}(\boldsymbol{\mu}_k -\boldsymbol{m}_0)^\top \boldsymbol{\Lambda}_k (\boldsymbol{\mu}_k - \boldsymbol{m}_0) \right\} \\
    &\qquad \times B(\boldsymbol{W}_0, \nu_0) | \boldsymbol{\Lambda}_k |^{(\nu_0 - D - 1) / 2} \exp \left\{ -\frac{1}{2} \mathrm{Tr} \{ \boldsymbol{W}_0^{-1} \boldsymbol{\Lambda}_k \} \right\}\biggl] \\
    &\qquad \times \Biggl[ \prod_{k=1}^KC(\boldsymbol{\eta}_0)\pi_k^{\eta_{0,k}-1}\biggl]\\
    &\qquad \times \biggl[\prod_{j=1}^KC(\boldsymbol{\zeta}_{0,j})\prod_{k=1}^K a_{j,k}^{\zeta_{0,j,k}-1}\Biggr],\\

where :math:`B(\boldsymbol{W}_0, \nu_0)` and :math:`C(\boldsymbol{\eta}_0)` are defined as follows:

.. math::
    B(\boldsymbol{W}_0, \nu_0) &= | \boldsymbol{W}_0 |^{-\nu_0 / 2} \left( 2^{\nu_0 D / 2} \pi^{D(D-1)/4} \prod_{i=1}^D \Gamma \left( \frac{\nu_0 + 1 - i}{2} \right) \right)^{-1}, \\
    C(\boldsymbol{\eta}_0) &= \frac{\Gamma(\sum_{k=1}^K \eta_{0,k})}{\Gamma(\eta_{0,1})\cdots\Gamma(\eta_{0,K})},\\
    C(\boldsymbol{\zeta}_{0,j}) &= \frac{\Gamma(\sum_{k=1}^K \zeta_{0,j,k})}{\Gamma(\zeta_{0,j,1})\cdots\Gamma(\zeta_{0,j,K})}. 

The apporoximate posterior distribution in the :math:`t`-th iteration of a variational Bayesian method is as follows:

* :math:`\boldsymbol{x}^n = (\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n) \in \mathbb{R}^{D \times n}`: given data
* :math:`\boldsymbol{z}^n = (\boldsymbol{z}_1, \boldsymbol{z}_2, \dots , \boldsymbol{z}_n) \in \{ 0, 1 \}^{K \times n}`: latent classes of given data
* :math:`\boldsymbol{m}_{n,k}^{(t)} \in \mathbb{R}^{D}`: a hyperparameter
* :math:`\kappa_{n,k}^{(t)} \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\nu_{n,k}^{(t)} \in \mathbb{R}`: a hyperparameter :math:`(\nu_n > D-1)`
* :math:`\boldsymbol{W}_{n,k}^{(t)} \in \mathbb{R}^{D\times D}`: a hyperparameter (a positive definite matrix)
* :math:`\boldsymbol{\eta}_n^{(t)} \in \mathbb{R}_{> 0}^K`: a hyperparameter
* :math:`\boldsymbol{\zeta}_{n,j}^{(t)} \in \mathbb{R}_{> 0}^K`: a hyperparameter

.. math::
    &q(\boldsymbol{z}^n, \boldsymbol{\mu},\boldsymbol{\Lambda},\boldsymbol{\pi},\boldsymbol{A}) \nonumber \\
    &= q^{(t)}(\boldsymbol{z}^n) \left\{ \prod_{k=1}^K \mathcal{N}(\boldsymbol{\mu}_k|\boldsymbol{m}_{n,k}^{(t)},(\kappa_{n,k}^{(t)} \boldsymbol{\Lambda}_k)^{-1})\mathcal{W}(\boldsymbol{\Lambda}_k|\boldsymbol{W}_{n,k}^{(t)}, \nu_{n,k}^{(t)}) \right\} \mathrm{Dir}(\boldsymbol{\pi}|\boldsymbol{\eta}_n^{(t)})\left\{\prod_{j=1}^K\mathrm{Dir}(\boldsymbol{a}_j|\boldsymbol{\zeta}_{n,j}^{(t)})\right\}, \\
    &= q^{(t)}(\boldsymbol{z}^n) \Biggl[ \prod_{k=1}^K \left( \frac{\kappa_{n,k}^{(t)}}{2\pi} \right)^{D/2} |\boldsymbol{\Lambda}_k|^{1/2} \exp \left\{ -\frac{\kappa_{n,k}^{(t)}}{2}(\boldsymbol{\mu}_k -\boldsymbol{m}_{n,k}^{(t)})^\top \boldsymbol{\Lambda}_k (\boldsymbol{\mu}_k - \boldsymbol{m}_{n,k}^{(t)}) \right\} \\
    &\qquad \times B(\boldsymbol{W}_{n,k}^{(t)}, \nu_{n,k}^{(t)}) | \boldsymbol{\Lambda}_k |^{(\nu_{n,k}^{(t)} - D - 1) / 2} \exp \left\{ -\frac{1}{2} \mathrm{Tr} \{ ( \boldsymbol{W}_{n,k}^{(t)} )^{-1} \boldsymbol{\Lambda}_k \} \right\} \Biggr] \\
    &\qquad \times C(\boldsymbol{\eta}_n^{(t)})\prod_{k=1}^K \pi_k^{\eta_{n,k}^{(t)}-1}\left[\prod_{j=1}^K C(\boldsymbol{\zeta}_{n,j}^{(t)})\prod_{k=1}^K a_{j,k}^{\zeta_{n,j,k}^{(t)}-1}\right],\\

where the updating rule of the hyperparameters is as follows.

.. math::
    N_k^{(t)} &= \sum_{i=1}^n \gamma^{(t)}_{i,k}, \\
    M_{j,k}^{(t)} &= \sum_{i=2}^n \xi^{(t)}_{i,j,k},\\
    \bar{\boldsymbol{x}}_k^{(t)} &= \frac{1}{N_k^{(t)}} \sum_{i=1}^n \gamma^{(t)}_{i,k} \boldsymbol{x}_i, \\
    S_k^{(t)} &= \frac{1}{N_k^{(t)}}\sum_{i=1}^n \gamma^{(t)}_{i,k} (x_i-\bar{\boldsymbol{x}}_k^{(t)})(x_i-\bar{\boldsymbol{x}}_k^{(t)})^{\top},\\
    \boldsymbol{m}_{n,k}^{(t+1)} &= \frac{\kappa_0\boldsymbol{\mu}_0 + N_k^{(t)} \bar{\boldsymbol{x}}_k^{(t)}}{\kappa_0 + N_k^{(t)}}, \\
    \kappa_{n,k}^{(t+1)} &= \kappa_0 + N_k^{(t)}, \\
    (\boldsymbol{W}_{n,k}^{(t+1)})^{-1} &= \boldsymbol{W}_0^{-1} + N_k^{(t)}S_k^{(t)} + \frac{\kappa_0 N_k^{(t)}}{\kappa_0 + N_k^{(t)}}(\bar{\boldsymbol{x}}_k^{(t)}-\boldsymbol{\mu}_0)(\bar{\boldsymbol{x}}_k^{(t)}-\boldsymbol{\mu}_0)^\top, \\
    \nu_{n,k}^{(t+1)} &= \nu_0 + N_k^{(t)},\\
    \eta_{n,k}^{(t+1)} &= \eta_{0,k} + \gamma^{(t)}_{1,k}, \\
    \zeta_{n,j,k}^{(t+1)} &= \zeta_{0,j,k}+M_{j,k}^{(t)}.

The approximate posterior distribution of the latent variable :math:`q^{(t+1)}(z^n)` is calculated by the forward-backward algorithm as follows.

.. math::
    \ln \rho_{i,k}^{(t+1)} &= \frac{1}{2} \Biggl[\, \sum_{d=1}^D \psi \left( \frac{\nu_{n,k}^{(t+1)} + 1 - d}{2} \right) + D \ln 2 + \ln | \boldsymbol{W}_{n,k}^{(t+1)} | \notag \\
    &\qquad - D \ln (2 \pi ) - \frac{D}{\kappa_{n,k}^{(t+1)}} - \nu_{n,k}^{(t+1)} (\boldsymbol{x}_i - \boldsymbol{m}_{n,k}^{(t+1)})^\top \boldsymbol{W}_{n,k}^{(t+1)} (\boldsymbol{x}_i - \boldsymbol{m}_{n,k}^{(t+1)}) \Biggr], \\
    \ln \tilde{\pi}_k^{(t+1)} &= \psi (\eta_{n,k}^{(t+1)}) - \psi \left( \textstyle \sum_{k=1}^K \eta_{n,k}^{(t+1)} \right) \\
    \ln \tilde{a}_{j,k}^{(t+1)} &= \psi (\zeta_{n,j,k}^{(t+1)}) - \psi \left( \textstyle \sum_{k=1}^K \zeta_{n,j,k}^{(t+1)} \right) \\
    \alpha^{(t+1)} (\boldsymbol{z}_i) &\propto
    \begin{cases}
    \prod_{k=1}^{K} \left( \rho_{i,k}^{(t+1)}\right)^{z_{i,k}} \sum_{\boldsymbol{z}_{i-1}} \left[\prod_{k=1}^{K}\prod_{j=1}^{K}\left(\tilde{a}^{(t+1)}_{j,k}\right)^{z_{i-1,j}z_{i,k}}\alpha^{(t+1)}(\boldsymbol{z}_{i-1})\right] & (i>1)\\
    \prod_{k=1}^{K}\left( \rho_{1,k}^{(t+1)} \tilde{\pi}_k^{(t+1)} \right)^{z_{1,k}} & (i=1)
    \end{cases} \\
    \beta^{(t+1)} (\boldsymbol{z}_i) &\propto
    \begin{cases}
    \sum_{\boldsymbol{z}_{i+1}} \left[ \prod_{k=1}^{K} \left( \rho_{i+1,k}^{(t+1)}\right)^{z_{i+1,k}} \prod_{k=1}^{K}\prod_{j=1}^{K}\left(\tilde{a}^{(t+1)}_{j,k}\right)^{z_{i,j}z_{i+1,k}}\beta^{(t+1)}(\boldsymbol{z}_{i+1})\right] & (i<n)\\
    1 & (i=n)
    \end{cases} \\
    q^{(t+1)}(\boldsymbol{z}_i) &\propto \alpha^{(t+1)}(\boldsymbol{z}_i)\beta^{(t+1)}(\boldsymbol{z}_i) \\
    \gamma^{(t+1)}_{i,k} &= \sum_{\boldsymbol{z}_i} q^{(t+1)}(\boldsymbol{z}_i) z_{i,k}\\
    q^{(t+1)}(\boldsymbol{z}_{i-1}, \boldsymbol{z}_{i}) &\propto \alpha^{(t+1)}(\boldsymbol{z}_{i-1}) \prod_{k=1}^{K} \left( \rho_{i,k}^{(t+1)}\right)^{z_{i,k}} \prod_{k=1}^{K}\prod_{j=1}^{K}\left(\tilde{a}^{(t+1)}_{j,k}\right)^{z_{i-1,j}z_{i,k}} \beta^{(t+1)}(\boldsymbol{z}_i) \\
    \xi^{(t+1)}_{i,j,k} &= \sum_{\boldsymbol{z}_{i-1}} \sum_{\boldsymbol{z}_i} q^{(t+1)}(\boldsymbol{z}_{i-1}, \boldsymbol{z}_{i}) z_{i-1,j} z_{i,k}

The approximate predictive distribution is as follows:

* :math:`\boldsymbol{x}_{n+1} \in \mathbb{R}^D`: a new data point
* :math:`(a_{\mathrm{p},j,k})_{1\leq j,k\leq K} \in [0, 1]^{K\times K}`: the parameters of the predictive transition probability of latent classes, (:math:`\sum_{k=1}^K a_{\mathrm{p},j,k}=1`)
* :math:`\boldsymbol{\mu}_{\mathrm{p},k} \in \mathbb{R}^D`: the parameter of the predictive distribution
* :math:`\boldsymbol{\Lambda}_{\mathrm{p},k} \in \mathbb{R}^{D \times D}`: the parameter of the predictive distribution (a positive definite matrix)
* :math:`\nu_{\mathrm{p},k} \in \mathbb{R}_{>0}`: the parameter of the predictive distribution

.. math::
    &p(x_{n+1}|x^n) \\
    &\approx \sum_{k=1}^K \left( \sum_{j=1}^K \gamma_{n,j}^{(t)} a_{\mathrm{p},j,k} \right) \mathrm{St}(x_{n+1}|\boldsymbol{\mu}_{\mathrm{p},k},\boldsymbol{\Lambda}_{\mathrm{p},k}, \nu_{\mathrm{p},k}) \\
    &= \sum_{k=1}^K \left( \sum_{j=1}^K \gamma_{n,j}^{(t)} a_{\mathrm{p},j,k} \right)\Biggl[ \frac{\Gamma (\nu_{\mathrm{p},k} / 2 + D / 2)}{\Gamma (\nu_{\mathrm{p},k} / 2)} \frac{|\boldsymbol{\Lambda}_{\mathrm{p},k}|^{1/2}}{(\nu_{\mathrm{p},k} \pi)^{D/2}} \nonumber \\
    &\qquad \qquad \qquad \qquad \qquad \times \left( 1 + \frac{1}{\nu_{\mathrm{p},k}} (\boldsymbol{x}_{n+1} - \boldsymbol{\mu}_{\mathrm{p},k})^\top \boldsymbol{\Lambda}_{\mathrm{p},k} (\boldsymbol{x}_{n+1} - \boldsymbol{\mu}_{\mathrm{p},k}) \right)^{-\nu_{\mathrm{p},k}/2 - D/2} \Biggr],

where the parameters are obtained from the hyperparameters of the predictive distribution as follows:

.. math::
    a_{\mathrm{p},j,k} &= \frac{\zeta_{n,j,k}^{(t)}}{\sum_{k=1}^K \zeta_{n,j,k}^{(t)}}, \\
    \boldsymbol{\mu}_{\mathrm{p},k} &= \boldsymbol{m}_{n,k}^{(t)}, \\
    \boldsymbol{\Lambda}_{\mathrm{p},k} &= \frac{\kappa_{n,k}^{(t)} (\nu_{n,k}^{(t)} - D + 1)}{\kappa_{n,k}^{(t)} + 1} \boldsymbol{W}_{n,k}^{(t)}, \\
    \nu_{\mathrm{p},k} &= \nu_{n,k}^{(t)} - D + 1.
"""
from ._hiddenmarkovnormal import GenModel
from ._hiddenmarkovnormal import LearnModel

__all__ = ["GenModel","LearnModel"]
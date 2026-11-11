---
header-includes:
  - \usepackage{bm}
mathspec: true
---

<!-- Document Author
Kohei Horinouchi <horinouchi@aoni.waseda.jp>
-->

# Variational Lower Bound for Sparse Linear Regression

This note derives the variational lower bound (ELBO), variational posterior update rules,
and the predictive distribution for the sparse linear regression model with the
Gaussian–exponential scale mixture representation of the Laplace prior.

## Model

$$
\begin{align}
    p(\boldsymbol{y} | \boldsymbol{X}, \boldsymbol{\theta}, \tau)
    &= \left(\frac{\tau}{2\pi}\right)^{n/2}
    \exp\left\{
        -\frac{\tau}{2}
        (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta})^\top
        (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta})
    \right\}, \\
    p(\boldsymbol{\theta} | \tau, \boldsymbol{v})
    &= \mathcal{N}\left(
        \boldsymbol{\theta} \middle| \boldsymbol{0},
        \tau^{-1} \mathrm{diag}(\boldsymbol{v})
    \right), \\
    p(\boldsymbol{v} | \boldsymbol{\lambda}_0)
    &= \prod_{j=1}^{d}
    \mathrm{Exp}\left(v_j \middle| \frac{\lambda_{0,j}^2}{2}\right), \\
    p(\tau | \alpha_0, \beta_0)
    &= \mathrm{Gam}(\tau | \alpha_0, \beta_0).
\end{align}
$$

where $\boldsymbol{\lambda}_0 = (\lambda_{0,1}, \ldots, \lambda_{0,d})^\top \in \mathbb{R}_{>0}^d$ is a hyperparameter vector (Laplace regularization strength for each dimension).

## Variational Distribution

We use the mean-field factorization:

$$
\begin{align}
    q(\boldsymbol{\theta}, \tau, \boldsymbol{v})
    &= q(\boldsymbol{\theta}, \tau) \prod_{j=1}^{d} q(v_j), \\
    q(\boldsymbol{\theta}, \tau)
    &= \mathcal{N}\left(
        \boldsymbol{\theta} \middle| \boldsymbol{\mu}_n,
        (\tau \boldsymbol{\Lambda}_n)^{-1}
    \right)
    \mathrm{Gam}(\tau | \alpha_n, \beta_n), \\
    q(v_j)
    &= \mathrm{GIG}(v_j | p_{n,j}, a_{n,j}, b_{n,j}),
\end{align}
$$

where the generalized inverse Gaussian (GIG) distribution is parameterized as:

$$
\begin{align}
    \mathrm{GIG}(x | p, a, b)
    = \frac{(a/b)^{p/2}}{2 K_p(\sqrt{ab})}
    x^{p-1}
    \exp\left\{-\frac{1}{2}(ax + bx^{-1})\right\}, \quad x > 0,
\end{align}
$$

and $K_p(\cdot)$ is the modified Bessel function of the second kind.

## Update Rules for Variational Parameters

Taking the coordinate ascent in the ELBO yields the following closed-form updates.

### Update for $q(\boldsymbol{\theta}, \tau)$

$$
\begin{align}
    \boldsymbol{\Lambda}_n
    &= \boldsymbol{X}^\top \boldsymbol{X}
    + \mathrm{diag}\!\left(
        \mathbb{E}_{q(v_1)}[v_1^{-1}], \ldots, \mathbb{E}_{q(v_d)}[v_d^{-1}]
    \right), \\
    \boldsymbol{\mu}_n
    &= \boldsymbol{\Lambda}_n^{-1} \boldsymbol{X}^\top \boldsymbol{y}, \\
    \alpha_n
    &= \alpha_0 + \frac{n}{2}, \\
    \beta_n
    &= \beta_0 + \frac{1}{2}
    \left(
        \boldsymbol{y}^\top \boldsymbol{y}
        - \boldsymbol{\mu}_n^\top \boldsymbol{\Lambda}_n \boldsymbol{\mu}_n
    \right).
\end{align}
$$

Note: the prior mean of $\boldsymbol{\theta}$ is $\boldsymbol{0}$, so there is no $\boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0$ term compared to standard Bayesian linear regression.

### Update for $q(v_j)$

The optimal variational distribution for each $v_j$ is:

$$
\begin{align}
    q^*(v_j) \propto \exp\left\{
        -\frac{1}{2}\!\left(
            \left[\lambda_{0,j}^2\right] v_j
            + \left[\frac{\alpha_n}{\beta_n}\mu_{n,j}^2
              + \left(\boldsymbol{\Lambda}_n^{-1}\right)_{jj}\right] v_j^{-1}
        \right)
    \right\} v_j^{-1/2},
\end{align}
$$

which is $\mathrm{GIG}(v_j | p_{n,j}, a_{n,j}, b_{n,j})$ with:

$$
\begin{align}
    p_{n,j} &= \frac{1}{2}, \\
    a_{n,j} &= \lambda_{0,j}^2, \\
    b_{n,j} &= \frac{\alpha_n}{\beta_n} \mu_{n,j}^2
              + \left(\boldsymbol{\Lambda}_n^{-1}\right)_{jj}.
\end{align}
$$

## Expectations of the Variational Distributions

### Expectations of $q(\tau)$

Let $\ell_\tau := \mathbb{E}_{q(\tau)}[\ln \tau] = \psi(\alpha_n) - \ln \beta_n$ and $\bar{\tau} := \mathbb{E}_{q(\tau)}[\tau] = \alpha_n / \beta_n$,
where $\psi(\cdot)$ is the digamma function.

### Expectations of $q(v_j) = \mathrm{GIG}(v_j | p_{n,j}, a_{n,j}, b_{n,j})$

Let $z_{n,j} := \sqrt{a_{n,j} b_{n,j}}$. The moments of the GIG distribution are:

$$
\begin{align}
    m_j^{(r)} &:= \mathbb{E}_{q(v_j)}[v_j^r]
    = \left(\frac{b_{n,j}}{a_{n,j}}\right)^{r/2}
    \frac{K_{p_{n,j}+r}(z_{n,j})}{K_{p_{n,j}}(z_{n,j})}, \\
    m_j^{\mathrm{log}} &:= \mathbb{E}_{q(v_j)}[\ln v_j]
    = \frac{1}{2} \ln \frac{b_{n,j}}{a_{n,j}}
    + \frac{\partial}{\partial p_{n,j}} \ln K_{p_{n,j}}(z_{n,j}).
\end{align}
$$

Since $p_{n,j} = 1/2$ and $K_{1/2}(z) = K_{-1/2}(z) = \sqrt{\pi/(2z)} e^{-z}$:

$$
\begin{align}
    m_j^{(-1)} &= \mathbb{E}_{q(v_j)}[v_j^{-1}]
    = \sqrt{\frac{a_{n,j}}{b_{n,j}}}
    = \frac{\lambda_{0,j}}{\sqrt{
        \dfrac{\alpha_n}{\beta_n} \mu_{n,j}^2
        + \left(\boldsymbol{\Lambda}_n^{-1}\right)_{jj}
    }}, \\
    m_j^{(1)} &= \mathbb{E}_{q(v_j)}[v_j]
    = \sqrt{\frac{b_{n,j}}{a_{n,j}}} \left(1 + \frac{1}{z_{n,j}}\right),
\end{align}
$$

For $m_j^{\mathrm{log}}$, since $p_{n,j} = 1/2$ and $K_p(z) = K_{-p}(z)$:

$$
\begin{align}
    m_j^{\mathrm{log}}
    = \frac{1}{2} \ln \frac{b_{n,j}}{a_{n,j}}
    + \left.\frac{\partial}{\partial p} \ln K_p(z_{n,j})\right|_{p=1/2}.
\end{align}
$$

## Variational Lower Bound (ELBO)

The ELBO is:

$$
\begin{align}
    \mathcal{L}(q)
    &= \mathbb{E}_q[\ln p(\boldsymbol{y}, \boldsymbol{\theta}, \tau, \boldsymbol{v} | \boldsymbol{X})]
    - \mathbb{E}_q[\ln q(\boldsymbol{\theta}, \tau, \boldsymbol{v})] \\
    &= \mathcal{L}_{\boldsymbol{y}}
    + \mathcal{L}_{\boldsymbol{\theta}}
    + \mathcal{L}_{\boldsymbol{v}}
    + \mathcal{L}_{\tau}
    + \mathcal{H}_{\boldsymbol{\theta},\tau}
    + \sum_{j=1}^d \mathcal{H}_{v_j}.
\end{align}
$$

Let $\boldsymbol{S} := \boldsymbol{\Lambda}_n^{-1}$.

### Likelihood term

$$
\begin{align}
    \mathcal{L}_{\boldsymbol{y}}
    &= \frac{n}{2} \ell_\tau
    - \frac{n}{2} \ln(2\pi)
    - \frac{1}{2}\left\{
        \bar{\tau}
        (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\mu}_n)^\top
        (\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\mu}_n)
        + \mathrm{tr}(\boldsymbol{X}^\top \boldsymbol{X} \boldsymbol{S})
    \right\}.
\end{align}
$$

### Gaussian scale-mixture prior term

$$
\begin{align}
    \mathcal{L}_{\boldsymbol{\theta}}
    &= \frac{d}{2} \ell_\tau
    - \frac{d}{2} \ln(2\pi)
    - \frac{1}{2} \sum_{j=1}^d m_j^{\mathrm{log}}
    - \frac{1}{2} \sum_{j=1}^d m_j^{(-1)}
    \left(\bar{\tau} \mu_{n,j}^2 + S_{jj}\right).
\end{align}
$$

### Exponential prior term

$$
\begin{align}
    \mathcal{L}_{\boldsymbol{v}}
    &= \sum_{j=1}^d \left\{
        \ln \frac{\lambda_{0,j}^2}{2}
        - \frac{\lambda_{0,j}^2}{2} m_j^{(1)}
    \right\}.
\end{align}
$$

### Gamma prior term

$$
\begin{align}
    \mathcal{L}_\tau
    &= \alpha_0 \ln \beta_0
    - \ln \Gamma(\alpha_0)
    + (\alpha_0 - 1) \ell_\tau
    - \beta_0 \bar{\tau}.
\end{align}
$$

### Entropy of $q(\boldsymbol{\theta}, \tau)$

$$
\begin{align}
    \mathcal{H}_{\boldsymbol{\theta},\tau}
    &= \frac{d}{2}\{1 + \ln(2\pi)\}
    - \frac{d}{2} \ell_\tau
    - \frac{1}{2} \ln |\boldsymbol{\Lambda}_n|
    + \alpha_n
    - \ln \beta_n
    + \ln \Gamma(\alpha_n)
    + (1 - \alpha_n) \psi(\alpha_n).
\end{align}
$$

### Entropy of $q(v_j)$

$$
\begin{align}
    \mathcal{H}_{v_j}
    &= \ln\{2 K_{p_{n,j}}(z_{n,j})\}
    - \frac{p_{n,j}}{2} \ln \frac{a_{n,j}}{b_{n,j}}
    - (p_{n,j} - 1) m_j^{\mathrm{log}}
    + \frac{1}{2}\left(a_{n,j} m_j^{(1)} + b_{n,j} m_j^{(-1)}\right).
\end{align}
$$

With $p_{n,j} = 1/2$ and $a_{n,j} = \lambda_{0,j}^2$, and using $K_{1/2}(z) = \sqrt{\pi/(2z)} e^{-z}$, so $2K_{1/2}(z) = \sqrt{2\pi/z}\, e^{-z}$:

$$
\begin{align}
    \ln\{2 K_{1/2}(z_{n,j})\}
    = \frac{1}{2}\ln(2\pi) - \frac{1}{2}\ln z_{n,j} - z_{n,j}.
\end{align}
$$

## Predictive Distribution

Given the variational posterior $q(\boldsymbol{\theta}, \tau) = \mathcal{N}(\boldsymbol{\theta} | \boldsymbol{\mu}_n, (\tau \boldsymbol{\Lambda}_n)^{-1}) \mathrm{Gam}(\tau | \alpha_n, \beta_n)$,
the approximate predictive distribution for a new point $\boldsymbol{x}_{n+1}$ is obtained by marginalizing over $q$:

$$
\begin{align}
    p(y_{n+1} | \boldsymbol{X}, \boldsymbol{y}, \boldsymbol{x}_{n+1})
    \approx \int \mathcal{N}(y_{n+1} | \boldsymbol{x}_{n+1}^\top \boldsymbol{\theta}, \tau^{-1})
    q(\boldsymbol{\theta}, \tau)\, d\boldsymbol{\theta}\, d\tau
    = \mathrm{St}(y_{n+1} | m_{\mathrm{p}}, \lambda_{\mathrm{p}}, \nu_{\mathrm{p}}),
\end{align}
$$

where:

$$
\begin{align}
    m_{\mathrm{p}} &= \boldsymbol{x}_{n+1}^{\top} \boldsymbol{\mu}_n, \\
    \lambda_{\mathrm{p}} &= \frac{\alpha_n}{\beta_n}
    \left(1 + \boldsymbol{x}_{n+1}^{\top} \boldsymbol{\Lambda}_n^{-1} \boldsymbol{x}_{n+1}\right)^{-1}, \\
    \nu_{\mathrm{p}} &= 2\alpha_n.
\end{align}
$$

This follows from the fact that the joint $q(\boldsymbol{\theta}, \tau)$ has the same form as the exact posterior in standard Bayesian linear regression; see `bayesml/linearregression/__init__.py` for the derivation.

<!-- Document Author
Yuji Iikubo <yuji-iikubo.8@fuji.waseda.jp>
-->

The logistic regression model with the Gaussian prior distribution.

The stochastic data generative model is as follows:

* $d \in \mathbb N$: a dimension
* $\boldsymbol{x} \in \mathbb{R}^d$: an explanatory variable. If you consider an intercept term, it should be included as one of the elements of $\boldsymbol{x}$.
* $y\in\{ 0, 1\}$: an objective variable
* $\boldsymbol{w}\in\mathbb{R}^{d}$: a parameter

$$
\begin{align}
    p(y|\boldsymbol{x},\boldsymbol{w}) &= \left\{ \frac{1}{1 + \exp( - \boldsymbol{w}^\top \boldsymbol{x} ) } \right\}^y \left\{ \frac{ \exp( - \boldsymbol{w}^\top \boldsymbol{x} ) }{1 + \exp( - \boldsymbol{w}^\top \boldsymbol{x} )} \right\}^{1 - y}.
\end{align}
$$

The prior distribution is as follows:

* $\boldsymbol{\mu}_0 \in \mathbb{R}^d$: a hyperparameter
* $\boldsymbol{\Lambda}_0 \in \mathbb{R}^{d\times d}$: a hyperparameter (a positive definite matrix)

$$
\begin{align}
    p(\boldsymbol{w}) &= \mathcal{N}(\boldsymbol{w}|\boldsymbol{\mu}_0, \boldsymbol{\Lambda}_0^{-1})\\
    &= \frac{|\boldsymbol{\Lambda}_0|^{1/2}}{(2 \pi)^{d/2}} \exp \left\{ -\frac{1}{2} (\boldsymbol{w} - \boldsymbol{\mu}_0)^\top \boldsymbol{\Lambda}_0 (\boldsymbol{w} - \boldsymbol{\mu}_0) \right\}.
\end{align}
$$

The apporoximate posterior distribution in the $t$-th iteration of a variational Bayesian method is as follows:

* $n \in \mathbb N$: a sample size
* $\boldsymbol{x}^n = (\boldsymbol{x}_1, \boldsymbol{x}_2, \dots , \boldsymbol{x}_n) \in \mathbb{R}^{n \times d}$
* $\boldsymbol{y}^n = (y_1, y_2, \dots , y_n) \in \{0,1\}^n$
* $\boldsymbol{\mu}_n^{(t)}\in \mathbb{R}^d$: a hyperparameter
* $\boldsymbol{\Lambda}_n^{(t)} \in \mathbb{R}^{d\times d}$: a hyperparameter (a positive definite matrix)

$$
\begin{align}
    q(\boldsymbol{w}) &= \mathcal{N}(\boldsymbol{w}|\boldsymbol{\mu}_n^{(t)}, (\boldsymbol{\Lambda}_n^{(t)})^{-1})\\
    &= \frac{|\boldsymbol{\Lambda}_n^{(t)}|^{1/2}}{(2 \pi)^{d/2}} \exp \left\{ -\frac{1}{2} (\boldsymbol{w} - \boldsymbol{\mu}_n^{(t)})^\top \boldsymbol{\Lambda}_n^{(t)} (\boldsymbol{w} - \boldsymbol{\mu}_n^{(t)}) \right\}.
\end{align}
$$

where the updating rules of the hyperparameters are as follows:

* $\boldsymbol{\xi}_n^{(t)} = (\xi_{1}^{(t)}, \xi_{2}^{(t)}, \dots, \xi_{n}^{(t)}) \in \mathbb{R}_{\geq 0}^n$: a variational parameter

$$
\begin{align}
    \boldsymbol{\Lambda}_n^{(t)} &= \boldsymbol{\Lambda}_0 + 2 \sum_{i=1}^{n} \lambda(\xi_i^{(t)}) \boldsymbol{x}_i \boldsymbol{x}_i^\top,\\
    \boldsymbol{\mu}_n^{(t)} &= (\boldsymbol{\Lambda}_n^{(t)})^{-1} \left(\boldsymbol{\Lambda}_0 \boldsymbol{\mu}_0 + \sum_{i=1}^{n} (y_i - 1/2) \boldsymbol{x}_{i} \right),\\
    \xi_i^{(t)} &= \left[ \boldsymbol{x}_{i}^\top \left\{ (\boldsymbol{\Lambda}_n^{(t)})^{-1} + \boldsymbol{\mu}_n^{(t)} \boldsymbol{\mu}_n^{(t)\top} \right\} \boldsymbol{x}_{i} \right]^{1/2}, 
\end{align}
$$

where $\lambda(\cdot)$ is defined as follows:

$$
\begin{align}
    \lambda(\xi) &= \frac{1}{2\xi} \left\{ \frac{1}{1+\exp(-\xi)} - \frac{1}{2} \right\}.
\end{align}
$$

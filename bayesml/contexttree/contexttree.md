<!--
Document Author
Koshi Shimada <shimada.koshi.re@gmail.com>
-->

The stochastic data generative model is as follows:

* $\mathcal{X}=\{1,2,\ldots,K\}$ : a space of a source symbol
* $x^n = x_1 x_2 \cdots x_n \in \mathcal{X}^n~(n\in\mathbb{N})$ : an source sequence
* $D_\mathrm{max} \in \mathbb{N}$ : the maximum depth of context tree models
* $T$ : a context tree model, $K$-ary regular tree whose depth is smaller than or equal to $D_\mathrm{max}$, where "regular" means that all inner nodes have $K$ child nodes
* $\mathcal{T}$ : a set of $T$
* $s$ : a node of a context tree model
* $\mathcal{I}(T)$ : a set of inner nodes of $T$
* $\mathcal{L}(T)$ : a set of leaf nodes of $T$
* $\mathcal{S}(T)$ : a set of all nodes of $T$, i.e., $\mathcal{S}(T) = \mathcal{I}(T) \cup \mathcal{L}(T)$
* $s_T(x^{n-1}) \in \mathcal{L}(T)$ : a leaf node of $T$ corresponding to $x^{n-1} = x_1 x_2\cdots x_{n-1}$
* $\boldsymbol{\theta}_s = (\theta_{1|s}, \theta_{2|s}, \ldots, \theta_{K|s})$ : a parameter on a leaf node, where $\theta_{k|s}$ denotes the occurrence probability of $k\in\mathcal{X}$

$$
\begin{align}
    p(x_n | x^{n-1}, \boldsymbol{\theta}_T, T)=\theta_{x_n|s_T(x^{n-1})}.
\end{align}
$$

The prior distribution is as follows:

* $g_{0,s} \in [0,1]$ : a hyperparameter assigned to $s \in \mathcal{S}(T)$
* $\beta_0(k|s) \in\mathbb{R}_{>0}$ : a hyperparameter of the Dirichlet distribution
* $\boldsymbol{\beta}_0(s) = (\beta_0(1|s), \beta_0(2|s), \ldots, \beta_0(K|s)) \in\mathbb{R}^{K}_{>0}$
* $C(\boldsymbol{\beta}_0(s)) = \frac{\Gamma\left(\sum_{k=1}^{K} \beta_0(k|s)\right)}{\prod_{k=1}^{K} \Gamma\left(\beta_0(k|s)\right)}$

For $\boldsymbol{\theta}_s$ on $s\in\mathcal{L}(T)$, the Dirichlet distribution is assumed as the prior distribution as follows:
$$
\begin{align}
    p(\boldsymbol{\theta}_s|T) = \mathrm{Dir}(\boldsymbol{\theta}_s|\,\boldsymbol{\beta}_0(s)) = C(\boldsymbol{\beta}_0(s)) \prod_{k=1}^{K} \theta_{k|s}^{\beta_0(k|s)-1}.
\end{align}
$$

For $T \in \mathcal{T}$,
$$
\begin{align}
    p(T)=\prod_{s \in \mathcal{I}(T)} g_{0,s} \prod_{s' \in \mathcal{L}(T)} (1-g_{0,s'}),
\end{align}
$$
where $g_{0,s}=0$ if the depth of $s$ is $D_\mathrm{max}$.

The posterior distribution is as follows:

* $g_{n,s} \in [0,1]$ : the updated hyperparameter
* $T_\mathrm{max}$ : a superposed context tree, $K$-ary perfect tree whose depth is $D_\mathrm{max}$
* $s_\lambda$ : the root node
* $\beta_n(k|s) \in\mathbb{R}_{>0}$ : a hyperparameter of the posterior Dirichlet distribution
* $\boldsymbol{\beta}_n(s) = (\beta_n(1|s), \beta_n(2|s), \ldots, \beta_n(K|s)) \in\mathbb{R}^{K}_{>0}$
* $I \{ \cdot \}$: the indicator function

For $\boldsymbol{\theta}_s \in\mathcal{L}(T_\mathrm{max})$,

$$
\begin{align}
    p(\boldsymbol{\theta}_s|x^n) = \mathrm{Dir}(\boldsymbol{\theta}_s|\,\boldsymbol{\beta}_n(s)) = C(\boldsymbol{\beta}_n(s)) \prod_{k=1}^{K} \theta_{k|s}^{\beta_n(k|s)-1},
\end{align}
$$

where the updating rule of the hyperparameter is as follows:

$$
\begin{align}
    \beta_n(k|s) = \beta_0(k|s) + \sum_{i=1}^n I \left\{ \text{$s$ is the ancestor of $s_{T_\mathrm{max}}(x^{i-1})$ and $x_i=k$ } \right\}.
\end{align}
$$

For $T \in \mathcal{T}$,

$$p(T|x^{n-1})=\prod_{s \in \mathcal{I}(T)} g_{n,s} \prod_{s' \in \mathcal{L}(T)} (1-g_{n,s'}),$$

where the updating rules of the hyperparameter are as follows:

$$g_{n,s} =
\begin{cases}
    g_{0,s} & \text{if $n=0$}, \\
    \frac{ g_{n-1,s} \tilde{q}_{s_{\mathrm{child}}} (x_n|x^{n-1}) }
    { \tilde{q}_s(x_n|x^{n-1}) } & \text{otherwise},
\end{cases}$$
where $s_{\mathrm{child}}$ is the child node of $s$ on the path from $s_\lambda$ to $s_{T_\mathrm{max}}(x^n)$ and

$$
\begin{align}
    \tilde{q}_s(x_n|x^{n-1}) =
    \begin{cases}
        q_s(x_n|x^{n-1}) & \text{if $s\in\mathcal{L}(T_\mathrm{max})$}, \\
        (1-g_{n-1,s}) q_s(x_n|x^{n-1}) + g_{n-1,s} \tilde{q}_{s_{\mathrm{child}}}(x_n|x^{n-1}) & \text{otherwise}.
    \end{cases}
\end{align}
$$

Here,

$$
\begin{align}
    q_s(x_n|x^{n-1}) = \frac{ \beta_{n-1}(x_n|s) }
    {\sum_{k'=1}^{K} \beta_{n-1}(k'|s)}.
\end{align}
$$

The predictive distribution is as follows:

* $\boldsymbol{\theta}_\mathrm{p} = (\theta_{\mathrm{p},1}, \theta_{\mathrm{p},2}, \ldots, \theta_{\mathrm{p},K})$ : a parameter of the predictive distribution, where $\theta_{\mathrm{p},k}$ denotes the occurrence probability of $k\in\mathcal{X}$.
$$
\begin{align}
p(x_n|x^{n-1}) = \theta_{\mathrm{p},x_n},
\end{align}
$$
where the updating rule of the parameters of the pridictive distribution is as follows.

$$
\begin{align}
\theta_{\mathrm{p}, k} = \tilde{q}_{s_\lambda}(k|x^{n-1})
\end{align}
$$

References

* Matsushima, T.; and Hirasawa, S. Reducing the space complexity of a Bayes coding algorithm using an expanded context tree, *2009 IEEE International Symposium on Information Theory*, 2009, pp. 719-723, https://doi.org/10.1109/ISIT.2009.5205677
* Nakahara, Y.; Saito, S.; Kamatsuka, A.; Matsushima, T. Probability Distribution on Full Rooted Trees. *Entropy* 2022, 24, 328. https://doi.org/10.3390/e24030328

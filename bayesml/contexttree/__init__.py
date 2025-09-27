# Document Author
# Koshi Shimada <shimada.koshi.re@gmail.com>
r"""
The stochastic data generative model is as follows:

* :math:`\mathcal{X}=\{1,2,\ldots,K\}` : a space of a source symbol
* :math:`x^n = x_1 x_2 \cdots x_n \in \mathcal{X}^n~(n\in\mathbb{N})` : an source sequence
* :math:`D_\mathrm{max} \in \mathbb{N}` : the maximum depth of context tree models
* :math:`T` : a context tree model, :math:`K`-ary regular tree whose depth is smaller than or equal to :math:`D_\mathrm{max}`, where "regular" means that all inner nodes have :math:`K` child nodes
* :math:`\mathcal{T}` : a set of :math:`T`
* :math:`s` : a node of a context tree model
* :math:`\mathcal{I}(T)` : a set of inner nodes of :math:`T`
* :math:`\mathcal{L}(T)` : a set of leaf nodes of :math:`T`
* :math:`\mathcal{S}(T)` : a set of all nodes of :math:`T`, i.e., :math:`\mathcal{S}(T) = \mathcal{I}(T) \cup \mathcal{L}(T)`
* :math:`s_T(x^{n-1}) \in \mathcal{L}(T)` : a leaf node of :math:`T` corresponding to :math:`x^{n-1} = x_1 x_2\cdots x_{n-1}`
* :math:`\boldsymbol{\theta}_s = (\theta_{1|s}, \theta_{2|s}, \ldots, \theta_{K|s})` : a parameter on a leaf node, where :math:`\theta_{k|s}` denotes the occurrence probability of :math:`k\in\mathcal{X}`

.. math::
    p(x_n | x^{n-1}, \boldsymbol{\theta}_T, T)=\theta_{x_n|s_T(x^{n-1})}.

The prior distribution is as follows:

* :math:`g_{0,s} \in [0,1]` : a hyperparameter assigned to :math:`s \in \mathcal{S}(T)`
* :math:`\beta_0(k|s) \in\mathbb{R}_{>0}` : a hyperparameter of the Dirichlet distribution
* :math:`\boldsymbol{\beta}_0(s) = (\beta_0(1|s), \beta_0(2|s), \ldots, \beta_0(K|s)) \in\mathbb{R}^{K}_{>0}`
* :math:`C(\boldsymbol{\beta}_0(s)) = \frac{\Gamma\left(\sum_{k=1}^{K} \beta_0(k|s)\right)}{\prod_{k=1}^{K} \Gamma\left(\beta_0(k|s)\right)}`

For :math:`\boldsymbol{\theta}_s` on :math:`s\in\mathcal{L}(T)`, the Dirichlet distribution is assumed as the prior distribution as follows:

.. math::
    p(\boldsymbol{\theta}_s|T) = \mathrm{Dir}(\boldsymbol{\theta}_s|\,\boldsymbol{\beta}_0(s)) = C(\boldsymbol{\beta}_0(s)) \prod_{k=1}^{K} \theta_{k|s}^{\beta_0(k|s)-1}.

For :math:`T \in \mathcal{T}`,

.. math::
    p(T)=\prod_{s \in \mathcal{I}(T)} g_{0,s} \prod_{s' \in \mathcal{L}(T)} (1-g_{0,s'}),

where :math:`g_{0,s}=0` if the depth of :math:`s` is :math:`D_\mathrm{max}`.

The posterior distribution is as follows:

* :math:`g_{n,s} \in [0,1]` : the updated hyperparameter
* :math:`T_\mathrm{max}` : a superposed context tree, :math:`K`-ary perfect tree whose depth is :math:`D_\mathrm{max}`
* :math:`s_\lambda` : the root node
* :math:`\beta_n(k|s) \in\mathbb{R}_{>0}` : a hyperparameter of the posterior Dirichlet distribution
* :math:`\boldsymbol{\beta}_n(s) = (\beta_n(1|s), \beta_n(2|s), \ldots, \beta_n(K|s)) \in\mathbb{R}^{K}_{>0}`
* :math:`I \{ \cdot \}`: the indicator function

For :math:`\boldsymbol{\theta}_s \in\mathcal{L}(T_\mathrm{max})`,

.. math::
    p(\boldsymbol{\theta}_s|x^n) = \mathrm{Dir}(\boldsymbol{\theta}_s|\,\boldsymbol{\beta}_n(s)) = C(\boldsymbol{\beta}_n(s)) \prod_{k=1}^{K} \theta_{k|s}^{\beta_n(k|s)-1},

where the updating rule of the hyperparameter is as follows:

.. math::
    \beta_n(k|s) = \beta_0(k|s) + \sum_{i=1}^n I \left\{ s \ \mathrm{is \ the \ ancestor \ of} \ s_{T_\mathrm{max}}(x^{i-1}) \ \mathrm{and} \ x_i=k \right\}.

For :math:`T \in \mathcal{T}`,

.. math::
    p(T|x^{n-1})=\prod_{s \in \mathcal{I}(T)} g_{n,s} \prod_{s' \in \mathcal{L}(T)} (1-g_{n,s'}),

where the updating rules of the hyperparameter are as follows:

.. math::
    g_{n,s} =
    \begin{cases}
        g_{0,s}, & n=0, \\
        \frac{ g_{n-1,s} \tilde{q}_{s_{\mathrm{child}}} (x_n|x^{n-1}) }
        { \tilde{q}_s(x_n|x^{n-1}) } & \mathrm{otherwise},
    \end{cases}

where :math:`s_{\mathrm{child}}` is the child node of :math:`s` on the path from :math:`s_\lambda` to :math:`s_{T_\mathrm{max}}(x^n)` and

.. math::
    \tilde{q}_s(x_n|x^{n-1}) =
    \begin{cases}
        q_s(x_n|x^{n-1}) & s\in\mathcal{L}(T_\mathrm{max}), \\
        (1-g_{n-1,s}) q_s(x_n|x^{n-1}) + g_{n-1,s} \tilde{q}_{s_{\mathrm{child}}}(x_n|x^{n-1}) & \mathrm{otherwise}.
    \end{cases}

Here,

.. math::
    q_s(x_n|x^{n-1}) = \frac{ \beta_{n-1}(x_n|s) }
    {\sum_{k'=1}^{K} \beta_{n-1}(k'|s)}.

The predictive distribution is as follows:

* :math:`\boldsymbol{\theta}_\mathrm{p} = (\theta_{\mathrm{p},1}, \theta_{\mathrm{p},2}, \ldots, \theta_{\mathrm{p},K})` : a parameter of the predictive distribution, where :math:`\theta_{\mathrm{p},k}` denotes the occurrence probability of :math:`k\in\mathcal{X}`.

.. math::
    p(x_n|x^{n-1}) = \theta_{\mathrm{p},x_n},

where the updating rule of the parameters of the pridictive distribution is as follows.

.. math::
    \theta_{\mathrm{p}, k} = \tilde{q}_{s_\lambda}(k|x^{n-1})

References

* Matsushima, T.; and Hirasawa, S. Reducing the space complexity of a Bayes coding algorithm using an expanded context tree, *2009 IEEE International Symposium on Information Theory*, 2009, pp. 719-723, https://doi.org/10.1109/ISIT.2009.5205677
* Nakahara, Y.; Saito, S.; Kamatsuka, A.; Matsushima, T. Probability Distribution on Full Rooted Trees. *Entropy* 2022, 24, 328. https://doi.org/10.3390/e24030328
"""
from ._contexttree import GenModel
from ._contexttree import LearnModel

__all__ = ["GenModel", "LearnModel"]
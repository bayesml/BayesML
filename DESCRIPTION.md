## Purpose

BayesML contributes to wide society through promoting education, research, and application of machine learning based on Bayesian statistics and Bayesian decision theory.

## Characteristics

* **Easy-to-use:**
  * You can use pre-defined Bayesian statistical models by simply importing it. You don't need to define models yourself like PyMC or Stan.
* **Bayesian Decision Theoretic API:**
  * BayesML's API corresponds to the structure of decision-making based on Bayesian decision theory. Bayesian decision theory is a unified framework for handling various decision-making processes, such as parameter estimation and prediction of new data. Therefore, BayesML enables intuitive operations for a wider range of decision-making compared to the fit-predict type API adopted in libraries like scikit-learn. Moreover, many of our models also implement fit-predict functions.
* **Model Visuialization Functions:**
  * All packages have methods to visualize the probabilistic data generative model, generated data from that model, and the posterior distribution learned from the data in 2~3 dimensional space. Thus, you can effectively understand the characteristics of probabilistic data generative models and algorithms through the generation of synthetic data and learning from them.
* **Fast Algorithms Using Conjugate Prior Distributions:**
  * Many of our learning algorithms adopt exact calculation methods or variational Bayesian methods that effectively use the conjugacy between probabilistic data generative models and prior distributions. Therefore, they are much faster than general-purpose MCMC methods and are also suitable for online learning. Although some algorithms adopt MCMC methods, but they use MCMC methods specialized for each model, taking advantage of conjugacy.

## Star Us on GitHub

If you found this library helpful, please consider giving us a star on GitHub.  
It really helps the project grow.

**[Open the repo and hit ☆ Star in the top right!](https://github.com/bayesml/BayesML)**

[![Star on GitHub](https://img.shields.io/github/stars/bayesml/BayesML?style=social)](https://github.com/bayesml/BayesML)

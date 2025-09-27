"""Your First Library for Bayesian Machine Learning

## Purpose

BayesML contributes to wide society thourgh promoting education, 
research, and application of machine learning based on Bayesian 
statistics and Bayesian decision theory.

## Characteristics

* **Easy-to-use:**
  * You can use pre-defined Bayesian statistical models by simply importing it. 
    You don't need to define models yourself like PyMC or Stan.
* **Bayesian Decision Theoretic API:**
  * BayesML's API corresponds to the structure of decision-making based on 
    Bayesian decision theory. Bayesian decision theory is a unified framework for 
    handling various decision-making processes, such as parameter estimation and 
    prediction of new data. Therefore, BayesML enables intuitive operations for 
    a wider range of decision-making compared to the fit-predict type API adopted in 
    libraries like scikit-learn. Moreover, many of our models also implement 
    fit-predict functions.
* **Model Visuialization Functions:**
  * All packages have methods to visualize the probabilistic data generative model, 
    generated data from that model, and the posterior distribution learned from 
    the data in 2~3 dimensional space. Thus, you can effectively understand 
    the characteristics of probabilistic data generative models and algorithms through 
    the generation of synthetic data and learning from them.
* **Fast Algorithms Using Conjugate Prior Distributions:**
  * Many of our learning algorithms adopt exact calculation methods or variational 
    Bayesian methods that effectively use the conjugacy between probabilistic data 
    generative models and prior distributions. Therefore, they are much faster than 
    general-purpose MCMC methods and are also suitable for online learning. 
    Although some algorithms adopt MCMC methods, but they use MCMC methods specialized 
    for each model, taking advantage of conjugacy.
"""
DOCLINES = (__doc__ or '').split("\n")

from setuptools import setup, find_packages

setup(
    name='bayesml',
    version='0.3.1',
    packages=find_packages(),
    author='Yuta Nakahara et al.',
    author_email='y.nakahara@waseda.jp',
    url='https://bayesml.github.io/BayesML/',
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    long_description_content_type='text/markdown',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: BSD License',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11',
                 'Topic :: Scientific/Engineering'
                 ],
    install_requires=['numpy >= 1.20',
                      'scipy >= 1.7',
                      'matplotlib >= 3.5',
                      'scikit-learn >= 1.1'],
    python_requires='~=3.7',
)

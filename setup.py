"""A library for Bayes statistics, Bayes decision theory, and Bayes machine learning

## Purpose

BayesML is a library designed for promoting research, education, 
and application of machine learning based on Bayesian statistics 
and Bayesian decision theory. 
Through these activities, BayesML aims to contribute to society.

## Characteristics

BayesML has the following characteristics.

* The structure of the library reflects the philosophy of 
  Bayesian statistics and Bayesian decision theory: 
  updating the posterior distribution learned from the data 
  and outputting the optimal estimate based on the Bayes criterion.
* Many of our learning algorithms are much faster than general-purpose 
  Bayesian learning algorithms such as MCMC methods because they 
  effectively use the conjugate property of a probabilistic data 
  generative model and a prior distribution. 
  Moreover, they are suitable for online learning.
* All packages have methods to visualize the probabilistic data 
  generative model, generated data from that model, and the posterior 
  distribution learned from the data in 2~3 dimensional space. 
  Thus, you can effectively understand the characteristics of 
  probabilistic data generative models and algorithms through 
  the generation of synthetic data and learning from them.
"""
DOCLINES = (__doc__ or '').split("\n")

from setuptools import setup, find_packages

setup(
    name='bayesml',
    version='0.3.0',
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

# manic

![License](https://img.shields.io/badge/license-MIT-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Build](https://github.com/dscamiss/manic/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/dscamiss/manic/graph/badge.svg?token=ZWTBITN49T)](https://codecov.io/gh/dscamiss/manic)

An PyTorch implementation of the Mechanic learning rate scale tuner [1].  

The reference implementation can be found in [this repository](https://github.com/optimizedlearning/mechanic).

## Brief background

Mechanic works alongside an optimizer and learning rate scheduler to tune the learning rate scale.

To make this concrete, suppose that the optimizer is vanilla SGD, so that one gradient descent iteration is
$\theta_{t+1} \leftarrow \theta_t - \alpha_t \nabla_\theta L(\theta_t)$, where $\alpha_t$ is the learning rate.  Generally
speaking, the learning rate scheduler needs to be tuned to select the base learning rate.  This amounts to searching (by some
process, usually *ad hoc*) for a scale factor $\sigma$ such that $\theta_{t+1} \leftarrow \theta_t - \sigma \alpha_t \nabla_\theta L(\theta_t)$ has good convergence properties.  

Mechanic *automatically* selects the scale factor at each gradient descent iteration.  The selection
process is entirely on-line, and comes with fairly strong theoretical guarantees with respect to convergence properties.
The cost is computational overhead, which is not severe since Mechanic is a first-order method 
(it only uses model parameters, gradients, and simple derived quantities).

# Installation

```
git clone https://github.com/dscamiss/manic
pip install manic
```

# Usage

# TODO

- [ ] Add state save/restore

# References

[1] Ashok Cutkosky, Aaron Defazio, and Harsh Mehta, Mechanic: A Learning Rate Tuner, [arXiv:2306.00144](https://arxiv.org/abs/2306.00144)

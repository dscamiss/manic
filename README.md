# manic (WIP)

![License](https://img.shields.io/badge/license-MIT-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Build](https://github.com/dscamiss/manic/actions/workflows/python-package.yml/badge.svg)
[![codecov](https://codecov.io/gh/dscamiss/manic/graph/badge.svg?token=ZWTBITN49T)](https://codecov.io/gh/dscamiss/manic)

An alternative PyTorch implementation of the Mechanic learning scheduler [1].

The reference implementation, written by the authors of [1], can be found in [this repository](https://github.com/optimizedlearning/mechanic).

# Installation

```
git clone https://github.com/dscamiss/manic
pip install manic
```

# Usage

# TODO

- [x] Add LR scheduler implementation
- [ ] Add LR scheduler state save/restore
- [ ] Add example(s)
- [ ] Add comparison with reference implementation
- [ ] Wrap base LRScheduler in addition to base optimizer

# References

[1] Ashok Cutkosky, Aaron Defazio, and Harsh Mehta, Mechanic: A Learning Rate Tuner, [arXiv:2306.00144](https://arxiv.org/abs/2306.00144)

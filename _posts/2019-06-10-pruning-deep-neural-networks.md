---
layout: post
comments: true
title: "Pruning Deep Neural Networks"
date: 2019-06-10 12:07:00
tags: review
image: "A3C_vs_A2C.png"
---

> Deep neural networks are usually over-parametrized which leads to high computational cost and memory overhead at inference time. In this post we are going to review some background and recent pruning algorithms.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## Background 

### Classical Pruning Algorithms

The core idea in pruning is to find a saliency for the weight parameters and remove those with low saliency with the belief that these will influence the model least.

The classical pruning algorithm was developed very early on by Lecun. The algorithm is based on constructing a local model of the loss function and analytically predict the effect of perturbing the parameter vectors. The proposed algorithm approximate the loss function $$\mathcal{L}$$ by a Taylor series. A perturbation $$\Delta \Theta$$ of the parameter will change the loss function by: 

$$
\begin{aligned}
& \Delta \mathcal{L} =  \frac{\partial \mathcal{L}}{\partial \theta}^{T}\Delta \Theta +\frac {1}{2}{\Delta \Theta}^{T} H \Delta \Theta + O(||\Delta \Theta||^{3})
\end{aligned}
$$

Where $$H$$ is the Hessian matrix. Usually pruning is done when the model is trained and the parameter vector is then at local minimum of $$\mathcal{L}$$, and the first term of the right hand side of above equation can be neglected. The quadratic approximation also assume that the loss function is nearly quadratic, so that the last term can be neglected. So we end up with: 

$$
\begin{aligned}
& \Delta \mathcal{L} = \frac {1}{2}{\Delta \Theta}^{T} H \Delta \Theta
\end{aligned}
$$

which seems a  a very good saliency metric.

#### Optimal Brain Damage (OBD)
Because computing the full Hessian in deep networks is intractable, the Hessian matrix $$H$$ is approximated by a diagonal matrix in OBD. If we prune a weight $$\theta_{q}$$, then the corresponding change in weights as well as the loss are:

$$
\begin{aligned}
& {\Delta \Theta}_{q} =-\theta_{q}^{*}  &  {\Delta \mathcal{L}}_{OBD} = \frac {1}{2}({\theta_{q}}^{*})^{2} H_{qq}
\end{aligned}
$$

Note that, OBD assumes a diagonal approximation for calculation of Hessian. In other words, it assumes all the weights are uncorrelated, such that removing one, will not affect the others.

#### Optimal Brain Sergeon (OBS)

$$
\begin{aligned}
\min _{q}\big[\min_{\Delta \Theta} \frac {1}{2}{\Delta \Theta}^{T} H {\Delta \Theta} & \text{s.t. } e_{q}^{T}{\Delta \Theta}+\theta_{q}^{*}=0\big]
\end{aligned}
$$

### Curvature Approximation
TBD
#### K-FAC
TBD
#### EKFAC
[[paper](https://arxiv.org/pdf/1806.03884.pdf)\|[code](https://github.com/wiseodd/natural-gradients)]
## Pruning Using EKFAC and OBD
[[paper](https://arxiv.org/pdf/1905.05934.pdf)\|[code](https://github.com/alecwangcq/EigenDamage-Pytorch)]


## Summary

TBD


## References

[1] LeCun et al.["Optimal Brain Damage."](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf). 1990.

[2] Hassibi et al. ["Optimal Brain Sergeon"](https://papers.nips.cc/paper/749-optimal-brain-surgeon-extensions-and-performance-comparisons.pdf). 1993.

[3] Thomas George, et al. ["Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis."](https://arxiv.org/pdf/1806.03884.pdf). 2018.

[4] Chaoqi Wang, et al. ["EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis."](https://arxiv.org/pdf/1905.05934.pdf). 2019.

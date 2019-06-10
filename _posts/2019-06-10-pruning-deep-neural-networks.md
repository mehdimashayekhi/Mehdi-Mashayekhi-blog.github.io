---
layout: post
comments: true
title: "Pruning deep neural networks"
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

The classical pruning algorithm was developed very early on by LeCun. The algorithm is based on constructing a local model of the loss function and analytically predict the effect of perturbing the parameter vectors. The proposed algorithm approximate the loss function $$\mathcal{L}$$ by a Taylor series. A perturbation $$\Delta \Theta$$ of the parameter will change the loss function by: 

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
TBD
#### Optimal Brain Sergeon (OBS)
TBD

### Curvature Approximation
TBD
#### K-FAC
TBD
#### EKFAC
TBD
## Pruning Using EKFAC and OBD
TBD


## Summary

TBD


## References

[1] ["Optimal Brain Damage."](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf) - (LeCun et al., 1990).

[2] ["Optimal Brain Sergeon"](https://papers.nips.cc/paper/749-optimal-brain-surgeon-extensions-and-performance-comparisons.pdf,  (Hassibi et al., 1993).

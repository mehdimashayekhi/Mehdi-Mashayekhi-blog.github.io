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

OBS was developed by Hassibi et.al, and it shares the same basic approach as OBD, in which, it trains a network to local minimum in error (w star) and then prunes a weight that leads to the smallest increase in the training error. The main difference between OBD and OBS is, OBS relaxes the diagonal approximation of the Hessian (i.e., uses full hessian), and it not only prune a single weight, but it takes into account the correlation between weights, and updates the rest of weights to compensate.

In OBS, the importance of each weight is calculated by solving the following constrained optimization problem:

$$
\begin{aligned}
\min _{q}\big[\min_{\Delta \Theta} \frac {1}{2}{\Delta \Theta}^{T} H {\Delta \Theta}  &  \text{s.t. } e_{q}^{T}{\Delta \Theta}+\theta_{q}^{*}=0\big]
\end{aligned}
$$

Solving above equation yields the optimal weight change and the corresponding change in error:

$$
\begin{aligned}
& \Delta \Theta =-\frac {\theta_{q}^{*}}{H_{qq}^{-1}}H^{-1} e_{q} &  {\Delta \mathcal{L}}_{OBS} = \frac {1}{2}\frac {(\theta_{q}^{*})^{2}}{H_{qq}^{-1}}
\end{aligned}
$$

### Hessian Approximation Using Fisher
As you saw for pruning, we need to calculate/approximate the Hessian. We can use Fisher matrix to approximate Hessian. Assume the function $$z = f(x, \theta)$$ is parametrized by $$\theta$$, and the loss function is $$\mathcal{L}(y,z)=- \log p(y \vert z)$$. Then the Hessian $$H$$ at a local minimum is equivalent to generalized Gauss-Newton matrix $$G$$:

$$
\begin{aligned}
& H = \mathbb{E}\Big[ J_{f}^{T} H_{l} J_{f} + \underbrace{\sum_{j=1}^m [\nabla_z \mathcal{L}(y,z) \vert_{z = f(x, \theta)} ]_{j} H _{[f_{j}]}}_{\approx 0}\Big] \\
&= \mathbb{E}\Big[ J_{f}^{T} H_{l} J_{f}\Big]= G
\end{aligned}
$$

Also Fisher (to be precise “empirical Fisher”) and generalized Gauss-Newton matrix are also identical. As a result we can approximate $$H$$ with Fisher. 
Assume we have a dataset $$D_{train}$$ containing (input, target) examples (x,y), and a neural network $$f_{\theta}(x)$$ with parameter vector $$\theta$$. Then Empirical Fisher is defined as follow:

$$
\begin{aligned}
& F = \mathbb{E}_{(x,y)\in D_{train} }\Big[ \nabla_\theta \nabla_\theta^{T} \Big]
\end{aligned}
$$

#### A block-wise Kronecker-factored (K-FAC) Fisher Approximation
Martens & Grosse [[paper](https://arxiv.org/pdf/1503.05671.pdf)] proposed an approximation to the Fisher as a Kronecker product F ≈ A ⊗ B which involves two smaller matrices. Specifically for a layer $$l$$ that receives input $$h$$ of size $$d_{in}$$ and computes linear pre-activations $$a = W^{T}h $$ of size $$d_{out}$$ followed by some non-linear activation, let the backpropagated gradient on $$a$$ be  $$\delta = \frac{\partial \mathcal{l}}{\partial a}$$. The gradients on parameter $$\theta^{l} = W$$ will be $$\Delta_{W}=\frac{\partial \mathcal{l}}{\partial W}=vec(h\partial^{T})$$.

The Kronecker factored approximation of corresponding $$ F = \mathbb{E}\Big[ \nabla_W \nabla_W^{T} \Big]$$ will use $$A= \mathbb{E}\big[hh^T\big]$$ and use $$B= \mathbb{E}\big[\delta \delta ^T\big]$$ which have matrices of size $$d_{in}*d_{in}$$, and $$d_{out}*d_{out}$$ respectively, whereas the Full $$F$$ would be of size $$d_{in}d_{out}* d_{in}d_{out}$$. Using Kronecker approximation, approximate entries of $$F^{l}$$ as follows: $$F_{iji’j’}^{l} = \mathbb{E}\Big[ \nabla_{W_{ij}}\nabla_{W_{i'j'}}^{T} \Big] =\mathbb{E}\Big[(h_{j}\delta_{j})((h_{i'}\delta_{j'})\Big]\approx \mathbb{E}\Big[(h_{i}h_{i'})\Big]\mathbb{E}\Big[(\delta_{j}\delta_{j'})\Big]$$

[[paper](https://arxiv.org/pdf/1806.03884.pdf)\|[code](https://github.com/wiseodd/natural-gradients)]
## Extending OBD and OBS to Structured Pruning
tbdS

$$
\begin{aligned}
& {\Delta \mathcal{L}}_{OBD} = \frac {1}{2} {\theta_{i}^{*}}^{T} F(i) {\theta_{i}}^{*}
\end{aligned}
$$

and 

$$
\begin{aligned}
& {\Delta \Theta}_{q} =-\theta_{i}^{*}  &  {\Delta \mathcal{L}}_{i} = \frac {1}{2} S_{ii} {\theta_{i}^{*}}^{T} A {\theta_{i}}^{*}
\end{aligned}
$$

tbdss

$$
\begin{aligned}
& \Delta \Theta =-\frac {\theta_{q}^{*}}{H_{qq}^{-1}}H^{-1} e_{q} &  {\Delta \mathcal{L}}_{OBS} = \frac {1}{2}\frac {{\theta_{i}^{*}}^{T} A {\theta_{i}}^{*}}{{S^-1}_{ii}}
\end{aligned}
$$




## EigenDamage: Structured Pruning in a KFE
TBD
[[paper](https://arxiv.org/pdf/1905.05934.pdf)\|[code](https://github.com/alecwangcq/EigenDamage-Pytorch)]


## Summary

TBD


## References

[1] LeCun et al.["Optimal Brain Damage."](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf). 1990.

[2] Hassibi et al. ["Optimal Brain Sergeon"](https://papers.nips.cc/paper/749-optimal-brain-surgeon-extensions-and-performance-comparisons.pdf). 1993.

[3] Thomas George, et al. ["Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis."](https://arxiv.org/pdf/1806.03884.pdf). 2018.

[4] Chaoqi Wang, et al. ["EigenDamage: Structured Pruning in the Kronecker-Factored Eigenbasis."](https://arxiv.org/pdf/1905.05934.pdf). 2019.

[5] James Martens and Roger Grosse ["Optimizing Neural Networks with Kronecker-factored Approximate Curvature"](https://arxiv.org/pdf/1503.05671.pdf). 2016.

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

## Background on Pruning Algorithms

The most popular pruning algorithms are OBD and OBS. Both of these algorithms are based on constructing a local model of the loss function and analytically predict the effect of perturbing the parameter vectors. They approximate the loss function L by a Taylor series. A perturbation delta theta of the parameter will change the loss function by: 

$$
\begin{aligned}
& \Delta L =  (\frac{\mathbf{\partial{L}}\ \mathbf{\partial{\Delta \Theta}}^\top})\mathbf{\Delta \Theta}
\end{aligned}
$$

### OBD
TBD
### OBS
TBD


## Summary

TBD


## References

[1] ["Optimal Brain Damage."](http://yann.lecun.com/exdb/publis/pdf/lecun-90b.pdf) - (LeCun et al., 1990).

[2] ["Optimal Brain Sergeon"](https://papers.nips.cc/paper/749-optimal-brain-surgeon-extensions-and-performance-comparisons.pdf,  (Hassibi et al., 1993). 

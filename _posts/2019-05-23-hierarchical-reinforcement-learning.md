---
layout: post
comments: true
title: "Hierarchical Reinforcement Learning"
date: 2019-05-23 00:15:06
tags: reinforcement-learning long-read
image: "A3C_vs_A2C.png"
---

> Abstract: In this post, we are going to look deep into Hierarchical Reinforcement Learning (HRL), and one of the moset important HRL algorithm proposed in recent years which is Options Framework, and its subsequent extentionns--option-critic.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}


## What is Reinforcement learning (RL)

Policy gradient is an approach to solve reinforcement learning problems. If you haven’t looked into the field of reinforcement learning, please first read the section ["A (Long) Peek into Reinforcement Learning >> Key Concepts"]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning %}#key-concepts) for the problem definition and key concepts.


### Notations

Here is a list of notations to help you read through equations in the post easily.

{: class="info"}
| Symbol | Meaning |
| ----------------------------- | ------------- |
| $$s \in \mathcal{S}$$ | States. |
| $$a \in \mathcal{A}$$ | Actions. |
| $$r \in \mathcal{R}$$ | Rewards. |
| $$S_t, A_t, R_t$$ | State, action, and reward at time step t of one trajectory. I may occasionally use $$s_t, a_t, r_t$$ as well. |
| $$\gamma$$ | Discount factor; penalty to uncertainty of future rewards; $$0<\gamma \leq 1$$. |
| $$G_t$$ | Return; or discounted future reward; $$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$. |
| $$P(s’, r \vert s, a)$$ | Transition probability of getting to the next state s’ from the current state s with action a and reward r. |
| $$\pi(a \vert s)$$ | Stochastic policy (agent behavior strategy); $$\pi_\theta(.)$$ is a policy parameterized by θ. |
| $$\mu(s)$$ | Deterministic policy; we can also label this as $$\pi(s)$$, but using a different letter gives better distinction so that we can easily tell when the policy is stochastic or deterministic without further explanation. Either $$\pi$$ or $$\mu$$ is what a reinforcement learning algorithm aims to learn. |
| $$V(s)$$ | State-value function measures the expected return of state s; $$V_w(.)$$ is a value function parameterized by w.|
| $$V^\pi(s)$$ | The value of state s when we follow a policy π; $$V^\pi (s) = \mathbb{E}_{a\sim \pi} [G_t \vert S_t = s]$$. |
| $$Q(s, a)$$ | Action-value function is similar to $$V(s)$$, but it assesses the expected return of a pair of state and action (s, a); $$Q_w(.)$$ is a action value function parameterized by w. |
| $$Q^\pi(s, a)$$ | Similar to $$V^\pi(.)$$, the value of (state, action) pair when we follow a policy π; $$Q^\pi(s, a) = \mathbb{E}_{a\sim \pi} [G_t \vert S_t = s, A_t = a]$$. |
| $$A(s, a)$$ | Advantage function, $$A(s, a) = Q(s, a) - V(s)$$; it can be considered as another version of Q-value with lower variance by taking the state-value off as the baseline. |



### What is Hard in RL

The goal of reinforcement learning is to find an optimal behavior strategy for the agent to obtain optimal rewards. The **policy gradient** methods target at modeling and optimizing the policy directly. The policy is usually modeled with a parameterized function respect to θ, $$\pi_\theta(a \vert s)$$. The value of the reward (objective) function depends on this policy and then various algorithms can be applied to optimize θ for the best reward.

The reward function is defined as:


$$
J(\theta) 
= \sum_{s \in \mathcal{S}} d^\pi(s) V^\pi(s) 
= \sum_{s \in \mathcal{S}} d^\pi(s) \sum_{a \in \mathcal{A}} \pi_\theta(a \vert s) Q^\pi(s, a)
$$

where $$d^\pi(s)$$ is the stationary distribution of Markov chain for $$\pi_\theta$$ (on-policy state distribution under π). 
For simplicity, the θ parameter would be omitted for the policy $$\pi_\theta$$ when the policy is present in the subscript of other functions; for example, $$d^{\pi}$$ and $$Q^\pi$$ should be $$d^{\pi_\theta}$$ and $$Q^{\pi_\theta}$$ if written in full.
Imagine that you can travel along the Markov chain’s states forever, and eventually, as the time progresses, the probability of you ending up with one state becomes unchanged --- this is the stationary probability for $$\pi_\theta$$. $$d^\pi(s) = \lim_{t \to \infty} P(s_t = s \vert s_0, \pi_\theta)$$ is the probability that $$s_t=s$$ when starting from $$s_0$$ and following policy $$\pi_\theta$$ for t steps. Actually, the existence of the stationary distribution of Markov chain is one main reason for why PageRank algorithm works. If you want to read more, check [this](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/).

It is natural to expect policy-based methods are more useful in the continuous space. Because there is an infinite number of actions and (or) states to estimate the values for and hence value-based approaches are way too expensive computationally in the continuous space. For example, in [generalized policy iteration]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning %}#policy-iteration), the policy improvement step $$\arg\max_{a \in \mathcal{A}} Q^\pi(s, a)$$ requires a full scan of the action space, suffering from the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).

Using *gradient ascent*, we can move θ toward the direction suggested by the gradient $$\nabla_\theta J(\theta)$$ to find the best θ for $$\pi_\theta$$ that produces the highest return. 


## What is Hierarchical Reinforcement learning (HRL)

Tons of policy gradient algorithms have been proposed during recent years and there is no way for me to exhaust them. I'm introducing some of them that I happened to know and read about.


### Options Framework

**REINFORCE** (Monte-Carlo policy gradient) relies on an estimated return by [Monte-Carlo]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning %}#monte-carlo-methods) methods using episode samples to update the policy parameter $$\theta$$. REINFORCE works because the expectation of the sample gradient is equal to the actual gradient:


$$
\begin{aligned}
\nabla_\theta J(\theta)
&= \mathbb{E}_\pi [Q^\pi(s, a) \nabla_\theta \ln \pi_\theta(a \vert s)] & \\
&= \mathbb{E}_\pi [G_t \nabla_\theta \ln \pi_\theta(A_t \vert S_t)] & \scriptstyle{\text{; Because } Q^\pi(S_t, A_t) = \mathbb{E}_\pi[G_t \vert S_t, A_t]}
\end{aligned}
$$

Therefore we are able to measure $$G_t$$ from real sample trajectories and use that to update our policy gradient. It relies on a full trajectory and that’s why it is a Monte-Carlo method. 

The process is pretty straightforward:
1. Initialize the policy parameter θ at random.
2. Generate one trajectory on policy $$\pi_\theta$$: $$S_1, A_1, R_2, S_2, A_2, \dots, S_T$$.
3. For t=1, 2, ... , T:
    1. Estimate the the return $$G_t$$;
    2. Update policy parameters: $$\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla_\theta \ln \pi_\theta(A_t \vert S_t)$$

A widely used variation of REINFORCE is to subtract a baseline value from the return $$G_t$$ to *reduce the variance of gradient estimation while keeping the bias unchanged* (Remember we always want to do this when possible). For example, a common baseline is to subtract state-value from action-value, and if applied, we would use advantage $$A(s, a) = Q(s, a) - V(s)$$ in the gradient ascent update. This [post](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/) nicely explained why a baseline works for reducing the variance, in addition to a set of fundamentals of policy gradient.



### The Option-critic Architecture

Two main components in policy gradient are the policy model and the value function. It makes a lot of sense to learn the value function in addition to the policy, since knowing the value function can assist the policy update, such as by reducing gradient variance in vanilla policy gradients, and that is exactly what the **Actor-Critic** method does. 


Actor-critic methods consist of two models, which may optionally share parameters:
* **Critic** updates the value function parameters w and depending on the algorithm it could be action-value $$Q_w(a \vert s)$$ or state-value $$V_w(s)$$.
* **Actor** updates the policy parameters θ for $$\pi_\theta(a \vert s)$$, in the direction suggested by the critic.

Let’s see how it works in a simple action-value actor-critic algorithm.
1. Initialize s, θ, w at random; sample $$a \sim \pi_\theta(a \vert s)$$.
2. For $$t = 1 \dots T$$:
    1. Sample reward $$r_t \sim R(s, a)$$ and next state $$s’ \sim P(s’ \vert s, a)$$;
    2. Then sample the next action $$a’ \sim \pi_\theta(a’ \vert s’)$$;
    3. Update the policy parameters: $$\theta \leftarrow \theta + \alpha_\theta Q_w(s, a) \nabla_\theta \ln \pi_\theta(a \vert s)$$;
    4. Compute the correction (TD error) for action-value at time t: <br/>$$\delta_t = r_t + \gamma Q_w(s’, a’) - Q_w(s, a)$$ <br/>and use it to update the parameters of action-value function:<br/> $$w \leftarrow w + \alpha_w \delta_t \nabla_w Q_w(s, a)$$
    5. Update $$a \leftarrow a’$$ and $$s \leftarrow s’$$.

Two learning rates, $$\alpha_\theta$$ and $$\alpha_w$$, are predefined for policy and value function parameter updates respectively.

### A3C

[[paper](https://arxiv.org/abs/1602.01783)\|[code](https://github.com/dennybritz/reinforcement-learning/tree/master/PolicyGradient/a3c)]

**Asynchronous Advantage Actor-Critic** ([Mnih et al., 2016](https://arxiv.org/abs/1602.01783)), short for **A3C**, is a classic policy gradient method with a special focus on parallel training. 

In A3C, the critics learn the value function while multiple actors are trained in parallel and get synced with global parameters from time to time. Hence, A3C is designed to work well for parallel training.

Let’s use the state-value function as an example. The loss function for state value is to minimize the mean squared error, $$J_v(w) = (G_t - V_w(s))^2$$ and gradient descent can be applied to find the optimal w. This state-value function is used as the baseline in the policy gradient update.

Here is the algorithm outline:
1. We have global parameters, θ and w; similar thread-specific parameters, θ’ and w’.
2. Initialize the time step $$t = 1$$
3. While $$T <= T_\text{MAX}$$:
    1. Reset gradient: dθ = 0 and dw = 0.
    2. Synchronize thread-specific parameters with global ones: θ’ = θ and w’ = w.
    3. $$t_\text{start}$$ = t and sample a starting state $$s_t$$.
    4. While ($$s_t$$ != TERMINAL) and $$t - t_\text{start} <= t_\text{max}$$:
        1. Pick the action $$A_t \sim \pi_{\theta’}(A_t \vert S_t)$$ and receive a new reward $$R_t$$ and a new state $$s_{t+1}$$.
        2. Update t = t + 1 and T = T + 1
    5. Initialize the variable that holds the return estimation $$R = \begin{cases} 
	0 & \text{if } s_t \text{ is TERMINAL} \\
	V_{w’}(s_t) & \text{otherwise}
	\end{cases}
	$$
    6. For $$i = t-1, \dots, t_\text{start}$$:
        1. $$R \leftarrow \gamma R + R_i$$; here R is a MC measure of $$G_i$$.
        2. Accumulate gradients w.r.t. θ’: $$d\theta \leftarrow d\theta + \nabla_{\theta’} \log \pi_{\theta’}(a_i \vert s_i)(R - V_{w'}(s_i))$$;<br/>Accumulate gradients w.r.t. w’: $$dw \leftarrow dw + 2 (R - V_{w’}(s_i)) \nabla_{w’} (R - V_{w’}(s_i))$$.
    7. Update asynchronously θ using dθ, and w using dw.

A3C enables the parallelism in multiple agent training. The gradient accumulation step (6.2) can be considered as a parallelized reformation of minibatch-based stochastic gradient update: the values of w or θ get corrected by a little bit in the direction of each training thread independently.


## Quick Summary

After reading through all the algorithms above, I list a few building blocks or principles that seem to be common among them:

* Try to reduce the variance and keep the bias unchanged to stabilize learning.
* Off-policy gives us better exploration and helps us use data samples more efficiently.
* Experience replay (training data sampled from a replay memory buffer);
* Target network that is either frozen periodically or updated slower than the actively learned policy network;
* Batch normalization;
* Entropy-regularized reward;
* The critic and actor can share lower layer parameters of the network and two output heads for policy and value functions.
* It is possible to learn with deterministic policy rather than stochastic one.
* Put constraint on the divergence between policy updates.
* New optimization methods (such as K-FAC).
* Entropy maximization of the policy helps encourage exploration.
* Try not to overestimate the value function.
* TBA more.


---

*If you notice mistakes and errors in this post, don't hesitate to contact me at [lilian dot wengweng at gmail dot com] and I would be very happy to correct them right away!*

See you in the next post :D


## References

[1] jeremykun.com [Markov Chain Monte Carlo Without all the Bullshit](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/)

[2] Richard S. Sutton and Andrew G. Barto. [Reinforcement Learning: An Introduction; 2nd Edition](http://incompleteideas.net/book/bookdraft2017nov5.pdf). 2017.

[3] John Schulman, et al. ["High-dimensional continuous control using generalized advantage estimation."](https://arxiv.org/pdf/1506.02438.pdf) ICLR 2016.

[4] Thomas Degris, Martha White, and Richard S. Sutton. ["Off-policy actor-critic."](https://arxiv.org/pdf/1205.4839.pdf) ICML 2012.

[5] timvieira.github.io [Importance sampling](http://timvieira.github.io/blog/post/2014/12/21/importance-sampling/)

[6] Mnih, Volodymyr, et al. ["Asynchronous methods for deep reinforcement learning."](https://arxiv.org/abs/1602.01783) ICML. 2016. 

[7] David Silver, et al. ["Deterministic policy gradient algorithms."](https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf) ICML. 2014.

[8] Timothy P. Lillicrap, et al. ["Continuous control with deep reinforcement learning."](https://arxiv.org/pdf/1509.02971.pdf) arXiv preprint arXiv:1509.02971 (2015).

[9] Ryan Lowe, et al. ["Multi-agent actor-critic for mixed cooperative-competitive environments."](https://arxiv.org/pdf/1706.02275.pdf) NIPS. 2017.

[10] John Schulman, et al. ["Trust region policy optimization."](https://arxiv.org/pdf/1502.05477.pdf) ICML. 2015.

[11] Ziyu Wang, et al. ["Sample efficient actor-critic with experience replay."](https://arxiv.org/pdf/1611.01224.pdf) ICLR 2017.

[12] Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc Bellemare. ["Safe and efficient off-policy reinforcement learning"](http://papers.nips.cc/paper/6538-safe-and-efficient-off-policy-reinforcement-learning.pdf) NIPS. 2016.

[13] Yuhuai Wu, et al. ["Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation."](https://arxiv.org/pdf/1708.05144.pdf) NIPS. 2017.

[14] kvfrans.com [A intuitive explanation of natural gradient descent](http://kvfrans.com/a-intuitive-explanation-of-natural-gradient-descent/)

[15] Sham Kakade. ["A Natural Policy Gradient."](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf). NIPS. 2002.

[16] ["Going Deeper Into Reinforcement Learning: Fundamentals of Policy Gradients."](https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/) - Seita's Place, Mar 2017.

[17] ["Notes on the Generalized Advantage Estimation Paper."](https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/) - Seita's Place, Apr, 2017.

[18] Gabriel Barth-Maron, et al. ["Distributed Distributional Deterministic Policy Gradients."](https://arxiv.org/pdf/1804.08617.pdf) ICLR 2018 poster.

[19] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor."](https://arxiv.org/pdf/1801.01290.pdf) arXiv preprint arXiv:1801.01290 (2018).

[20] Scott Fujimoto, Herke van Hoof, and Dave Meger. ["Addressing Function Approximation Error in Actor-Critic Methods."](https://arxiv.org/abs/1802.09477) arXiv preprint arXiv:1802.09477 (2018).

[21] Tuomas Haarnoja, et al. ["Soft Actor-Critic Algorithms and Applications."](https://arxiv.org/abs/1812.05905) arXiv preprint arXiv:1812.05905 (2018).

[22] David Knowles. ["Lagrangian Duality for Dummies"](https://cs.stanford.edu/people/davidknowles/lagrangian_duality.pdf) Nov 13, 2010.

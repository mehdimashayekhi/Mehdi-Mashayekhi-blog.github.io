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

Reinforcement learning is learning what to do—mapping states to actions—so as to maximize a numerical reward. The learner is not told which actions to perform, but instead must learn which actions lead to the most reward by trying them. Actions not only may affect the immediate reward, but also the next situation and, through that, all future rewards. Trial-and-error searching, and delayed reward—are the two most important unique features of reinforcement learning.


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

## What is Hierarchical Reinforcement learning (HRL)

Hierarchical reinforcement learning is a promising method which extends conventional reinforcement learning approaches, by exploiting temporal abstraction, where decisions are not needed at each step, but rather invoke the execution of temporally-extended actions which follow their own policies until termination. 

### Main limitations of RL, in Comparison to HRL

- Data inefficiency
- Scaling
- Generalization
- Abstraction


## HRL Algorithms

Tons of policy gradient algorithms have been proposed during recent years and there is no way for me to exhaust them. I'm introducing some of them that I happened to know and read about.


### Options Framework

[[paper](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)\|[code](https://github.com/mehdimashayekhi/Some-RL-Implementation)]

**Options** (Monte-Carlo policy gradient) relies on an estimated return by [Monte-Carlo]({{ site.baseurl }}{% post_url 2018-02-19-a-long-peek-into-reinforcement-learning %}#monte-carlo-methods) methods using episode samples to update the policy parameter $$\theta$$. REINFORCE works because the expectation of the sample gradient is equal to the actual gradient:


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

#### Options
TBD
#### MonteCarlo Model Learning
TBD
#### Intra-Option Model Learning
For Markov options, special temporal-difference methods can be used to learn usefully about the model of an option before the option terminates. We call these intra-option methods.



### The Option-critic Architecture

[[paper](https://arxiv.org/pdf/1609.05140.pdf)\|[code](https://github.com/mehdimashayekhi/Some-RL-Implementation)]

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

### Summary of Other HRL Algorithms
- Feudal Learning
- Successor Features (SFs) and Generalized policy improvement (GPI)
- Subgoal Discovery

## Quick Summary and Future Research

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

*If you notice mistakes and errors in this post, please don't hesitate to contact me at [mehdi dot mashayekhi dot 65 at gmail dot com] and I would be very happy to correct them right away!*



## References

[1] Richard S. Sutton and Andrew G. Barto. [Reinforcement Learning: An Introduction; 2nd Edition](http://incompleteideas.net/book/bookdraft2017nov5.pdf). 2017.

[2] Richard S. Sutton, et al.[Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf). 1999.

[3] Pierre-Luc Bacon, et al.[The Option-Critic Architecture](https://arxiv.org/pdf/1609.05140.pdf). 2016.

[4] André Barreto, et al.[Successor Features for Transfer in Reinforcement Learning](https://papers.nips.cc/paper/6994-successor-features-for-transfer-in-reinforcement-learning.pdf). 2017.

[5] André Barreto, et al.[Feudal Reinforcement Learning](http://www.cs.toronto.edu/~fritz/absps/dh93.pdf). 1993.

[6] Tejas D. Kulkarni, et al.[Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/abs/1604.06057). 2016.

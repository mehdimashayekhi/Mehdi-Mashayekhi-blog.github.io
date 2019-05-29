---
layout: post
comments: true
title: "Hierarchical Reinforcement Learning"
date: 2019-05-23 00:15:06
tags: reinforcement-learning
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
| $$I_o(s)$$ | An initiation function (precondition) for an option. |
| $$\pi_o(a \vert s)$$ | An internal policy (behavior) for an option. |
| $$\beta_o(s)$$ | a termination function (post-condition) for an option. |

### Some of Main Ingredients of Reinforcement Learning
- Value Functions
- Temporal Difference (TD) Learning
- Advantage Function

### Main limitations of RL

- Data inefficiency
- Scaling
- Generalization
- Abstraction

## What is Hierarchical Reinforcement learning (HRL)
Imagine you want to build a cooking robot. In order for the robot to cook we can imagine various levels of action abstraction:
- High-level actions: choosing a recipe, making a grocery list, getting groceries and finally cook.
- Medium-level actions: getting a pot, putting ingredients in the  pot, string until smooth, checking the recipe  
- Low-level actions: wrist and arm movement while stirring, pouring

All these steps have to be seamlessly integrated. Dividing the actions into various levels of abstraction is called temporal abstraction. 
Hierarchical reinforcement learning is a promising method which extends conventional reinforcement learning approaches, by exploiting temporal abstraction-- multiple-time step ``macro`` actions.
Temporal abstraction allows us to only exploring/computing values for interesting states, and transfer learning across problems/regions.


## HRL Algorithms

In the following we deep dive into one of the most important HRL algorithms which is ``Options Framework``, and summarize some of other HRL algorithms.



### Options Framework

[[paper](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf)\|[code](https://github.com/mehdimashayekhi/Some-RL-Implementation)]

**Option** is defined by a tuple tuple ($$I_o(s)$$, $$\pi_o(a \vert s)$$, $$\beta_o(s)$$). 

![OPTIONS]({{ '/assets/images/option.png' | relative_url }})
{: class="center" style="width: 70%;"}
*Fig. 1. Decision Making with Options. (Image source: [Richard S. Sutton, et al, 1999](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf))*

#### Monte Carlo Model Learning
What is meant by model learning is to determine the transition dynamics of an option (i.e., $$p_{ss’}^{o}$$), and the expected reward under an option (i.e., $$r_s^{o}$$), given experience and knowledge of $$o$$ (i.e., of its $$I$$, $$\pi$$, and $$\beta$$). A monte carlo approach is to execute the option to termination many times in each state
s, recording in each case the resultant next state s′, and cumulative discounted reward r. An incremental learning rule for this could update its model after each execution of $$o$$ by :

$$
\begin{aligned}
& r_s^{o} = r_s^{o} + \alpha [r-r_s^{o}] \\
& p_{sx}^{o} = p_{sx}^{o} + \alpha[\gamma^k \delta_{s’x}- p_{sx}^{o}]
\end{aligned}
$$

for all x ∈ S^+, where \delta_{s’x} = 1 if s′ = x and is 0 else, and where the step-size parameter, α, may be constant or may depend on the state, option, and time. For example here, α is 1 divided by the number of times that o has been experienced in s, then these updates maintain the estimates as sample averages of the experienced outcomes. However the averaging is done, we call these SMDP model-learning methods because, they are based on jumping from initiation to termination of each option.

#### Intra-Option Model Learning
For Markov options, special temporal-difference methods can be used to learn usefully about the model of an option before the option terminates. We call these intra-option methods.



### The Option-critic Architecture

[[paper](https://arxiv.org/pdf/1609.05140.pdf)\|[code](https://github.com/mehdimashayekhi/Some-RL-Implementation)]

![OPTION CRITIC]({{ '/assets/images/OPTION_CRITIC.png' | relative_url }})
{: class="center" style="width: 70%;"}
*Fig. 1. Decision Making with Options. (Image source: [Pierre-Luc Bacon, et al, 2016](https://arxiv.org/pdf/1609.05140.pdf))*


### Summary of Other HRL Algorithms
- Feudal Learning
- Successor Features (SFs) and Generalized policy improvement (GPI)
- Subgoal Discovery

## Quick Summary and Future Research

TBD


---

*If you notice mistakes and errors in this post, please don't hesitate to contact me at [mehdi dot mashayekhi dot 65 at gmail dot com]*



## References

[1] Richard S. Sutton and Andrew G. Barto. [Reinforcement Learning: An Introduction; 2nd Edition](http://incompleteideas.net/book/bookdraft2017nov5.pdf). 2017.

[2] Richard S. Sutton, et al.[Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf). 1999.

[3] Pierre-Luc Bacon, et al.[The Option-Critic Architecture](https://arxiv.org/pdf/1609.05140.pdf). 2016.

[4] André Barreto, et al.[Successor Features for Transfer in Reinforcement Learning](https://papers.nips.cc/paper/6994-successor-features-for-transfer-in-reinforcement-learning.pdf). 2017.

[5] André Barreto, et al.[Feudal Reinforcement Learning](http://www.cs.toronto.edu/~fritz/absps/dh93.pdf). 1993.

[6] Tejas D. Kulkarni, et al.[Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/abs/1604.06057). 2016.

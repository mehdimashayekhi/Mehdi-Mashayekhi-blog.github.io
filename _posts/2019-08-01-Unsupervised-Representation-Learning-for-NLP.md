---
layout: post
comments: true
title: "Unsupervised Representation Learning for NLP"
date: 2019-08-01 00:15:06
tags: nlp review
image: "A3C_vs_A2C.png"
---

> (WIP)Recently there has been a significant progress in regards to unsupervised representation learning in the domain of natural language processing such as OpenAI-GPT, BERT, Transformer-XL, and XLNet. XLNet has achieved SOTA performance on many NLP tasks. Just shortly after XLNet, Facebook AI’s model RoBERTa, outperformed XLNet on GLUE benchmark. The backbone of all these models is based on attention, and language modeling. In my opinion, there are really some interesting ideas in the XLNet paper, and in this post we are going to review this model, and look at some code snippets to fully understand the ideas present in the paper.


<!--more-->

{: class="table-of-content"}
* TOC
{:toc}


## Background
Given a text sequence $$\mathbf{x} = (x_1, \dots, x_T)$$,  AR language
modeling factorizes the likelihood into a forward product $$p(\mathbf{x}) = \prod_{t=1}^T p(x_{t}\mid \mathbf{x}_{<t})$$ or a backward one $$p(\mathbf{x}) = \prod_{t=1}^T p(x_{t}\mid \mathbf{x}_{>t})$$.
#### Autoregressive Language Modeling
$$
\begin{aligned}
& \max_{\theta} \log p_\theta(\mathbf{x}) = \sum_{t=1}^{T} \log p_\theta(x_t\mid \mathbf{x}_{<t})= \sum_{t=1}^{T} \log  \frac{\exp({h_{\theta}(\mathbf{x}_{1:t-1})}^\top e(x_t))}{\sum_{x'} \exp({h_{\theta}(\mathbf{x}_{1:t-1})}^\top e(x'))}
\end{aligned}
$$

#### Autoencoding

$$
\begin{aligned}
& \max_{\theta} \log p_\theta(\mathbf{\bar{x}} \mid \hat{\mathbf{x}}) \approx \sum_{t=1}^{T} m_{t} \log p_\theta(x_t\mid \mathbf{\hat{x}})= \sum_{t=1}^{T} m_{t} \log  \frac{\exp({H_{\theta}(\mathbf{\hat{x}}_{t})}^\top e(x_t))}{\sum_{x'} \exp({H_{\theta}(\mathbf{\hat{x}}_{t})}^\top e(x'))}
\end{aligned}
$$


## Optimization Objective: Permutation Language Modeling
TBD

![OPTIONS]({{ '/assets/images/permutation_example.png' | relative_url }})
{: class="center" style="width: 80%;"}
*Fig. 1. Illustration of the permutation language modeling objective for predicting x3 given the same input sequence x but with different factorization orders. (Image source: [Zhilin Yang, et al., 2019](https://arxiv.org/pdf/1906.08237.pdf))*

$$
\begin{aligned}
& \max_{\theta}  \mathbb{E}_{z \sim {Z_T} }\Big[\sum_{t=1}^{T} \log p_\theta(x_{z_t}\mid \mathbf{x}_{z<t})\Big]
\end{aligned}
$$

## Architecture: Two-Stream Self-Attention for Target-Aware Representations
TBD

![OPTIONS]({{ '/assets/images/Two-Stream-Self-Attention.png' | relative_url }})
{: class="center" style="width: 90%;"}
*Fig. 2. (a): Content stream attention, which is the same as the standard self-attention. (b): Query stream attention, which does not have access information about the content $$x_{z_t}$$ . (c): Overview of the permutation language modeling training with two-stream attention. (Image source: [Zhilin Yang, et al., 2019](https://arxiv.org/pdf/1906.08237.pdf))*

$$
\begin{aligned}
&  p_\theta(X_{z_t}=x\mid \mathbf{x}_{z<t})= \frac{\exp({e(x)^\top g_{\theta}(\mathbf{x}_{\mathbf{z}<t},z_t)} )}{\sum_{x'} \exp({e(x)^\top g_{\theta}(\mathbf{x}_{\mathbf{z}<t},z_t)})}
\end{aligned}
$$
### Two-Stream Self-Attention
TBD

$$g_{z_t}^{(m)} \leftarrow Attention(\mathbf{Q}=g_{z_t}^{(m-1)},\mathbf{KV}=h_{z<t}^{(m-1)}; \theta)$$

$$h_{z_t}^{(m)} \leftarrow Attention(\mathbf{Q}=h_{z_t}^{(m-1)},,\mathbf{KV}=h_{z<t}^{(m-1)}; \theta)$$

### Partial Prediction
TBD

## Incorporating Ideas from Transformer-XL
TBD
### Segment-level Recurrence
![OPTIONS]({{ '/assets/images/Transformer-XL-segment-level recurrence.png' | relative_url }})
{: class="center" style="width: 90%;"}
*Fig. 3. Transformer-XL with segment-level recurrence. (Image source: [Zhilin Yang, et al., 2019](https://arxiv.org/pdf/1901.02860.pdf))*
### Relative Positional Encodings

## Modeling Multiple Segments
## Code Snippets 
TBD
### Data Creation
TBD
### Modeling
TBD
### Training
TBD


---

*If you notice mistakes and errors in this post, please don't hesitate to contact me at [mehdi dot mashayekhi dot 65 at gmail dot com]*

---
Cited as:
```
@article{mashayekhi2019ULM,
  title   = " Unsupervised Representation Learning for NLP",
  author  = "Mashayekhi, Mehdi",
  journal = " https://mehdimashayekhi.github.io/mehdi-mashayekhi-blog.github.io/ ",
  year    = "2019",
  url     = " https://mehdimashayekhi.github.io/mehdi-mashayekhi-blog.github.io/2019/08/01/Unsupervised-Representation-Learning-for-NLP.html"
}
```


## References

[1] Ashish Vaswani, et al. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf). 2017.

[2] Jacob Devlin, et al. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf). 2018.

[3] Alec Radford, et al. [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). 2018.

[4] Zhilin Yang, et al. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf). 2019.

[5] Zhilin Yang, et al. [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf). 2019.

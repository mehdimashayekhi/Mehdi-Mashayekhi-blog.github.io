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
TBD
#### Autoregressive Language Modeling
$$
\begin{aligned}
& \max_{\theta} \log p_\theta(\mathbf{x}) = \sum_{t=1}^{T} \log p_\theta(x_t\mid \mathbf{x}_{<t})= \sum_{t=1}^{T} \log  \frac{\exp({h_{\theta}(\mathbf{x}_{1:t-1})}^\top e(x_t))}{\sum_{x'} \exp({h_{\theta}(\mathbf{x}_{1:t-1})}^\top e(x'))}
\end{aligned}
$$

#### Autoencoding

$$
\begin{aligned}
& \max_{\theta} \log p_\theta\big(\mathbf{\bar{x}} \mid \mathbf{\hat{x}}\big) \approx \sum_{t=1}^{T} m_{t} \log p_\theta(x_t\mid \mathbf{\hat{x}})= \sum_{t=1}^{T} \log  \frac{\exp({h_{\theta}(\mathbf{x}_{1:t-1})}^\top e(x_t))}{\sum_{x'} \exp({h_{\theta}(\mathbf{x}_{1:t-1})}^\top e(x'))}
\end{aligned}
$$


## Optimization Objective: Permutation Language Modeling
TBD

## Architecture: Two-Stream Self-Attention for Target-Aware Representations
TBD

## Incorporating Ideas from Transformer-XL
TBD

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

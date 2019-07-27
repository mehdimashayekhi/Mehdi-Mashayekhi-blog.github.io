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
Recently there has been a significant progress in regards to unsupervised representation learning in the domain of natural language processing such as [OpenAI GPT](https://blog.openai.com/language-unsupervised/), and [BERT](https://arxiv.org/abs/1810.04805). Among different unsupervised pre-training objectives, autoregressive (**AR**) language modeling and autoencoding (**AE**) have been the two most successful pre-training objectives. 

AR language modeling estimates the probability distribution of a text corpus with an autoregressive model. Given a text sequence $$\mathbf{x} = (x_1, \dots, x_T)$$,  AR language
modeling factorizes the likelihood into a forward product $$p(\mathbf{x}) = \prod_{t=1}^T p(x_{t}\mid \mathbf{x}_{<t})$$ or a backward one $$p(\mathbf{x}) = \prod_{t=1}^T p(x_{t}\mid \mathbf{x}_{>t})$$.

Specifically, AR language modeling maximizes the likelihood under the following forward autoregressive factorization:

$$
\begin{aligned}
& \max_{\theta} \log p_\theta(\mathbf{x}) = \sum_{t=1}^{T} \log p_\theta(x_t\mid \mathbf{x}_{<t})= \sum_{t=1}^{T} \log  \frac{\exp({h_{\theta}(\mathbf{x}_{1:t-1})}^\top e(x_t))}{\sum_{x'} \exp({h_{\theta}(\mathbf{x}_{1:t-1})}^\top e(x'))}
\end{aligned}
$$

Where $${h_{\theta}(\mathbf{x}_{1:t-1})}$$ is a context representation created by neural models such as [Transformer](https://arxiv.org/pdf/1706.03762.pdf), and $$e(x)$$ denotes the embedding of $$x$$. As you can see, AR language modeling is only trained to encode a uni-directional forward or backward context, and is not effective at modeling deep bidirectional contexts. But, most language understanding tasks typically need bidirectional context information. As a result, there is a gap between AR language modeling and effective pretraining. 

There has been some attempts for bidirectional AR language modeling such as [ELMo](https://arxiv.org/abs/1802.05365). ElLMo simply concatenated the left-to-right and right-to-left information, meaning that the representation couldn’t take advantage of both left and right contexts simultaneously.

In contrast, AE based pretraining does not perform density estimation, but it works towards reconstructing the original data from corrupted input. BERT is a notable example of AE. BERT replaces language modeling with a modified objective called “masked language modeling”. In this model, words in a sentence are randomly erased and replaced with a special token `[MASK]` with some small probability. Then, the model is trained to recover the erased tokens.  As density estimation is not part of the objective, BERT can utilize bidirectional contexts for reconstruction which also closes the bidirectional information gap in AR language modeling and improves performance.

Given the input $$\mathbf{x}$$ and the corrupted input $$\hat{\mathbf{x}}$$, and the masked tokens $$\bar{\mathbf{x}}$$, BERT optimizes the following objective by reconstructing the masked tokens from the corrupted input: 

$$
\begin{aligned}
& \max_{\theta} \log p_\theta(\mathbf{\bar{x}} \mid \hat{\mathbf{x}}) \approx \sum_{t=1}^{T} m_{t} \log p_\theta(x_t\mid \mathbf{\hat{x}})= \sum_{t=1}^{T} m_{t} \log  \frac{\exp({H_{\theta}(\mathbf{\hat{x}}_{t})}^\top e(x_t))}{\sum_{x'} \exp({H_{\theta}(\mathbf{\hat{x}}_{t})}^\top e(x'))}
\end{aligned}
$$

where $$m_{t} = 1$$ indicates $$x_t$$ is masked, and $$H_{\theta}$$ is a Transformer that maps a length-T text sequence $$x$$ into a sequence of hidden vectors $$H_{\theta}(\mathbf{x}) = [H_{\theta}(\mathbf{x}_{1}), \dots, H_{\theta}(\mathbf{x}_{T}) ]$$. 

Even though BERT achieves better performance than pretraining approaches that are based on autoregressive language modeling, there are two main issues with BERT. One is discrepancy between pretraining and fine tuning, since the `[MASK]` tokens are absent during fine tuning. Second is, BERT assumes the predicted tokens are independent of each other given the unmasked tokens, this is the reason to have $$\approx$$ in the above equation.

Considering these pros and cons of AR and AE, the researchers from CMU and Google proposed [XLNet](https://arxiv.org/pdf/1906.08237.pdf), a generalized autoregressive pretraining method that leverages the best of these two modeling. More specifically, XLNet offers the following advantages: 

1. Enables learning bidirectional contexts by simply maximizing the expected likelihood over **all possible permutations of the factorization order**. In contrast to a fixed forward or backward factorization. As a result, in expectation, each position learns to utilize contextual information from all positions, and captures bidirectional context. 
2. Since it does not rely on data corruption, it does not suffer from the pretrain-finetune discrepancy
3. Eliminating the independence assumption, since the autoregressive objective provides a natural way to use the product rule for factorizing the joint probability of the predicted tokens. 
4. Incorporating ideas from Transformer-XL including Segment-level Recurrence, and Relative Positional Encodings.
5. Because the factorization order is arbitrary and the target is ambiguous, a new reformulation of the Transformer-XL model is needed to eliminate the ambiguity. 

In the following we will go over the details of the XLNet model and its implementation details. 

## Optimization Objective: Permutation Language Modeling

Let $$Z_T $$ be the set of all possible permutations of the length-T index sequence [1, 2, . . . , T ]. And let $$z_t $$ and $$\mathbf{z}_{<t} $$ to denote the t-th element and the first t−1 elements of a permutation $$ \mathbf{z}< \in Z_T $$. Then, XLNet proposes the following permutation language modeling objective: 

$$
\begin{aligned}
& \max_{\theta}  \mathbb{E}_{z \sim {Z_T} }\Big[\sum_{t=1}^{T} \log p_\theta(x_{z_t}\mid \mathbf{x}_{z<t})\Big]
\end{aligned}
$$

Fig. 1, shows an example of predicting token x3 given the same input sequence x but under different factorization orders. For example under the factorization order [3 -> 2 -> 4 -> 1], token $$x_3$$ does not have any context to attend (except the previous memories).

![OPTIONS]({{ '/assets/images/permutation_example.png' | relative_url }})
{: class="center" style="width: 80%;"}
*Fig. 1. Illustration of the permutation language modeling objective for predicting x3 given the same input sequence x but with different factorization orders. (Image source: [Zhilin Yang, et al., 2019](https://arxiv.org/pdf/1906.08237.pdf))*

Basically, for a text sequence $$x$$, we sample a factorization order z at a time and decompose the likelihood $$p_{\theta} (x)$$ according to factorization order. Note that the in the above objective, only factorization order is permuted, and not the sequence order. It means, the initial sequence order is kept, and positional encodings corresponding to the initial sequence are used. The proposed method, relies on a appropriate `attention mask ` in Transformer to achieve permutation of the factorization order. 

let’s look at an example to better understand the difference between BERT and XLNet. Consider a sentence [New, York, is, a, city ], and assume both BERT and XLNet choose the two words [New, York] as the prediction targets and maximize `log p(New York | is a city)`. Also assume that XLNet samples the factorization order [is, a, city, New, York]. In this case, BERT and XLNet reduce to the following objectives respectively: 

$$
\begin{aligned}
& \jmath_{BERT} = \log p(New | is \ a \ city)+\log p(York | is \ a \ city)\\

& \jmath_{XLNet} = \log p(New | is \ a \ city)+\log p(York |\color{red}{New} \ is \ a \ city)

\end{aligned}
$$

## Architecture: Two-Stream Self-Attention for Target-Aware Representations

$$
\begin{aligned}
&  p_\theta(X_{z_t}=x\mid \mathbf{x}_{z<t})= \frac{\exp({e(x)^\top g_{\theta}(\mathbf{x}_{\mathbf{z}<t},z_t)} )}{\sum_{x'} \exp({e(x)^\top g_{\theta}(\mathbf{x}_{\mathbf{z}<t},z_t)})}
\end{aligned}
$$

![OPTIONS]({{ '/assets/images/Two-Stream-Self-Attention.png' | relative_url }})
{: class="center" style="width: 90%;"}
*Fig. 2. (a): Content stream attention, which is the same as the standard self-attention. (b): Query stream attention, which does not have access information about the content $$x_{z_t}$$ . (c): Overview of the permutation language modeling training with two-stream attention. (Image source: [Zhilin Yang, et al., 2019](https://arxiv.org/pdf/1906.08237.pdf))*

### Two-Stream Self-Attention
TBD

$$g_{z_t}^{(m)} \leftarrow Attention(\mathbf{Q}=g_{z_t}^{(m-1)},\mathbf{KV}=h_{z<t}^{(m-1)}; \theta)$$

$$h_{z_t}^{(m)} \leftarrow Attention(\mathbf{Q}=h_{z_t}^{(m-1)},,\mathbf{KV}=h_{z\leq{t}}^{(m-1)}; \theta)$$

### Partial Prediction
TBD

$$
\begin{aligned}
& \max_{\theta}  \mathbb{E}_{ \mathbf{z} \sim {Z_T} }\Big[\log p_\theta(x_{z>c}\mid \mathbf{x}_{z\leq{c}}) \Big] = \mathbb{E}_{ \mathbf{z} \sim {Z_T} }\Big[\sum_{t=c+1}^{ \mathbf{|z|}} \log p_\theta(x_{z>c}\mid \mathbf{x}_{z\leq{c}})\Big]
\end{aligned}
$$

## Incorporating Ideas from Transformer-XL
TBD
### Segment-level Recurrence
![OPTIONS]({{ '/assets/images/Transformer-XL-segment-level recurrence.png' | relative_url }})
{: class="center" style="width: 90%;"}
*Fig. 3. Transformer-XL with segment-level recurrence. (Image source: [Zhilin Yang, et al., 2019](https://arxiv.org/pdf/1901.02860.pdf))*
### Relative Positional Encodings

![OPTIONS]({{ '/assets/images/positional_encoding.png' | relative_url }})
{: class="center" style="width: 80%;"}
*Fig. 4. Relative positional encoding illustration.

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

[4] Matthew E. Peters , et al. [Deep contextualized word representations ](https://arxiv.org/abs/1802.05365). 2018.

[5] Zhilin Yang, et al. [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/pdf/1901.02860.pdf). 2019.

[6] Zhilin Yang, et al. [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf). 2019.

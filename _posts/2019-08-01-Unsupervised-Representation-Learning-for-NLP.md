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

Consider that we have a query of length `seq_len`, a memory of  length `mem_len`, and key of length `klen= mem_len+seq_len`. Note that the relative distance $$(i-j)$$ between query $$q_i$$ and key vector $$k_j$$ can only be integer form 0 to `mem_len+seq_len-1`. So, eventually we want a positional matrix of shape `[(seq_len)  x (mem_len+seq_len)]`.   To get the desired matrix, it’s efficient  to construct a matrix of shape `[(seq_len)  x (mem_len+2*seq_len)]`, and then sliced it out to get the desired parts and reshape. Fig. 4 shows an example in which we have a memory of length 3, a sequence of length 5, and a tensor of shape `[5 x (3+2*5)]`. And imagine query in reversed order. So the relative distance for the first token of query, varies from 5 to 12, and so on and so forth. The red rectangles show the desired slices for each token of the query. Look at the code snippets below on the implementation details.  

![OPTIONS]({{ '/assets/images/positional_encoding_illu.png' | relative_url }})
{: class="center" style="width: 80%;"}
*Fig. 4. Relative positional encoding illustration.

## Modeling Multiple Segments

TBD

## Code Snippets 

Note that the authors, have released the full implementation [here]( https://github.com/zihangdai/xlnet). My goal here is to summarize the implementation of important components of the model, provide additional comments, and have the algorithm and implementation in one place.

### Data Creation
TBD
### Modeling
TBD

#### XLNet Model

Here is how we initialize the model:

```python
class XLNet(nn.Module):
    """
        Args:
        
        inp_k: int32 Tensor in shape [seq_len, bsz], the input token IDs.
        seg_id: int32 Tensor in shape [seq_len, bsz], the input segment IDs.
        input_mask: float32 Tensor in shape [seq_len, bsz], the input mask.
          0 for real tokens and 1 for padding.
        mems: a list of float32 Tensors in shape [mem_len, bsz, d_model], memory
          from previous batches. The length of the list equals n_layer.
          If None, no memory is used.
        perm_mask: float32 Tensor in shape [seq_len, seq_len, bsz].
          If perm_mask[i, j, k] = 0, i attend to j in batch k;
          if perm_mask[i, j, k] = 1, i does not attend to j in batch k.
          If None, each position attends to all the others.
        target_mapping: float32 Tensor in shape [num_predict, seq_len, bsz].
          If target_mapping[i, j, k] = 1, the i-th predict in batch k is
          on the j-th token.
          Only used during pretraining for partial prediction.
          Set to None during finetuning.
        inp_q: float32 Tensor in shape [seq_len, bsz].
          1 for tokens with losses and 0 for tokens without losses.
          Only used during pretraining for two-stream attention.
          Set to None during finetuning.

        n_layer: int, the number of layers.
        d_model: int, the hidden size.
        n_head: int, the number of attention heads.
        d_head: int, the dimension size of each attention head.
        d_inner: int, the hidden size in feed-forward layers.
        n_token: int, the vocab size.

        mem_len: int, the number of tokens to cache.
        reuse_len: int, the number of tokens in the currect batch to be cached
          and reused in the future.
        bi_data: bool, whether to use bidirectional input pipeline.
          Usually set to True during pretraining and False during finetuning.

      """
    def __init__(self, n_token, n_layer, n_head, d_head, d_inner, d_model,
                 attn_type, bi_data, reuse_len, mem_len):
        super(XLNet, self).__init__()

        self.n_token = n_token
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.d_model = d_model
        self.attn_type = attn_type
        self.bi_data = bi_data
        self.reuse_len = reuse_len
        self.mem_len = mem_len

        self.embedding = nn.Embedding(n_token, d_model)

        # untie the biases in attention.
        self.r_w_bias = nn.Parameter(torch.randn(self.n_layer,
                                                  self.n_head,self.d_head))
        self.r_r_bias = nn.Parameter(torch.randn(self.n_layer,
                                                  self.n_head, self.d_head))

        ##### Segment embedding
        self.r_s_bias = nn.Parameter(torch.randn(self.n_layer,
                                                  self.n_head,self.d_head))
        self.seg_embed = nn.Parameter(torch.randn(self.n_layer, 2,
                                                   self.n_head, self.d_head))

        self.mask_emb = nn.Parameter(torch.randn(1, 1, d_model))

        # post-attention projection (back to `d_model`)
        self.proj_o = nn.Parameter(torch.randn(self.d_model,
                                                self.n_head, self.d_head))

        #### Project hidden states to a specific head with a 4D-shape.
        self.q_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head))
        self.k_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head))
        self.v_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head))
        self.r_proj_weight = nn.Parameter(torch.randn(self.d_model,
                                                       self.n_head, self.d_head))

        self.layer_norm = nn.LayerNorm(d_model)

        self.conv1 = nn.Linear(d_model, d_inner)
        self.conv2 = nn.Linear(d_inner, d_model)
        self.relu = nn.ReLU(inplace=True)

        self.softmax_b = nn.Parameter(torch.zeros(self.n_token))
    
```
Here is the forward pass of the tensorflow model:

```python
    def forward(self, inp_k, seg_id, input_mask, mems, perm_mask, target_mapping, inp_q):
        new_mems = []
        qlen = inp_k.shape[0] # qlen=seq_len
        klen = mlen + qlen

        # data mask: input mask & perm mask
        data_mask = perm_mask

        # all mems can be attended to
        mems_mask = torch.zeros([data_mask.shape[0], mlen, bsz],dtype=torch.float32)   # shape: [seq_len,mem_len,bsz]
                                 
        data_mask = torch.cat([mems_mask, data_mask], dim=1)   # shape: [seq_len, mem_len+seq_len, bsz ] 
        attn_mask = data_mask[:, :, :, None] # shape: [seq_len, mem_len+seq_len, bsz,1 ]

        non_tgt_mask = -torch.eye(qlen, dtype=torch.float32) # [qlen, qlen]
        non_tgt_mask = torch.cat([torch.zeros([qlen, mlen], dtype=torch.float32),non_tgt_mask],dim=-1) # [qlen, klen]                               
        non_tgt_mask = (attn_mask + non_tgt_mask[:, :, None, None]).gt(0).type(dtype=torch.float32)   # query token can attend to itself because of -eye

        # As mentioned in the paper the first layer query stream is initialized with a trainable vector, i.e. g^(0)_i = w
        # while the content stream is set to the corresponding word embedding, i.e. h(0) = e(xi).

        ##### Word embedding
        lookup_table = self.embedding
        word_emb_k = lookup_table(input_mask) # shape [seq_len x bsz * d_model]
        word_emb_q = self.mask_emb.repeat(target_mapping.shape[0], bsz, 1) # shape [num_predict x bsz * d_model]

        #### Figure 2(a), Content Stream(Original Attention), h^(0)_t = e(x_i) = e(inp_k)
        output_h = self.Dropout(word_emb_k) #[seq_len x bsz x dmodel]
        #### Query Stream, g^(0)_t = w
        #### the first layer query stream is initialized with a trainable vector
        output_g = self.Dropout(word_emb_q) # shape [num_predict x bsz * d_model]

        ##### Segment embedding
        # paper
        # Given a pair of positions i and j in the sequence, if i and j are from the same segment
        # `1` indicates not in the same segment
        # Convert `seg_id` to one-hot `seg_mat`
        mem_pad = torch.zeros([mlen, bsz], dtype=torch.int32)
        cat_ids = torch.cat([mem_pad, seg_id], dim=0) # shape [klen x bsz]
        #compare every element of one row of seg_id with all the elements in all rows of cat_ids
        seg_mat = (~torch.eq(seg_id[:, None], cat_ids[None, :])).type(torch.long)
        seg_mat = torch.eye(2, dtype=torch.float32)[seg_mat] # [qlen x klen x bsz x 2] one hot

        ##### Positional encoding
        pos_emb = self.relative_positional_encoding(
            qlen, klen, self.d_model, self.clamp_len, self.attn_type, self.bi_data,
            bsz=bsz, dtype=torch.float32)
        pos_emb = self.Dropout(pos_emb)  #[(klen+qlen) x 1 x d_model]

        ##### Attention layers
        if mems is None:
            mems = [None] * self.n_layer

        for i in range(self.n_layer):
            # cache new mems
            new_mems.append(self._cache_mem(output_h, mems[i], self.mem_len, self.reuse_len))

            # segment bias
            r_s_bias_i = self.r_s_bias[i]
            seg_embed_i = self.seg_embed[i]

            # output_h, output_g SHAPES  =  #[seq_len x bsz x dmodel] ;#[num_predict x bsz x dmodel]

            output_h, output_g = self.two_stream_rel_attn(
                h=output_h, # [seq_len x bsz x d_model]
                g=output_g,  # [num_predict x bsz x d_model]
                r=pos_emb,   #[(klen+qlen+1) x 1 x d_model]
                r_w_bias= self.r_w_bias[i], #[n_layer x n_head x d_head]
                r_r_bias= self.r_r_bias[i], #[n_layer x n_head x d_head]
                seg_mat=seg_mat, # [qlen x klen x bsz x 2] one hot
                r_s_bias=r_s_bias_i, #[n_layer x n_head x d_head]
                seg_embed=seg_embed_i, #[2 x n_head x d_head]
                attn_mask_h=non_tgt_mask, #[seq_len, mem_len+q_len, bsz,1 ]  query can attend to itself
                attn_mask_g=attn_mask,    # [seq_len, mem_len+q_len, bsz,1 ]  can not attend to itself
                mems=mems[i],             # [mem_len x bsz x d_model]
                target_mapping=target_mapping) # [num_predict, seq_len, 1(=bsz)]

            output_g = self.positionwise_ffn(inp=output_g) # [num_predict x bsz x d_inner]
            output_h = self.positionwise_ffn(inp=output_h) # [seq_len x bsz x d_inner]

        if inp_q is not None:
            output = self.Dropout(output_g)
        else:
            output = self.Dropout(output_h)

        logits = torch.einsum('ibd,nd->ibn', output, lookup_table.weight) + self.softmax_b # [num_predict x bsz x n_token]

        return logits, new_mems
```

Similar to [Transformer](https://arxiv.org/pdf/1706.03762.pdf), we use sine and cosine functions of different frequencies, to get the positional encoding:

$$ PE_{(pos,2i)} = sin(\frac{pos}{10000^{\frac{2i}{d_model}}})$$

$$ PE_{(pos,2i+1)} = cos(\frac{pos}{10000^{\frac{2i}{d_model}}})$$

```python
def relative_positional_encoding(qlen, klen, d_model):
    freq_seq = torch.arange(0, d_model, 2.0)
    inv_freq = 1 / (10000 ** (freq_seq / d_model))
    beg, end = klen - 1, -1 # note that klen = mlen + qlen
    pos_seq = torch.arange(beg, end, -1.0)
    sinusoid_inp = torch.einsum('i,d->id', pos_seq, inv_freq) # shape [(klen+qlen) x 1 x d_model/2]
    pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1) # shape [(klen+qlen) x d_model]
    pos_emb = pos_emb[:, None, :] # shape [(klen+qlen) x 1 x d_model]
    return pos_emb
```

Here we cache hidden states into memory. Note tha, as mentioned in the [Transformer-XL](https://arxiv.org/pdf/1901.02860.pdf), we stop the gradient for the memory.  

```python
def _cache_mem(self, curr_out, prev_mem, mem_len, reuse_len=None):
    with torch.no_grad():
        if mem_len is None or mem_len == 0:
            return None
        else:
            if reuse_len is not None and reuse_len > 0:
                curr_out = curr_out[:reuse_len]

            if prev_mem is None:
                new_mem = curr_out[-mem_len:]
            else:
                new_mem = torch.cat([prev_mem, curr_out], dim=0)[-mem_len:]

        return new_mem
```

Here is the implementation details of the two-stream attention with a Transformer-XL backbone. This function implements the following formulas, described in A.2 appendix of the paper:

$$\hat{h_{z_t}^{(m)}} = \text{LayerNorm}(h_{z_t}^{(m-1)} + \text{RelAttn}(h_{z_t}^{(m-1)},\big[\tilde{h}^{(m-1)}, h_{\mathbf{z}\leq{t}}^{m-1}]))$$

$$\hat{g_{z_t}^{(m)}} = \text{LayerNorm}(g_{z_t}^{(m-1)} + \text{RelAttn}(g_{z_t}^{(m-1)},\big[\tilde{h}^{(m-1)}, h_{\mathbf{z}\leq{t}}^{m-1}]))$$

\tilde{
```python
    def two_stream_rel_attn(self, h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias,
                            seg_embed, attn_mask_h, attn_mask_g, target_mapping):
        scale = 1 / (self.d_head ** 0.5)  # this is scaling factor in Scaled Dot-Product Attention

        # content based attention score
        if mems is not None and len(mems.size()) > 1:
            cat = torch.cat([mems, h], dim=0)  # cat shape [(mem_len+seq_len) x bsz x d_model]
        else:
            cat = h  

        # content-based key head
        k_head_h = self.head_projection(cat, 'k')   # k shape [(mem_len+seq_len) x bsz x n_head x d_head]

        # content-based value head
        v_head_h = self.head_projection(cat, 'v')  # v shape [(mem_len+seq_len) x bsz x n_head x d_head]

        # position-based key head
        k_head_r = self.head_projection(r, 'r')    # r shape [(k_len+q_len+1) x 1 x n_head x d_head]

        ##### h-stream
        # content-stream query head
        q_head_h = self.head_projection(h, 'q')   # q shape [seq_len x bsz x n_head x d_head]

        # core attention ops
        # hˆ(m)_zt = LayerNorm(h^(m-1)_zt + RelAttn(h^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        attn_vec_h = self.rel_attn_core(
            q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
            r_r_bias, r_s_bias, attn_mask_h, scale) # [q_len x bsz x n_head x d_head]

        # post processing
        output_h = self.post_attention(h, attn_vec_h)  #[seq_len x bsz x dmodel]; seq_len or q_len

        ##### g-stream
        # query-stream query head
        q_head_g = self.head_projection(g, 'q')  # [num_predict x bsz x n_head x d_head]

        # core attention ops
        # gˆ(m)_zt = LayerNorm(g^(m-1)_zt + RelAttn(g^(m-1)_zt + [h~^(m-1), hT(m-1)_z<=t]))
        if target_mapping is not None:
            q_head_g = torch.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping) # [seq_len x bsz x n_head x d_head]
            attn_vec_g = self.rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
                r_r_bias, r_s_bias, attn_mask_g, scale)  # only qury and att mask are different; # [q_len x bsz x n_head x d_head]
            attn_vec_g = torch.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping) # [num_predict x bsz x n_head x d_head]
        else:
            attn_vec_g = self.rel_attn_core(
                q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat, r_w_bias,
                r_r_bias, r_s_bias, attn_mask_g, scale)

        # post processing
        output_g = self.post_attention(g, attn_vec_g) #[num_predict x bsz x dmodel]

        return output_h, output_g
```

Here is the implementation details of the core relative positional attention operations. Note that the backbone of this, is the scaled Dot-Product Attention described in [Transformer](https://arxiv.org/pdf/1706.03762.pdf):


$$ \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^\top}{\sqrt{d_k}})V $$

```python
def rel_attn_core(self, q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                  r_w_bias, r_r_bias, r_s_bias, attn_mask, scale):

    """Core relative positional attention operations."""

    # content based attention score
    ac = torch.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h) # [seq_len x (mem_len+seq_len) x bsz x n_head]

    # position based attention score
    bd = torch.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)  # [seq_len x (klen+q_len+1) x bsz x n_head]
    bd = self.rel_shift(bd, klen=ac.shape[1])  #[seq_len x klen x bsz x n_head]

    # segment based attention score
    if seg_mat is None:
        ef = 0
    else:
        ef = torch.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed) # [seq_len x bsz x n_head x 2]
        ef = torch.einsum('ijbs,ibns->ijbn', seg_mat, ef)                 # [seq_len x k_len x bsz x n_head]

    # merge attention scores and perform masking
    attn_score = (ac + bd + ef) * scale  # [seq_len x k_len x bsz x n_head]
    if attn_mask is not None:
        # attn_score = attn_score * (1 - attn_mask) - 1e30 * attn_mask
        attn_score = attn_score - 1e30 * attn_mask  # if att_mask is one; then the attn_prob becomes zero

    # attention probability
    attn_prob = F.softmax(attn_score, dim=1)
    attn_prob = self.DropAttn(attn_prob) # [seq_len x k_len x bsz x n_head]; seq_len is equal q_len

    # attention output
    attn_vec = torch.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h) # [q_len x bsz x n_head x d_head]

    return attn_vec
```

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

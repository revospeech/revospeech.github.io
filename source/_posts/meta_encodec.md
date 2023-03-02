---
title: 音频编解码 | 论文笔记 | Encodec
date: 2023-01-15 11:40:36
tags: [音频编解码, 音频生成, 语音合成]
categories: [论文笔记]
copyright_info: true
toc: true
mathjax: true
comment: false
---

**Encodec** 是 Meta AI 于 2022 年 10 月份发表的神经网络音频编解码方法，达到了比之前介绍的 Google 提出的 **SoundStream** 更优的效果。从思想上和 SoundStream 几乎没有差别，沿用了 Encoder-Decoder 结构和 VQ 向量量化方法。本文对 Encodec 与 SoundStream 一致的部分不予赘述，只针对相关的改进进行分析和总结。SoundStream 的论文解读详见[链接](https://revospeech.github.io/2023/01/14/lyra_v2_soundstream)。

<!-- more -->

| 会议/期刊 | 年份 | 题目 | 链接 |
| :---: | :---: | :---: | :---: |
| arxiv | 2022 | High Fidelity Neural Audio Compression | [https://arxiv.org/abs/2210.13438](https://arxiv.org/abs/2210.13438) |

<br>

神经网络编解码器对音频的压缩通常是有损的，实际可用的编解码器需要解决两大问题：第一，编解码器需要覆盖到各种各样的音频，即泛化性强，对于神经网络模型来说就是：不能过拟合到训练集、不能只在某种类型的音频上有比较好的效果；第二，编解码器需要高效地进行音频压缩，高效体现在`实时性`（从时间的角度）和`低比特率`（从数据大小的角度）两方面。

Meta AI 提出的 Encodec 仍然是集中解决上述问题。为了提高模型的泛化性，一方面模型训练时使用更大更多样的训练数据集，另一方面在训练时使用了 GAN 的思想，引入判别器使得解码恢复的音频质量更高。为了提高模型的高效性，一方面对模型的复杂度限制在单核 CPU 上要求是实时的，另一方面引入 RVQ（残差向量量化）进行量化，最大程度降低压缩后音频的比特率。

### Encodec 模型结构

#### Encoder-Decoder

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/encodec_model.jpg" width = "750"/>

如上图所示，Encodec 采取和 SoundStream 几乎完全一样的 Encodecor-Decoder 结构。对于 24 kHz 的音频，Encodec 和 SoundStream 一样，经过 Encoder 之后进行了 320 倍的降采样，帧率降低至 24000 / 320 = 75 Hz；此外，Encodec 的目标输入还增加了对 48 kHz 音频的支持，经过 320 倍降采样后帧率降低至 150 Hz。

Encodec 支持流式和非流式两种场景，流式适合于低时延的应用，非流式适合音频质量要求更高的应用：对于非流式场景，设每个一维卷积的卷积核大小为 K，跳步 stride 为 S，需要 padding ${K - S}$ 个时间步，非流式时将 padding 平分到第一个时间步之前和最后一个时间步之后；对于流式场景，所有的 padding 都放在第一个时间步之前。Encodec 在非流式场景下，将音频按照 1s 进行片段的切分，每两个片段之间保留 10 ms 的重叠（避免毛刺噪声）；此外，Encodec 在输入编码器前对音频进行 normalize，解码器的输出再反归一化（de-normalize），因此编码器端还需要将 normalize 的均值方差信息传送到解码器端。

#### RVQ 残差向量量化

和 SounStream 一样，Encodec 也用到了 RVQ 多阶段量化的方法，将多个 RVQ 模块叠加，可以在训练和推理时经过不同个数的量化器模块，进而实现不同的比特率，具有很高的灵活性。[SoundStream 论文笔记](https://revospeech.github.io/2023/01/14/lyra_v2_soundstream)中我们介绍了 RVQ 的 codebook 初始化、参数更新方法和 EMA（指数移动平均）的训练技巧，此处对 RVQ 的细节再进行一些补充。

##### Straight-Through Estimator

参考 [VQ-VAE](https://arxiv.org/pdf/1711.00937.pdf) 论文中的表述，设 Encoder 的输出为 $z_e(x)$，经过向量量化器之后，$z_e(x)$ 会被映射到 codebook 中和 $z_e(x)$ 欧氏距离最小的向量 $e_{k}$ 得到量化后的向量 $z_q(x)$，数学表达式为：

$$z_q(x) = e_k, k = arg\min_j \vert\vert z_e(x) - e_j \vert\vert_2, j = 1, 2, ..., N$$

上式中 $N$ 代表 codebook 码本的大小，即包含向量的总数。该公式中，从量化前的 Encoder 输出 $z_e(x)$，到量化后的码本中的向量 $z_q(x)$ ，是一个求最值（欧氏距离最小）的操作，没有定义反向传播时梯度的计算方法，Encodec 采用了和 VQ-VAE 一样的操作，直接将 $z_q(x)$ 的梯度拷贝给 $z_e(x)$，属于一种 **Straight-Through Estimator** 的梯度更新策略。

> **Straight-Through Estimator** 在 Yoshua Bengio 的[论文](https://arxiv.org/pdf/1308.3432.pdf) 中有详细的介绍，简单理解就是将神经网络反向传播时中某一处的梯度直接作为另外一处的梯度值，相当于这两处之间的网络近似为恒等函数（Identity Function）。VQ 训练 codebook 中的向量是 Straight-Through Estimator 方法的典型应用之一，Encodec 也采用了该方法。

<br>
<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/vq-vae.jpg" width = "750"/>

上图是 VQ-VAE 的图例，可以作为 Encoder + VQ + Decoder 框架的典型结构，本文中 RVQ 里的每个 quantizer 都是按照 VQ-VAE 中的 VQ 方法来实现的。下面使用示例代码进行讲解。代码来源：https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py 。

```python
def forward(self, latents: Tensor) -> Tensor:
    # 1. 输入 latents 是 encoder 的输出
    latents = latents.permute(0, 2, 3, 1).contiguous()
    # [B x D x H x W] -> [B x H x W x D]

    latents_shape = latents.shape
    flat_latents = latents.view(-1, self.D)  # [BHW x D]

    # 2. 计算各 latent 向量和 codebook 各 embedding 之间的欧氏距离
    # Compute L2 distance between latents and embedding weights
    dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
           torch.sum(self.embedding.weight ** 2, dim=1) - \
           2 * torch.matmul(flat_latents, self.embedding.weight.t())  
           # [BHW x K]

    # 3. 根据欧式距离最小的原则选择相应的 codebook index
    # 注意此处使用了 torch.argmin 函数，这个函数操作是不可导的
    # 本代码采用的是 straight-through estimator
    # Get the encoding that has the min distance
    encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

    # 4. 根据获取的 index 得到量化后的 embedding
    # Convert to one-hot encodings
    device = latents.device
    encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
    encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

    # Quantize the latents
    quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
    quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

    # 5. 计算 VQ 相关的损失函数
    # Compute the VQ Losses
    # # 对应于「损失函数分析」中的第三项损失函数
    commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
    # # 对应于「损失函数分析」中的第二项损失函数
    embedding_loss = F.mse_loss(quantized_latents, latents.detach())

    vq_loss = commitment_loss * self.beta + embedding_loss

    # 6. 关键的一步：straight-through estimator
    # Add the residue back to the latents
    quantized_latents = latents + (quantized_latents - latents).detach()

    return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]
```

上述代码给出了 VQ 向量量化的 Pytorch 实现。其中，计算 Encoder 输出在 codebook 中距离最近向量 index 的代码是：
```python
encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)
```
`torch.argmin` 是无法求导的；因此，体现 Straight-Through Estimator 思想的关键代码是：
```python
quantized_latents = latents + (quantized_latents - latents).detach()
```
这行代码表示：正向传播时，量化后的结果 quantized_latents 保持不变；但在反向传播时，detach()的部分梯度为 0，因此量化后的 quantized_latents 和输入的量化前 latents 的梯度相同，体现了 Straight-Through Estimator 的思想，从而解决了 VQ 在计算欧氏距离最小的 codebook 编号时 argmin 函数不可导的问题。事实上，VQ 还有其他方式规避不可导的问题，比如**基于 Gumbel-Softmax 的量化**，本文最后会补充介绍。

---

##### VQ 的损失函数分析

一个完整的 VQ 模型，损失函数一般包含三部分，如下面公式所示，后文将详细介绍。

$$ L_{total} = L_{rec} + \vert\vert sg[z_e(x)] - e \vert\vert_2^2 + \vert\vert z_e(x) - sg[e] \vert\vert_2^2$$

1. **重建损失函数** $L_{rec}$：用于 Encoder 的输入（编码前）和 Decoder 的输出（解码后）之间的差异。$L_{rec}$ 只会影响到 Encoder 和 Decoder 的参数更新，由于前文提到的 Straight-Through Estimator 直接将 $z_q(x)$ 的梯度拷贝给 $z_e(x)$，因此 $L_{rec}$ 不会影响 VQ 中 codebook 的各向量。

2. **VQ 的 codebook 损失函数**：第二项 $\vert\vert sg[z_e(x)] - e \vert\vert_2^2$ 用于缩小 codebook 中的 embedding 向量和 Encoder 的输出之间的距离。该项损失函数只用来更新 codebook 中的向量，对 Encoder 的参数不产生作用，因为增加了 $sg[z_e(x)]$（stop-gradient）的限制，相当于在 Encoder 的输出固定的情况下，让 codebook 中的向量和 Encoder 输出更接近。

3. **VQ 的 commitment 损失函数**：codebook 损失函数是从更新 codebook 的角度缩小 $e$ 与 $z_e(x)$ 的距离，VQ-VAE 引入了第三项 $\vert\vert z_e(x) - sg[e] \vert\vert_2^2$，称为 **commitment loss**，相当于固定 codebook 向量（增加了 $sg[e]$ 的限制），只更新 Encoder 的参数，使得 $z_e(x)$ 更接近于与其最近的 codebook 向量，避免 $z_e(x)$ 选择与之距离最近的 codebook 向量时，在不同的 codebook 向量中波动。对于多阶段 RVQ 量化器，设一共有 C 层量化，$z_c$ 表示第 c 层量化的输入，$q_c(z_c)$ 表示 $z_c$ 在当前量化层中距离最近的 codebook 向量，则 RVQ 整体的 commitment loss 可以定义为：

$$l_w = \sum_{c=1}^{C} \vert\vert z_c - sg[q_c(z_c)] \vert\vert_2^{2}$$

> **使用 commitment loss 的动机**
> 在 VQ-VAE 中，Encoder 输出向量和 codebook 码本向量处于相同空间中，两者都通过梯度下降算法进行训练，但是更新速度可能不同。如果 Encoder 参数的训练速度比 codebook 快，那么 Encoder 将会不断地调整其输出向量的位置，但 codebook 向量不能及时更新，会导致 Encoder 的输出空间不断地扩大，出现以下问题：1) **过拟合**：Encoder 输出空间不断扩大，有更多的自由度来适应训练数据，导致过拟合训练数据；2) **训练不稳定**：Encoder 的输出在选择与之距离最近的 codebook 向量时“反复横跳”，导致训练不稳定、难以收敛等问题。
> 总之，commitment loss 是为了控制编码器输出的向量，将其映射到最近的码本向量，保证编码器的输出不会过分“自由”，避免过拟合和训练不稳定问题，从而提高模型的泛化能力和稳定性。

三部分损失函数各自只负责 VQ 中一部分参数的更新：第一项用来优化 Encoder 和 Decoder 不用来优化 codebook，第二项只用来优化 codebook 不涉及 Encoder 和 Decoder，第三项只用来优化 Encoder 不涉及 codebook 和 Decoder。反向总结，Encoder 由第一项和第三项损失函数共同优化，codebook 由第二项损失函数来优化，Decoder 只由第一项损失函数优化。

> Encoder 的输出 $z_e(x)$ 经过量化器得到的 $z_q(x)$ 作为 Decoder 的输入，直接作用于重建损失函数的计算，损失函数的变化也会通过梯度反向传播影响到 Encoder 的参数更新。因此，量化和重建的过程不只是为了学习到更好的量化器 VQ 部分，也会使得 Encoder 学到如何改变输出从而降低重建损失函数。

---

#### 语言建模与熵编码

Encoder-Decoder 结构和 RVQ 的使用，都是在 SoundStream 论文中已经使用的方法，那么 Encodec 相比于 SoundStream 的优越之处究竟体现在哪里呢？本部分将介绍 Encodec 真正提出的一些新思想。

##### 语言建模

为了让模型能够在单核 CPU 上达到实时的端到端音频编解码，Encodec 提出了一种推理加速的设计。使用一个轻量级的 Transformer 语言模型，共包含 5 层 TransformerEncoder，其中 Self-Attention 的 head 和 $d_{model}$ 维度分别为 8 和 200，Positionwise Feedforward 层的隐层大小为 800，不使用 dropout。每个 Self-Attention 层的感受野是因果的，只关注过去 3.5 秒的信息；同时，Encodec 还通过改变正余弦绝对位置编码（positional embedding）的偏移量（offset），来模拟当前音频片段来自于长音频中间部分的情况。

Encoder 的降采样倍数固定时，音频压缩的比特率只与 RVQ 包含的量化层数 $N_q$ 有关，对应于 $N_q$ 个 codebook。
- 设 $t-1$ 时刻 Encoder 的输出为 $z_e^{(t-1)}(x)$，在 $N_q$ 个 codebook（codebook 大小均设为 $N_C$，用 $\log_2 N_C$ 比特来表征 codebook 中的向量编号）中对应的编号分别为 $m_1, m_2, ..., m_{N_q}$。
- 将这 $N_q$ 个编号分别用各自 codebook 对应的 embedding table 映射到连续空间中得到 $d_{model}$ 维度的 embeddings，再将$N_q$个 embeddings 相加，得到 $e_q^{(t-1)}(x)$。

Transformer 语言模型正是在 $e_q^{(t)}(x)$ 这个层面上进行建模的，根据历史信息预测得到的 $e_q^{(t)}(x)$ ，被送入到 $N_q$ 个全连接层，对应于 $N_C$ 类的分类任务。训练时，第$i$个全连接层经过 softmax 之后的预测目标是第$i$个 RVQ 量化后的编号；推理时，第$i$个全连接层经过 softmax 后的输出，是第$i$个 codebook 上 $N_C$ 个编号的概率分布。

引入 Transformer 语言建模的好处很明显：原始的多阶段 RVQ 量化方法，需要逐级在每个 codebook 上进行一次量化，这一过程无法一次性完成；但是引入语言模型后，第 $t$ 时刻每个 codebook 上的编号是直接通过 Transformer 输入的历史信息（$t-1$时刻及之前的输入）**一次性预测**得到的。总之，Encodec 通过引入轻量级的 Transformer 语言模型，以及额外学习 $N_q$ 个 embedding table 将对应 codebook 的向量编号映射到连续空间，将多阶段的量化转换成了多个 codebook 编号的一次性预测，从而能够加速推理。

##### 熵编码

熵编码是一种根据数据的统计特征（比如频率分布）将数据编码为变长编码的技术，核心思想是将出现频率更高的符号用更短的编码来表示，常见的熵编码方法包括霍夫曼编码和算术编码。相比于霍夫曼编码，算术编码能够更有效地压缩数据（[参考链接](https://www.jianshu.com/p/959938932f73)），但同时也需要更高的计算成本。Encodec 使用的是**基于区间的算术编码**，也可以称为**区间编码**，再对 Transformer 预测出的概率分布进行压缩。

1. [**定长编码与霍夫曼编码**](https://www.jianshu.com/p/959938932f73)

以一段英文字符串为例，"AABABCABAB" 共 10 个字符，A 出现了 5 次，B 出现了 4 次，C 出现了 1 次。ABC 3 种符号，如果用定长编码进行编码，每种符号至少需要 $\lceil{log_2 3}\rceil = 2$ 个 bit 来表示，那么字符串编码后的长度为 $ 10 * 2 = 20 ~ bit$。但是这种编码方式肯定是存在冗余的，因为 2 bit 实际上最多能够用来表征 4 种符号。

如果采用霍夫曼编码，对于出现频次更少高的 A，可以使用更短的编码，如下图所示，根据频率的高低（作为权重）构建霍夫曼树：
(1) B 和 C 的权重（频率）最小，对应的两个结点形成一个新的二叉树，根结点的权重为 B 和 C 的权重之和 0.5；
(2) A 的权重也是 0.5，和 B/C 的根结点再形成一个新的二叉树；
霍夫曼编码时，左子树统一编码为 0，右子树统一编码为 1；因此，A 的编码为 0，B 的编码为 10，C 的编码为 11。使用霍夫曼编码后，整个字符串的编码长度为 $5\times 1 + 4 \times 2 + 1 \times 2 = 15~bit$。

霍夫曼编码相比于定长编码，将所需 bit 数从 20 降低至 15，但是还没有逼近香农定理的熵极限值。因为霍夫曼仍然是采用整数个 bit 对符号进行编码，比如 A 和 B 两个符号，出现概率分别为 0.4 和 0.1，但是都采用了相同 bit 的编码。根据信息学理论，字符串 "AABABCABAB" 的信息熵为：
$$
\begin{align}
H(x) & = -\sum_x P(x)\log_2 P(x) \\\\
     & = - (0.5 \times \log_2 0.5 + 0.4 \times \log_2 0.4 + 0.1 \times \log_2 0.1) \\\\
     & = 1.361
\end{align}
$$

2. **算术编码**

为了帮助更好地理解 Encodec 所用的区间编码，下面先对 "AABABCABAB" 进行算术编码，帮助理解其思想。
算术编码会先对 [0, 1] 区间根据概率进行划分：
(1) 第一次概率划分：A: [0, 0.5), B:[0.5, 0.9), C:[0.9, 1]
第 1 个字符为 A，那么首先选中 A 的区间 [0, 0.5) 作为新目标区间，按照概率进行划分：
(2) 第二次概率划分：A:[0, 0.25), B: [0.25, 0.45), C:[0.45, 0.5)
下一个出现的字符仍然是 A，继续按照概率对 [0, 0.25) 区间进行划分：
(3) 第三次概率划分：A:[0, 0.125), B:[0.125, 0.225), C:[0.225, 0.25)
下一个出现的字符是 B，继续按照概率对 [0.125, 0.225) 区间进行划分：
(4) 第四次概率划分：A:[0.125, 0.175), B: [0.175, 0.215), C:[0.215, 0.225)
依此类推，直到最后一个字符 B，下表给出了每个字符对应的目标区间。

<!-- <img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/ari_code.jpg" width = "550"/> -->


| 当前字符 | 当前目标区间 |
| :---: | :---: |
| A | [0, 0.5) |
| A | [0, 0.25) |
| B | [0.125, 0.225) |
| A | [0.125, 0.175) |
| B | [0.15, 0.17) |
| C | [0.168, 0.17) |
| A | [0.168, 0.169) |
| B | [0.1685, 0.1689) |
| A | [0.1685, 0.1687) |
| B | [0.1686, 0.16868) |


完成上述操作之后，最终的目标区间是 [0.1686, 0.16868)，在其中任选一个**二进制表示最短**的小数，比如 0.16864，二进制为：0.00101011001011，只保留小数点之后的二进制编码：00101011001011，bit 长度为 14 位，比哈夫曼编码还要少 1 位。算术编码的解码也很直接，将二进制编码还原为小数 0.16864，根据其所处的区间位置，对应的字符串是唯一的，反向对应得到字符串为 "AABABCABAB"。

3. **区间编码（基于区间的算术编码）：待补充**


<!-- 以上就是对基于区间的算术编码的解释。由于不同计算机架构上浮点小数的表示可能存在细微差异，还包括一些浮点数近似的问题，因此编码器和解码器之间可能无法完全准确对应进而导致解码错误。 -->

#### 训练目标

Encodec 的训练目标包含重建损失函数、GAN 的损失函数以及 RVQ 的 commitment loss。

##### 重建损失函数

包含时域损失函数 $l_t$ 和频域损失函数 $l_f$ 两部分，其中时域采用 $L_1$ 损失函数，频率采用 $L_1$ 和 $L_2$ 两个损失函数。${x}$ 表示音频的原始波形，$\hat{x}$ 表示预测时域波形；$S_{i}$ 表示不同参数的 STFT 提取得到的 64 维的梅尔特征，${i}$ 和 SoundStream 中的尺度因子 $s$ 作用一样。

$$ l_t(x, \hat{x}) = \vert\vert x - \hat{x} \vert\vert_1 $$
$$ l_f(x, \hat{x}) = \frac{1}{\vert \alpha \vert \cdot \vert s \vert} \sum_{\alpha_{i} \in \alpha} \sum_{i \in e} \vert \vert S_i(x) - S_i(\hat{x}) \vert\vert_{1} + \alpha_{i} \vert\vert S_{i}(x) - S_{i}(\hat{x}) \vert\vert_2$$

##### 判别器损失函数

为了提高解码器音频的音质，Encodec 也采用了 SoundStream 中的 GAN 训练思想，整个编解码器作为生成器，然后增加多个判别器。相比于 SoundStream 使用的 STFT 判别器，Encodec 将其扩充为多尺度的 STFT（Multi-Scale STFT）判别器，如下图所示：

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/ms_stft.jpg" width = "750"/>

每个 STFT 内部结构基本与 SoundStream 中介绍的相同，只不过在计算 STFT 时，采用了五组不同的窗长 [2048, 1024, 512, 256, 128]。此外，Encodec 扩展了对 48 kHz 采样率的支持，窗长相应地进行了加倍，每两个 batch 数据更新一次判别器的参数；Encodec 还增加对双通道音频的支持，两个通道独立处理即可。GAN 的生成器和判别器损失函数，以及增加的 feature matching 损失函数与 SoundStream 一直，不再赘述。

> 一个需要注意的 trick：论文发现判别器更倾向于优化解码器端的参数，为了弱化这一问题，在模型参数更新时，24 kHz 情况下解码器参数更新进行 2/3 概率的加权，48 kHz 情况下权重系数为 0.5。
> 此外，和 SoundStream 一样，通过控制训练过程中音频经过的 RVQ 层数，可以实现不同比特率的编解码：24 kHz 采取 1.5/3/6/12 kbps 四种不同的比特率，48 kHz 采取 3/6/12/24 kbps 的比特率。

##### 整体的损失函数
$$L_{model} = \lambda_t \cdot l_t(x, \hat{x}) + \lambda_f \cdot l_f(x, \hat{f}) + \lambda_g \cdot l_g(\hat{x}) + \lambda_{feat} \cdot l_{feat}(x, \hat{x}) + \lambda_{w} \cdot l_w(w)$$
其中，$l_g(\hat{x})$ 包含了生成器和判别器两个损失函数，$l_w(w)$ 是 RVQ 的 commitment loss。

##### 损失函数均衡器
Encodec 提出了一种损失函数均衡器（Loss Balancer），用于提高训练的稳定性。Loss Balancer 能够根据不同损失函数传回的梯度大小进行规整，避免参数更新受到单个损失函数梯度的过度影响。

假设损失函数 $l_i$ 只与模型的输入 $\hat{x}$ 有关，定义以下两个数值：$g_i = \frac{\partial l_i}{\partial \hat{x}}$ 表示损失函数 $l_i$ 相对于 $\hat{x}$ 的梯度，$\<\vert\vert g_i \vert\vert_2 \>_{\beta}$ 表示梯度 $g_i$ 在最后一批训练数据上的指数移动平均（EMA）。

给定各种损失函数的权重 $\lambda_{i}$，定义下面的新梯度：
$$\tilde{g_{i}} = R \frac{\lambda_{i}}{\sum_j \lambda_{j}} \cdot \frac{g_i}{\<\vert\vert g_i \vert\vert_2 \>_{\beta}}$$

在参数更新时，反向传播的梯度使用 $\sum_{i}\tilde{g_i}$ 而不是原本的 $\sum_i \lambda_i g_i$。论文中 $R=1$, EMA 的 $\beta=0.999$。除了 VQ 的 commitment loss $l_w$ 之外（commitment loss 与编解码器的输出无关），其他的损失函数加入到损失函数均衡器中，所以上式中 $\sum_j \lambda_{j}$ 的 $\lambda_{j} \in \\{\lambda_{t}, \lambda_{f}, \lambda_{g}, \lambda_{feat}\\}$。

### 实验与结论

#### 实验准备
**数据集**：24 kHz 在干净语音、带噪语音和音乐等各种音频上评测；48 kHz 只在音乐音频上评测。干净语音来自于 DNS 比赛和 Common Voice 数据集；音乐音频来自于 Jamendo 数据集；其他音频来自 AudioSet 和 FSD50K。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table-a-1.jpg" width = "750"/>

**基线**：为了更好地和前人的工作进行全方位对比，基线包括 Opus、EVS、MP3 以及 Google 提出的 Lyra v2（SoundStream），其中 SoundStream 使用了官方实现版和优化版（包括判别器中间层增加了 LayerNorm，能够有效提升音质）。

**评测指标**：主观评测使用 MUSHRA，客观评测适用 ViSQOL 和 SI-SNR 两种。

#### 实验结果

##### 不同比特率下的音质对比

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/result-figure.jpg" width = "700"/>

从图中可以看出，Encodec 的效果是明显好于 SoundStream 模型的，采用熵编码能够达到稍微更好的效果。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table-1.jpg" width = "700"/>

上图所示的表格中有更详尽的数值结果。Encodec 在 3kbps 下的音频质量甚至略好于 SoundStream 在 6bkps 下的结果；此外，经过熵编码之后，Encodec 的比特率可以进一步降低大约 25%-40%，编解码效率进一步提升。

##### 针对判别器的消融实验

在音频和语音领域的 GAN 训练中，HiFi-GAN 使用了多周期判别器（MPD, Multi-Period Discriminator）和多尺度判别器（MSD, Multi-Scale Discriminator），SoundStream 采用的是 STFT 判别器，本文对 STFT 判别器进行了扩展，提出多尺度 STFT（MS-STFT）判别器。通过下表的实验结果，发现 MS-STFT + MPD 能够达到最好的效果，论文认为只使用 MS-STFT 即可达到很好的效果。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table-2.jpg" width = "700"/>


##### 流式编解码的对比实验
表格中给出了流式/非流式编解码的效果对比：

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table-3.jpg" width = "700"/>

<!-- ##### Balancer 的消融实验 -->


##### 双声道音频实验
对于双通道的音频，使用了算术编码的 Encodec 在 4.2 kbps 比特率下就能达到和 24 kbps 的 Opus 基本一样的效果。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table-4.jpg" width = "700"/>

##### 单核 CPU 下的时延与实时性
与 SoundStream 相比，模型的实时率有所变差，但是 24 kHz 下仍然能够满足实际应用的实时性需求，但是 48 kHz 下实时率小于 1，只能用于音频的离线压缩场景。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table-5.jpg" width = "700"/>

<br>

##### TODO

- [ ] 区间编码 + 源码示例
- [ ] Encodec 基于 Gumbel-Softmax 的量化器
- [ ] 补充完善相关的参考文献
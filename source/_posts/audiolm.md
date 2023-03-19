---
title: 音频生成 | 基于语言建模的 AudioLM
date: 2023-01-16 11:40:36
tags: [音频生成, 语音合成]
categories: [论文笔记]
copyright_info: true
toc: true
mathjax: true
comment: false
---

音频生成（Audio Generation）是最近非常热门的方向，是**AIGC**的具体应用之一。相比于**语音**，**音频**包含的意义更广泛，不仅包含语音识别/语音合成所针对的人说话声，还包括音乐声、环境声、动物声等各种各样的声音。

本文介绍 2022 年 9 月份 Google 提出的 AudioLM，将语言建模的思想应用在音频生成任务上，能够生成高质量的音频，并保持音频长时间范围的连续性和一致性。语言建模最近在文本生成、图像生成、视频生成等各类**生成式 AI**任务中均得到了成功应用，比如 2023 年 1 月微软提出的语音合成模型 [**VALL-E**](https://arxiv.org/pdf/2301.02111.pdf)，同月 Google 提出的图像生成 SOTA 模型 [**MUSE**](https://arxiv.org/abs/2301.00704) 也采用了类似的技巧。


| 会议/期刊 | 年份 | 题目 | 链接 |
| :---: | :---: | :---: | :---: |
| arxiv | 2022 | AudioLM: a Language Modeling Approach to Audio Generation | [https://arxiv.org/abs/2209.03143](https://arxiv.org/abs/2209.03143) |

### 论文概述

本文提出的 AudioLM，能够在长时间范围内保持生成音频的一致性和连贯性，在语音和钢琴声的**续写**任务上都进行了效果验证。从模型的构成来看，AudioLM 可以说是**集大成者**：将音频编解码的 SoundStream 模型、自监督表征的 wav2vec-Bert 模型以及强大的 Transformer 语言模型进行了结合。
- 1. wav2vec-Bert 用于提取粗粒度的语义 token（Coarse Semantic Tokens）表征，在粗粒度语义 token 上自回归建模，能够对音频的局部信息（比如语音的音素、钢琴曲的旋律）和全局信息（比如语音的内容、钢琴曲的和声和节奏）进行有效建模，但是建模粒度比较粗糙，无法保证合成音频的细节质量。
- 2. SoundStream 用于提取细粒度的声学 token（Fine Acoustic Tokens）表征，音频编码时的 token 保留了音频波形细节的信息，通过解码器能够恢复原始波形，实现音频的高质量生成。
- 3. 将 wav2vec-Bert 与 SoundStream 两个模型相结合，均当作音频的离散化 tokenizer，再使用 Transformer 语言模型对两类 token 同时建模，从**语义内容**和**声学细节**两个角度保证了合成音频的质量。

AudioLM 使用的 Transformer 语言模型，并不直接在音频波形/采样点级别进行建模（音频的采样点序列太长，使用 Self-Attention 计算复杂度过高），而是在预训练模型抽取的离散 token 上进行建模（对应采样率通常很低），极大降低计算量的同时最大程度保留音频的可恢复性。

### AudioLM 模型

#### 模型构成

AudioLM 一共包含三个大模块：
- **tokenizer 模型：** 将音频采样点序列（高采样率）$x \in R^{T}$ 映射为离散 token 序列（低采样率），数学表达式为：
$$y = enc(x), y = (y_1, ..., y_{T'})$$
离散 token 序列的长度 $T'$ 通常远小于采样点序列的长度 $T$。
- **语言模型：** 语言模型的输入是离散化后的 token 序列 y，训练时的目标是最大化似然函数**LLH（Likelihood）**，推理时采用自回归的方式预测序列 $\hat{y}$。
$$LLH = \prod_{t=1}^{T'} p(y_t \vert y_{\lt t})$$
- **detokenizer 模型：** 将预测得到的 token 序列 $\hat{y}$ ，还原为原始的音频波形 $\hat{x} = dec(\hat{y})$。

需要说明的是，AudioLM 训练时，tokenizer 和 detokenizer 都是在大量的音频数据集上预训练好的模型，具有很好的泛化性，因此这些参数固定不变，只需训练语言模型部分的参数即可。

#### 不同的离散表征

针对音频生成任务，将输入音频离散化为 token 时，主要有两点要求：一方面，离散的 token 满足较低比特率的同时，能够恢复出高质量的音频；另一方面，离散的 token 能够获取到音频的长时间粒度下的语义表征，使得后续的语言模型能够利用语义信息，保持音频节奏/语义内容等方面的连贯性。然而，基于前人在音频领域的研究，这两个要求的出发点不同，往往是存在矛盾的，因为一般来说更 compact 的语义表征会伴随着波形细节信息的明显丢失。

针对第一个要求，选用 SoundStream 的出发点是 SoundStream 本身用于音频压缩，目标恰好是在低比特率下也能恢复出高质量原始音频；而针对第二个要求，wav2vec-BERT 则是从语义的角度进行离散特征的建模，符合模型的设想。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/two_tokens.jpg" width = "600"/>

##### Acoustic Tokens

AudioLM 的声学 token 是用 SoundStream 抽取得到的，具体原理详见相关论文笔记[链接](https://revospeech.github.io/2023/01/14/lyra_v2_soundstream/)。对于 16 kHz 的音频，经过编码器降采样 320 倍后采样率变为 16000/320 = 50 Hz，设音频的采样点数为 $T$，Encoder 输出的 embedding 经过多层 RVQ 进行量化后，得到 $M^{\frac{T}{320} \times N_q}$ 的 token 矩阵，其中 $N_q$ 表示 RVQ 量化器的个数。矩阵中每个元素都是在 1 到 $N_c$ 范围内的整数，其中 $N_c$ 代表每个量化器中 codebook 的大小（向量的个数）。

##### Semantic Tokens

AudioLM 的语义 token 是用 wav2vec-BERT 抽取得到的。如下图所示，wav2vec-Bert 由多个 Conformer Block 构成，训练目标结合了两种主流的自监督表征学习方法：wav2vec 的对比学习（Constrastive Learning）目标和 BERT 的 MLM （Masked Language Model）。AudioLM 在 MLM 损失函数对应的中间层输出 embedding 上进行 k-means 聚类，预设 $K$ 个聚类中心，将每个聚类中心的 index 作为该类别所有向量的离散化语义 token。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/wav2vec_bert.jpg" width = "600"/>

论文发现，在 k-means 聚类之前，需要将 wav2vec-BERT 中间层输出的 embedding 先进行正则化，使得 embedding 的均值为 0、方差为 1；正则化之后再聚类能够显著提升语义 token 在音素分类等任务上的区分效果。

由于 wav2vec-BERT 的降采样力度更大，语义 token 的采样率为 25 Hz。对于采样点数为 $T$ 的 16 kHz 音频，相当于进行了 640 倍的降采样，所以 wav2vec-BERT 离散化的语义 token 序列为 $z = (z_1, ..., z_{\frac{T}{640}}), z_i \in \\{1, ..., K\\}$。

##### 两种 Token 的评测

AudioLM 论文先对两种 token 各自的优缺点进行了实验验证与分析，包含两组实验：一组用于衡量 token 的语义能力，评测指标是音素区分任务的错误率；另一组用于衡量 token 的声学特性，对两种不同的 token 分别训练一个 SoundStream 结构的解码器（以最小化重建损失函数为训练目标），评测时对 token 进行重建后评价音频质量，采用客观评价指标 ViSQOL。实验结果如下文表格所示，wav2vec-BERT 得到的 token 在音素区分任务上具有明显更低的错误率，而且错误率随着比特率的提升而降低，表明 wav2vec-BERT 提取的 token 语义能力更强；而 SoundStream 的音频重建效果更好，比 wav2vec-BERT 显著更好，具有更好的声学表征能力。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table1.jpg" width = "700"/>

##### 两种 Token 的联合

为了充分发挥两种不同 Tokenizer 的优势，AudioLM 提出了层次化建模的方法，先获取整个输入音频的语义 token，再将语义 token 作为模型的条件输入，预测声学 token。

层次化建模的思想有一个前提假设：给定过去时刻的语义 token（$z_{\lt t}$），当前时刻的语义 token（$z_{t}$）与过去时刻的声学 token（$y_{\lt t}$）之间是条件独立的，近似认为当前语义 token 与声学 token 的历史信息没有关系，数学表达式为：

$$p(z_t \vert z_{\lt t}, y_{\lt t}) \approx p(z_t \vert z_{\lt t})$$


#### AudioLM 建模三阶段

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/three_stage.jpg" width = "780"/>

上图给出了 AudioLM 建模的三阶段流程，建模粒度从粗到细：从最粗粒度的语义 token，到声学 token 中的粗粒度表征，再到声学 token 中的细粒度表征，层层递进，逐步合成高质量的音频。

##### 阶段一：语义建模

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230306/audiolm-stage1.jpg" width = "700"/>

阶段一用 wav2vec-BERT 抽取音频的离散语义 token，在语义 token 上进行自回归建模 $p(z_t \vert z_{\lt t})$，用于建模长时间范围内的语义结构。

##### 阶段二：粗粒度声学建模

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230306/audiolm-stage2.jpg" width = "700"/>

阶段二采用和语义建模类似的方法，但只在 SoundStream 的前 $Q'$ 个量化器的输出上进行自回归建模，同时将第一阶段得到的语义 token 作为条件输入。由于 SoundStream 的 RVQ 属于多阶段量化，声学 token 也具有一定的层次结构，论文认为：前若干个量化器（**粗粒度量化器，coarse quantizer**）可以认为主要用来恢复说话人特性、录制环境等偏向全局的粗粒度信息，而剩下的量化器（**细粒度量化器，fine quantizer**）则更侧重于波形的细节信息。

阶段二主要建模 $p(y_t^{q} \vert z, y_{\lt t}^{\leq Q'}, y_{t}^{\lt q}), q \leq Q'$，其中：
- $z$ 表示第一阶段得到的离散语义 token 序列；
- $y_{\lt t}^{\leq Q'}$ 表示之前时刻，前 $Q'$ 个粗粒度量化器的声学 token 序列，并且将矩阵展开（flatten）；
- $y_{t}^{\lt q}$ 表示当前时刻下，前 $q-1 (q \leq Q')$ 个量化器的 token 序列，同样进行展开。

##### 阶段三：细粒度声学建模

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230306/audiolm-stage3.jpg" width = "700"/>

阶段三在细粒度量化器输出的声学 token 上进行建模，使用 $Q'$ 个粗粒度量化器的 token 作为条件输入，对 $p(y_t^{q} \vert y^{\leq Q'}, y_{\lt t}^{\gt Q'}, y_t^{\lt q}), Q' \lt q \leq Q$ 进行建模，其中：
- $y^{\leq Q'}$ 表示前 $Q'$ 个量化器输出的 token 序列，将矩阵展开；
- $y_{\lt t}^{\gt Q'}$ 表示之前时刻，第 $Q'+1$ 到第 $Q$ 个量化器（一共 $Q$ 个量化器）输出的 token 序列，并且展开；
- $y_t^{\lt q}$ 表示当前时刻下，前 $q-1 (Q' \lt q \leq Q)$ 个量化器的 token 序列，并且展开。

实际上，$y^{\leq Q'}$ 和 $y_{\lt t}^{\gt Q'}$ 相当于 $t$ 时刻之前，全部 $Q$ 个量化器输出的 token 序列在时间维度上进行了展开，以及第 $t$ 时刻将第 $q$ 个量化器之前输出的 token 作为条件输入。阶段三没有考虑第一阶段的语义 token 序列，是增加了条件独立假设：给定粗粒度量化器的 token 时，认为细粒度量化器输出的声学 token 与第一阶段的语义 token 是条件独立的。


#### AudioLM 的推理

- **非条件式音频生成：** 第一种推理方式，模型没有任何外部条件输入，直接通过随机采样得到语义 token，然后作为粗粒度声学建模的条件输入，其输出再作为细粒度声学建模的条件输入，经过 detokenizer 即可得到生成的音频。虽然是非条件式生成，但是后续实验验证了生成音频整体的一致性。
- **声学生成：** 第二种推理方式，模型以真实的音频作为条件输入，wav2vec-BERT 基于真实音频得到的语义 token，再通过第二三阶段的处理，经过 detokenizer 生成得到音频。这种推理方式下，所生成音频的语义信息是来自于真实音频的，只不过经过了模型的后续处理变得更多样，但是语义内容没有发生变化。
- **音频续写生成：** 第三种推理方式，给定一段音频的前缀或者 prompt，然后生成后续的音频。对于给定 prompt：
	+ 先抽取 prompt 对应的语义 token 序列和粗粒度量化器的 token 序列；
	+ 第一阶段：利用 prompt 真实音频的语义 token 序列，预测之后的语义 token 序列；
	+ 第二阶段：已有 prompt 真实音频对应的语义 token 序列、第一阶段预测出的后续语义 token 序列，以及根据 prompt 真实音频提取的粗粒度声学 token 序列；将三个序列拼接，预测出后续的粗粒度声学 token 序列；
	+ 第三阶段：将第二阶段获取到的完整的粗粒度声学 token 序列作为细粒度模型的输入，预测得到完整的细粒度声学 token 序列；
	+ 最后，基于粗粒度和细粒度的声学 token 序列，使用 detokenizer（SoundStream 的解码器）生成音频。

### 实验与分析

#### 实验设计与准备

**两种实验任务:** 分别是语音续写和钢琴曲续写：语音续写需要生成的音频满足音色、录音环境、韵律方面的一致性，并且语义上保证准确和连贯；钢琴曲续写则需要生成的音频满足旋律、和声和节奏上的连续性。语音续写是基于 3 秒的语音 prompt 继续生成 7 秒的音频；钢琴曲续写是基于 4 秒的音频 prompt 继续生成 20 秒的音频。

**数据集：** 语音数据采用无标签的 6 万小时的 Libri-Light 数据；钢琴曲使用的是谷歌自有的 4 万小时数据，覆盖初学者到钢琴家级别、不同声学环境、不同曲调的钢琴曲声音。

**实验配置：** wav2vec-Bert 采用 6 亿参数量的模型，输出 embedding 来自 MLM 部分的第 7 层，离散化聚类时类别个数 $K=1024$；SoundStream 采用 320 倍降采样，$Q=12$ 个量化器，前 $Q'=4$个量化器作为第二阶段的粗粒度声学 token，后面 8 个量化器用于第三阶段的细粒度声学 token；而 Transformer 自回归语言模型的参数：12 层，self-attention 包含 16 个 head、embedding 维度为 1024，feed-forward 层的隐层大小为 4096，使用的是 $T5$ 相对位置编码。

#### 语义 token 的信息

本实验用于测试语义 token 是否保留了语音的内容信息。使用 LibriSpeech test-clean 中 4-10 秒内的音频，采用第二类推理方式**声学生成**，每条音频随机生成三个样本，使用 Conformer-Transducer 的 Large 模型对生成的样本进行识别，计算得到 CER 和 WER，用于衡量生成音频在语义内容上的正确性和一致性。

从下表的实验结果可以看出，AudioLM 生成的语音在内容上的一致性还是相对比较高的，尤其是考虑到**声学生成**任务中，新生成的音频可能会带有一些额外的噪声，导致语音识别系统的效果也会有些降低。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table2.jpg" width = "520"/>

#### 声学 token 的信息
同样地，本部分用于验证声学 token 包含的声学特性的信息。使用 LibriSpeech 中的 291 个说话人的音频，在第二种（**声学生成**）和第三种（**音频续写**）推理方式下，对新生成的音频和原始音频进行说话人分类，实验结果如下图所示。从中可以得到一些结论：**声学生成**因为只保留了真实音频的语义 token，不包含说话人音色方面的信息，所以说话人分类准确率很低；但是**音频续写**以 prompt 的真实声学 token 作为部分条件输入，保持了音色的连续性，因此说话人分类准确率很高。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table3.jpg" width = "520"/>

#### 语言学信息评测
论文还针对基于语义 token 进行语言建模的方法进行更细致的评价，采用以下两种评价指标（均来自于 ZeroResource 2021 年的比赛），验证 AudioLM 在语义内容上的建模效果：
- **sWUGGY**：两个发音相近的词，一个真实存在，一个并不是真正的词，如果能够给真实存在的词更高的概率，说明模型效果越好；
- **sBLIMP**：模型需要赋予语法正确的句子比语法错误的句子更高的概率，准确率越高，说明模型的语义能力越强。

下图是两个评测的实验结果，可以看出，AudioLM 的语义建模能力很强。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230307/table-iv.jpg" width = "520"/>

> **实验细节说明：**
> 实验评测使用 ZeroResource 2021 年比赛的开发集，针对 sWUGGY 和 sBLIMP 两个评测，分别有 10000 和 6300 组测试数据对。sWUGGY 评测时，只考虑在 LibriSpeech 中出现过的词（集内词）。使用模型输出的似然值作为评价概率高低的依据，但是 sBLIMP 评测的正样本和负样本序列长度明显不同，为了避免句子长度对评测的影响，将似然值除以序列长度，正则化之后作为判断句子正确与否的指标。
> AudioLM 力压包括 BERT, HuBert, RoBERTA, CPC_BERT 等在内的一众模型，达到了最优的效果。

#### 钢琴曲续写实验
针对钢琴曲的续写，论文单独使用 4 万小时钢琴声训练了一个 AudioLM 模型，不过对 SoundStream 模块的参数进行了更改，采用 3 层的 RVQ，每层 codebook 的大小为 $2^{14} = 16384$，所以去除了第三阶段的训练，而是在第二阶段一次性预测所有的声学 token。实验组音频是使用 AudioLM 进行续写，对照组音频是去除了 wav2vec-BERT 提取的语义 token 建模，主观评测显示 83.3% 评测数据对中，AudioLM 续写的钢琴曲更受评测人青睐。


### 参考文献/链接
- **AudioLM 示例音频**：[https://google-research.github.io/seanet/audiolm/examples](https://google-research.github.io/seanet/audiolm/examples/)
- **AudioLM 官方博客**：[https://ai.googleblog.com/2022/10/audiolm-language-modeling-approach-to.html](https://ai.googleblog.com/2022/10/audiolm-language-modeling-approach-to.html).
- **SoundStream**：Zeghidour, Neil, et al. "Soundstream: An end-to-end neural audio codec." IEEE/ACM Transactions on Audio, Speech, and Language Processing 30 (2021): 495-507. [[pdf]](https://arxiv.org/pdf/2107.03312.pdf)
- **SoundStream 论文笔记**：[https://revospeech.github.io/2023/01/14/lyra_v2_soundstream](https://revospeech.github.io/2023/01/14/lyra_v2_soundstream/).
- **wav2vec-BERT**：Chung, Yu-An, et al. "W2v-bert: Combining contrastive learning and masked language modeling for self-supervised speech pre-training." 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2021. [[pdf]](https://arxiv.org/pdf/2108.06209.pdf)
- **ZeroChallenge 2021 评测指标**：[https://zerospeech.com/challenge_archive/2021/02_track1/#evaluation](https://zerospeech.com/challenge_archive/2021/02_track1/#evaluation)


### demo 视频
<iframe width="560" height="315" src="https://www.youtube.com/embed/_xkZwJ0H9IU" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
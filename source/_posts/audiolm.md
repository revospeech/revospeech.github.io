---
title: 音频生成 | 论文笔记 | AudioLM
date: 2023-01-16 11:40:36
tags: [音频生成, 语音合成]
categories: [论文笔记]
copyright_info: true
toc: true
mathjax: true
comment: false
---

音频生成（Audio Generation）是最近非常热门的一个方向，可以视为**AIGC**的具体应用之一。本文介绍的是 2022 年 9 月份 Google 提出的 AudioLM 工作，将语言模型的建模思想应用在音频生成任务上，能够生成高质量的音频，并且保持音频风格等方面的长期一致性与连贯性。论文的思想是比较直接的，最近在图像生成领域也有所应用，比如 2023 年 1 月份 Google 使用类似的思想提出了图像生成的 SOTA 模型 [**MUSE**](https://arxiv.org/abs/2301.00704)。

<!-- more -->


**音频**相比于**语音**，其包含的意义更广泛，不仅包含语音识别/语音合成所针对的人说话的声音，还包括音乐声、环境声、动物声音等各种各样的声音，音频生成主要针对的就是各类声音的生成任务。AudioLM 从语言模型入手，是因为语言模型已经在文本生成、图像生成中得到了成功的应用。

| 会议/期刊 | 年份 | 题目 | 链接 |
| :---: | :---: | :---: | :---: |
| arxiv | 2022 | AudioLM: a Language Modeling Approach to Audio Generation | [https://arxiv.org/abs/2209.03143](https://arxiv.org/abs/2209.03143) |

### 论文概述

本文提出的 AudioLM，能够在长时间范围内保持生成音频的一致性和连贯性，在语音和钢琴声的**续写（continuation）** 任务上都得到了验证。AudioLM 可以说是集大成者：将音频压缩中的 SoundStream 模型、自监督表征的 wav2vec-Bert 模型以及强大的基于 Transformer 的语言模型进行了成功的结合。
- 一方面，wav2vec-Bert 用于提取粗粒度的语义 token（Coarse Semantic Tokens）表征，基于粗粒度语义 token 进行自回归建模，能够对音频的短时局部信息（比如语音的音素、钢琴曲的旋律）和长时全局信息（比如语音的语义内容、钢琴曲的和声和节奏）进行有效地建模，但是无法保证合成音频的细节质量。
- 另一方面，SoundStream，用于提取细粒度的声学 token（Fine Acoustic Tokens）表征，能够保留音频波形细节的信息，通过音频编解码，实现音频的高质量生成。
- 将 wav2vec-Bert 与 SoundStream 相结合，均当作音频的 tokenizer，再使用同一个语言模型对两类 token 同时建模，可以说是 AudioLM 最大的亮点。

AudioLM 使用的 Transformer 语言模型，并不直接在音频的波形/采样点级别进行建模（音频的采样点序列太长，使用 Self-Attention 不现实），而是在外部模型抽取的离散 token 空间上进行建模，极大降低计算量的同时最大程度保留音频的可恢复性。

### AudioLM 模型

#### 模型构成

AudioLM 一共包含三个大模块：
- **tokenizer 模型：** 将音频采样点序列（高采样率） $x \in R^{T}$ 映射为离散的 token 序列（低采样率），token 序列的长度 $T'$ 远小于采样点序列的 $T$，通常远小于
$$y = enc(x), y = (y_1, ..., y_{T'})$$
- **只有 decoder 的语言模型：** 语言模型的输入是离散化后的 token 序列 y，训练时的目标是最大化似然函数，推理时采用自回归的方式预测序列 $\hat{y}$。
$$LLH(likelihood) = \prod_{t=1}^{T'} p(y_t \vert y_{\lt t})$$
- **detokenizer 模型：** 将预测得到的 token 序列 $\hat{y}$ 还原为原始的音频波形 $\hat{x} = dec(\hat{y})$。

AudioLM 训练时，tokenizer 和 detokenizer 都是在音频数据集上预训练好的，这些参数固定不变，只需要训练语言模型部分的参数即可。


#### 不同的离散表征

将音频离散化为 token 时，音频生成任务主要有两点要求：一方面，希望离散的 token 满足较低比特率的同时，能够恢复出高质量的音频；另一方面，希望离散的 token 能够获取到音频的长时间粒度下的语义表征，从而保持音频风格/节奏/语义内容等方面的连贯性。基于前人在音频领域的研究，以上两个要求的出发点是完全不一样的，事实上往往也是存在矛盾的，因为通常更 compact 的语义表征会伴随着波形细节信息的丢失。

针对第一个要求，SoundStream 的出发点是音频压缩任务，恰好要求在低比特率下也能恢复出高质量原始音频；而针对第二个要求，wav2vec-BERT 则是从语义的角度进行离散特征的建模。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/two_tokens.jpg" width = "600"/>

##### Acoustic Tokens
AudioLM 的声学 token 是用 SoundStream 抽取得到的。SoundStream 的具体原理详见[链接](https://revospeech.github.io/2023/01/14/lyra_v2_soundstream/)。对于 16 kHz 的音频，经过编码器降采样 320 倍后采样率变为 16000/320 = 50 Hz。对于采样点数为 $T$ 的 16 kHz 音频，Encoder 输出的 embedding 经过多层 RVQ 进行量化后，得到的是 $M^{\frac{T}{320} \times Q}$ 的 token 矩阵，其中 Q 表示 RVQ 量化器的个数。矩阵中每个元素都是在 $1-N$ 范围内的整数，其中 $N$ 代表每个量化器中 codebook 的大小（向量的个数）。

##### Semantic Tokens

AudioLM 的语义 token 是用 wav2vec-BERT 抽取得到的。wav2vec-BERT 结合了两种自监督表征学习的方法，同时用到了 wav2vec 的对比学习（Constrastive Learning）目标和 BERT 的 MLM （Masked Language Model）目标。如下图所示，wav2vec-Bert 由多个 Conformer Block 构成，AudioLM 基于 MLM 相应中间层输出的 embedding 进行 k-means 聚类，预设 $K$ 个聚类中心，将每个聚类中心的 index 作为该类别所有向量的语义 token 表征。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/wav2vec_bert.jpg" width = "600"/>

在 k-means 聚类之前，需要将 wav2vec-BERT 中间层输出的 embedding 先进行正则化，正则化后均值为 0、方差为 1，论文发现这样正则化之后聚类能够显著提升语义 token 在音素分类等任务上的区分效果。

wav2vec-BERT 的降采样力度更大，语义 token 的采样率为 25 Hz，相当于 16 kHz 的音频进行了 640 倍的降采样。对于采样点数为 $T$ 的 16 kHz 音频，使用 wav2vec-BERT 离散化后得到语义 token 序列为 $z = (z_1, ..., z_{\frac{T}{640}}), z_i \in {1, ..., K}$。

##### 两种 Token 的评测

AudioLM 论文先对两种 token 各自的优缺点进行了实验验证与分析，包含两组实验：一组用于衡量 token 的语义能力，进行音素的区分，评测指标是音素区分的错误率；一组用于衡量 token 的声学特性，根据 token 对原始音频进行重建，评测指标是音频质量的客观评价标准 ViSQOL。实验结果如下文表格所示，wav2vec-BERT 得到的 token 语义能力更强，而 SoundStream 的音频恢复效果更好，两者可以优势互补。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table1.jpg" width = "700"/>


##### 两种 Token 的联合

为了充分发挥两种不同 tokenizer 的优势，AudioLM 提出了层次化建模的方法，先获取整个输入音频序列语义 token 的表征，然后将语义 token 作为模型的条件输入，预测声学 token。

层次化建模的思想有一个前提假设：给定过去时刻的语义 token，当前时刻的语义 token 与过去时刻的声学 token 之间是条件独立的，也就是近似认为当前语义 token 与声学 token 的历史信息没有关系，数学表达式为：
$$p(z_t \vert z_{\lt t}, y_{\lt t}) \approx p(z_t \vert z_{\lt t})$$


#### AudioLM 建模三阶段

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/three_stage.jpg" width = "780"/>

##### 阶段一：语义建模
先用 wav2vec-BERT 抽取音频的离散语义 token，在语义 token 上进行自回归建模 $p(z_t \vert z_{\lt t})$，用于建模长时间范围内的时序结构。

##### 阶段二：粗粒度声学建模

阶段二采用和语义建模类似的方法，但只在 SoundStream 的前 $Q'$ 个量化器的输出上进行自回归建模，同时将第一阶段得到的语义 token 作为条件输入。由于 SoundStream 的 RVQ 属于多阶段量化，声学 token 也具有一定的层次结构，前若干个量化器（**粗粒度量化器，coarse quantizer**）可以认为主要用来恢复说话人特性、录制环境等偏向全局的粗粒度信息，而之后的量化器（**细粒度量化器，fine quantizer**）则更侧重于波形的细节信息。

阶段二主要建模 $p(y_t^{q} \vert z, y_{\lt t}^{\leq Q'}, y_{t}^{\lt q})$，其中：
- $z$ 表示第一阶段得到的离散语义 token 序列
- $y_{\lt t}^{\leq Q'}$ 表示之前时刻，前 $Q'$ 个粗粒度量化器的声学 token 序列（展开）
- $y_{t}^{\lt q}$ 表示当前时刻下，前 $q-1 (q \leq Q')$ 个量化器的 token 序列（展开）

实际上，$y_{\lt t}^{\leq Q'}$和$y_{t}^{\lt q}$相当于对 SoundStream 量化后的 token 矩阵在时间维度上进行了展开（flatten）。

##### 阶段三：细粒度声学建模

阶段三在细粒度量化器输出的声学 token 上进行建模，使用 $Q'$个粗粒度量化器的 token 作为条件输入，对 $p(y_t^{q} \vert y^{\leq Q'}, y_{\lt t}^{\gt Q'}, y_t^{\lt q})$ 进行建模，其中：
- $y^{\leq Q'}$ 表示前 $Q'$ 个量化器输出的 token 序列（展开）
- $y_{\lt t}^{\gt Q'}$ 表示之前时刻，第 $Q'+1$ 到第 $Q$ 个量化器（一共 $Q$ 个量化器）输出的 token 序列（展开）
- $y_t^{\lt q}$ 表示当前时刻下，前 $q-1 (Q' \lt q \leq Q)$ 个量化器的 token 序列（展开）

实际上，$y^{\leq Q'}$ 和 $y_{\lt t}^{\gt Q'}$ 相当于 t 时刻之前，全部 Q 个 量化器输出的 token 序列在时间维度上进行了展开，以及第 $t$ 时刻将第 $q$ 个量化器之前输出的 token 作为条件输入。阶段三没有考虑第一阶段获得的语义 token 序列，是增加了条件独立假设：给定粗粒度量化器的输出 token，细粒度量化器输出的声学 token 与第一阶段的语义 token 是条件独立的。


#### AudioLM 的推理

- **非条件式音频生成：** 第一种推理方式，模型没有任何外部条件输入，直接通过随机采样得到语义 token，然后作为粗粒度声学建模的条件输入，输出再作为细粒度声学建模的条件输入，经过 detokenizer 即可得到生成的音频。虽然是非条件式生成，但是后续实验验证了音频整体的一致性。
- **声学生成：** 第二种推理方式，模型以真实的音频作为条件输入，wav2vec-BERT 基于真实音频得到语义 token，再通过第二三阶段的处理，经过 detokenizer 得到生成的音频。这种推理方式，生成的音频其语义信息是来自于真实音频的，只不过经过了模型的处理变得更加多样，但是语义内容没有发生变化。
- **音频续写生成：** 第三种推理方式，给定一段音频的前缀或者 prompt，然后生成后续的音频。对于给定的 prompt：
	1. 预先抽取 prompt 对应的语义 token 序列和粗粒度量化器的 token 序列
	2. 第一阶段：利用 prompt 真实音频的语义 token 序列，预测之后的语义 token 序列
	3. 第二阶段：已有：prompt 真实音频对应的语义 token 序列、第一阶段预测出的后续语义 token 序列、根据 prompt 真实音频提取的粗粒度声学 token 序列；将这三个序列拼接，预测出后续的粗粒度声学 token 序列
	4. 第三阶段：将第二阶段获取到的完整的粗粒度声学 token 序列作为细粒度模型的输入，预测得到完整的声学 token 序列
	5. 最后，根据完整的声学 token 序列，使用 detokenizer（SoundStream 的解码器）生成音频


### 实验与分析

#### 实验设计与准备

**两种实验任务:** 分别是语音续写和钢琴曲续写：语音续写需要生成的音频满足音色、录音环境、韵律方面的一致性，并且语言准确、语义连贯；音乐续写需要生成的音频满足旋律、和声和节奏的连续性。

**数据集：** 语音数据采用无标签的 6 万小时的 Libri-Light 数据。

**实验配置：** wav2vec-Bert 采用 6 亿参数量的模型，输出 embedding 来自 MLM 部分的第 7 层，离散化聚类时类别个数 $K=1024$；SoundStream 采用 320 倍降采样，$Q=12$ 个量化器，前 $Q'=4$个量化器作为第二阶段的粗粒度 token，后面 8 个量化器用于第三阶段的细粒度 token；Transformer 自回归语言模型的参数：12 层，self-attention 包含 16 个 head、embedding 维度为 1024，feed-forward 层的隐层大小为 4096，使用的是相对位置编码。

#### 语义 token 的信息

使用 LibriSpeech test-clean 中 4-10 秒内的音频，采用第二类推理方式**声学生成**，每条音频随机生成三个样本，使用 Conformer-Transducer 的 Large 模型对生成的样本进行识别，计算得到 CER 和 WER，用于衡量生成音频在语义内容上的正确性和一致性。从下表的实验结果可以看出，AudioLM 生成的语音在内容上的一致性还是比较高的，尤其是考虑到**声学生成**任务中，新生成的音频可能会带有一些额外的噪声，因此语音识别系统的效果也会有些降低。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table2.jpg" width = "520"/>

#### 声学 token 的信息
同样地，使用 LibriSpeech 中的 291 个说话人的音频，在第二种（**声学生成**）和第三种（**音频续写**）推理方式下新生成的音频和原始音频进行说话人分类，实验结果如下图所示。从中可以得到一些结论：**声学生成**因为只保留了真实音频的语义 token，不包含说话人音色方面的信息，所以说话人分类准确率很低；但是音频续写以 prompt 的真实声学 token 作为部分条件输入，保持了音色的连续性，因此说话人分类准确率很高。

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230227/table3.jpg" width = "520"/>


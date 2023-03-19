---
title: 语音合成 | 论文笔记 | VALL-E
date: 2023-01-20 11:40:36
tags: [语音合成, 音频编解码]
categories: [论文笔记]
copyright_info: true
toc: true
mathjax: true
comment: false
---

之前介绍了 **AudioLM** 和 **MusicLM** 两篇基于 **SoundStream** 的音频/音乐生成的论文，思想都是：将音频编解码的 SoundStream 模型作为音频信号离散化的 tokenizer，在量化后的声学 token 上进行语言模型建模，最大程度地保留了音频生成所需的细节信息，这也成为了一种行之有效的音频生成方法论。本篇介绍的论文 VALL-E 将这一思想应用于语音合成任务（之前的 AudioLM 也可以做语音的续写生成，但不是以文本为条件（Condition）的 Text-to-Speech 的语音合成），使用更优的 **Encodec** 音频编解码器，实现了一种有效的 zero-shot TTS 模型。

<br>

| 会议/期刊 | 年份 | 题目 | 链接 |
| :---: | :---: | :---: | :---: |
| arxiv | 2023 | Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers | [https://arxiv.org/abs/2301.02111](https://arxiv.org/abs/2301.02111) |

<br>

VALL-E 将语音合成视为一种**条件语言建模**（Conditional Language Modeling）的任务，使用神经网络音频编解码器的中间结果作为音频的离散表征，在此表征的基础上进行语言建模。VALL-E 使用 6 万小时量级的英语语音数据（语音合成的数据量也卷起来了）进行预训练，在对未见过的目标说话人进行 zero-shot 推理时，只需要 3 秒的音频作为 **prompt**（也可称为前缀），即可实现高自然度 + 高音色相似度的语音合成，在语音的情感、声学环境等方面也能和 prompt 的语音保持一致，体现出 VALL-E 已经具备 **in-context learning** 的能力。


### 论文背景与概述

语音合成的训练数据通常对音频质量要求较高，网络上的语音数据虽然很多，但是绝大多数质量较低、不够干净，无法用来训练语音合成模型；同时，高质量音频的稀缺，也导致目前的语音合成系统通常泛化性较差，对于训练数据中没有出现过的说话人（zero-shot），语音合成的音色相似度和自然度都显著降低。

针对 zero-shot TTS 问题，目前的工作大多采用 Speaker / Style Encoder 等**基于参考音频**的方法，从参考音频中提取全局的说话人特性信息，并显式注入到 TTS 模型中。VALL-E 则完全跳脱出了这种思想，只需要从 prompt 语音抽取离散化的声学 token，基于声学 token 和输入的音素序列，使用条件语言模型对后续的声学 token 进行预测，然后从声学 token 解码恢复出语音波形。

> NLP 文本生成任务中大语言模型的启发：一方面，更大的**数据量**可以显著提高模型的泛化性；另一方面，**语言模型**相比于 VAE / GAN / Diffusion 等生成式模型，可能是一种更简单有效的生成模型。NLP 领域近几年涌现了很多值得借鉴的大语言模型（**LLMs**，Large Language Models）工作（比如 GPT 系列），之后在其他文章中会另做梳理。


VALL-E 工作的贡献可以概括为：
1. VALL-E 使用音频编解码器中的编码结果而不是梅尔特征作为 TTS 系统建模时的中间表征，并采用条件语言模型的建模方法，具有上下文学习（In-Context Learning）的能力，推理阶段可以使用目标说话人语音作为 prompt 直接实现 zero-shot 语音合成，不需要对模型进行 finetune；
2. VALL-E 使用了大量的数据，采用的是半监督的方式：一部分有标注语音训练 ASR 模型，然后对大量的无标注语音进行识别得到伪标签，用于模型的训练；
3. 对于同样的文本输入，VALL-E 能够根据 prompt 的不同而生成不同说话人、不同情感、不同录音环境的多样的语音；
4. VALL-E 在 zero-shot 场景下合成的语音具有 SOTA 的音色相似度和自然度。


持续更新...










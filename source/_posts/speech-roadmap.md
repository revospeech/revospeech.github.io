---
title: 开篇 | RevoSpeech 智能语音工作指南
date: 2023-01-01
tags: [语音识别, 语音合成]
categories: [技术思考]
copyright_info: true
toc: true
mathjax: true
comment: false
# aging: false
# aging_days: 200
sticky: 999
---

智能语音是当今科技发展的热门方向之一。随着人工智能技术的不断进步，智能语音技术日趋成熟，在各个领域的应用也在不断增多，目前已在语音搜索、智能家居、语音助理等多个领域得进行落地，并且随着元宇宙、AIGC 等新产业的兴起焕发出新的活力。[**RevoSpeech**](https://revospeech.github.io) 旨在推动智能语音的落地和普及，基于学术界近十年在语音处理、语音识别、语音合成等方向的技术突破，总结归纳智能语音的技术要点，密切跟进前沿科研动向的同时，展望智能语音乃至人工智能技术的未来发展。

本文主要梳理未来一年（甚至更长时间范围）内，RevoSpeech 计划在智能语音方向的发展指南（Road Map），主要着眼于技术总结和前沿论文跟进，同时也将对智能语音领域的数据工程及应用落地等问题进行探讨。

> 注意：本工作指南将主要着眼于**语音识别**和**语音合成**两大技术。

# 语音识别

## 语音数据库构建

### 开源语音数据梳理

- 整理目前语音社区内的大规模语音数据，持续跟进开源数据
- 覆盖英文、中文、韩语、日语、俄语、法语等多语种语音数据
- **持续更新中**：[speech-datasets-collection](https://github.com/RevoSpeechTech/speech-datasets-collection)

### 自建语音数据库

#### 数据库构建流程

- 参考论文
	- LibriSpeech: [pdf](http://www.danielpovey.com/files/2015_icassp_librispeech.pdf), MLS (LibriVox): [pdf](https://arxiv.org/pdf/2012.03411.pdf)
	- GigaSpeech: [pdf](https://arxiv.org/pdf/2106.06909.pdf), WenetSpeech: [pdf](https://arxiv.org/pdf/2110.03370.pdf)
	- JTubeSpeech: [pdf](https://arxiv.org/pdf/2112.09323.pdf), SPGISpeech: [pdf](https://arxiv.org/pdf/2104.02014.pdf), The People's Speech: [pdf](https://arxiv.org/pdf/2111.09344.pdf)

- 数据获取来源
	- Videos: [YouTube](https://www.youtube.com), [bilibili](https://www.bilibili.com)
	- Podcasts: [google](https://podcasts.google.com), [apple](https://www.apple.com/hk/en/apple-podcasts)
	- Misc: [archive.org](https://archive.org)

- 数据获取工具
	- [ytb-dlp](https://github.com/yt-dlp/yt-dlp), [you-get](https://github.com/soimort/you-get)

- 数据清洗流程（待梳理）
	- 文本归一化: [NeMo](https://github.com/NVIDIA/NeMo-text-processing), [Wenet](https://github.com/wenet-e2e/WeTextProcessing), [SpeechIO](https://github.com/speechio/chinese_text_normalization)
	- 强制对齐: [Kaldi](https://github.com/kaldi-asr/kaldi), CTC, [DSAlign](https://github.com/mozilla/DSAlign)
	- 音频切分与校验：[Kaldi cleanup](https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/cleanup/clean_and_segment_data_nnet3.sh)


#### 数据库构建计划
**_v1.0 阶段_**

- 目标语言：中文/英文
- 数据来源：youtube / bilibili / podcast 自带字幕文件的音视频数据
- 对齐模型：kaldi 使用 multi-cn + wenetspeech 的 nnet3 模型
- 处理思路：基于字幕或文本进行长音频切分并进行 cleanup 处理
- 方案评测：抽查数据准确度 (要求 97% 以上)，确认方案可行性
- 最终产出：中文/英文各 10k 小时以上

**_v2.0 阶段_**

- 目标语言：中文/英文
- 数据来源：youtube / bilibili 不带外挂字幕但存在硬字幕的视频
- 额外功能：需要额外进行字幕定位和 OCR 文本识别功能
- 处理思路：以 OCR 识别出的结果为伪标签，再进行 cleanup 处理
- 方案评测：抽查数据准确度 (要求 97% 以上)，确认方案可行性
- 最终产出：中文/英文各 10k 小时以上

**_v3.0 阶段_** (长期计划)

- 多次迭代模型，提高 Kaldi 中文/英文对齐模型的多领域泛化能力
- 扩充目标一：中文/英文的不同方言/口音的语音数据
- 扩充目标二：法语/德语/西班牙/韩语/日语 等多语种的语音数据
- 扩充目标三：Audio-Visual 语音和图像多模态的数据库

---

## 传统语音识别

以 **Kaldi** 为代表的传统语音识别，目前在学术界/产业界仍然保持着相对的优势。虽然识别准确率与端到端模型已经存在比较明显的差距，但其中声学特征、MMI 损失函数、WFST 解码器、语言模型重打分等构件在端到端语音识别中仍然是重要的技术。因此，在传统语音识别部分，RevoSpeech 将着力推进 Kaldi-Revo 项目，优化传统语音识别的训练框架，充分发挥其模型轻量、落地成本低等优势。

### 训练流程优化

#### 声学特征
- 理论梳理：MFCC/FBank/LPC 等声学特征的提取流程
- 代码研读：Kaldi 特征提取部分的代码整理
- 工程实践：C++ 动态库及接口封装知识

#### 声学模型

- 核心训练流程：GMM-HMM → DNN-HMM(nnet3) → LF-MMI(chain) 
- 发音词典构建：在 GMM 早期加入更多可能的发音，统计训练数据的发音概率进行筛选
- 前沿声学模型：
	- 候选结构：TDNN-F-SAN 或 Multi-Stream TDNN-F-SAN
	- 流式识别：保证实时性、准确性、时延低

#### 语言模型
- N-gram 语言模型：
	- 通用向语言模型的训练
- 语言模型重打分：
	- 候选结构：Transformer/Transformer-XL 两种模型的双向重打分
	- 加速思想：词级别语言模型 → 字级别语言模型
	- 工程实践：Transformer-XL 参考 NVIDIA 的 [Benchmark](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL)

#### 解码器
- 理论学习：
	- 参考教程：语音识别：原理与应用(第二版), [代码研读](https://github.com/snsun/kaldi-decoder-code-reading)
- 热词方案：基于前缀树的方案在 Kaldi 解码器代码上实现
- 解码器复现
	- HMM 的 WFST Decoder 的实现方法
	- CTC 的 WFST Decoder 的实现方法
- 解码器优化
	- BigLM Decoder 的实现方法
	- [Async Decoder](https://www.danielpovey.com/files/2021_icassp_async_biglm_decoder.pdf)
	- [LET-Decoder](https://www.danielpovey.com/files/2021_spl_lazy_evaluation_decoder.pdf)

### 开源数据集实验
**_v1.0 阶段_**

- 中文：aishell / aidatatang_200zh / MagicData-RAMC / multi_cn / WenetSpeech 
- 英文：LibriSpeech / TEDLium / GigaSpeech 
- 中英 Code-Switching：ASCEND / TALCS 
- 英文：尝试 **grapheme 级别的 chenone** 建模方式 [pdf](https://arxiv.org/pdf/1910.01493.pdf)
- 产出：整理对比实验结果，梳理得到 Benchmark

**_v2.0 阶段_**

- 自监督特征：Wav2vec2 / WavLM / HuBert / Data2vec 
- 多通道/远场：aishell4 / alimeeting / [CHiME(5|6|7)](https://www.chimechallenge.org) 
- 多语种 ASR

**_v3.0 阶段_**(长期计划)
- 多模态 (AVSR)：LRS2-BBC / LRS3-TED / [CMLR](https://www.vipazoo.cn/CMLR.html)
- 多模态自监督：AV-HuBert
- 歌词识别/转写：DAMP / DALI
- 多模态歌词识别：[N20EM](https://n20em.github.io)
- 其他参考工作：[pkwrap](https://github.com/idiap/pkwrap), [apam](https://github.com/idiap/apam), [wav2vec-lfmmi](https://github.com/idiap/wav2vec-lfmmi)


### 工具化实践
- CPU 服务：[vosk-server](https://github.com/alphacep/vosk-server) / [vosk-api](https://github.com/alphacep/vosk-api) 实践及优化
- GPU 服务：Nvidia [Triton](https://github.com/NVIDIA/DeepLearningExamples/tree/master/Kaldi/SpeechRecognition) GPU 服务化实践
- Android 端侧：[vosk-android](https://github.com/alphacep/vosk-android-demo) 实践及优化
- 参考书籍：语音识别服务实战

---

## 端到端语音识别

### 基础知识准备

- 训练准则：CTC / RNN-T / Pruned RNN-T 
- 模型结构：Transformer / Conformer (Squeezeformer) / Emformer / Zipformer 
- 多种解码方案：
	- CTC：greedy search / beam search / prefix beam search
	- RNN-T：greedy search / beam search / TSD / ALSD / NSC / MAES 
- 流式语音识别：Emformer / Zipformer 
- 语言模型适应：N-gram / Transformer LM 并入解码过程
- 非自回归的端到端 ASR
   - Mask-CTC 系列模型
   - Paraformer
- 进阶课题
	- 低延迟解码：RNN-T 的 FastEmit / TrimTail 
	- 内部语言模型估计与适应：ILME / ILMA / ILMT
	- 热词 (context-biasing)：WFST / 前缀树 

### ASR Benchmark

- 开源数据集实验
	- 中文：aishell / aidatatang_200zh / MagicData-RAMC / multi_cn / WenetSpeech 
	- 英文：LibriSpeech / TEDLium / GigaSpeech 
	- 中英 Code-Switching：ASCEND / TALCS 
	- 自监督特征：Wav2vec2 / WavLM / HuBert / Data2vec 
	- 多通道/远场：aishell4 / alimeeting / [CHiME(5|6|7)](https://www.chimechallenge.org) 
	- 多模态 (AVSR)：LRS2-BBC / LRS3-TED / [CMLR](https://www.vipazoo.cn/CMLR.html)
	- 多模态自监督：AV-HuBert
	- 歌词识别/转写：DAMP / DALI
	- 多模态歌词识别：[N20EM](https://n20em.github.io)

### 工程化实践
- Wenet Runtime
- Icefall (sherpa / sherpa-ncnn)
- 预计产出：开源大规模预训练模型，提供中文/英文/中英混等语音识别服务

---

# 语音合成

## 深度学习生成模型
- 理论基础：自回归模型 / VAE / GAN / Flow / Diffusion
- 参考资料：[udlbook](https://udlbook.github.io/udlbook/) / [pml-book2](https://probml.github.io/pml-book/book2.html)

## 数据采集与处理
- 与语音识别中的数据处理基本一致，不再赘述
- 强制对齐通常采用 [MFA](https://montreal-forced-aligner.readthedocs.io/) 工具进行离线对齐

## 前端文本分析

- 文本预处理：文本归一化
- 中文 / 英文 g2p 模块，包含多音字词的发音预测
- 基于文本的分词和停顿预测模块
- 功能产出：输出带有停顿信息的音素序列

## 声学模型

- 自回归：Tacotron / Tacotron2
- 非自回归：FastSpeech / FastPitch / FastSpeech2 / SpeedySpeech
- 经典文章：DeepVoice 1/2/3 、Parallel tacotron 1/2
- 基于 VAE：VARA-TTS / VAENAR-TTS / PVAE / GMVAE-Tacotron / VAE-TTS / BVAE-TTS
- 基于 GAN：GAN exposure / Multi-SpectroGAN
- 基于 Flow：Flow-TTS / Glow-TTS / RAD-TTS / Flowtron
- 基于 Diffusion：Guided-TTS / Grad-TTS / Diff-TTS / PriorGrad-AM / DiffWave / FastDiff / ProDiff
- 其他：DurIAN / VoiceLoop / ParaNet

## Vocoder 声码器

- 经典之作：WORLD / WaveNet / WaveRNN
- WaveRNN 系列：subscale WaveRNN / MultiBand WaveRNN / Universal WaveRNN / LPCNet
- LPCNet 进阶版：FeatherWave / Full-band LPCNet / Bunched LPCNet / Gaussian LPCNet
- 基于 VAE：Wave-VAE
- 基于 GAN：WaveGAN / GAN-TTS / Parallel WaveGAN / VocGAN / MelGAN / Hifi-GAN / Fre-GAN
   - 基于 MelGAN：MultiBand MelGAN
   - 通用声码器：Avocodo / BigVGAN
- 基于 Flow：Parallel WaveNet / WaveGlow / WaveFlow / SqueezeWave / FloWaveNet
- 基于 Diffusion：WaveGrad /  PriorGrad-Vocoder / DiffWave

## 完全端到端语音合成

- FastSpeech 2s / EfficientTTS-Wav / Wave-Tacotron
- EATS / JETS：FastSpeech 2 + Hifi-GAN
- VITS / NaturalSpeech

## TTS 进阶方向

- 表现力 TTS / 情感化 TTS / 个性化 TTS
- 语音转换：Voice Conversion
- 歌声合成：WeSinger / WeSinger2 / Diffsinger / Multi-Singer / Learn2sing 1+2
- 语音编辑：RetrieverTTS / A3T

## 工程实践
- 参考项目：espnet / PaddleSpeech 

---

# 预期产出

## 语音数据库构建

- 搜集开源语音数据：**speech-datasets-collection**
	- [https://github.com/RevoSpeechTech/speech-datasets-collection](https://github.com/RevoSpeechTech/speech-datasets-collection)
	- 目前持续更新中，[网页版](https://revospeech.github.io/2023/02/13/2023-02-13-speech-datasets/)

- 数据采集功能模块：**speech-miner**
	- 支持语音数据的自动下载 + 预处理 + 格式规范化

- 数据对齐清洗模块：**speech-cleaner**
	- 支持语音数据的清洗处理、对齐及筛选

- 字幕 OCR 提取功能模块：**subtitle-extractor**
	- 支持一般视频的字幕定位 + OCR 文本识别

## 传统语音识别

- Kaldi Benchmark 框架：**kaldi-revo**
	- 使用更准确的 nnet3 模型获取对齐结果
	- 支持 TDNN-F-SAN + Multi-Stream TDNN-F-SAN 模型
	- 支持 Transformer / Transformer-XL 重打分
	- 多个 egs 及 Benchmark 实验结果
	- 支持基于前缀树的热词方案配置
	- 支持 Audio-Visual 语音识别

- 英文 g2p 发音词典构建：**english-lexicon-builder**
	- 发音词典引入 g2p 工具的多样性
	- 发布基于 GigaSpeech 的开源英文发音词典

- Audio-Visual 语音识别辅助模块：**visual-feature-extractor**
	- 多个候选模型提取图像模态的特征

- 统一的语言模型重打分框架：**neural-lm-rescorer**
	- 支持：RNN/LSTM / SRU / Transformer / Transformer-XL / GPT-2
	- 支持双向重打分操作
	- 支持单机多卡下的大规模文本数据训练
	- 开源大规模语料预训练的语言模型
	- 加速和 Benchmark 结果参考 Nvidia 相关工作

## 端到端语音识别

- 在现有框架基础上进行功能增强
	- [Icefall](https://github.com/k2-fsa/icefall) (主): Conformer, Emformer, Zipformer 
	- [Wenet](https://github.com/wenet-e2e/wenet) (次): U2, U2++ 
	- [Espnet](https://github.com/espnet/espnet): 借鉴前沿新工作
	- [Fairseq](https://github.com/facebookresearch/fairseq): 借鉴前沿新工作

## 语音合成
- 在现有框架基础上进行功能增强
	- [Espnet](https://github.com/espnet/espnet): 借鉴前沿新工作
	- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN): 声码器

## 开源预训练模型
- 语音数据处理
	- 强制对齐模型
- 语音识别模型
	- Kaldi 声学模型
	- Transformer 等强语言模型
	- 端到端 ASR 框架的预训练模型
- 语音合成模型
	- 多说话人 TTS 模型
	- 通用声码器


---
title: 生成模型基础 | 论文笔记 | GAN 的基础（一）
date: 2023-01-21 11:40:36
tags: [生成模型, 语音合成, 声码器]
categories: [论文笔记]
copyright_info: true
toc: true
mathjax: true
comment: false
---

**生成对抗网络（GAN，Generative Adversarial Nets）**是 Yoshua Bengio 团队在 2014 年提出的，一作是 [Ian Goodfellow](https://www.iangoodfellow.com)。Yoshua Bengio 团队坚持认为：深度学习的目标是发现更丰富、更加层次化的模型，能够用来表示各种数据的概率分布。2014 年时，深度学习方法在判别式模型方面已经有很多突出的成果，用于将高维输入特征映射到不同的类别标签，基于梯度的反向传播方法优化模型。然而当时深度学习在生成式模型方面的影响性成果较少，主要是因为：当时提出的生成模型在最大似然估计（MLE，Maximum Likelihood Estimation）的过程中需要近似很多难处理的概率运算（近似推断之类），而 GAN 正是为了规避之前生成模型的缺点而提出的一种新型的生成模型。

在智能语音领域，GAN 的应用非常广泛，在语音合成、语音降噪等生成式任务中是一类重要的方法。先在本文介绍下 GAN 的基础知识，包括 GAN 模型的基本思想、数学分析、优化方法及训练目标等，后续会针对语音合成的声码器等具体任务介绍研究者提出的各种模型。本文主要涉及下面的三篇论文：

| 会议/期刊 | 年份 | 题目 | 链接 |
| :---: | :---: | :---: | :---: |
| NeurIPS | 2014 | Generative Adversarial Nets | [[pdf]](https://papers.nips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) |
| NeurIPS | 2016 | Improved Techniques for Training GANs | [[pdf]](https://papers.nips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf) |
| ICCV | 2017 | Least Squares Generative Adversarial Networks | [[pdf]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Mao_Least_Squares_Generative_ICCV_2017_paper.pdf) |


GAN（生成对抗网络）包含生成器（Generator，生成式模型）和判别器（Discriminator，判别式模型）两大模块，GAN 的核心思想正是在于**生成器和判别器的对抗（Adversarial）**。在实际应用中，通常只用到生成器部分，用于生成图像、语音或音频等，但实际上训练出效果好的生成器离不开判别器，判别器的作用是区分样本究竟是从生成模型生成的假样本还是真实样本。GAN 实际上提出了一套新的框架，和具体的训练算法、模型结构和优化算法都无关。

对于生成器和判别器间存在的对抗关系，论文给出一个形象的比喻：生成器相当于造假币的人，而判别器则相当于警察；所谓的对抗关系是指，造假币的人会为了让警察无法区分货币的真假而提升造假币的能力，同时警察也会提高自己区分真假币的能力（判别器的同步优化），优化时生成和判别两个模型的能力都会提升。对于最终训练完成的生成器，理想的状态是判别器无法区分样本究竟是真实的还是生成的，从而达到“以假乱真”的效果。

### GAN 之前的生成模型

在 GAN 被提出之前，大多数生成模型的工作集中于**已经给定具体概率分布函数 pdf** 的情况。已知概率分布函数时可以通过最大对数似然目标来训练，比较典型的有效方法是深度玻尔兹曼机（DBM，Deep Boltzman Machine），但实际上生成模型处理的似然函数都非常复杂，往往需要对似然函数的梯度进行大量近似运算。

为了避免上述大量梯度近似的问题，有研究者提出**生成式机器**（Generative Machine），不需要表征出具体的似然函数形式，也可以生成所需概率分布的样本，典型的工作是**生成式随机网络**，可以直接用反向传播来训练，但生成随机网络一般依赖于马尔科夫链，GAN 消除了这一依赖并进行了拓展。

在 GAN 工作的同时，有其他研究者延续了生成随机网络的思想，提出了更通用的随机反向传播方法，能够对方差有限的高斯分布进行反向传播，更新均值和协方差的参数，用于变分自编码器（VAE，Variational AutoEncoder）的训练。VAE 是另一种生成式模型，之后在另外的文章中会详细解读。从思想上来看，VAE 和 GAN 都是使用两个网络协同训练，但是 VAE 中的第二个网络作为判别模型时进行了近似推断，此外 GAN 不能够建模离散的数据，VAE 不能使用离散的隐变量。

之前有类似于 GAN 的**采用判别式模型协助训练生成模型**的工作。PM（Predictability Minimization）的作者 [Jürgen Schmidhuber](https://people.idsia.ch/~juergen/) 曾经和 GAN 的作者 [Ian Goodfellow](https://www.iangoodfellow.com) 产生过争议（[YouTube 链接](https://www.youtube.com/watch?v=HGYYEUSm-0Q)），PM 的思路是第一个网络的每个隐层单元都朝着和第二个网络当前输出结果不同的方向去训练，思想上是有几分相似的地方。Ian Goodfellow 阐明了 GAN 和 PM 的三点不同之处：
1. GAN 的生成器和判别器之间的对抗训练，是模型训练的唯一目标，该目标已经足够训练网络；但是 PM 严格来说相当于辅助隐层单元训练的正则项，不是训练的主要目标；
2. 在 PM 中，两个网络的输出之间进行对比，其中一个网络的目标使得两者的输出更加接近，另一个网络的目标是使得两者的输出不同；但是 GAN 没有对比两个网络的输出，而是将第一个网络得到的高维特征信息作为第二个网络的输入，目标是让第二个网络对输入丧失真假的判别能力；
3. PM 的训练目标是单纯的优化问题，最小化定义的目标函数；但是 GAN 包含了两层目标的同步优化，即博弈中的 Minimax，在两层优化目标中间找到一个相对的平衡点。

### GAN 的数学含义

原始的论文中为了简化问题，生成器和判别器都使用普通的多层感知机（MLP，Multi-Layer Perceptron），将生成模型的输入设定为随机噪声。

GAN 的生成器目的是从数据 $x$ 中学习分布 ${p_g}$，$g$ 表示 generator。GAN 对问题的形式做了个变换，引入噪声变量 $z$，先定义 $z$ 的先验分布为 $p_z(z)$，GAN 的生成器是将输入的噪声 $z$ 映射到数据 $x$，这一过程用 $G(z) = G(z;θ_g)$ 来表示，$z$ 是作为输入的噪声变量，$θ_g$ 表示生成器 MLP 的参数。

GAN 的判别器可以定义成 $D(x;θ_d)$ ，其中 $x$ 表示判别器的输入，$θ_d$ 表示判别器 MLP 的参数。$D(x;θ_d)$ 的输出是一个标量，用来表示输入 $x$ 来自真实样本（而不是生成器生成）的概率。

确立了 GAN 的生成器和判别器之后，两者各自的训练目标分别为：
- 生成器：最小化“生成样本被判别为 0 的概率”的损失函数
- 判别器：最大化“表示真假分类准确性的损失函数”，希望将生成的样本正确分类为 0（0 表示假的），将真实的样本正确分类为 1（1 表示真的）

> 一方面，判别器对应于判别任务，假设真实样本 $x$ 被分类为 1 的概率为 $p_1 = D(x)$，生成样本被分类为 1 的概率为 $p_2 = D(G(z))$，在整个数据集上求期望（平均）后，交叉熵损失函数为：
> $$\max E [\log(p_1) + \log(1-p_2)]  = \max E [\log(D(x)) + \log(1-D(G(z)))]$$
> 另一方面，生成器需要最小化生成样本被判别为 0 的“概率”，可以理解成最大化生成样本被判别为 1 的“概率”，同样也是在整个数据集上求期望，所以交叉熵损失函数为：
> $$ \max E \log(p_2) = \min E \log(1-p_2) = \min E \log(1-D(G(z)))$$
> 从生成器和判别器的损失函数来看，核心的博弈是 $\log(1-D(G(z)))$ 部分，在生成器和判别器中，这部分的优化目标是矛盾的：判别器中的优化方向是**最大化**（让判别器将生成的样本分类为假类），生成器中的优化方向是**最小化**（让判别器将生成的样本分类为真类），这种矛盾的优化目标决定了 GAN 的两层目标的同步优化，在训练过程中体现为判别器和生成器交替训练。将生成器和判别器的优化目标联合起来即为 GAN 的整体优化目标：
> $$ \min_D \max_G V(D, G) = E_{x \in p_{data}(x)} \log(D(x)) + E_{x \in p_{z}(z)} \log(1-D(G(z))) $$

其实 GAN 的这一优化目标可能无法让 G 学习的足够好，比如在学习的起始阶段，G 的生成能力很差，和真实样本之间存在明显的差别，因此判别器 D 会高置信地将生成样本判别为类别 0（假样本），这种情况下 $\log(1-D(G(z)))$对应的梯度为 0 导致无法训练。论文为了解决这一问题，在训练的起始阶段，要求 G 的训练目标不是最小化 $\log(1-D(G(z)))$ ，而是最大化 $\log(D(G(z))$，两者目标上是一致的。

### GAN 原理的直观表示

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230222/gan-pic-sample.jpg" width = "700"/>

上图直观地表示出 GAN 的训练原理，其中 $z$ 表示噪声分布，$x$ 表示数据分布，$z$ 到 $x$ 带有箭头的过程，表示噪声输入经过生成器后的输出。

1. 图(a)：GAN 的初始阶段，生成器输出的数据分布和真实的样本分布存在较大差异，此时判别器还没更新，因此对两个分布之间的判别能力也比较差；
2. 图(b)：先优化判别器，生成器的参数固定不变，上述两个分布固定的情况下，能够学出效果更好的判别器；
3. 图(c)：再优化生成器，判别器的参数固定不变，让生成的样本被判别为真实样本的概率更大，图中体现为生成器输出的数据分布和真实样本数据分布更加接近；
4. 图(d)：回过头继续优化判别器，生成器的参数固定不变，根据生成样本的新分布，更新判别器在该情况下的判别效果；...

生成器和判别器如此交替更新，直到达到类似图 (d) 所示的理想情况，生成样本分布和真实样本分布完全一样，判别器也完全无法区分样本究竟来自哪个分布，意味着生成的样本已经达到了“以假乱真”的地步。

### GAN 的具体流程

<img src="https://cdn.staticaly.com/gh/revospeech/image-hosting@master/20230223/gan-train-proc.jpg" width = "650"/>

对于每次训练迭代：
1) 先执行 k (k 是超参数) 步，固定生成器，更新判别器：
- 从噪声的先验分布 $p_z$ 中采样 m 个噪声样本，通过生成器得到对应的 $G(z)$
- 从真实数据分布中 $p_{data}$ 中采样 m 个真实样本
- 优化交叉熵损失函数 $\max E [\log(D(x)) + \log(1-D(G(z)))]$，更新判别器的参数

2) 再执行 1 步，固定判别器，更新生成器：
- 从噪声先验分布中采样 m 个噪声样本
- $\min E \log(1-D(G(z)))$ 作为目标更新生成器参数
以上步骤迭代重复很多次，即为 GAN 的一般训练过程

### GAN 原理的数学分析

GAN 的生成器的目标，是让生成的样本分布和真实样本的分布完全相同。设生成器从数据中训练学习到的分布为 $p_g$，真实的理论上的样本分布为 $p_{data}$，GAN 的数学原理是：
> GAN 在经过 MiniMax 博弈之后，有个全局的最优点，$p_g = p_{data}$

1. 固定生成器 G 时，求判别器 D 的最优解：

对于优化目标 $ \min_D \max_G V(D, G) = E_{x \in p_{data}(x)} \log(D(x)) + E_{x \in p_{z}(z)} \log(1-D(G(z))) $，只关注内层的 max 目标时，目标可以表示为：

$$
\begin{align}
    V(G,D) & = \int_{x} p_{data}(x) \log (D(x)) dx + \int_{z}p_{z}(z) \log (1 - D(g(z))) dz \\\\
    & = \int_{x} p_{data}(x) \log (D(x)) + p_g(x) \log (1-D(x)) dx 
\end{align}
$$

其中，第二个等号只是进行了 $ t = g(z) $ 的替换，对 $t$ 积分，但是 $t$ 和 $x$ 只是符号表示，所以可以合并积分项。目标函数求导后值为 0 时，求解得到 D 的最优解是：

$$ D_G^{\star}(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)} $$

2. 将 D 的最优解代回，得到固定 D 优化 G 时的目标函数：
$$ 
\begin{align}
    C(G) & = max_D V(G,D) \\\\
    & = E_{x \in p_{data}}[\log D_G^{\star}(x)] + E_{z \in p_z}[\log (1 - D_G^{\star}(G(z)))] \\\\
    & = E_{x \in p_{data}}[\log D_G^{\star}(x)] + E_{x \in p_g}[\log (1 - D_G^{\star}(x)] \\\\
    & = E_{x \in p_{data}} [\log\frac{p_{data}(x)}{p_{data}(x) + p_g(x)}] + E_{x \in p_g} [\log\frac{p_{g}(x)}{p_{data}(x) + p_g(x)}]
\end{align}
$$

注意，上述公式中 $ p_{data} $ 和 $ p_{g} $ 都是随机变量的概率分布：$ p_{data} $ 表示真实样本的分布，$ p_{g} $ 表示生成样本的分布。

> **数学基础知识复习**
> 1. 若连续随机变量 $x \in p(x)$, 则 $x$ 的期望计算方式如下：
> $$ E(x) = \int_{-\infty}^{+\infty} xp(x) dx$$
> 2. **KL (Kullback-Leibler) 散度** 常被用于衡量两个分布（如 $p_x$ 和 $q_x$）间的相似程度，数学表达式为：
> $$ D_{KL}(p || q) = \int_{-\infty}^{+\infty} p(x) \log \frac{p(x)}{q(x)} dx$$
> 计算 KL 散度时对两个分布的顺序是有要求的，属于非对称关系，即 $ D_{KL}(p || q) \neq D_{KL}(q || p) $。
> 3. **JS (Jensen-Shannon) 散度** 也可用于衡量两个分布间的相似程度，解决了 KL 散度的非对称的问题，数学表达式为：
> $$ D_{JS}(p || q) = \frac{1}{2} D_{KL}(p || \frac{p+q}{2}) + \frac{1}{2} D_{KL}(q || \frac{p+q}{2}) $$
> JS 散度的计算具有对称性，即 $D_{JS}(p || q) = D_{JS}(q || p)$。
> 4. 不论是 KL 散度还是 JS 散度，数值都具有非负性。KL 散度的非负性也被称为[吉布斯不等式](https://zh.wikipedia.org/zh-hans/%E5%90%89%E5%B8%83%E6%96%AF%E4%B8%8D%E7%AD%89%E5%BC%8F)，KL 散度值为 0 当且仅当 $p = q$ 时得到。JS 散度非负性很容易从 KL 散度的非负性推导出。

回过头来看 $C(G)$ 的表达式可以变换为：
$$
\begin{align}
C(G) & = E_{x \in p_{data}} [\log\frac{p_{data}(x)}{p_{data}(x) + p_g(x)}] + E_{x \in p_g} [\log\frac{p_{g}(x)}{p_{data}(x) + p_g(x)}] \\\\
& = \int_{-\infty}^{+\infty} p_{data}(x) \log \frac{p_{data}(x)}{p_{data}(x) + p_{g}(x)} dx + \int_{-\infty}^{+\infty} p_{g}(x) \log \frac{p_{g}(x)}{p_{data}(x) + p_{g}(x)} dx \\\\
& = [\log\frac{1}{2} + \int_{-\infty}^{+\infty} p_{data}(x) \log \frac{p_{data}(x)}{\frac{p_{data}(x) + p_{g}(x)}{2}} dx] + [\log\frac{1}{2} + \int_{-\infty}^{+\infty} p_{g}(x) \log \frac{p_{g}(x)}{\frac{p_{data}(x) + p_{g}(x)}{2}} dx] \\\\
& = -log4 + D_{KL}(p_{data}||\frac{p_{data} + p_g}{2}) + D_{KL}(p_{g}||\frac{p_{data} + p_g}{2}) \\\\
& = -log4 + 2 \times D_{JS}(p_{data} || p_g)
\end{align}
$$

因此，$C(G) >= -log4$ 当且仅当 $p_g = p_{data}$ 时得到，这也证明出了 GAN 优化目标的全局最优结果是生成器的分布和训练数据的分布完全相同。

GAN 原论文中的实验主要是图像生成任务，此处不再过多展开，具体可以参考本文结尾的李沐老师的讲解视频。GAN 被提出后，在代码实现和实际训练过程中也暴露出一些问题，后续将具体介绍一些 GAN 中常用的方法（trick）作为补充。

<br>

<div style="position: relative; width: 100%; height: 0; padding-bottom: 75%;">
    <iframe src="//player.bilibili.com/player.html?aid=634089974&bvid=BV1rb4y187vD&cid=439574005&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" style="position: absolute; width: 100%; height: 100%; left: 0; top: 0;"></iframe>
</div>


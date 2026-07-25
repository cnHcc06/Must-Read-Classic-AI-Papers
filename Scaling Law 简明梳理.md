<h1 align="center"> 《Scaling Laws for Neural Language Models》<br> 论文梳理以及不同流派的挑战 </h1>

Scaling Law 可以说是最难概述的一篇论文了。它是OpenAI关于大模型规模定律的开山之作，它通过研究各个尺度下模型的行为，总结出了模型交叉熵损失L和参数量、计算量、数据量、批量大小等之间的复杂关系，并以这几组关系为基础求解优化问题，从而给出了效益最大化的LLM训练策略。  

简单来说，论文就是在研究这样一个核心函数：

$$ L(N,B,S) $$

其中又有 $D=BS,C=6ND$ ,关系非常复杂，但论文通过严密的逻辑娓娓道来，一步步探究其中的规律。  

<h2 align="center">现象观测和基础规律（Section 1-3）</h2> 

论文首先在 $Batch size=2^{19}$ 以及 $Context \space Length=1024$ 的条件下训练了大量Transformer模型，发现交叉熵损失L与模型大小N、数据及大小D、计算量C均呈现幂律关系：  
当D、C足够大时，

$$ L(N)=(N_c/N)^{\alpha_N} \quad \alpha_N \approx 0.076$$

当D有限，N足够大，C训练至收敛前停下，

$$ L(D)=(D_c/D)^{\alpha_D} \quad \alpha_D \approx 0.095$$

而当C有限，D足够大，N最优，
把计算量 $C$ 转化至 $B_{crit}$ 下的最优计算量 $C^{min}$，有规律

$$ L(C^{min})=(C^{min}_c/C^{min})^{\alpha_C} \quad \alpha_C^{min} \approx 0.050$$

![基本幂律](https://i-blog.csdnimg.cn/img_convert/41cde4304e8fab6b9d89e6a13684c227.jpeg)
这些公式是在**其他条件都足够好**的情况下得出的纯经验拟合，它们确立了大模型最基础的规律——加算力、堆参数、加数据，性能一定会提升，且变化趋势是可以预测的。

<h2 align="center">联合缩放与过拟合（Section 4）</h2> 

但是基础规律都只是最理想的情况，在L(N)中如果D不够大，过拟合了是什么规律？在L(D)中万一N不够大，模型欠拟合了又如何？  

### 联合缩放公式(4.1)

基于这种疑惑，论文探究了N和D同时变化时模型的优化结果。L(D,N)的表达式必然满足一些性质，比如当 $D \rightarrow \infty 时应退化为 L(N)$ 而当 $N \rightarrow \infty$ 时应退化为L(D)  

根据大量实验的结果，论文找到了一个表达式：

$$ L(N, D) = \left[ \left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D} \right]^{\alpha_D} $$

不论过拟合还是欠拟合，这个公式和实验数据拟合的很好。当发生过拟合时，N很大而D相对不足，这在公式就体现为 $D_c/D$ 项远大于 $(N_c/N)^{\alpha_N/\alpha_D}$ 项，损失由数据量 D 决定（而非模型容量 N）
### 过拟合的量化(4.2)

作者也不止步于此，而是乘胜追击，进一步定义了一个与过拟合有关的量：

$$\delta L = \frac{L(N,D)}{L(N,\infty)}-1$$

这个公式首次量化了“过拟合”这个概念，当 $\delta L$ 大于随机初始化的噪声0.02时就被认为是过拟合。  
照这个标准计算，为了避免过拟合，数据集规模应当按照如下比例扩大:

$$ D \propto N^{\alpha_N / \alpha_D}=N^{0.74}$$

<h2 align="center">训练动态和时间维度(Section 5)</h2> 

不仅如此，论文还引入训练步数 $S$ 这个标度，来解析训练过程。

OpenAI团队在[MKAT18]研究中总结了这样的规律：当目标的 $Loss$ 固定，步数S和数据量D存在这样的关系：

$$\left(\frac{S}{S_{min}}-1\right)\left(\frac{D}{D_{min}}-1\right)=1$$

其中 $S_{min}，D_{min}$ 即为能够达到$L$的最小步数和数据量，它们都由L决定。  

### 用 $B_{crit}$ 校准计算量和步数(5.1)

论文[MKAT18]中定义当 $S=2S_{min},\quad D=2D_{min}$ 时，训练处于时间和算力的折衷点。此时的batch size就被定义为 $B_{crit}$ ，

$$B_{crit}=\frac{D}{S}=\frac{D_{min}}{S_{min}}$$

实验测试出

$$B_{crit}(L)=\frac{B_*}{L^{1/\alpha_B}} \quad \alpha_B \approx 0.21$$

接着带入 $D=BS$ 就得出：

$$S_{min}(S)=\frac{S}{1+B_{crit}(L)/B}$$

$$D_{min}=\frac{D}{1+B/B_{crit}(L)}$$

$S_{min}(S)$就被定义为实际 $S$ 下的有效步数，这是一把统一的标尺，能为不同实验条件提供统一的度量。

带入 $C=6ND$ 便得到：

$$C_{min}=\frac{C}{1+B/B_{crit}(L)}$$

这个公式通过引入临界批量 $B_{crit}(L)$ ，将实际消耗的计算量 $C$
 校准 为“如果使用最优 Batch Size 理论上需要的最小计算量 $C_{min}$ ” , 从而消除batch size并非最优造成的干扰。  

这为论文中的计算量 $C$ 和步数 $S$ 的研究提供了必要的数学校正工具

### 模型规模N 和 训练进度S 的解耦(5.2)

有了 $S_{min}$ 作为一把统一的标尺，就能研究不同尺度模型的训练过程了。  
校正过后，实验数据就呈现出了清晰的规律。于是作者以迅雷不及掩耳之势给出拟合：

$$L(N,S_{min})=\left(\frac{N_c}{N}\right)^{\alpha_N}+\left(\frac{S_c}{S_{min}}\right)^{\alpha_S}$$

( $\alpha_S \approx 0.076$ )  
这个公式将 $Loss$ 拆解为“模型容量项”和“训练不足”项。它直观地说明了大模型不仅最终 Loss 低，而且更容易训练——因为第一项（基准 Loss）更低，所以达到同样性能目标所需的优化步数 $S_{min}$ 更少。

<h2 align="center">最佳算力分配(Section 6)</h2> 

### 优化问题计算资源分配

老板准备1个月后开发布会，给的资源是100张A100，应该用多少数据训多大的模型效果最好？

基础规律告诉我们，只要砸钱增加规模（不论是N、D还是C），模型性能一定会提升。但没有方向地砸钱必然会造成巨大的效率浪费。

于是论文利用 Section 5 的数学工具将问题形式化为约束优化，严谨求解。核心推导如下：

> **约束条件**（来自 5.1）：
> 
> $$C_{min} = 6N \cdot B_{crit}(L) \cdot S_{min}, \quad B_{crit}(L) = \frac{B_*}{L^{1/\alpha_B}}$$
> 
> 即最优 Batch Size 下的有效计算量 = 参数量 × 临界批量 × 步数。
>
> **目标函数**（来自 5.2 解耦公式）：
> 
> $$L(N, S_{min}) = \underbrace{\left(\frac{N_c}{N}\right)^{\alpha_N}}_{\text{模型容量项}} + \underbrace{\left(\frac{S_c}{S_{min}}\right)^{\alpha_S}}_{\text{训练不足项}}$$
>
> **优化问题**：给定算力预算 $C$，选择 $N$ 和 $S_{min}$ 使 $L$ 最小。

由约束解出 $S_{min} = \dfrac{C}{6N B_{crit}(L)}$，代入目标函数消去 $S_{min}$，得 $L$ 关于 $N$ 的单变量隐函数。求解 $\partial L / \partial N = 0$ 并代入实测指数 $\alpha_N \approx 0.076$ 、  $\alpha_S \approx 0.076$ 、 $\alpha_B \approx 0.21$ ，数值拟合幂律即得最优分配比例。逻辑链为：

$$C \xrightarrow{\text{约束}} S_{min}(N) \xrightarrow{\text{代入}} L(N) \xrightarrow{\frac{\partial L}{\partial N}=0} N_{opt}(C) \xrightarrow{\text{回代}} S_{opt}, D_{opt}$$

最终结论——当给定算力C,

$$ N \propto C^{0.73},\quad S \propto C^{0.03},\quad D \propto C^{0.27} $$

这是全文最震撼的结论，直接指导了后世所有大模型（如 GPT-3, GPT-4）的训练策略：

* 当算力 $C$ 增加 10 倍时，**模型参数 $N$ 应增加约 5.4 倍**（ $10^{0.73} \approx 5.4$ ）。  
* 训练步数 $S$ 几乎不需要增加（仅增加约 $10^{0.03} \approx 1.07$ 倍）。  
* 数据量 $D$ 只需适当增加。

**核心启示**：**Big Models are Better**。与其把算力花在训练小模型更长时间上，不如把算力花在训练更大的模型上。大模型训练得更少一点（不要求完全收敛），反而更划算。

### 来自 Chinchilla Scaling Law 的挑战

 **Chinchilla 的修正（2022, Hoffmann et al.）**  
 Kaplan 的上述结论在后来的 Chinchilla 论文中受到挑战。其 Section 3.1 的 **Approach 2** 采用更直接的约束形式重新求解同一优化问题：
 $$C \approx 6ND$$
 与 Kaplan 的核心区别在于：Chinchilla 不引入 $B_{crit}(L)$ 校准，而是直接固定总计算量 $C$，令 $D = C/(6N)$，枚举不同 $N$ 和对应 $D$ 的组合，实测 Loss 寻找最优。
 
 这一简化约束下的最优分配比例与 Kaplan 截然不同：
 $$N \propto C^{0.50}, \quad D \propto C^{0.50}$$
 即**模型大小与数据量应当等比放大**。这推翻了 Kaplan "侧重堆参数" 的结论，强调数据同样重要——Chinchilla（70B 参数，1.4T tokens）能以远小于 Gopher（280B 参数，300B tokens）的算力消耗实现更优性能。
 
 两篇论文的约束本质相同（$C=6ND$），差异在于是否用 $B_{crit}$ 校准：Kaplan 在 $B_{crit}$ 框架下允许 $S$ 弹性伸缩，得出 $N$ 应增长更快的结论；Chinchilla 在固定训练设置下直接测量，发现 $N$ 和 $D$ 应等比例增长。

### Llama 关于计算效率的反思

假设遵循计算效率最优来研发LLM，那么根据Scaling Law，给定模型大小，可以推算出最优的计算量，进一步根据最优计算量就能推算出需要的token数量，然后训练就行。 机器学习与人工智能

**但是计算效率最优这个观点是针对训练阶段而言的，并不是推理阶段，实际应用中推理阶段效率更实用。**

Meta在LLaMA[8]的观点是：给定模型的目标性能，并不需要用最优的计算效率在最快时间训练好模型，而应该在更大规模的 数据上，训练一个相对更小模型，这样的模型在推理阶段的成本更低，尽管训练阶段的效率不是最优的（同样的算力其实能获得更优的模型，但是模型尺寸也会更大）。根据Scaling Law，10B模型只需要200B的数据，但是作者发现7B的模型性能在1T的数据后还能继续提升。

具体而言，确定模型尺寸后，Scaling Law给到的只是最优的数据量，或者说是一个至少的数据量，实际在训练中观察在各个指标上的性能表现，只要还在继续增长，就可以持续增加训练数据。

![LLaaMA best training figure](https://i-blog.csdnimg.cn/img_convert/a54fec4412482a9a3c0d7ce86169f298.jpeg)

<h1 align="center">LLM究竟能不能解冻？</h1>

<p align="center">部分内容参考deepseek-v4-pro T=0.3</p>

<h2 align="center"> BLIP-2 的迷惑设计 </h2>

BLIP-2论文里有一些惑操作，例如Figure 2中 self-attention 看起来是分开的，实际是连在一起的。

<img src="https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/LLM%E8%83%BD%E5%90%A6%E8%A7%A3%E5%86%BB%E6%8F%92%E5%9B%BE/BLIP-2%E9%97%AE%E9%A2%98%E6%8F%92%E5%9B%BE%E5%AF%B9%E6%AF%94.png" alt="第一阶段图" width="700" load="lazy">

### Q-Former 突然不给文本了？

在训练架构上，为了支撑Q-Former **轻量化、浅注入的特性**，BLIP-2必须添加第一阶段VL-Representation Learning 来补偿。  

但别扭的地方来了，到了第二阶段训练时Generative Learning，BLIP就不在Q-Former端输入文本了，而是直接从LLM的底部送入。

Q-Former在 Stage 1 学好了图文互信息最大化，Stage 2突然不给文本了，这样Self Attention 的性能必然会退化。

<img src="https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/LLM%E8%83%BD%E5%90%A6%E8%A7%A3%E5%86%BB%E6%8F%92%E5%9B%BE/BLIP%20-2%E9%98%B6%E6%AE%B5%E4%BA%8C.png" alt="第二阶段训练图" width="1000" load="lazy">

### 计算量的限制  

这本质上是 **信息提取效率**的问题：

如果让文本提示从Q-Former进入，Q-Former就能提取与这段文字相关的信息；而如果从LLM输入，Q-Former只能提取宽泛的和语言有关的信息。从图像信息提取的效率上来讲，不应该从Q-Former输入文本更高效吗？

在测试中，作者确实考虑到了这个问题，因此在4.3的问答任务重Q-Former和LLM的底部都输入了文本：

> In order to extract image features that are more relevant to
the question, we additionally condition Q-Former on the
question.

> Specifically, the question tokens are given as
input to the Q-Former and interact with the queries via the
self-attention layers (4.3开头)

但在**训练时**，作者还必须考虑到计算量的问题。

如果Stage 2 训练时文本也从Q-Former进入，那么在生成推理时，**每生成一个 token，整个 Q-Former 必须重新跑一遍**，因为文本序列变了，自注意力和交叉注意力的输出全变了。这意味着对于一个 50 token 的回答，你要跑 50 次 Q-Former。

所以即便从"信息提取效率"的角度上合理——文本条件化的视觉提取理论上更精准，但自回归生成的循环依赖导致**训练计算量过大**，所以训练只能暂时错位。

值得一提的是,BLIP团队的后续工作 **InstructBLIP** 就把所有推理任务文本提示都变为两次输入，这究竟是复盘反思的结果，还是传说中的模块化研究？

<img src="https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/LLM%E8%83%BD%E5%90%A6%E8%A7%A3%E5%86%BB%E6%8F%92%E5%9B%BE/BLIP-2%E6%8F%92%E5%9B%BE3.jpg" alt="毕导图（差异化）" width="400" load="lazy">

<h2 align="center"> LLM 究竟能不能解冻？ </h2>

### BLIP-2的设计逻辑和LLM的解冻困境

BLIP最反直觉的地方，也就是它的核心贡献——它没有像Flamingo那样桥接的门控交叉注意力模块像三明治一样一层一层插进LLM里，而是直接接在了LLM底部。

**但从直觉上看，Q-Former输出的视觉表征这并不是LLM可以直接理解的东西。**

- 如果LLM被完全冻结，那么它本质上是在理解自然语言序列，直接送入视觉表征不是强人所难吗？
- 但如果LLM被解冻，那就更是违背原则——Q-Former就是为了减少训练参数，这一解冻就由多回去了。

虽然这听起来确实像是"逼一个只懂中文的人去读摩尔斯电码"，但BLIP-2的成功告诉我们，**LLM仍然可以学会，只不过学习过程会更加艰难，需要精细的训练设计**（Stage 1辅助）。

原因在于LLM是在连续的词嵌入空间上处理，**Q-Former给出的视觉表征，和词嵌入向量在数学上没有本质区别**。这个高维空间中可以表示大多数自然语言的意义，如果**表示视觉意义的向量**能被低损映射到**自然语义空间**中，LLM的理解就能正常进行。

<img src="https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/LLM%E8%83%BD%E5%90%A6%E8%A7%A3%E5%86%BB%E6%8F%92%E5%9B%BE/BLIP-2%E6%8F%92%E5%9B%BE4.jpg" alt="3b1b图" width="500" load="lazy">

关于 Prefix Tuning 的研究也得出同样的结果：你不需要自然语言指令，只需在 LLM 输入前加几个完全抽象的、可学习的嵌入向量，LLM 就能按你的意图完成任务。

但问题在于，视觉信息和语言信息并非同构，**视觉意义很难无损映射成自然语义**，映射时一些空间结构和图像细节信息容易丢失。

这是LLM本身固有的局限。那么一个想法随之而来，能不能把LLM解冻，让LLM也能处理视觉语义？

### 结论：不仅可行，而且是主流做法

为了解决自然语义空间的局限，后续的有许许多多的工作都围绕这一点展开，也分成了很多不同的做法：

- 有在Q-Former前也加入文本提示的**InstructBLIP**，
- 也有把Stage 2换成高质量微调投影层的**GPT-4 Mini**，
- 还有类似Flamingo的视觉专家分支**CogVLM**，

它们无一例外都选择了微调LLM，只是微调程度不同

| 解冻程度 | 代表性工作 | 效果 |
|---------|-----------|------|
| **全冻结** | BLIP-2 Stage 2 | LLM 只是一个推理引擎，不参与视觉理解<br>——效果好但天花板低 |
| **LoRA 微调**或 <br> **Adapter插入**  | LLaVA, LLaMA-Adapter | 仅训练低秩适配器（参数量极小），LLM 的原始权重不动 |
| **部分层解冻** | LLaVA（某些版本） | 解冻最后几层，让 LLM 学会"读"视觉 token |
| **全量微调** | LLaVA-1.5 full fine-tune,<br> CogVLM, Qwen-VL | 解冻整个 LLM，上限最高 |

- 非常值得一提的是 **LoRA（Low-Rank Adaptation）** 优雅的解决方案：与其解冻整个层进行全参数微调，不如给每一层（包括早期和后期层）并行插入**低秩适配矩阵**：

  ```text
  原始：  h = W·x          （W 冻结）
  LoRA：  h = W·x + B·A·x  （B、A 低秩，可训练）
  ```

  这是是最流行的做法，效果非常接近全量微调，LLaMA-Adapter、LLaVA（LoRA 版本）、mPLUG-Owl 等都验证了 LoRA 的可行性。

BLIP-2之前考虑灾难性遗忘、解冻LLM会让训练参数增加，但在精良的数据集和合适的策略下，灾难性遗忘是可以被解决的，训练的步数也不多，实际计算量的增加并不是很恐怖。随着硬件的升级，微调LLM的性能增益还是胜过了训练上的困难。

### VLM 处理LLM的趋势总结

从BLIP-2到LLaVA、InstructBLIP、CogVLM 的整个演进逻辑。核心趋势是：

> **减少中间模块（Q-Former），增加 LLM 参与度（部分解冻），用更强的训练数据和策略弥补架构简化带来的损失。**

BLIP-2 的 Q-Former 是一个精巧的过渡方案——它在当时解决了"如何把视觉信息喂给冻结 LLM"的问题。但后续工作很快发现：与其费心翻译，不如让 LLM 自己学这门"外语"。LLM 的学习能力远超我们为它设计的任何适配器。

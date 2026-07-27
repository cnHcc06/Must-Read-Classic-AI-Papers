# 训练如何把CoT内化，让“不写特殊Prompt”也管用

在论文 Discussion 里，作者洋洋得意地说到：
> In all experiments, chain-of-thought reasoning is elicited simply by prompting an off-the-shelf language model. No language models were finetuned in the process of writing this paper.

在Prompt 里写思维链，你训练是方便了，刷分也上去了，但却苦了API用户：每遇到一个问题都要写数个CoT，体验感直线下降。可以说这篇论文中的CoT完全没有实用价值。

但奇怪的是，自从2023年LLM逐渐开始兴起，我就从来没有在输入中写过CoT。即使有一段时间网传要提示词工程，现在也不见踪影。

模厂似乎把推理的提示藏起来了，压缩到了用户看不到的地方。这是怎么做到的呢？

## 为什么现在不用写特殊Prompt，模型也能展现相当推理能力？

**简短结论：**  

现代主流“聊天/推理型”大模型在训练阶段就已经被大规模喂过“分步推理/思维链（CoT）”的数据与奖励信号，并在RL/PPO/DPO等对齐训练里被教会“挑出更好的推理路径”。  
因此，哪怕你在Prompt里不显式要求“一步步思考”，模型也更倾向于自发输出结构化推理、先想再答。  
这相当于厂商把CoT“内化”到了模型里；同时，研究也发现CoT式路径本身就常分布在模型的备择token里，解码方式不同会影响你看到的是不是显式的“分步文本”。
下面分点展开。

---

## 一、训练成本其实可控

> For rationale-augmented training and
finetuning methods, it is costly to create a large set of high quality rationales, which is much more
complicated than simple input–output pairs used in normal machine learning.

论文 Introduction 中提到手写CoT训练集开销过大。但其实把思路打开，这个问题并不存在。

对于数学等成熟学科，论文、教科书、ArXiv中都有大量的推理过程，根本不愁没有数据，微调成本其实可控。

### 训练端就把CoT“喂”了进去：指令微调与SFT包含大量分步示例

- 厂商会在指令微调（SFT）阶段加入大量“问题 + 推理过程 + 答案”的样本，让模型学会“输出带过程的回复”。例如：
  - Qwen2.5-Math在预训练与后训练中都系统加入并合成大量数学推理数据；官方明确其支持以Chain-of-Thought和Tool-Integrated Reasoning解决中英数学题，并在后训练中引入CoT/TIR数据与SFT数据、配合RM与GRPO做强化。
- 这类训练会让模型把“分步解题/分步解释”当作一种默认的响应风格，尤其对STEM/推理类任务。因此，你不写“逐步解释”，模型也更可能自发给出步骤。

### RL/偏好对齐把“先想再答”固化为策略

- OpenAI的o1系列明确是“用大规模RL训练来使用链式思维”的推理模型，会“先想再答——可以在给用户最终回答之前产生长链式思维”。训练让模型学会打磨思维过程、尝试不同策略并识别错误。
- Meta Llama 3官方博客也提到：通过偏好排序训练（PPO/DPO），模型在推理与编码任务上表现明显提升，并且“模型知道如何产出正确答案，但不知道如何选择；训练偏好排序使其学会选出来”。这说明对齐训练会鼓励模型选择更合理的推理路径并在最终输出中体现。

这里给出的仍然是RLHF论文（Instruct GPT）里的那一套组合拳：先SFT扎根，再RL放大。事实证明，如果是追求更好的数学能力，“大力出奇迹”的微调—也是一条必经之路。

---

## 二、系统提示/助手设定在后台扮演“隐形CoT提示”

- 多数产品化模型在服务端都挂有系统提示（system prompt），类似“当遇到推理/数学问题时，请分步解释、检查中间结果、避免直接给最终答案”这类要求。你看上去是“不写特殊Prompt”，但厂商已经替你加了默认的推理引导指令（具体内容一般不公开）。
- GPT‑4系统卡也提到模型在“可操控性”上更强，能更好推断用户意图，减少对手工Prompt工程的依赖。这背后与系统提示和对齐训练密不可分。

---

## 三、更大、更强的基座本身就“能推理”

论文的核心结论之一
> we find that chain-of-thought reasoning is an emergent property of model scale that allows sufficiently large language models to perform reasoning tasks that otherwise have flat scaling curves (8 Conclusion)

“只有模型规模大到一定程度，CoT Prompting 才有效”。这间接说明 Scaling Up 让模型学到了如何推理，而其他大模型的报告也指出了相似的结论：

- Qwen2.5技术报告将预训练数据量从7T tokens扩展到18T，明确指出这为常识、专家知识和推理能力打下强基础。更大的基座+高质量数据本身就会让推理变得“自然”，即便没有显式CoT提示，复杂问题的准确率也会更高。
- Meta Llama 3也在官方博客中强调，相比Llama 2，预训练与后训练的改进带来了“推理、代码生成、指令遵循”等能力的显著提升，可见能力来源不只是Prompt，而是训练与规模的综合效果。

---

## 四、CoT路径本来就“藏在”模型里：解码方式会改变你能否看到它

其实论文3.1的实验细节里也提了一嘴

> We sample from the models via greedy decoding (though follow-up work shows
chain-of-thought prompting can be improved by taking the majority final answer over many sampled generations (Wang et al., 2022a)).

- 近期研究发现，即使不写CoT提示，仅仅改变解码策略，也能从预训练模型中“掏出”隐式的CoT推理路径（查看Top‑k备择token），说明模型内部在推理时就可能候选了“分步”的续写，只是贪婪解码未必选到显式的CoT形式。
- 换言之，现代模型在回答推理问题时，即使你不说“一步步想”，模型也常会自发展开分步表述；训练只是把这种倾向变得更稳定、更可触发。

---

## 总结：为什么“不写特殊Prompt”也能推得很像模像样？

- 训练层面：SFT阶段被喂了大量“带步骤的数据”；RL/偏好对齐把“先想再答、选更好路径”写进了策略；系统提示在后台做“隐形CoT提示”。
- 模型层面：更大基座、高质量数据让推理更自然；研究发现CoT路径本就常分布在候选token里，解码方式决定你看得见几分。  

所以，你看到的“不写CoT也自动推理”，本质上是厂商把当年的Prompt工程“固化”到了训练和系统提示里，模型更默认以“分步推理”的方式响应。这并不意味着CoT不再重要——它只是从“你在Prompt里写”，变成了“厂商提前在训练与配置里帮你写好了”。

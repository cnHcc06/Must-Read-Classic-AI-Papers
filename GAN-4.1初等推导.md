<h1 align="center">《GAN》 Section 4 定理初等推导及解析</h1>

<p align="center">部分内容由GLM 5.2思考及Qwen 3.7 Max思考生成，并由deepseek v4 pro审核</p>

GAN论文架构简单，实验配置很常规；14年那会图像生成好像很缺benchmark，GAN论文连定量分析都没做，也没什么好说的。作者为了提高工作量和专业性，只能抄起数学武器证明这个方法的合理性，这一定程度上弥补了实验的不足。

<h2 align="center">初等方法推导4.1 $p_g = p_{data}$</h2> 

看着Theorem 1 大大的黑箱，不禁要问：没学过 **Kullback-Leibler divergence** 和 **Jensen-Shannon divergence** 怎么办？🤔

其实Theorem 1只用 **代数变形** 和 **基本对数性质** 就能推导。

我们要证明的目标是：当且仅当 $p_g=p_{data}$ 时，C(G) 取得最小值。

### 第一步：把最优判别器 $D^{\ast}$ 代入 $V (D,G)$

根据论文前面的推导，对于固定的生成器 $G$，最优的判别器 $D^{\ast} (x)$ 是：

$$D^{\ast} (x) = \frac{p_{data} (x)}{p_{data} (x) + p_g (x)}$$

带入 $V (D, G)$，我们得到(4)：

$$C (G) =\mathbb{E}_{x \sim p_{data}} \left[ \log \frac{p_{data}
   (x)}{p_{data} (x) + p_g (x)} \right] +\mathbb{E}_{x \sim p_g} \left[ \log
   \frac{p_g (x)}{p_{data} (x) + p_g (x)} \right] $$

### 第二步：简化+变形

把 $p_g (x)$ 和 $p_{data} (x)$ 视为常数，令 
$y = p_g (x), z = p_{data} (x)$ ，实际上 $p_g(x)$ 是优化变量。

$$
  C (G) = \int_x p_{data} (x) \log \left(
  \frac{p_{data} (x)}{p_{data} (x) + p_g (x)} \right) + p_g (x) \log \left(
  \frac{p_g (x)}{p_{data} (x) + p_g (x)} \right) d x\\
  = \int_x z \log \left( \frac{z}{y + z} \right) + y \log \left(
  \frac{y}{y + z} \right) d x
$$

其中 $\int_x z{dx}= \int_x y{dx} =1$

直接看这个式子很难看出最小值在哪。论文里用了一个非常巧妙的技巧：  
我们在方括号里加上 $\log 2$ 再减去 $\log 2$ ，式子值不变，但形式会变得很有趣：

$$C (G) = \int_x z \log \left( \frac{z}{y + z} \cdot 2 \cdot \frac{1}{2}
   \right) + y \log \left( \frac{y}{y + z} \cdot 2 \cdot \frac{1}{2} \right)
   dx$$

$$C (G) = \log \frac{1}{2} + \log \frac{1}{2} + \text{} \int_x z \log \left(
   \frac{2 z}{y + z} \right) + y \log \left( \frac{2 y}{y + z} \right) d x$$

### 第三步：利用一个基本不等式

现在，我们只需要判断"剩下的部分"到底是多少。
我们要用到数学分析中一个最基础的不等式： $\ln x \geq  \left( 1 - \frac{1}{x} \right)$ （当且仅当 $x = 1$ 时取等号）。(这里log就表示自然对数)于是：

$$z \log \left( \frac{2 z}{y + z} \right) \geq z \cdot  \left( 1 - \frac{y + z}{2 z} \right) = \frac{z - y}{2}$$

$$同理y \log \left( \frac{2 y}{y + z} \right) \geq  \frac{y - z}{2}$$

把不等式代入积分抵消： $$C (G) \geq - \log 4 + 0 = - \log 4 $$
当且仅当 $\frac{2 z}{y + z} = \frac{2 y}{y + z} = 1，即p_g (x) =
p_{data} (x) 等号成立$

通过代数变形，我们将 $C (G)$ 拆成了"常数项 $- \log
4$"加上"非负项"。因为"非负项"只有在 $p_g = p_{data}$ 时才为
0，所以这就证明了 $C (G)$ 的最小值是 $- \log
4$，且只在 $p_g = p_{data}$ 时取到。

---

<h2 align="center">逐句解析4.2证明 算法的可行性</h2> 

Proposition 2 的证明非常简练，用到了一些凸优化的数学性质。如果没学过凸分析，这段话确实很难懂。它核心就在说明一件事：
> **Algorithm 1 (交替训练D和G的算法)，等价于在给 $C(G)=sup_D U(p_g,D)$ 做梯度下降。**

### 📌 先回顾前提条件

Proposition 2 有几个前提假设：
1. **G 和 D 有足够的容量**（即非参数极限，可以表示任意分布/函数）
2. **每步训练中，D 都能被训练到当前 G 下的最优**（即 $D_G^*$）
3. **$p_g$ 的更新方向是改善目标函数**（即沿梯度下降方向更新）

在这些条件下，结论是：**$p_g$ 会收敛到 $p_{data}$**。

### 第一步：把 $V(G, D)$ 看成关于 $p_g$ 的函数

> $$V(G, D) = U(p_g, D)$$

这里的关键视角转换是：不再把 $V$ 看成关于网络参数的函数，而是直接看成关于**生成分布 $p_g$** 的函数。

> $U(p_g, D)$ 对于 $p_g$ 是**凸的（convex）**。  

这是因为：

$$U(p_g, D) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D(x))]$$

其中第一项与 $p_g$ 无关，第二项是关于 $p_g$ 的**线性函数**（期望本质上是积分 $\int p_g(x) \log(1-D(x)) dx$），而线性函数自然是凸的。

---

### 第二步：关键数学定理——"凸函数族的上确界的次梯度"（Danskin定理）

论文中提到的这段话：

> “The subderivatives of a supremum of convex functions include the derivative of the function at the point where the maximum is attained.”

翻译成白话：要优化一个“一堆凸函数取最大值”得到的函数，你可以直接用“当前最大的那个凸函数”的梯度。

> 换言之，如果 $f(x) = \sup_{\alpha \in A} f_\alpha(x)$，且每个 $f_\alpha(x)$ 都对 $x$ 是凸的，那么：
> **在取得上确界的那个 $\beta = \arg\sup_\alpha f_\alpha(x)$ 处， $f_\beta(x)$ 的次梯度（subderivative）属于 $f(x)$ 的次梯度集合。**
即： $\partial f_\beta(x) \in \partial f(x)$

**直觉理解：** 想象你有很多条凸曲线 $f_\alpha(x)$，你取它们在每个点的最大值形成一条新的曲线 $f(x)$。在某个点 $x$ 处，如果 $f_\beta$ 是"最高的那条曲线"，那么 $f$ 在这个点的"斜率"就包含了 $f_\beta$ 的斜率。

---

### 第三步：将这个定理应用到 GAN 中

在我们的问题中：

| 定理中的符号 | GAN 中对应 |
|---|---|
| $x$ | $p_g$（生成分布） |
| $\alpha$ | $D$（判别器） |
| $f_\alpha(x)$ | $U(p_g, D)$（对于固定 D，关于 $p_g$ 的函数） |
| $f(x) = \sup_D U(p_g, D)$ | $C(G) = \max_D V(G, D)$（虚拟训练准则） |

根据定理：
- 我们在 Algorithm 1 中，**先把 D 训练到最优** $D_G^*$（即找到使 $U(p_g, D)$ 最大的那个 $D$）
- 然后**用这个最优 $D_G^*$ 下的梯度来更新 $p_g$**
- 根据定理，这个梯度就是 $C(G) = \sup_D U(p_g, D)$ 的（次）梯度！

**这意味着：** 
>Algorithm 1 的交替更新过程，本质上等价于对 $C(G) = \sup_D U(p_g, D)$ 做梯度下降！

---

### 第四步：结合第一步得出结论

第一步中提到：$U(p_g, D)$ 对于 $p_g$ 是凸的  
由于 **一族凸函数的上确界仍是凸函数** ，因此
> - $C(G) = \sup_D U(p_g, D)$ 也是关于 $p_g$ 是**凸函数**


> - 它有**唯一的全局最小值**，在 $p_g = p_{data}$ 处取得（此时 $C(G) = -\log 4$）

由于：
- $C(G)$ 是凸的且有唯一全局最优
- Algorithm 1 等价于对 $C(G)$ 做梯度下降
- 只要步长足够小（sufficiently small updates）

> **因此 $p_g$ 必然收敛到 $p_{data}$。** ✅

---

## 🎯 总结：证明的逻辑链

```
Algorithm 1 的交替训练：先训 D 到最优 → 再用该 D 的梯度更新 pg
       │
       ▼
利用"凸函数族上确界的次梯度"定理
       │
       ▼
等价于对 C(G) = sup_D U(pg, D) 做梯度下降
       │
       ▼
由 U(pg, D)凸得到C(G)=sup_D U(pg, D)凸，
且唯一最小值在 pg = pdata
       │
       ▼
步长足够小 → pg 收敛到 pdata  ∎
```

---

## ⚠️ 补充说明

证明最后也指出了一个**现实差距**：在实际中，G 是用多层感知机（MLP）参数化的，我们优化的是参数 $\theta_g$ 而不是直接优化分布 $p_g$ 本身。MLP 引入了**多个临界点（critical points）**，所以理论上不能保证收敛到全局最优。但作者指出，MLP 在实践中表现很好，是一个合理的选择。

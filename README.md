<div align="center">
    <h1> 🚩Must Read List of Classic AI Papers </h1>
</div>

<p align="center">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License"></a>
    <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs welcome"></a>
</p>

## Introduction

Welcome to the **AI Classic Paper** Learning repository! This repo compiles some of the **most influential classic papers in the history of AI development**.

This list is arranged along the main learning thread, covering the complete evolution path from *the fundamentals of deep learning* to *large models*, *generative AI*, and *embodied intelligence*.

>Learrning thread  
>CNN → Transformer → LLM → Diffusion → Multimodal → Agent → Physical AI

## 📖Overview

- [🏗️ 深度学习基础 (Deep Learning Backbone)](#Backbone-id)
- [🧠 大语言模型 (Large Language Models - LLM)](#LLM)
- [🎨 生成式 AI (Generative AI - AIGC)](#AIGC)
- [🔗 多模态 AI (Multimodal AI)](#Multimodel)
- [🤖 AI 智能体 (AI Agent)](#ai-agent)
- [🦾 具身智能 (Physical AI)](#physical-id)

<a id="Backbone-id"></a>

## 1. 🏗️ 深度学习基础 (Deep Learning Backbone)

这些论文奠定了现代深度学习、序列建模和计算机视觉的基础。

| # | 论文名称 | 年份 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| 1 | [**Long Short-Term Memory (LSTM)**](https://ieeexplore.ieee.org/abstract/document/6795963) | 1997 | 解决 RNN 长期依赖问题 |
| 2 | [**ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)**](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) <br><details><summary>Extend Reading</summary> [AlexNet为什么能成功？](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/AlexNet%E4%B8%BA%E4%BB%80%E4%B9%88%E8%83%BD%E6%88%90%E5%8A%9F%EF%BC%9F.md)</details>  | 2012 | 深度学习复兴 |
| 3 | [**Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG)**](https://arxiv.org/abs/1409.1556) | 2014 | 深层 CNN 设计思想 |
| 4 | [**Deep Residual Learning for Image Recognition (ResNet)**](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) <br><details><summary>Extend Reading</summary> [CNN技术迭代史](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/CNN%E5%8F%91%E5%B1%95%E5%8E%86%E7%A8%8B.md)</details> | 2015 | Residual Connection, 现代CV基石 |
| 5 | [**Attention Is All You Need**](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) <br><details><summary>Extend Reading</summary> <ol><li>[初学attention最困惑的问题](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/%E5%88%9D%E5%AD%A6attention%E6%9C%80%E5%9B%B0%E6%83%91%E7%9A%84%E9%97%AE%E9%A2%98.md) </li> <li> [审稿人意见](https://papers.nips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Reviews.html)</li></ol> </details> | 2017 | Transformer 诞生，现代 AI 最重要论文之一 |

<a id="LLM"></a>

## 2. 🧠 大语言模型 (Large Language Models - LLM)

开启预训练、大语言模型与推理时代。

| # | 论文名称 | 年份 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| 6 | [**BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**](https://aclanthology.org/N19-1423/) <br><details><summary>Extend Reading</summary> <ol><li>[预训练-微调鸿沟](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/%E9%A2%84%E8%AE%AD%E7%BB%83-%E5%BE%AE%E8%B0%83%E9%B8%BF%E6%B2%9F.md)</li>  </ol> </details> | 2018 | Encoder 预训练模型代表 |
| 7 | [**Language Models are Few-Shot Learners (GPT-3)**](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html?utm_source=transaction&utm_medium=email&utm_campaign=linkedin_newsletter) <br><details><summary>Extend Reading</summary> <ol><li> wait for adding </li>  </ol> </details> | 2020 | Few-shot Learning，大模型 Scaling 时代 |
| 8 | **Scaling Laws for Neural Language Models** | 2020 | 提出 Scaling Law，指导大模型训练 |
| 9 | **Training Language Models to Follow Instructions with Human Feedback (InstructGPT)** | 2022 | RLHF，让 LLM 真正可用 |
| 10 | **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** | 2022 | CoT 推理能力，引爆 Reasoning 研究 |

<a id="AIGC"></a>

## 3. 🎨 生成式 AI (Generative AI - AIGC)

现代生成模型的发展路线。

| # | 论文名称 | 年份 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| 11 | **Generative Adversarial Nets (GAN)** | 2014 | GAN 开创生成模型新时代 |
| 12 | **Denoising Diffusion Probabilistic Models (DDPM)** | 2020 | Diffusion 模型奠基论文 |
| 13 | **High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)** | 2022 | Latent Diffusion，大规模开源文生图基础 |

<a id="Multimodel"></a>

## 4. 🔗 多模态 AI (Multimodal AI)

视觉、语言、多模态融合的发展。

| # | 论文名称 | 年份 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| 14 | **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** | 2021 | Vision-Language 对齐基础 |
| 15 | **Flamingo: a Visual Language Model for Few-Shot Learning** | 2022 | Few-shot 多模态模型代表 |
| 16 | **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** | 2023 | Q-Former 架构经典 |
| 17 | **Visual Instruction Tuning (LLaVA)** | 2023 | 开源 Vision-Language Model 代表 |

<a id="ai-agent"></a>

## 5. 🤖 AI 智能体 (AI Agent)

Agent 时代的重要起点。

| # | 论文名称 | 年份 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| 18 | **ReAct: Synergizing Reasoning and Acting in Language Models** | 2022 | Reasoning + Tool Use，Agent 开山之作 |
| 19 | **Toolformer: Language Models Can Teach Themselves to Use Tools** | 2023 | Tool Calling 思想来源 |

<a id="physical-id"></a>

## 6. 🦾 具身智能 (Physical AI)

机器人与具身智能。

| # | 论文名称 | 年份 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| 20 | **RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control** | 2023 | Vision-Language-Action (VLA) 代表作 |

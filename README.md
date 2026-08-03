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
| 7 | [**Scaling Laws for Neural Language Models**](https://arxiv.org/pdf/2001.08361/1000)<br><details><summary>Extend Reading</summary> [论文梳理以及不同流派的挑战](http://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/Scaling%20Law%20%E7%AE%80%E6%98%8E%E6%A2%B3%E7%90%86.md) | 2020 | 提出 Scaling Law，指导大模型训练 |
| 8 | [**Language Models are Few-Shot Learners (GPT-3)**](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html?utm_source=transaction&utm_medium=email&utm_campaign=linkedin_newsletter) <br><details><summary>Extend Reading</summary> <ol><li> [制作dataset：去重还是过采样？](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/%E5%88%B6%E4%BD%9C%E8%AF%AD%E8%A8%80dataset%EF%BC%9A%E5%8E%BB%E9%87%8D%E8%BF%98%E6%98%AF%E8%BF%87%E9%87%87%E6%A0%B7%EF%BC%9F.md) </li> <li> [辨析：任务微调和指令微调](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/%E8%BE%A8%E6%9E%90%EF%BC%9A%E4%BB%BB%E5%8A%A1%E5%BE%AE%E8%B0%83%E5%92%8C%E6%8C%87%E4%BB%A4%E5%BE%AE%E8%B0%83.md)</li></ol> </details> | 2020 | Few-shot Learning，大模型 Scaling 时代 |
| 9 | [**Training Language Models to Follow Instructions with Human Feedback (InstructGPT)**](https://proceedings.neurips.cc/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract.html) <br><details><summary>Extend Reading</summary> <ol><li> [从人工标注到AI杂交](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/From%20RLHF%20to%20RLAIF.md)</li> <li>[现代大模型如何“减毒”？](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/%E7%8E%B0%E4%BB%A3%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%A6%82%E4%BD%95%E5%87%8F%E6%AF%92.md)</li> </ol> | 2022 | RLHF，让 LLM 真正可用 |
| 10 | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html) <br><details><summary>Extend Reading</summary> [为什么我从来没写过CoT Prompting？](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/%E4%B8%8D%E5%86%99CoT%20Prompting%EF%BC%8C%E5%A6%82%E4%BD%95%E6%8E%A8%E7%90%86%EF%BC%9F.md) | 2022 | CoT 推理能力，引爆 Reasoning 研究 |

<a id="AIGC"></a>

## 3. 🎨 生成式 AI (Generative AI - AIGC)

现代生成模型的发展路线。

| # | 论文名称 | 年份 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| 11 | [**Generative Adversarial Nets (GAN)**](https://arxiv.org/pdf/1406.02661) <br><details><summary>Extend Reading</summary> [Section 4 初等推导解说](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/GAN-4.1%E5%88%9D%E7%AD%89%E6%8E%A8%E5%AF%BC.md) | 2014 | GAN 开创生成模型新时代 |
| 12 | [**Denoising Diffusion Probabilistic Models (DDPM)**](https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html) <br><details><summary>Extend Reading</summary> [[视频]这些公式是什么意思？《DDPM》论文科普](https://www.bilibili.com/video/BV1fS7nznEUf/?spm_id_from=333.337.search-card.all.click&vd_source=fc3d9baef95f4f9022b0c3a1dc357933)  | 2020 | Diffusion 模型奠基论文 |
| 13 | [**High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)**](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html?utm_source=rns.dwaiai.de) <br><details><summary>Extend Reading</summary> [GAN的顽疾——模式坍塌](https://github.com/cnHcc06/Must-Read-Classic-AI-Papers/blob/main/GAN%E7%9A%84%E9%A1%BD%E7%96%BE--mode%20collapse.md) | 2022 | Latent Diffusion，大规模开源文生图基础 |

<a id="Multimodel"></a>

## 4. 🔗 多模态 AI (Multimodal AI)

视觉、语言、多模态融合的发展。

| # | 论文名称 | 年份 | 核心贡献 |
| :--- | :--- | :--- | :--- |
| 14 | [**Learning Transferable Visual Models From Natural Language Supervision (CLIP)**](https://proceedings.mlr.press/v139/radford21a) | 2021 | Vision-Language 对齐基础 |
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

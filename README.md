# 重庆大学大数据架构与技术课程

本仓库包含重庆大学2024年秋大数据架构与技术课程的实验资源，基于学校提供的[原开源项目](https://github.com/Wanghui-Huang/CQU_bigdata)进行了优化和扩展，包含**最新的工具版本**、**简化的环境配置流程**、以及对**更便捷高效的开发工具**（如 Jupyter Notebook）的全面支持。

---

## 快速导航

- [重庆大学大数据架构与技术课程](#重庆大学大数据架构与技术课程)
  - [快速导航](#快速导航)
  - [优化与扩展概要](#优化与扩展概要)
  - [资料分布说明](#资料分布说明)
    - [实验一：搭建分布式环境](#实验一搭建分布式环境)
    - [实验二：词频统计](#实验二词频统计)
    - [实验三：手写数字识别](#实验三手写数字识别)

---

## 优化与扩展概要

> 在本人最初跟着原有开源教程进行实验时，发现原教程因时间久远而普遍出现**链接失效**，**版本老旧**和**库包弃用**等问题，同时步骤方面也有一些可供改善与优化的地方，故建此仓库解决上述问题。

1. **最新的工具版本支持**
   - 替换原项目中的过时工具，使用最新版本的 Ubuntu (22.04)、Hadoop (3.3.6)、Spark (3.5.1)等，避免兼容性问题。
   - 适配新的开发环境，推荐使用 Ubuntu 22.04 LTS (代替 CentOS)和 OpenJDK 1.8。

2. **简化环境配置**
   - 专门提供更详细的环境配置指南，从学生视角出发帮助同学们更快更好搭建可运行环境。推荐配合原项目操作指南使用。
   - 全程命令行操作，避免使用图形界面引发性能问题。如果对命令行操作不习惯，推荐使用辅助工具：Xshell 7 与 Xftp 7。

3. **引入更便捷高效的开发工具**
   - 全面采用 **Anaconda** 和 **Jupyter Notebook**，简化开发流程，优化学习体验。
   - 支持交互式编程、数据可视化、Markdown 文档记录等功能。

4. **逐步优化的实验流程**
   - 每个实验均提供对应的详细说明及其相关代码，覆盖从环境搭建到具体应用的完整流程。
   - 最终呈现的所有内容均通过了身边朋友以及同学的验证，可以放心地跟着指南一步步进行操作。

---

## 资料分布说明

### 实验一：搭建分布式环境

**主要内容**：

1. 完成 Hadoop & Spark 单机版以及分布式环境搭建。

**核心文件**：

- [实验一环境搭建指南](https://github.com/CQULeaf/Big-Data_Course_Resources/blob/main/Lab1_SettingUpEnv/%E9%A1%B9%E7%9B%AE%E5%AE%9E%E9%AA%8C%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)

---

### 实验二：词频统计

**主要内容**：

1. 配置好 **Anaconda** 与 **Jupyter Notebook**。
2. 使用 **VSCode 远程连接**至云服务器进行开发。
3. 使用单机版 Hadoop & Spark 并结合 **Jieba 分词**完成词频统计。
4. 对统计结果**可视化**呈现。
5. 换用 **Bert 分词**完成词频统计并加以对比分析。

**核心文件**：

- [实验二环境搭建指南](https://github.com/CQULeaf/Big-Data_Course_Resources/blob/main/Lab2_WordCount/%E5%AE%9E%E9%AA%8C%E4%BA%8C%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)
- 数据分析与可视化代码：`WordCountUsingBert.ipynb`、`WordCountUsingJieba.ipynb`

---

### 实验三：手写数字识别

**主要内容**：

1. 使用**分布式** Hadoop & Spark 完成手写数字识别大数据分析任务。
2. 对于未经预处理的数据使用KNN进行**可分性测试**。
3. 使用如 OpenCV 等进行**数据预处理**。
4. 使用机器学习方法如**逻辑回归LR**与**随机森林RF**进行模型训练并对比分析。
5. 使用深度学习方法如**卷积神经网络CNN**进行模型训练以提高准确率。
6. 导出训练的模型并植入开发的网页中。

**核心文件**：

- [实验三环境搭建指南](https://github.com/CQULeaf/Big-Data_Course_Resources/blob/main/Lab3_CHN/%E5%AE%9E%E9%AA%8C%E4%B8%89%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA%E6%8C%87%E5%8D%97.md)
- 数据集可分性测试：`ModelBasedKNN.ipynb`
- 数据预处理：`DataPreprocessing.ipynb`
- 机器学习模型训练：`LRStandalone.py`，`RFLocal.ipynb`
- 深度学习模型训练：`ModelBasedCNN.ipynb`
- 开发的网页：`handwrittenWords_frontend`

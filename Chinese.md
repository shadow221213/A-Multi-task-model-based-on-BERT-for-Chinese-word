<!--
 * @Description:
 * @Author: shadow221213
 * @Date: 2023-10-06 17:21:27
 * @LastEditTime: 2025-07-24 23:15:42
-->
# <div align="center">基于BERT的中文多任务模型</div>

<div align="center">
    <a href="https://github.com/shadow221213/SerpentAI-based-for-Binding-of-Isaac/blob/master/README.md">
        English
    </a>
    |
    <a href="https://github.com/shadow221213/SerpentAI-based-for-Binding-of-Isaac/blob/master/Chinese.md">
        简体中文
    </a>
</div>

**持续更新中————**

## 名词解释

中文分词是指将一句话分割成一个个单独的词（类似于“断句”）

文本分类是指将一段话归类为某类信息（如体育、财经）

命名实体识别是指具有特殊含义的词（如人名、地名）

多任务系统是通过同一预训练模型作为参数共享层编码，对每个任务采用不同的解码层解析出对应的结果

## 项目简介

本项目采用`python==3.9.21`和`torch==2.6.0`编写。

如果你需要下载`torch`，请使用这些选项去下载依赖。

```
# ROCM 6.1 (Linux only)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.1
# ROCM 6.2.4 (Linux only)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
# CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# CUDA 12.6
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
# CPU only
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

如果您使用的是`GPU`版本，可以使用`pip install -r requirements.txt`下载依赖项。

如果您使用的是`CPU`版本，可以使用`pip install -r requirements_cpu.txt`下载依赖项。

请使用`python main.py --mtl --augment --train --evaluate --freeze_cls`命令使用增强数据训练并评估模型。  
如需查看更多详细命令，请使用`python main.py --help`或 `pyhton main.py -h`命令查看。

可以通过`tensorboard --logdir=./output/mtl_aug`查看训练过程中的性能图。

查询资料发现中文分词、文本分类和命名实体识别三个任务经常同时出现并使用，想到为什么不能将他们合并构建在一起，于是产生了本项目的多任务系统。

对分词任务采用BiLSTM（双向LSTM）提取上下文信息便于分词（理解歧义词），
对分类任务采用Linear做归类处理（因为BERT已经足够处理分类的大部分信息），
对命名实体识别任务结合分词的数据结果并采用BiLSTM（双向LSTM）提取上下文信息，有助于准确率提升（因为命名实体识别要划分词的含义，对词的边界很重要）。
具体方案请见[论文](./paper/A%20Multi-task%20model%20based%20on%20BERT%20for%20Chinese%20word.pdf)。

## 实验结果

![](./paper/result.png)

![](./paper/cls_augment.png)

|   任务类型   | 精确率(P) | 召回率(R) |   F1   |
| :----------: | :-------: | :-------: | :----: |
|   中文分词   |  91.98%   |  91.43%   | 91.69% |
|   文本分类   |  90.34%   |  78.61%   | 80.39% |
| 命名实体识别 |  92.85%   |  76.39%   | 78.35% |

最终实现多任务间协同提升，即实现多任务输出的同时，各个任务的准确率仍一定提升。
<!--
 * @Description:
 * @Author: shadow221213
 * @Date: 2023-10-06 17:21:27
 * @LastEditTime: 2025-07-24 15:27:20
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

中文分词是指将一句话分割成一个个单独的词（类似于“断句”）

文本分类是指将一段话归类为某类信息（如体育、财经）

命名实体识别是指具有特殊含义的词（如人名、地名）

多任务系统是通过同一预训练模型作为参数共享层编码，对每个任务采用不同的解码层解析出对应的结果

查询资料发现中文分词、文本分类和命名实体识别三个任务经常同时出现并使用，想到为什么不能将他们合并构建在一起，于是产生了本项目的多任务系统。

对分词任务采用BiLSTM（双向LSTM）提取上下文信息便于分词（理解歧义词），
对分类任务采用Linear做归类处理（因为BERT已经足够处理分类的大部分信息），
对命名实体识别任务结合分词的数据结果并采用BiLSTM（双向LSTM）提取上下文信息，有助于准确率提升（因为命名实体识别要划分词的含义，对词的边界很重要）。

## 结果如下：

中文分词：
![](./datasets/isaac/ppo_model/all_model/run_reward_chart.jpg)
文本分类：
![](./datasets/isaac/ppo_model/all_model/run_alive_time_chart.jpg)
命名实体识别：
![](./datasets/isaac/ppo_model/all_model/run_boss_hp_chart.jpg)

最终实现多任务间协同提升，即实现多任务输出的同时，各个任务的准确率仍一定提升。
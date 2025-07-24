<!--
 * @Description:
 * @Author: shadow221213
 * @Date: 2023-10-06 17:14:05
 * @LastEditTime: 2025-07-24 23:14:47
-->
# <div align="center">A Multi-task model based on BERT for Chinese word</div>

<div align="center">
    <a href="https://github.com/shadow221213/A-Multi-task-model-based-on-BERT-for-Chinese-word/blob/master/README.md">
        English
    </a>
    |
    <a href="https://github.com/shadow221213/A-Multi-task-model-based-on-BERT-for-Chinese-word/blob/master/Chinese.md">
        简体中文
    </a>
</div>

**Continuously updated ----**

## Glossary

Chinese word Segmentation refers to dividing a sentence into individual words (similar to “punctuation”).

Text Classification refers to categorizing a passage into a specific type of information (e.g., sports, finance).

Named Entity Recognition (NER) refers to words with special meanings (e.g., proper nouns, place names).

A Multi-task system uses the same pre-trained model as a parameter-sharing layer for encoding, and employs different decoding layers for each task to produce the corresponding results.

## Project Introduction

Upon researching the literature, it was found that the three tasks of Chinese word Segmentation, Text Classification, and NER often appear together and are used in conjunction. This led to the idea of combining them into a single system, resulting in the Multi-task system proposed in this project.

This project is written in `python==3.9.21`, for the specific environment, please use `pip install -r requirements.txt` to download.

For the Word Segmentation task, BiLSTM (bidirectional LSTM) is used to extract contextual information to facilitate word Segmentation (understanding ambiguous words).
For the Text Classification task, a linear model is used for categorization (since BERT already handles most of the information required for Classification).
For the NER task, the results from Word Segmentation are combined, and BiLSTM (bidirectional LSTM) is used to extract contextual information, which helps improve accuracy (since NER involves determining the meaning of words, and the boundaries of words are crucial).
For the specific program, please see [paper](./paper/A%20Multi-task%20model%20based%20on%20BERT%20for%20Chinese%20word.pdf).

## Experimental Results

![](./paper/result.png)

![](./paper/cls_augment.png)

|        Task Type         | Precision(P) | Recall(R) |   F1   |
| :----------------------: | :----------: | :-------: | :----: |
|   Chinese Segmentation   |    91.98%    |  91.43%   | 91.69% |
|   Text Classification    |    90.34%    |  78.61%   | 80.39% |
| Named Entity Recognition |    92.85%    |  76.39%   | 78.35% |

Ultimately, achieve synergistic improvement across multiple tasks, i.e., while achieving output across multiple tasks, the accuracy of each task is still improved.
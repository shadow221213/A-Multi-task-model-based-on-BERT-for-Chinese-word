import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class TextDataset(Dataset):
    def __init__( self, csv_path, tokenizer, max_len=64, augment=False ):
        assert isinstance(tokenizer, PreTrainedTokenizerFast), "必须使用快速tokenizer（PreTrainedTokenizerFast）"

        if augment:
            splitext = os.path.splitext(csv_path)
            csv_path = splitext[0] + '_augment' + splitext[1]
            print("已进行数据增强")

        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 标签映射定义
        self.seg_label_map = {
            'B': 0,
            'I': 1,
            'O': 2
            }
        self.cls_label_map = {
            '体育': 0,
            '娱乐': 1,
            '彩票': 2,
            '房产': 3,
            '教育': 4,
            '时政': 5,
            '游戏': 6,
            '社会': 7,
            '财经': 8
            }
        self.ner_label_map = {
            'O':              0,
            'B-name':         1,
            'I-name':         2,
            'B-organization': 3,
            'I-organization': 4,
            'B-address':      5,
            'I-address':      6,
            'B-government':   7,
            'I-government':   8,
            'B-scene':        9,
            'I-scene':        10,
            'B-game':         11,
            'I-game':         12,
            'B-position':     13,
            'I-position':     14,
            'B-book':         15,
            'I-book':         16,
            'B-company':      17,
            'I-company':      18,
            'B-movie':        19,
            'I-movie':        20,
            }

    def __len__( self ):
        return len(self.data)

    def __getitem__( self, idx ):
        text = self.data['text'].iloc[idx]
        seg_labels = self.data['seg_label'].iloc[idx].split( )
        cls_label = self.cls_label_map[self.data['cls_label'].iloc[idx]]
        ner_labels = self.data['ner_label'].iloc[idx].split( )

        assert len(text) == len(seg_labels) == len(ner_labels), \
            f"文本长度({len(text)})与标签数量(seg:{len(seg_labels)}, ner:{len(ner_labels)})不匹配\n文本：{text}"

        # 文本编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
            return_tensors='pt'
            )

        # 标签对齐（处理subword问题）
        offset_mapping = encoding['offset_mapping'][0].tolist( )
        tokenized_text = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])

        # # 打印关键信息
        # print("\n=== 调试信息 ===")
        # print(f"原始文本：{text}")
        # print(f"字符数：{len(text)}")
        # print(f"原始标签数量：seg={len(seg_labels)}, ner={len(ner_labels)}")
        # print(f"分词结果：{tokenized_text}")
        # print(f"offset_mapping：{offset_mapping}")

        aligned_seg = []
        aligned_ner = []

        original_words = text.split( )  # 原始空格分词结果

        for token, (start, end) in zip(tokenized_text, offset_mapping):
            # 处理特殊标记、[UNK]等无效token
            if token in ['[CLS]', '[SEP]', '[PAD]'] or start == end == 0:
                # aligned_seg.append(self.seg_label_map['O'])
                # aligned_ner.append(self.ner_label_map['O'])
                aligned_seg.append(int(-100))
                aligned_ner.append(int(-100))
                continue

            char_indices = list(range(start, end))

            # 确保所有字符位置都有标签
            for char_idx in char_indices:
                try:
                    aligned_seg.append(self.seg_label_map[seg_labels[char_idx]])
                    aligned_ner.append(self.ner_label_map[ner_labels[char_idx]])
                except IndexError as e:
                    # 异常处理：打印详细信息并填充'O'
                    print(f"\n===== 标签对齐错误 =====")
                    print(f"文本：{text}")
                    print(f"Token：{token} (start={start}, end={end})")
                    print(f"原始标签数量：seg={len(seg_labels)}, ner={len(ner_labels)}")
                    print(f"当前索引：start={start}")
                    print(f"WARNING: 标签不足：文本『{text}』的分词结果为{original_words}，"
                          f"但seg_label有{len(seg_labels)}个标签，"
                          f"Token: {token}")
                    # aligned_seg.append(self.seg_label_map['O'])
                    # aligned_ner.append(self.ner_label_map['O'])
                    aligned_seg.append(int(-100))
                    aligned_ner.append(int(-100))

        # 填充到max_len
        aligned_seg = aligned_seg[:self.max_len] + [int(-100)] * (self.max_len - len(aligned_seg))
        aligned_ner = aligned_ner[:self.max_len] + [int(-100)] * (self.max_len - len(aligned_ner))

        # print("\n对齐结果：")
        # for token, seg, ner in zip(tokenized_text, aligned_seg, aligned_ner):
        #     print(f"Token: {token}\tseg={seg}\tner={ner}")

        return {
            'input_ids':      encoding['input_ids'].flatten( ),
            'attention_mask': encoding['attention_mask'].flatten( ),
            'seg_labels':     torch.tensor(aligned_seg, dtype=torch.long),
            'cls_labels':     torch.tensor(cls_label, dtype=torch.long),
            'ner_labels':     torch.tensor(aligned_ner, dtype=torch.long),
            'raw_text':       text
            }

if __name__ == '__main__':
    def set_seed( seed=42 ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True

    set_seed(221213)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", use_fast=True)
    test_dataset = TextDataset('./train.csv', tokenizer)

    for i, data in enumerate(test_dataset):
        pass

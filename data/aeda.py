import random


class Aeda:
    def __init__(self):
        self.PUNCS = {'，', '。', '！', '？', '；', '：', '、'}

    def __call__(self, text, seg_labels, ner_labels, punc_ratio=0.3):
        chars = list(text)
        insert_pos = random.sample(range(1, len(chars)), k=max(1, int(len(chars) * punc_ratio)))
        insert_pos.sort(reverse=True)

        for pos in insert_pos:
            chars.insert(pos, random.choice(tuple(self.PUNCS)))
            seg_labels.insert(pos, 'O')
            ner_labels.insert(pos, 'O')

        return ["".join(chars), " ".join(seg_labels), " ".join(ner_labels)]

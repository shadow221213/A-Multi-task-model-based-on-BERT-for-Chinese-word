import random


class Aeda:
    def __init__( self ):
        self.PUNCS = {'，', '。', '！', '？', '；', '：', '、'}

    def __call__( self, text, seg_labels, ner_labels, punc_ratio=0.3 ):
        chars = list(text)
        seg_label = seg_labels.copy()
        ner_label = ner_labels.copy()
        insert_pos = random.sample(range(1, len(chars)), k=max(1, int(len(chars) * punc_ratio)))
        insert_pos.sort(reverse=True)

        for pos in insert_pos:
            chars.insert(pos, random.choice(tuple(self.PUNCS)))
            seg_label.insert(pos, 'O')
            ner_label.insert(pos, 'O')

        return ["".join(chars), " ".join(seg_label), " ".join(ner_label)]

if __name__ == '__main__':
    text = '获得很大的共鸣及世界的注目。《芙蓉镇》也是，中国的问题不看这部电影你不会了解。'
    seg_labels = 'B I B I O B I O B I O B I O B I I I I O O O B I O B I B I B I B I O B I B I O'
    cls_labels = '娱乐'
    ner_labels = 'O O O O O O O O O O O O O O B-movie I-movie I-movie I-movie I-movie O O O O O O O O O O O O O O O O O O O O'

    seg_labels = seg_labels.split( )
    ner_labels = ner_labels.split( )

    aeda_texts = []
    aeda_results = []
    results = []
    aeda = Aeda( )
    for _ in range(3):
        aeda_result = aeda(text, seg_labels, ner_labels, punc_ratio=0.3)
        if aeda_result[0] not in aeda_texts:
            aeda_texts.append(aeda_result[0])
            aeda_results.append(aeda_result)

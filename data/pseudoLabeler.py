import torch


class PseudoLabeler:
    def __init__( self, model, tokenizer, max_len=128 ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.large_cls = ['游戏', '财经']
        self.mid_cls = ['房产', '体育']
        self.small_cls = ['彩票', '娱乐']
        self.too_small_cls = ['时政', '社会', '教育']
        self.device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')

    def _tokenize( self, text ):
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
        return encoding

    @torch.no_grad( )
    def __call__( self, text ):
        encoding = self._tokenize(text).to(self.device)
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        seg_logits = outputs['seg_logits']
        ner_logits = outputs['ner_logits']

        seg_probs = torch.softmax(seg_logits, dim=-1)
        ner_probs = torch.softmax(ner_logits, dim=-1)
        seg_conf, seg_pred = seg_probs.max(-1)
        ner_conf, ner_pred = ner_probs.max(-1)

        # 应用阈值
        if (seg_conf < self.word_threshold).any( ):
            seg_label = None
        else:
            seg_label = ' '.join(map(str, seg_pred[0].cpu( ).numpy( ).tolist( )))

        if (ner_conf < self.word_threshold).any( ):
            ner_label = None
        else:
            ner_label = ' '.join(map(str, ner_pred[0].cpu( ).numpy( ).tolist( )))

        return [seg_label, ner_label]

    def check_cls( self, cls_label ):
        # 根据类别占比调整阈值（2：9：18：30）
        if cls_label in self.large_cls:
            self.word_threshold = 0.95
            self.string_threshold = 0.975
            self.choice_num = 1
            self.aeda_num = 1
            self.backtrans_num = 0
        elif cls_label in self.mid_cls:
            self.word_threshold = 0.925
            self.string_threshold = 0.95
            self.choice_num = 4
            self.aeda_num = 4
            self.backtrans_num = 1
        elif cls_label in self.small_cls:
            self.word_threshold = 0.9
            self.string_threshold = 0.925
            self.choice_num = 8
            self.aeda_num = 8
            self.backtrans_num = 2
        elif cls_label in self.too_small_cls:
            self.word_threshold = 0.875
            self.string_threshold = 0.9
            self.choice_num = 10
            self.aeda_num = 16
            self.backtrans_num = 4
        else:
            raise ValueError(f"Unknow cls_labels: {cls_label}")

        return {
            'word_threshold':   self.word_threshold,
            'string_threshold': self.string_threshold,
            'choice_num':       self.choice_num,
            'aeda_num':         self.aeda_num,
            'backtrans_num':    self.backtrans_num
            }

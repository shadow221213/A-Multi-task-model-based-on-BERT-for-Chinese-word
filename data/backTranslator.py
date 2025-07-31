import torch
from transformers import MarianMTModel, MarianTokenizer


class BackTranslator:
    def __init__( self, max_len=128 ):
        src2tgt_name = 'Helsinki-NLP/opus-mt-zh-en'
        tgt2src_name = 'Helsinki-NLP/opus-mt-en-zh'
        self.device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')

        self.src2tgt_tok = MarianTokenizer.from_pretrained(src2tgt_name)
        self.src2tgt_model = MarianMTModel.from_pretrained(src2tgt_name).to(self.device)
        self.tgt2src_tok = MarianTokenizer.from_pretrained(tgt2src_name)
        self.tgt2src_model = MarianMTModel.from_pretrained(tgt2src_name).to(self.device)
        self.max_len = max_len

    @torch.no_grad( )
    def __call__( self, text: str ):
        # zh -> en
        tokens = self.src2tgt_tok(text, return_tensors='pt', padding=True).to(self.device)
        gen = self.src2tgt_model.generate(**tokens, num_beams=4, max_length=self.max_len, early_stopping=True)
        en = self.src2tgt_tok.batch_decode(gen, skip_special_tokens=True)[0]

        # en -> zh
        tokens = self.tgt2src_tok(en, return_tensors='pt', padding=True).to(self.device)
        gen = self.tgt2src_model.generate(**tokens, num_beams=4, max_length=self.max_len, early_stopping=True)
        zh = self.tgt2src_tok.batch_decode(gen, skip_special_tokens=True)[0]
        return zh.replace(' ', '')

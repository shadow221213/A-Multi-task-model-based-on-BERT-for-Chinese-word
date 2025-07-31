import argparse
import gc
import logging
import multiprocessing
import os
import random
import sys
import time
import warnings
from functools import lru_cache
from multiprocessing import Lock, cpu_count

import jieba
import jieba.posseg as pseg
import numpy as np
import OpenHowNet
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, MarianMTModel, MarianTokenizer
from MutiTask.model import MultiTaskModel, MultiTaskModelConfig
from utils.tools import parse_args_aug
from tqdm import tqdm

PUNCS = {'，', '。', '！', '？', '；', '：', '、'}

def aeda(text, seg_labels, ner_labels, punc_ratio = 0.3):
    chars = list(text)
    insert_pos = random.sample(range(1, len(chars)), k=max(1, int(len(chars) * punc_ratio)))
    insert_pos.sort(reverse=True)

    for pos in insert_pos:
        chars.insert(pos, random.choice(tuple(PUNCS)))
        seg_labels.insert(pos, 'O')
        ner_labels.insert(pos, 'O')

    return ["".join(chars), " ".join(seg_labels), " ".join(ner_labels)]

class BackTranslator:
    def __init__(self, src2tgt_name='Helsinki-NLP/opus-mt-zh-en', tgt2src_name='Helsinki-NLP/opus-mt-en-zh', max_len = 128):
        self.device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')

        self.src2tgt_tok = MarianTokenizer.from_pretrained(src2tgt_name)
        self.src2tgt_model = MarianMTModel.from_pretrained(src2tgt_name).to(self.device)
        self.tgt2src_tok = MarianTokenizer.from_pretrained(tgt2src_name)
        self.tgt2src_model = MarianMTModel.from_pretrained(tgt2src_name).to(self.device)
        self.max_len = max_len

    @torch.no_grad()
    def __call__(self, text: str):
        # zh -> en
        tokens = self.src2tgt_tok(text, return_tensors='pt', padding=True).to(self.device)
        gen = self.src2tgt_model.generate(**tokens, num_beams=4, max_length=self.max_len, early_stopping=True)
        en = self.src2tgt_tok.batch_decode(gen, skip_special_tokens=True)[0]

        # en -> zh
        tokens = self.tgt2src_tok(en, return_tensors='pt', padding=True).to(self.device)
        gen = self.tgt2src_model.generate(**tokens, num_beams=4, max_length=self.max_len, early_stopping=True)
        zh = self.tgt2src_tok.batch_decode(gen, skip_special_tokens=True)[0]
        return zh.replace(' ', '')

class PseudoLabeler:
    def __init__(self, model, tokenizer, max_len=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.large_cls = ['游戏', '财经']
        self.mid_cls = ['房产', '体育']
        self.small_cls = ['彩票', '娱乐']
        self.too_small_cls = ['时政', '社会', '教育']
        self.device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')

    def _tokenize(self, text):
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

    @torch.no_grad()
    def __call__(self, text):
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
        if (seg_conf < self.word_threshold).any() or (ner_conf < self.word_threshold).any():
            return None

        # 转为字符串
        seg_label = ' '.join(map(str, seg_pred[0].cpu().numpy().tolist()))
        ner_label = ' '.join(map(str, ner_pred[0].cpu().numpy().tolist()))

        return {
            'seg_label': seg_label,
            'ner_label': ner_label
        }

    def check_cls(self, cls_label):
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
            'word_threshold': self.word_threshold,
            'string_threshold': self.string_threshold,
            'choice_num': self.choice_num,
            'aeda_num': self.aeda_num,
            'backtrans_num': self.backtrans_num
        }

class SafeSynonymAugmenter:
    """安全同义词替换增强器（保持标签一致性）"""

    def __init__( self ):
        self.hownet = OpenHowNet.HowNetDict(init_sim=True)
        # 实体类型到HowNet义原的映射
        self.entity_sememes = {
            'address':      ['location', 'building', 'facilities'],
            'organization': ['group', 'organization', 'institution'],
            'government':   ['politics'],
            'scene':        ['Scene'],
            'game':         ['entertainment', 'recreation'],
            'position':     ['Occupation'],
            'book':         ['literature', 'publications', 'readings', 'document'],
            'company':      ['economy', 'commerce'],
            'movie':        ['artifact'],
            'name':         ['Name'],
            }
        self.punctuations = {'，', '。', '！', '？', '；', '：', '“', '”', '（', '）'}

        self.get_sememe = lru_cache(maxsize=1000)(self.hownet.get_sememe)
        self.calculate_similarity = lru_cache(maxsize=1000)(self.hownet.calculate_word_similarity)
        self.get_word_senses = lru_cache(maxsize=1000)(self.hownet.get_sense)
        self.get_nearest_words = lru_cache(maxsize=1000)(self.hownet.get_nearest_words)

    @lru_cache(maxsize=1000)
    def _check_pos_consistency( self, original_word, candidate_word ):
        """词性一致性检查（批量优化）"""
        return pseg.lcut(original_word)[0].flag == pseg.lcut(candidate_word)[0].flag

    @lru_cache(maxsize=1000)
    def get_safe_synonyms( self, word, seg_tag, ner_tag, word_level, string_level, choice_num=5 ):
        """获取符合实体类型约束的同义词"""
        if word in self.punctuations or len(word) == 1:  # 跳过标点和单字
            return [word]

        result = []
        if ner_tag != 'O':
            # # 实体词：从HowNet中筛选同类型义原的词语
            entity_type = ner_tag.split('-')[1] if '-' in ner_tag else None
            core_sememes = self.entity_sememes.get(entity_type, [])

            # 获取原始词语的所有义原
            original_sememes = set( )
            for sense in self.get_word_senses(word, 'zh'):
                original_sememes.update(sense.get_sememe_list( ))

            candidates = set( )
            for sememe_name in core_sememes:
                for sememe in self.get_sememe(sememe_name, 'en'):
                    for sense in sememe.get_senses( ):
                        candidate_sememes = set(sense.get_sememe_list( ))
                        # 优化条件：交集至少包含2个义原且语义相似度>0.9
                        similarity = self.calculate_similarity(word, sense.zh_word)
                        check_consistency = self._check_pos_consistency(word, sense.zh_word)

                        if len(candidate_sememes) == 1 and original_sememes & candidate_sememes:
                            if similarity >= string_level and check_consistency:
                                candidates.add(sense.zh_word)
                        if len(original_sememes & candidate_sememes) >= 2:
                            if similarity >= word_level and check_consistency:
                                candidates.add(sense.zh_word)

            result = [w for w in candidates if len(w) == len(word)]

            # 非实体词：使用Synonyms库获取同义词
        if result == [] and seg_tag != 'O' and seg_tag != 'S':
            similar_words = self.get_nearest_words(word, language='zh', K=choice_num, score=True)
            all_synonyms = set( )
            for sense_key, synonyms in similar_words.items( ):
                all_synonyms.update(synonyms)

            result = [w for (w, s) in all_synonyms if len(w) == len(word) and s >= word_level]

        result.append(word)
        return result

class DataAugmenterParallel:
    """并行数据增强处理器

    Attributes:
        input_csv (str): 输入数据文件路径
        output_csv (str): 输出数据文件路径
        chunksize (int): 数据分块大小，默认为1000
    """

    def __init__( self, input_csv, output_csv, temp_dir, chunksize=1000 ):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.data = pd.read_csv(input_csv)
        self.temp_dir = temp_dir
        self.chunksize = chunksize
        self.total_samples = self._calculate_total_samples( )

        self.num_workers = max(1, cpu_count( ) // 2)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.ctx = multiprocessing.get_context('spawn')
        self.counter = self.ctx.Value('i', 0)
        self.lock = self.ctx.Lock( )

    def run( self, max_len, args ):
        """执行并行增强流程"""
        # 主进程日志抑制
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
        logging.basicConfig(level=logging.CRITICAL)
        logger = logging.getLogger('OpenHowNet')
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False

        # 分块读取数据
        chunks, now = self._prepare_chunks( )
        if not chunks:
            print("没有需要处理的新数据")
            return

        output_dir = f"{args.output_dir}/mtl"
        best_model_path = f"{output_dir}/checkpoint-best"

        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, use_fast=True)
        device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')

        config = MultiTaskModelConfig(
            model_name=args.pretrained_model
            )

        try:
            best_model = MultiTaskModel.from_pretrained(best_model_path, config=config).to(device)
            print("成功加载最佳模型")
        except Exception as e:
            print(f"加载模型失败：{e}")

        try:
            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2',
                                        device="cuda" if torch.cuda.is_available( ) else "cpu")
            pseudo_labeler = PseudoLabeler(best_model, tokenizer, max_len=max_len)
            back_translator = BackTranslator(max_len=max_len)
        except Exception as e:
            print(f"加载伪标签模型失败：{e}")

        # 主进度条：跟踪总增强文本量
        bar_format = "{l_bar}{bar:50}{r_bar}{bar:-50b}"
        with tqdm(total=self.total_samples, desc="Total Progress", position=0, bar_format=bar_format) as pbar:
            pbar.n = now
            pbar.refresh( )
            # 创建进程池
            with self.ctx.Pool(processes=self.num_workers, initializer=self.init_child_process,
                               initargs=(print_lock, self.lock, self.counter),
                               maxtasksperchild=max(1, self.num_workers // 2)) as pool:
                results = []
                for chunk_idx, input_path in chunks:
                    output_path = os.path.join(self.temp_dir, f"output_chunk_{chunk_idx}.csv")
                    # 提交任务时绑定回调函数
                    results.append(pool.apply_async(
                        self._process_chunk,
                        (model, pseudo_labeler, back_translator, (input_path, output_path, max_len))
                        ))

                last_value = 0
                while not all(r.ready( ) for r in results):
                    with self.lock:  # 安全读取当前值
                        current_value = self.counter.value
                    pbar.update(current_value - last_value)
                    last_value = current_value
                    pbar.refresh( )
                    time.sleep(0.1)

        self._merge_results( )

    def _calculate_total_samples( self ):
        """计算总样本数（用于进度条）"""
        if not os.path.exists(self.input_csv):
            return 0
        with open(self.input_csv, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1  # 减去标题行

    def _merge_results( self ):
        output_files = sorted(
            [os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir) if f.startswith("output_chunk_")],
            key=lambda x: int(x.split("_")[-1].split(".")[0])
            )

        # 使用生成器减少内存占用
        data = pd.read_csv(self.input_csv)
        df_gen = (pd.read_csv(f) for f in output_files)
        # 合并并去重
        merged_df = pd.concat([data] + list(df_gen), ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=['text'], keep='first')
        merged_df = merged_df.sort_values(by=['cls_label'])
        merged_df.to_csv(self.output_csv, index=False)

        print(f"\n增强完成！原始样本: {len(data)}")
        print(f"新增样本: {len(merged_df) - len(data)}")
        print(f"去重后总计: {len(merged_df)}")

    def get_processed_chunks( self ):
        """获取已完成的chunk编号"""
        processed = set( )
        for f in os.listdir(self.temp_dir):
            if f.startswith("output_chunk_"):
                chunk_id = f.split("_")[2].split(".")[0]
                processed.add(int(chunk_id))
        return sorted(processed)

    def _prepare_chunks( self ):
        """准备数据分块（跳过已处理）"""
        processed = self.get_processed_chunks( )
        chunks = []
        sum = 0

        for i, chunk in enumerate(pd.read_csv(self.input_csv, chunksize=self.chunksize)):
            if i in processed:
                sum += len(chunk)
                continue
            chunk_path = os.path.join(self.temp_dir, f"input_chunk_{i}.csv")
            chunk.to_csv(chunk_path, index=False)
            chunks.append((i, chunk_path))

        return chunks, sum

    @staticmethod
    def init_child_process( plock, glock, gcounter ):
        """子进程初始化：抑制日志+继承锁"""
        global print_lock, lock, counter
        print_lock = plock
        lock = glock
        counter = gcounter

        # 子进程独立设置日志抑制（Windows必需）
        warnings.filterwarnings("ignore")
        warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
        logging.basicConfig(level=logging.CRITICAL)
        logger = logging.getLogger('OpenHowNet')
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        jieba.setLogLevel(logging.CRITICAL)

        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        # 显存优化
        if torch.cuda.is_available( ):
            torch.cuda.empty_cache( )

    @staticmethod
    def _process_chunk( model, pseudo_labeler, back_translator, args ):
        """处理单个数据块（静态方法以便多进程序列化）"""
        input_path, output_path, max_len = args
        try:
            chunk = pd.read_csv(input_path)
            augmenter = SafeSynonymAugmenter( )

            results = []

            for _, row in chunk.iterrows( ):
                text = row['text']
                seg_labels = row['seg_label'].split( )
                cls_labels = row['cls_label']
                ner_labels = row['ner_label'].split( )

                try:
                    pseudo_labels_num = pseudo_labeler.check_cls(cls_labels)
                except Exception as e:
                    print(f"分类区分失败：{e}")
                word_threshold = pseudo_labels_num['word_threshold']
                string_threshold = pseudo_labels_num['string_threshold']
                choice_num = pseudo_labels_num['choice_num']
                aeda_num = pseudo_labels_num['aeda_num']
                backtrans_num = pseudo_labels_num['backtrans_num']

                # 分词与安全增强
                new_texts = []
                new_text_set = set( )
                words = jieba.lcut(text)
                max_idx = min(len(seg_labels), len(ner_labels))

                for _ in range(choice_num):
                    new_words = []
                    char_index = 0

                    # 确保分词后的标签对齐
                    for word in words:
                        word_len = len(word)
                        if char_index >= max_idx or word in augmenter.punctuations or len(word) == 1:
                            new_words.append(word)
                            char_index += word_len
                            continue
                        # 安全同义词替换（保持长度一致）
                        synonyms = augmenter.get_safe_synonyms(word, seg_labels[char_index], ner_labels[char_index],
                                                               word_threshold, string_threshold, choice_num=choice_num)
                        new_word = random.choice(synonyms)
                        new_words.append(new_word if len(new_word) == len(word) else word)
                        char_index += word_len

                    new_text = "".join(new_words)
                    new_text_set.add(new_text)

                new_texts = list(new_text_set)
                original_embeddings = model.encode([text for _ in range(choice_num)])
                new_embeddings = model.encode(new_texts)
                sim_score = cosine_similarity(original_embeddings, new_embeddings).diagonal()

                # 重建增强后的文本
                results.extend({
                                   'text': t,
                                   'seg_label': row['seg_label'],
                                   'cls_label': row['cls_label'],
                                   'ner_label': row['ner_label']
                               } for t, s in zip(new_texts, sim_score) if s >= string_threshold)

                # AEDA
                try:
                    aeda_texts = []
                    aeda_results = []
                    for _ in range(aeda_num):
                        aeda_result = aeda(text, seg_labels, ner_labels, punc_ratio = 0.3)
                        if aeda_result[0] not in aeda_texts:
                            aeda_texts.append(aeda_result[0])
                            aeda_results.append(aeda_result)
                    results.extend({
                        'text': aeda_result[0],
                        'seg_label': aeda_result[1],
                        'cls_label': cls_labels,
                        'ner_label': aeda_result[2],
                    } for aeda_result in aeda_results)
                except Exception as e:
                    print(f"加载AEDA失败：{e}")

                # 回译
                trans_text_set = set( )
                if backtrans_num > 0:
                    pre_text = text
                    trans_text_set.add(text)
                    try:
                        for _ in range(backtrans_num):
                            trans_result = back_translator(pre_text)
                            # print(trans_result)
                            trans_text_set.add(trans_result)
                            pre_text = trans_result
                    except Exception as e:
                        print(f"加载回译模型失败：{e}")

                    try:
                        trans_texts = list(trans_text_set)
                        for trans_text in trans_texts:
                            pseudo_labels = pseudo_labeler(trans_text)
                            if pseudo_labels is None:
                                continue

                            results.extend({
                                'text': trans_text,
                                'seg_label': pseudo_labels['seg_label'],
                                'cls_label': cls_labels,
                                'ner_label': pseudo_labels['ner_label']
                            })
                    except Exception as e:
                        print(f"pseudo_labeler加载失败：{e}")

                # 更新进度
                if 'original_embeddings' in locals():
                    del original_embeddings
                if 'new_embeddings' in locals():
                    del new_embeddings
                if 'sim_score' in locals():
                    del sim_score
                if 'new_texts' in locals( ):
                    del new_texts
                if 'new_text_set' in locals( ):
                    del new_text_set
                if 'aeda_results' in locals( ):
                    del aeda_results
                if 'aeda_texts' in locals( ):
                    del aeda_texts
                if 'trans_texts' in locals( ):
                    del trans_texts
                if 'trans_text_set' in locals( ):
                    del trans_text_set
                if 'synonyms' in locals( ):
                    del synonyms

                gc.collect( )
                if torch.cuda.is_available( ):
                    torch.cuda.empty_cache( )
                with lock:
                    counter.value += 1

            results = pd.DataFrame(results, columns=['text', 'seg_label', 'cls_label', 'ner_label'])
            results = results.drop_duplicates(subset=['text'], keep='first')
            results.to_csv(output_path, index=False)

        except Exception as e:
            print(f"处理分块失败: {e}")
            return
        finally:
            # 显式释放资源
            if 'chunk' in locals( ):
                del chunk
            if 'results' in locals( ):
                del results

            gc.collect( )
            if torch.cuda.is_available( ):
                torch.cuda.empty_cache( )

if __name__ == '__main__':
    args = parse_args_aug()
    if args.delete:
        for f in os.listdir(args.temp_dir):
            os.remove(os.path.join(args.temp_dir, f))

    # 全局输出锁
    print_lock = Lock( )

    def set_seed( seed=42 ):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True

    set_seed(221213)

    augmenter = DataAugmenterParallel(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        temp_dir=args.temp_dir,
        chunksize=args.chunk_size
        )

    try:
        augmenter.run(128, args)
        print("\n增强完成！结果保存在:", args.output_csv)
    except KeyboardInterrupt:
        print("\n用户中断，已保存进度。下次运行将自动继续。")
    except Exception as e:
        print(f"运行错误: {str(e)}")

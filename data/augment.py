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
from tqdm import tqdm

global_semantic_model = None
string_level = 0.975
word_level = 0.95

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
    def get_safe_synonyms( self, word, seg_tag, ner_tag, choice_num=5 ):
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

    def run( self, num_augments ):
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

        # 主进度条：跟踪总增强文本量
        bar_format = "{l_bar}{bar:50}{r_bar}{bar:-50b}"
        with tqdm(total=self.total_samples, desc="Total Progress", position=0, bar_format=bar_format) as pbar:
            pbar.n = now
            pbar.refresh( )
            # 创建进程池
            with self.ctx.Pool(processes=self.num_workers, initializer=self.init_child_process,
                               initargs=(print_lock, self.lock, self.counter), maxtasksperchild=5) as pool:
                results = []
                for chunk_idx, input_path in chunks:
                    output_path = os.path.join(self.temp_dir, f"output_chunk_{chunk_idx}.csv")
                    # 提交任务时绑定回调函数
                    results.append(pool.apply_async(
                        self._process_chunk,
                        ((input_path, output_path, num_augments),)
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
    def _process_chunk( args ):
        """处理单个数据块（静态方法以便多进程序列化）"""
        input_path, output_path, num_augments = args
        try:
            chunk = pd.read_csv(input_path)
            augmenter = SafeSynonymAugmenter( )
            results = []

            model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2',
                                        device="cuda" if torch.cuda.is_available( ) else "cpu")

            for _, row in chunk.iterrows( ):
                text = row['text']
                seg_labels = row['seg_label'].split( )
                ner_labels = row['ner_label'].split( )

                # 分词与安全增强
                new_texts = []
                words = jieba.lcut(text)
                max_idx = min(len(seg_labels), len(ner_labels))

                for _ in range(num_augments):
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
                                                               choice_num=num_augments)
                        new_word = random.choice(synonyms)
                        new_words.append(new_word if len(new_word) == len(word) else word)
                        char_index += word_len

                    new_text = "".join(new_words)
                    new_texts.append(new_text)

                original_embeddings = model.encode([text for _ in range(num_augments)])
                new_embeddings = model.encode(new_texts)
                sim_score = cosine_similarity(original_embeddings, new_embeddings).diagonal( )

                # 重建增强后的文本
                results.extend({
                                   'text':      t,
                                   'seg_label': row['seg_label'],
                                   'cls_label': row['cls_label'],
                                   'ner_label': row['ner_label']
                                   } for t, s in zip(new_texts, sim_score) if s >= string_level)

                # 更新进度
                if 'original_embeddings' in locals():
                    del original_embeddings
                if 'new_embeddings' in locals():
                    del new_embeddings
                if 'sim_score' in locals():
                    del sim_score
                gc.collect( )
                if torch.cuda.is_available( ):
                    torch.cuda.empty_cache( )
                with lock:
                    counter.value += 1

            results = pd.DataFrame(results, columns=['text', 'seg_label', 'cls_label', 'ner_label'])
            results = results.drop_duplicates(subset=['text'], keep='first')
            results.to_csv(output_path, index=False)

        except Exception as e:
            print(f"处理分块失败: {str(e)}")
            return
        finally:
            # 显式释放资源
            if 'model' in locals( ):
                del model
            if 'chunk' in locals( ):
                del chunk
            if 'results' in locals( ):
                del results

            gc.collect( )
            if torch.cuda.is_available( ):
                torch.cuda.empty_cache( )

def parse_args( ):
    parser = argparse.ArgumentParser(description="preprocess data")
    parser.add_argument('--chunk_size', type=int, default=128, help='Batch size for preprocessing')
    parser.add_argument('--input_csv', type=str, default='./data/train.csv', help='Path to input data')
    parser.add_argument('--output_csv', type=str, default='./data/train_augment.csv', help='Path to output data')
    parser.add_argument('--temp_dir', type=str, default='./data/temp', help='Directory to save temp data')
    parser.add_argument('--delete', action='store_true', help='Enable delete temp files')

    return parser.parse_args( )

if __name__ == '__main__':
    args = parse_args()
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
        augmenter.run(5)
        print("\n增强完成！结果保存在:", args.output_csv)
    except KeyboardInterrupt:
        print("\n用户中断，已保存进度。下次运行将自动继续。")
    except Exception as e:
        print(f"运行错误: {str(e)}")

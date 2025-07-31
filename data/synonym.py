from functools import lru_cache
import jieba.posseg as pseg
import OpenHowNet


class Synonym:
    """安全同义词替换增强器（保持标签一致性）"""

    def __init__(self):
        self.hownet = OpenHowNet.HowNetDict(init_sim=True)
        # 实体类型到HowNet义原的映射
        self.entity_sememes = {
            'address': ['location', 'building', 'facilities'],
            'organization': ['group', 'organization', 'institution'],
            'government': ['politics'],
            'scene': ['Scene'],
            'game': ['entertainment', 'recreation'],
            'position': ['Occupation'],
            'book': ['literature', 'publications', 'readings', 'document'],
            'company': ['economy', 'commerce'],
            'movie': ['artifact'],
            'name': ['Name'],
        }
        self.punctuations = {'，', '。', '！', '？', '；', '：', '“', '”', '（', '）'}

        self.get_sememe = lru_cache(maxsize=1000)(self.hownet.get_sememe)
        self.calculate_similarity = lru_cache(maxsize=1000)(self.hownet.calculate_word_similarity)
        self.get_word_senses = lru_cache(maxsize=1000)(self.hownet.get_sense)
        self.get_nearest_words = lru_cache(maxsize=1000)(self.hownet.get_nearest_words)

    @lru_cache(maxsize=1000)
    def _check_pos_consistency(self, original_word, candidate_word):
        """词性一致性检查（批量优化）"""
        return pseg.lcut(original_word)[0].flag == pseg.lcut(candidate_word)[0].flag

    @lru_cache(maxsize=1000)
    def get_safe_synonyms(self, word, seg_tag, ner_tag, word_level, string_level, choice_num=5):
        """获取符合实体类型约束的同义词"""
        if word in self.punctuations or len(word) == 1:  # 跳过标点和单字
            return [word]

        result = []
        if ner_tag != 'O':
            # # 实体词：从HowNet中筛选同类型义原的词语
            entity_type = ner_tag.split('-')[1] if '-' in ner_tag else None
            core_sememes = self.entity_sememes.get(entity_type, [])

            # 获取原始词语的所有义原
            original_sememes = set()
            for sense in self.get_word_senses(word, 'zh'):
                original_sememes.update(sense.get_sememe_list())

            candidates = set()
            for sememe_name in core_sememes:
                for sememe in self.get_sememe(sememe_name, 'en'):
                    for sense in sememe.get_senses():
                        candidate_sememes = set(sense.get_sememe_list())
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
            all_synonyms = set()
            for sense_key, synonyms in similar_words.items():
                all_synonyms.update(synonyms)

            result = [w for (w, s) in all_synonyms if len(w) == len(word) and s >= word_level]

        result.append(word)
        return result

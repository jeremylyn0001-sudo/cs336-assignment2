# cs336_basics/bpe.py

import regex as re
from collections import defaultdict
import time 

# GPT-2 标准预分词正则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    def __init__(self):
        # 预编译正则：提速且稳定
        self.pat = re.compile(PAT)
        self.vocab = {}
        self.merge = []

    def train(self, input_path, vocab_size, special_tokens):
        """
        极致保险的 BPE 训练函数：逐行流式读取，不吃内存
        """
        # --- 阶段 1：初始化统计 (隔离 Special Tokens) ---
        word_freqs = defaultdict(int)
        
        # 构造特殊词识别正则
        special_pattern = None
        if special_tokens:
            sp_str = "|".join(re.escape(st) for st in special_tokens)
            special_pattern = re.compile(f"({sp_str})")

        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f: # 【最高保险】一行一行读，永远不爆内存
                # 先分离特殊字符块
                if special_pattern:
                    parts = special_pattern.split(line)
                else:
                    parts = [line]
                
                for part in parts:
                    if special_tokens and part in special_tokens:
                        # 重点：特殊词存成单元素元组 (长度为1，绝不参与合并)
                        word_freqs[(part.encode("utf-8"),)] += 1
                    elif part.strip() or part:
                        # 普通文本使用正则切成单词
                        for match in self.pat.finditer(part): # finditer 比 findall 更省内存
                            word = match.group()
                            bw = word.encode("utf-8")
                            # 将每个词切碎成基础字节元组 (b'h', b'e', b'l', b'l', b'o')
                            char_tuple = tuple(bw[i:i+1] for i in range(len(bw)))
                            word_freqs[char_tuple] += 1

        # 初始化词表
        self.vocab = {i: bytes([i]) for i in range(256)}
        for st in special_tokens:
            self.vocab[len(self.vocab)] = st.encode("utf-8")
        
        self.merge = []

        # 初始词对频率统计
        pair_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i+1])] += freq

        # --- 阶段 2：BPE 核心迭代循环 ---
        start_time = time.time() 
        while len(self.vocab) < vocab_size:
            if not pair_counts:
                break

            # 选冠军：频率最高且字典序最大
            best_pair, max_freq = max(pair_counts.items(), key=lambda item: (item[1], item[0]))
            if max_freq <= 0:
                break

            new_token = best_pair[0] + best_pair[1]
            self.merge.append(best_pair)
            self.vocab[len(self.vocab)] = new_token
            
            current_vocab_size = len(self.vocab)
            if current_vocab_size % 100 == 0:
                elapsed = time.time() - start_time
        # 预估剩下的时间（选做，非常有安全感）
                percent = (current_vocab_size - 256) / (vocab_size - 256) # 基础是256字节
        # 这里打印出：当前进度、时间、合并了什么词
                print(f"进度: {current_vocab_size}/{vocab_size} | "
                      f"已用时: {elapsed:.1f}秒 | "
                      f"当前合并词对: {best_pair} -> 频率: {max_freq}")

            # --- 增量更新 (只去动那些包含 best_pair 的词) ---
            new_word_freqs = {} 
            p0, p1 = best_pair
            changed_words = {} 
            
            for word_tuple, count in word_freqs.items():
                if p0 not in word_tuple or p1 not in word_tuple:
                    continue 
                
                for i in range(len(word_tuple) - 1):
                    pair_counts[(word_tuple[i], word_tuple[i+1])] -= count
                
                
                new_word = []
                idx = 0
                while idx < len(word_tuple):
                    if idx < len(word_tuple) - 1 and word_tuple[idx] == p0 and word_tuple[idx+1] == p1:
                        new_word.append(new_token)
                        idx += 2
                    else:
                        new_word.append(word_tuple[idx])
                        idx += 1
                
                new_tuple = tuple(new_word)
                changed_words[word_tuple] = (new_tuple, count)
                
            for old_word, (new_word, count) in changed_words.items():
                del word_freqs[old_word]
                word_freqs[new_word] += count
                
                for i in range(len(new_word) - 1):
                    pair_counts[(new_word[i], new_word[i+1])] += count


        return self.vocab, self.merge
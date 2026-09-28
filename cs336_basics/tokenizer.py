# cs336_basics/tokenizer.py
import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.byte_to_id = {v:k for k,v in self.vocab.items()}
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            pattern_str = "|".join(re.escape(st) for st in sorted_specials)
            self.special_pattern = re.compile(f"({pattern_str})")
        else:
            self.special_pattern = None
        self.cache = {} # 新增：专门存单词合并结果的缓存
    #解码器
    def decode(self, ids):
        words = []
        for id in ids:
            words.append(self.vocab[id])
        full_bytes = b"".join(words)
        decode_txt = full_bytes.decode("utf-8", errors="replace")
        return decode_txt
    #编码器
    def encode(self, text: str) -> list[int]:
        final_ids = []
        #1.特殊词的正则处理
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]
        #2.遍历片段
        for part in parts:
            if part in self.special_tokens:
                id = self.byte_to_id[part.encode("utf-8")]
                final_ids.append(id)
            else:
                for match in self.pat.finditer(part):
                    word = match.group()
                    
                    
                    # --- [新增 1]：查记事本 ---
                    if word in self.cache:
                        final_ids.extend(self.cache[word])
                        continue # 如果记事本里有，直接跳到下一个单词，省去下面的大循环
                    
                    word_bytes = word.encode("utf-8")
                    # 初始切分
                    word_tuple = tuple(word_bytes[i:i+1] for i in range(len(word_bytes)))
                    #查表merge，得到合并后的字节流
                    for p0, p1 in self.merges:
                        if p0 not in word_tuple or p1 not in word_tuple:
                            continue
                        
                        new_word = [] #开始查表
                        
                        i = 0
                        while i < len(word_tuple):
                            if i < len(word_tuple) - 1 and p0 == word_tuple[i] and p1 == word_tuple[i+1]:
                                new_word.append(p0+p1)
                                i+=2
                            else:
                                new_word.append(word_tuple[i])
                                i+=1
                                
                        word_tuple = tuple(new_word)
                    
                     # --- [新增 2]：计算完后，换成 ID 并记在账本上 ---
                    result_ids = [self.byte_to_id[token] for token in word_tuple]
                    self.cache[word] = result_ids # 存入记事本
                    
                    #合并完毕，得到最终词
                    for token in word_tuple:
                        id = self.byte_to_id[token]
                        final_ids.append(id)
        
        return final_ids
    #
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line) # 处理完一行，内存立即释放
        
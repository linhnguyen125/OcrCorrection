import itertools
import os
import re
import sys
from unicodedata import normalize
import lmdb
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NGRAM
from tool.utils import extract_phrases, gen_ngrams


cache = {}
error = 0
char_regrex = '^[_aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789 !\"\',\-\.:;?_\(\)]+$'
train_cnt = 0
val_cnt = 0

# open lmdb database
train_env = lmdb.open('./lmdb/train_lmdb') # tạo hoặc mở lmdb kích thước tối đa 10MB
val_env = lmdb.open('./lmdb/val_lmdb') # tạo hoặc mở lmdb kích thước tối đa 10MB

def write_cache(env, cache):
    """
    Lưu dữ liệu từ cache vào lmdb
    """
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)
            
# load data from file txt
file_path = './traindata/train_data.txt'
lines = open(file_path, 'r', encoding='utf-8').readlines()

phrases = itertools.chain.from_iterable(extract_phrases(text) for text in lines)
phrases = [p.strip() for p in phrases if len(p.split()) > 1]

tgt_env = val_env
val_data = round(len(phrases) * 0.15)

for p in tqdm(phrases, desc='Creating dataset ...'):
    if not re.match(char_regrex, p):
        continue

    for ngr in gen_ngrams(p, NGRAM):
        if len(" ".join(ngr)) < NGRAM * 7:
            ngram_text = " ".join(ngr)

            if val_cnt == val_data:
                if len(cache) > 0:
                    write_cache(tgt_env, cache)
                    cache = {}
                tgt_env = train_env

            if val_cnt < val_data:
                # write data
                textKey = 'text-%12d' % val_cnt
                val_cnt += 1
            else:
                # write data
                textKey = 'text-%12d' % train_cnt
                train_cnt += 1

            ngram_text = ngram_text.strip()
            ngram_text = ngram_text.rstrip()
            ngram_text = normalize("NFC", ngram_text)

            cache[textKey] = ngram_text.encode()
            
            if len(cache) % 1000 == 0:
                write_cache(tgt_env, cache)
                cache = {}
           
# Lưu nốt cache nếu đã thoát khỏi vòng lặp
if len(cache) > 0:
    write_cache(tgt_env, cache)
    
print('Done!')
print('val_cnt: ', val_cnt)
print('train_cnt: ', train_cnt)
    
cache = {}
cache['num-samples'] = str(train_cnt).encode()
write_cache(train_env, cache)

cache = {}
cache['num-samples'] = str(val_cnt).encode()
write_cache(val_env, cache)
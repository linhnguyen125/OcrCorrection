import itertools
import os
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

# Tạo môi trường LMDB với kích thước map_size lớn hơn
map_size1 = 10 * 1024 * 1024 * 1024  # 10 GB
map_size2 = 2 * 1024 * 1024 * 1024  # 2 GB

# open lmdb database
train_env = lmdb.open('./lmdb/train_item_lmdb', map_size=map_size1)
val_env = lmdb.open('./lmdb/val_item_lmdb', map_size=map_size2)

def write_cache(env, cache):
    """
    Lưu dữ liệu từ cache vào lmdb
    """
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)
            
# load data from file txt
file_path = './traindata/data_inventory_item.txt'
lines = open(file_path, 'r', encoding='utf-8').readlines()

phrases = itertools.chain.from_iterable(extract_phrases(text) for text in lines)
phrases = [p.strip() for p in phrases if len(p.split()) > 2]

tgt_env = val_env
val_data = round(len(phrases) * 0.15)

print('val_data: ', val_data)

# Tăng kích thước cache để giảm số lần ghi
CACHE_SIZE = 50000

for p in tqdm(phrases, desc='Creating dataset ...'):
    for ngr in gen_ngrams(p, NGRAM):
        ngram_text = " ".join(ngr)
        if len(ngram_text) < NGRAM * 7:
            if val_cnt == val_data:
                if len(cache) > 0:
                    write_cache(tgt_env, cache)
                    cache = {}
                tgt_env = train_env

            if val_cnt <= val_data:
                # write data
                textKey = 'text-%12d' % val_cnt
                val_cnt += 1
            else:
                # write data
                textKey = 'text-%12d' % train_cnt
                train_cnt += 1

            ngram_text = ngram_text.strip()
            ngram_text = normalize("NFC", ngram_text)
            if ngram_text:
                cache[textKey] = ngram_text.encode()
            
            if len(cache) % CACHE_SIZE == 0:
                write_cache(tgt_env, cache)
                cache = {}
        
# Lưu nốt cache nếu đã thoát khỏi vòng lặp
if len(cache) > 0:
    write_cache(tgt_env, cache)
    
print('Done!')
print('val_cnt: ', val_cnt-1)
print('train_cnt: ', train_cnt)
    
cache = {}
cache['num-samples'] = str(train_cnt).encode()
write_cache(train_env, cache)

cache = {}
cache['num-samples'] = str(val_cnt-1).encode()
write_cache(val_env, cache)
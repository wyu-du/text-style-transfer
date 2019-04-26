"""
args:
1 corpus file (tokenized)
2 K
prints K most frequent vocab items
"""
import os
from collections import Counter

dir_path = os.path.abspath(os.path.dirname(os.getcwd()))
print(dir_path)


vocab_size = 20000
with open(dir_path+'/data/hp/train.hp.txt', 'r', encoding='utf8') as f0:
    neg_corpus = f0.read().split('\n')
with open(dir_path+'/data/hp/train.new.txt', 'r', encoding='utf8') as f1:
    pos_corpus = f1.read().split('\n')   
all_corpus = neg_corpus + pos_corpus

c = Counter()
for line in all_corpus:
    for tok in line.strip().split():
        c[tok] += 1

with open(dir_path+'/data/hp/dict.20k', 'w', encoding='utf8') as f:
    f.write('<unk>\n')
    f.write('<pad>\n')
    f.write('<s>\n')
    f.write('</s>\n')
    for tok, _ in c.most_common(vocab_size):
        f.write(tok+'\n')



# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:20:47 2019

@author: Wanyu Du
"""

import os

dir_path = os.path.abspath(os.path.dirname(os.getcwd()))
print(dir_path)


with open(dir_path+'\\data\\yelp\\reference.1', 'r', encoding='utf8') as f0:
    corpus = f0.read().split('\n')

pos_corpus = []
neg_corpus = []
for line in corpus:
    sents = line.split('	')
    if len(sents)==2:
        pos_corpus.append(sents[0])
        neg_corpus.append(sents[1])
    
with open(dir_path+'\\data\yelp\sentiment.truth.1', 'w', encoding='utf8') as f1:
    for line in pos_corpus:
        f1.write(line+'\n')
with open(dir_path+'\\data\yelp\sentiment.truth.0', 'w', encoding='utf8') as f2:
    for line in neg_corpus:
        f2.write(line+'\n')
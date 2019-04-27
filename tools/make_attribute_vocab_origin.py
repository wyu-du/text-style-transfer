#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:43:42 2019

@author: wanyu
"""

"""
python make_attribute_vocab.py [vocab] [corpus1] [corpus2] r
subsets a [vocab] file by finding the words most associated with 
one of two corpuses. threshold is r ( # in corpus_a  / # in corpus_b )
"""
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

class SalienceCalculator(object):
    def __init__(self, pre_corpus, post_corpus):
        self.vectorizer = CountVectorizer()

        pre_count_matrix = self.vectorizer.fit_transform(pre_corpus)
        self.pre_vocab = self.vectorizer.vocabulary_
        self.pre_counts = np.sum(pre_count_matrix, axis=0)
        self.pre_counts = np.squeeze(np.asarray(self.pre_counts))

        post_count_matrix = self.vectorizer.fit_transform(post_corpus)
        self.post_vocab = self.vectorizer.vocabulary_
        self.post_counts = np.sum(post_count_matrix, axis=0)
        self.post_counts = np.squeeze(np.asarray(self.post_counts))


    def salience(self, feature, attribute='0', lmbda=1):
        assert attribute in ['0', '1']

        if feature not in self.pre_vocab:
            pre_count = 0.0
        else:
            pre_count = self.pre_counts[self.pre_vocab[feature]]

        if feature not in self.post_vocab:
            post_count = 0.0
        else:
            post_count = self.post_counts[self.post_vocab[feature]]
        
        if attribute == '0':
            return (pre_count + lmbda) / (post_count + lmbda)
        else:
            return (post_count + lmbda) / (pre_count + lmbda)




dir_path = os.path.abspath(os.path.dirname(os.getcwd()))
print(dir_path)

with open(dir_path+'/data/amazon/amazon_dict.20k', 'r', encoding='utf8') as f:
    vocab_corpus = f.read().split('\n')
vocab = set([w.strip() for i, w in enumerate(vocab_corpus)])


corpus0 = dir_path+'/data/amazon/sentiment.train.0'
corpus0 = [
    w if w in vocab else '<unk>' 
    for l in open(corpus0)
    for w in l.strip().split()
]

corpus1 = dir_path+'/data/amazon/sentiment.train.1'
corpus1 = [
    w if w in vocab else '<unk>' 
    for l in open(corpus1)
    for w in l.strip().split()
]

r = 15.0
sc = SalienceCalculator(corpus0, corpus1)
with open(dir_path+'/data/amazon/dict_attr_origin.20k', 'w', encoding='utf8') as f:
    for tok in vocab:
    #    print(tok, sc.salience(tok))
        if max(sc.salience(tok, attribute='0'), sc.salience(tok, attribute='1')) > r:
            f.write(tok+'\n')

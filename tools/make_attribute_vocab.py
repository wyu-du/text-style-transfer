"""
python make_attribute_vocab.py [vocab] [corpus1] [corpus2] r

subsets a [vocab] file by finding the words most associated with 
one of two corpuses. threshold is r ( # in corpus_a  / # in corpus_b )
"""
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import nltk

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

with open('./data/yelp/dict.20k', 'r', encoding='utf8') as f:
    vocab_corpus = f.read().split('\n')
vocab = set([w.strip() for i, w in enumerate(vocab_corpus)])


corpus0 = './data/yelp/sentiment.train.0'
corpus0_vocab = []
for line in open(corpus0):
    pos_tags = nltk.pos_tag(nltk.word_tokenize(line))

    for (word, tag) in pos_tags:
        if not tag.startswith("NN"):
            corpus0_vocab.append(word)

print("Get corpus0 vocab.")

corpus1 = './data/yelp/sentiment.train.1'
corpus1_vocab = []
for line in open(corpus1):
    pos_tags = nltk.pos_tag(nltk.word_tokenize(line))

    for (word, tag) in pos_tags:
        if not tag.startswith("NN"):
            corpus1_vocab.append(word)

print("Get corpus1 vocab.")

r = 15.0
sc = SalienceCalculator(corpus0_vocab, corpus1_vocab)
with open('./data/yelp/dict_attr.20k', 'w', encoding='utf8') as f:
    for tok in vocab:
        if(len(tok) == 0):
            continue
        #print(tok, sc.salience(tok))
        if max(sc.salience(tok, attribute='0'), sc.salience(tok, attribute='1')) > r:
            f.write(tok+'\n')



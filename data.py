"""Data utilities."""
import os
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import torch
from torch.autograd import Variable

from cuda import CUDA

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from tools.make_attribute_vocab import make_attribute

class CorpusSearcher(object):
    def __init__(self, query_corpus, key_corpus, value_corpus, vectorizer, make_binary=True, use_doc2vec=False):
        self.use_doc2vec = use_doc2vec

        if(use_doc2vec):
            documents = []
            cnt = 0
            for line in key_corpus:
                documents.append(TaggedDocument(line, [str(cnt)]))
                cnt += 1

            self.vectorizer = vectorizer(documents, min_count=5, size=100)
            self.key_corpus = key_corpus

            key_corpus_matrix = []
            for i in range(len(self.vectorizer.docvecs)):
                key_corpus_matrix.append(np.array(self.vectorizer.docvecs[str(i)]))
            self.key_corpus_matrix = np.array(key_corpus_matrix)
            
        else:
            self.vectorizer = vectorizer
            self.vectorizer.fit(key_corpus)
            # rows = docs, cols = features
            self.key_corpus_matrix = self.vectorizer.transform(key_corpus)
            if make_binary:
                self.key_corpus_matrix = (self.key_corpus_matrix != 0).astype(int) # make binary

        self.query_corpus = query_corpus
        self.key_corpus = key_corpus
        self.value_corpus = value_corpus
        
    def most_similar(self, key_idx, n=10):
        query = self.query_corpus[key_idx]

        if(self.use_doc2vec):
            query_vec = query.split()
            
            topn_vec = self.vectorizer.docvecs.most_similar([self.vectorizer.infer_vector(query_vec)], topn=n)

            # Convert tag to integer
            selected = []
            for (str_i, score) in topn_vec:
                i = int(str_i)
                selected.append((self.query_corpus[i], ' '.join(self.key_corpus[i]), self.value_corpus[i], i, score) )

        else:
            query_vec = self.vectorizer.transform([query])
            scores = np.dot(self.key_corpus_matrix, query_vec.T)
            scores = np.squeeze(scores.toarray()) 
        
            scores_indices = zip(scores, range(len(scores)))
            selected = sorted(scores_indices, reverse=True)[:n]
            # use the retrieved i to pick examples from the VALUE corpus
            selected = [
                (self.query_corpus[i], self.key_corpus[i], self.value_corpus[i], i, score) 
                for (score, i) in selected
            ]
    
        #print("\n\nQuery: " + query)
        #print("\n\tSelected: ")
        #for select in selected:
        #    print("\ti: " + str(select[3]) + "\tkey_corpus[i]: " + str(select[1]) + "\tvalue_corpus[i]: " + str(select[2]) + "\tscore: " + str(select[4]))
        #return []
        #if(self.use_doc2vec):
        #    print("\n\nQuery: " + query)
        #    print("\n\tSelected: ")
        #    for select in selected:
        #        print("\ti: " + str(select[3]) + "\tkey_corpus[i]: " + str(select[1]) + "\tvalue_corpus[i]: " + str(select[2]) + "\tscore: " + str(select[4]))
        #return

        return selected


def build_vocab_maps(vocab_file):
    assert os.path.exists(vocab_file), "The vocab file %s does not exist" % vocab_file
    unk = '<unk>'
    pad = '<pad>'
    sos = '<s>'
    eos = '</s>'

    lines = [x.strip() for x in open(vocab_file, encoding="utf8")]

    assert lines[0] == unk and lines[1] == pad and lines[2] == sos and lines[3] == eos, \
        "The first words in %s are not %s, %s, %s, %s" % (vocab_file, unk, pad, sos, eos)

    tok_to_id = {}
    id_to_tok = {}
    for i, vi in enumerate(lines):
        tok_to_id[vi] = i
        id_to_tok[i] = vi

    # Extra vocab item for empty attribute lines
    empty_tok_idx =  len(id_to_tok)
    tok_to_id['<empty>'] = empty_tok_idx
    id_to_tok[empty_tok_idx] = '<empty>'

    return tok_to_id, id_to_tok


def extract_attributes(line, tok_weights_dict):
    # Decide how many attributes to be picked
    attr_num = 3 if len(line) > 8 else 2 if len(line) > 4 else 1
    
    line_tok_dict = {}
    for tok in line:
        line_tok_dict[tok] = tok_weights_dict.get(tok, -1)
    
    attribute = sorted(line_tok_dict, key=lambda k: line_tok_dict[k], reverse=True)[:attr_num]
    
    content = []
    for tok in line:
        if tok not in attribute:
            content.append(tok)
    return line, content, attribute


def gen_train_data(src, tgt, config):
    tok_weights_dict = make_attribute(src, tgt)

    src_lines = [l.strip().split() for l in open(src, 'r', encoding="utf8")]
    src_lines, src_content, src_attribute = list(zip(
        *[extract_attributes(line, tok_weights_dict) for line in src_lines]
    ))
    src_tok2id, src_id2tok = build_vocab_maps(config['data']['src_vocab'])
    # train time: just pick attributes that are close to the current (using word distance)
    # we never need to do the TFIDF thing with the source because 
    # test time is strictly in the src => tgt direction
    src_dist_measurer = CorpusSearcher(
        query_corpus=[' '.join(x) for x in src_attribute],
        key_corpus=[' '.join(x) for x in src_attribute],
        value_corpus=[' '.join(x) for x in src_attribute],
        vectorizer=CountVectorizer(vocabulary=src_tok2id),
        make_binary=True
    )
    src = {
        'data': src_lines, 'content': src_content, 'attribute': src_attribute,
        'tok2id': src_tok2id, 'id2tok': src_id2tok, 'dist_measurer': src_dist_measurer
    }

    return src, tok_weights_dict


def gen_dev_data(src, tgt, tok_weights_dict, config):
    src_lines = [l.strip().split() for l in open(src, 'r', encoding="utf8")]
    src_lines, src_content, src_attribute = list(zip(
        *[extract_attributes(line, tok_weights_dict) for line in src_lines]
    ))
    src_tok2id, src_id2tok = build_vocab_maps(config['data']['src_vocab'])
    # train time: just pick attributes that are close to the current (using word distance)
    # we never need to do the TFIDF thing with the source because 
    # test time is strictly in the src => tgt direction
    src_dist_measurer = CorpusSearcher(
        query_corpus=[' '.join(x) for x in src_attribute],
        key_corpus=[' '.join(x) for x in src_attribute],
        value_corpus=[' '.join(x) for x in src_attribute],
        vectorizer=CountVectorizer(vocabulary=src_tok2id),
        make_binary=True
    )
    src = {
        'data': src_lines, 'content': src_content, 'attribute': src_attribute,
        'tok2id': src_tok2id, 'id2tok': src_id2tok, 'dist_measurer': src_dist_measurer
    }

    tgt_lines = [l.strip().split() for l in open(tgt, 'r', encoding="utf8")] if tgt else None
    tgt_lines, tgt_content, tgt_attribute = list(zip(
        *[extract_attributes(line, tok_weights_dict) for line in tgt_lines]
    ))
    tgt_tok2id, tgt_id2tok = build_vocab_maps(config['data']['tgt_vocab'])
    tgt_dist_measurer = CorpusSearcher(
        query_corpus=[' '.join(x) for x in src_content],
        key_corpus=[' '.join(x) for x in tgt_content],
        value_corpus=[' '.join(x) for x in tgt_attribute],
        vectorizer=CountVectorizer(vocabulary=src_tok2id),
        make_binary=True
    )
    tgt = {
        'data': tgt_lines, 'content': tgt_content, 'attribute': tgt_attribute,
        'tok2id': tgt_tok2id, 'id2tok': tgt_id2tok, 'dist_measurer': tgt_dist_measurer
    }

    return src, tgt


def sample_replace(lines, dist_measurer, sample_rate, corpus_idx):
    """
    replace sample_rate * batch_size lines with nearby examples (according to dist_measurer)
    not exactly the same as the paper (words shared instead of jaccaurd during train) but same idea
    """
    out = [None for _ in range(len(lines))]
    for i, line in enumerate(lines):
        if random.random() < sample_rate:
            sims = dist_measurer.most_similar(corpus_idx + i)[1:]  # top match is the current line
            try:
                line = next( (
                    tgt_attr.split() for src_cntnt, tgt_cntnt, tgt_attr, _, _ in sims
                    if tgt_attr != ' '.join(line) # and tgt_attr != ''   # TODO -- exclude blanks?
                ) )
            # all the matches are blanks
            except StopIteration:
                line = []
            line = ['<s>'] + line + ['</s>']

        # corner case: special tok for empty sequences (just start/end tok)
        if len(line) == 2:
            line.insert(1, '<empty>')
        out[i] = line

    return out


def get_minibatch(lines, tok2id, index, batch_size, max_len, sort=False, idx=None,
                  dist_measurer=None, sample_rate=0.0):
    """
    Prepare minibatch.
    Input:
        lines: input sequence list
        tok2id: token -> id dictionary
        index: current batch index
        batch_size: minibatch size
        max_len: maximum sequence length
        sort: whether to sort sequence by descending length
        idx: the index of the sequence
        dist_measure: replace sample_rate * batch_size lines with nearby examples (don't know which function to use!!)
        sample_rate: sampling rate for the sample_replace() method
    Output:
        input_lines: input sequence_id list (start with <s>), shape = (batch_size, max_len)
        output_lines: input sequence_id list (end with </s>), shape = (batch_size, max_len)
        lens: input sequence length list
        mask: input mask list, shape = (batch_size, max_len)
        idx: the index of the sequence
            
    """
    # FORCE NO SORTING because we care about the order of outputs
    #   to compare across systems
    lines = [
        ['<s>'] + line[:max_len] + ['</s>']
        for line in lines[index:index + batch_size]
    ]

    if dist_measurer is not None:
        lines = sample_replace(lines, dist_measurer, sample_rate, index)

    lens = [len(line) - 1 for line in lines]
    max_len = max(lens)

    unk_id = tok2id['<unk>']
    input_lines = [
        [tok2id.get(w, unk_id) for w in line[:-1]] +
        [tok2id['<pad>']] * (max_len - len(line) + 1)
        for line in lines
    ]
    
    output_lines = [
        [tok2id.get(w, unk_id) for w in line[1:]] +
        [tok2id['<pad>']] * (max_len - len(line) + 1)
        for line in lines
    ]

    mask = [
        ([1] * l) + ([0] * (max_len - l))
        for l in lens
    ]

    if sort:
        # sort sequence by descending length
        idx = [x[0] for x in sorted(enumerate(lens), key=lambda x: -x[1])]

    if idx is not None:
        lens = [lens[j] for j in idx]
        input_lines = [input_lines[j] for j in idx]
        output_lines = [output_lines[j] for j in idx]
        mask = [mask[j] for j in idx]

    input_lines = Variable(torch.LongTensor(input_lines))
    output_lines = Variable(torch.LongTensor(output_lines))
    mask = Variable(torch.FloatTensor(mask))

    if CUDA:
        input_lines = input_lines.cuda()
        output_lines = output_lines.cuda()
        mask = mask.cuda()

    return input_lines, output_lines, lens, mask, idx


def minibatch(src, tgt, idx, batch_size, max_len, model_type, is_test=False):
    """
    Generate minibatch.
    Input:
        src: {'data': src_lines (input seq list), 'content': src_content (input seq list, no attribute words), 
              'attribute': src_attribute (list of attribute words, from the dict_att),
              'tok2id': src_tok2id, 'id2tok': src_id2tok, 
              'dist_measurer': src_dist_measurer (list of attributes that are close to the current attributes)}
        tgt:{'data': tgt_lines (target seq list), 'content': tgt_content (target seq list, no attribute words), 
             'attribute': tgt_attribute (list of attributr words, from the dict_att),
             'tok2id': tgt_tok2id, 'id2tok': tgt_id2tok, 'dist_measurer': tgt_dist_measurer}
        idx: current batch index
        batch_size: size of the minibatch
        max_len: maximum sequence length
        model_type: type of models
        is_test: train or test
    Output:
        inputs: (src_content_lines (with <s>), src_content_lines (with </s>), lens, mask, idx)
        attributes:
            - 'delete': (attribute_ids (all 0 or all 1), None, None, None, None)
            - 'delete_retrieve': (target_attributes_list, target_attributes_list, lens, mask, idx)
            - 'seq2seq': (None, None, None, None, None)
        outputs: (target_data_lines, target_data_lines, lens, mask, idx)
            
    """
    if not is_test:
        use_src = random.random() < 0.5
        in_dataset = src if use_src else tgt
        out_dataset = in_dataset
        attribute_id = 0 if use_src else 1
    else:
        in_dataset = src
        out_dataset = tgt
        attribute_id = 1

    if model_type == 'delete':
        inputs = get_minibatch(
            in_dataset['content'], in_dataset['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

        # true length could be less than batch_size at edge of data
        batch_len = len(outputs[0])
        attribute_ids = [attribute_id for _ in range(batch_len)]
        attribute_ids = Variable(torch.LongTensor(attribute_ids))
        if CUDA:
            attribute_ids = attribute_ids.cuda()

        attributes = (attribute_ids, None, None, None, None)

    elif model_type == 'delete_retrieve' or model_type == 'pointer':
        inputs =  get_minibatch(
            in_dataset['content'], in_dataset['tok2id'], idx, batch_size, max_len, sort=True)
        attributes = get_minibatch(
            out_dataset['attribute'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1],
            dist_measurer=out_dataset['dist_measurer'], sample_rate=0.25)
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

    elif model_type == 'seq2seq':
        # ignore the in/out dataset stuff
        inputs = get_minibatch(
            src['data'], src['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            tgt['data'], tgt['tok2id'], idx, batch_size, max_len, idx=inputs[-1])
        attributes = (None, None, None, None, None)

    else:
        raise Exception('Unsupported model_type: %s' % model_type)

    return inputs, attributes, outputs


def unsort(arr, idx):
    """unsort a list given idx: a list of each element's 'origin' index pre-sorting
    """
    unsorted_arr = arr[:]
    for i, origin in enumerate(idx):
        unsorted_arr[origin] = arr[i]
    return unsorted_arr




import math
import numpy as np
import sys
from collections import Counter
import logging

import torch
from torch.autograd import Variable
import torch.nn as nn
import editdistance

import data
import models
from utils import word2id, id2word
from cuda import CUDA

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# BLEU functions from https://github.com/MaximumEntropy/Seq2Seq-PyTorch
#    (ran some comparisons, and it matches moses's multi-bleu.perl)
def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats

def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_edit_distance(hypotheses, reference):
    ed = 0
    for hyp, ref in zip(hypotheses, reference):
        ed += editdistance.eval(hyp, ref)

    return ed * 1.0 / len(hypotheses)


def get_precisions_recalls(inputs, preds, ground_truths):
    def precision_recall(src, tgt, pred):
        src_set = set(src)
        tgt_set = set(tgt)
        pred_set = set(pred)
    
        tgt_unique = tgt_set - src_set
        src_unique = src_set - tgt_set
        shared = tgt_set & src_set
        
        correct_shared = len(pred_set & shared)
        correct_tgt = len(pred_set & tgt_unique)
        
        incorrect_src = len(pred_set & src_unique)
        incorrect_unseen = len(pred_set - src_set - tgt_set)
        
        # words the model correctly introduced
        tp = correct_tgt
        # words the model incorrectly introduced
        fp = incorrect_unseen
        # bias words the model incorrectly kept
        fn = incorrect_src
        
        precision = tp * 1.0 / (tp + fp + 0.001)
        recall = tp * 1.0 / (tp + fn + 0.001)

        return precision, recall

    [precisions, recalls] = list(zip(*[
        precision_recall(src, tgt, pred) 
        for src, tgt, pred in zip(inputs, ground_truths, preds)
    ]))

    return precisions, recalls


def gen_ngram(sent, n=2):
    words = sent.split()
    ngrams = []
    for i, token in enumerate(words):
        if i<=len(words)-n:
            ngram = '-'.join(words[i:i+n])
            ngrams.append(ngram)
    return ngrams

def count_match(ref, dec, n=2):
    counts = 0.
    for d_word in dec:
        if d_word in ref:
            counts += 1
    return counts

def rouge_2(gold_sent, decode_sent):
    bigrams_ref = gen_ngram(gold_sent, 2)
    bigrams_dec = gen_ngram(decode_sent, 2)
    if len(bigrams_ref) == 0:
        recall = 0.
    else:
        recall = count_match(bigrams_ref, bigrams_dec, 2)/len(bigrams_ref)
    if len(bigrams_dec) == 0:
        precision = 0.
    else:
        precision = count_match(bigrams_ref, bigrams_dec, 2)/len(bigrams_dec)
    if recall+precision == 0:
        f1_score = 0.
    else:
        f1_score = 2*recall*precision/(recall+precision)
    return f1_score


def inference_bleu(model, src, tgt, config):
    """ decode and evaluate bleu """
    searcher, rouge_list, initial_inputs, preds, ground_truths, auxs = my_decode_dataset(model, src, tgt, config)

    bleu = get_bleu(preds, ground_truths)
    edit_distance = get_edit_distance(preds, ground_truths)
    precisions, recalls = get_precisions_recalls(initial_inputs, preds, ground_truths)

    precision = np.average(precisions)
    recall = np.average(recalls)

    initial_inputs = [' '.join(seq) for seq in initial_inputs]
    preds = [' '.join(seq) for seq in preds]
    ground_truths = [' '.join(seq) for seq in ground_truths]
    for i, seq in enumerate(auxs):
        if len(seq) == 1:
            auxs[i] = seq[0]
        else:
            auxs[i] = ' '.join(seq)

    return bleu, edit_distance, precision, recall, initial_inputs, preds, ground_truths, auxs


def inference_rouge(model, src, tgt, config):
    """ 
    decode and evaluate rouge
    
    args:
        src: src data object (i.e. data 1, not learnt by the model)
        tgt: target data object (i.e. data 0, learnt by the model)
    """
        
    searcher, rouge_list, initial_inputs, preds, ground_truths, auxs = my_decode_dataset(model, src, tgt, config)
        
    rouge = np.mean(rouge_list)
    edit_distance = get_edit_distance(preds, ground_truths)
    precisions, recalls = get_precisions_recalls(initial_inputs, preds, ground_truths)

    precision = np.average(precisions)
    recall = np.average(recalls)

    initial_inputs = [' '.join(seq) for seq in initial_inputs]
    preds = [' '.join(seq) for seq in preds]
    ground_truths = [' '.join(seq) for seq in ground_truths]
    for i, seq in enumerate(auxs):
        if len(seq) == 1:
            auxs[i] = seq[0]
        else:
            auxs[i] = ' '.join(seq)

    return rouge, edit_distance, precision, recall, initial_inputs, preds, ground_truths, auxs



def evaluate_lpp(model, src, tgt, config):
    """ evaluate log perplexity WITHOUT decoding
        (i.e., with teacher forcing)
    """
    weight_mask = torch.ones(len(tgt['tok2id']))
    if CUDA:
        weight_mask = weight_mask.cuda()
    weight_mask[tgt['tok2id']['<pad>']] = 0
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
    if CUDA:
        loss_criterion = loss_criterion.cuda()

    losses = []
    for j in range(0, len(src['data']), config['data']['batch_size']):
        # get batch
        input_content, input_aux, output = data.minibatch(src, tgt, j, config['data']['batch_size'], 
                                                          config['data']['max_len'], 
                                                          config['model']['model_type'],
                                                          is_test=True)
        input_content_src, _, srclens, srcmask, _ = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_data_tgt, output_data_tgt, _, _, _ = output

        decoder_logit, decoder_probs = model(input_content_src, srcmask, srclens,
                                             input_ids_aux, auxmask, auxlens, input_data_tgt, mode='train')

        loss = loss_criterion(decoder_logit.contiguous().view(-1, len(tgt['tok2id'])),
                              output_data_tgt.view(-1))
        losses.append(loss.item())

    return np.mean(losses)


def evaluate_rouge(model, src, tgt, config):
    """ 
    evaluate log perplexity WITH decoding
    
    args:
        src: src data object (i.e. data 0, learnt by the model)
        tgt: target data object (i.e. data 0, learnt by the model)
    """
    weight_mask = torch.ones(len(tgt['tok2id']))
    if CUDA:
        weight_mask = weight_mask.cuda()
    weight_mask[tgt['tok2id']['<pad>']] = 0
        
    searcher = models.GreedySearchDecoder(model)

    rouge_list = []
    decoded_results = []
    for j in range(0, len(src['data'])):
        # batch_size = 1
        input_content, input_aux, output = data.minibatch(src, src, j, 1, 
                                             config['data']['max_len'], 
                                             config['model']['model_type'])
        input_content_src, _, srclens, srcmask, _ = input_content
        input_ids_aux, _, auxlens, auxmask, _ = input_aux
        input_data_tgt, output_data_tgt, _, _, _ = output
        
        
        decoder_logit, decoded_data_tgt = searcher(input_content_src, srcmask, srclens,
                                                   input_ids_aux, auxmask, auxlens,
                                                   20, tgt['tok2id']['<s>'])
        decoded_sent = id2word(decoded_data_tgt, tgt)
        gold_sent = id2word(output_data_tgt, tgt)
        rouge = rouge_2(gold_sent, decoded_sent)
        rouge_list.append(rouge)
        decoded_results.append(decoded_sent)
        
        #print('Source content sentence:'+gold_sent)
        #print('Decoded data sentence:'+decoded_sent)

    return np.mean(rouge_list), decoded_results

def my_decode_dataset(model, src, tgt, config):
    searcher = models.GreedySearchDecoder(model)
    rouge_list = []
    initial_inputs = []
    preds = []
    ground_truths = []
    auxs = []
    
    for j in range(0, len(src['data'])):
        if j%100 == 0:
            logging.info('Finished decoding data: %d/%d ...'% (j, len(src['data'])))
        
        # batch_size = 1
        inputs, _, outputs = data.minibatch(src, tgt, j, 1, 
                                            config['data']['max_len'], 
                                            config['model']['model_type'], 
                                            is_test=True)
        input_content_src, _, srclens, srcmask, _ = inputs
        _, output_data_tgt, tgtlens, tgtmask, _ = outputs
       
        
        tgt_dist_measurer = tgt['dist_measurer']
        related_content_tgt = tgt_dist_measurer.most_similar(j, n=3)   # list of n seq_str
        # related_content_tgt = source_content_str, target_content_str, target_att_str, idx, score
        
        # Put all the retrieved attributes together
        retrieved_attrs_set = set()
        for single_data_tgt in related_content_tgt:
            sp = single_data_tgt[2].split()
            for attr in sp:
                retrieved_attrs_set.add(attr)
                    
        retrieved_attrs = ' '.join(retrieved_attrs_set)
        
        input_ids_aux, auxlens, auxmask = word2id(retrieved_attrs, None, tgt, config['data']['max_len'])
        
        n_decoded_sents = []
        
        input_ids_aux = Variable(torch.LongTensor(input_ids_aux))
        auxlens = Variable(torch.LongTensor(auxlens))
        auxmask = Variable(torch.LongTensor(auxmask))
            
        if CUDA:
            input_ids_aux = input_ids_aux.cuda()
            auxlens = auxlens.cuda()
            auxmask = auxmask.cuda()
            
        _, decoded_data_tgt = searcher(input_content_src, srcmask, srclens,
                                           input_ids_aux, auxmask, auxlens,
                                           20, tgt['tok2id']['<s>'])
        
        decode_sent = id2word(decoded_data_tgt, tgt)
        n_decoded_sents.append(decode_sent)
        #print('Source content sentence:'+''.join(related_content_tgt[0][1]))
        #print('Decoded data sentence:'+n_decoded_sents[0])
        input_sent = id2word(input_content_src, src)
        initial_inputs.append(input_sent.split())
        pred_sent = n_decoded_sents[0]
        preds.append(pred_sent.split())
        truth_sent = id2word(output_data_tgt, tgt)
        ground_truths.append(truth_sent.split())
        aux_sent = id2word(input_ids_aux, src)
        auxs.append(aux_sent.split())
        rouge_cur = rouge_2(truth_sent, pred_sent)
        rouge_list.append(rouge_cur)
    
    return searcher, rouge_list, initial_inputs, preds, ground_truths, auxs
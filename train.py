import json
import numpy as np
import logging
import argparse
import os
import time
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data
import models
from utils import attempt_load_model, word2id, id2word
import evaluation
from cuda import CUDA


def build_model(src, config):
    # ensure that the parameter initialization values are the same every time we strat training
    torch.manual_seed(config['training']['random_seed'])
    np.random.seed(config['training']['random_seed'])
    
    if config['model']['model_type'] == 'delete_retrieve':
        model = models.DeleteRetrieveModel(vocab_size=len(src['tok2id']), pad_id=src['tok2id']['<pad>'], config=config)
    else:
        model = models.DeleteModel(vocab_size=len(src['tok2id']), pad_id=src['tok2id']['<pad>'], config=config)
    return model


def train(config, working_dir):
    # load data
    src = data.gen_train_data(src=config['data']['src'], attribute_vocab=config['data']['attribute_vocab'], config=config)
    src_dev, tgt_dev = data.gen_dev_data(src=config['data']['src_dev'], tgt=config['data']['tgt_dev'],
                                         attribute_vocab=config['data']['attribute_vocab'], config=config)
    src_truth, tgt_truth = data.gen_dev_data(src=config['data']['src_truth'], tgt=config['data']['tgt_truth'],
                                             attribute_vocab=config['data']['attribute_vocab'], config=config)
    logging.info('Reading data done!')
    
    # build model
    model = build_model(src, config)
    logging.info('MODEL HAS %s params' %  model.count_params())
    
    # get most recent checkpoint
    model, start_epoch = attempt_load_model(model=model, checkpoint_dir=working_dir)
    
    # initialize loss criterion
    weight_mask = torch.ones(len(src['tok2id']))
    weight_mask[src['tok2id']['<pad>']] = 0
    loss_criterion = nn.CrossEntropyLoss(weight=weight_mask)
    
    if CUDA:
        model = model.cuda()
        weight_mask = weight_mask.cuda()
        loss_criterion = loss_criterion.cuda()
        
    # initialize optimizer
    if config['training']['optimizer'] == 'adam':
        lr = config['training']['learning_rate']
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif config['training']['optimizer'] == 'sgd':
        lr = config['training']['learning_rate']
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif config['training']['optimizer']=='adadelta':
        lr = config['training']['learning_rate']
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    else:
        raise NotImplementedError("Learning method not recommend for task")
    
    
    # start training
    start_since_last_report = time.time()
    losses_since_last_report = []
    best_metric = 0.0
    cur_metric = 0.0    # log perplexity or BLEU
    dev_loss = 0.0
    num_batches = len(src['content']) // config['data']['batch_size']

    for epoch in range(start_epoch, config['training']['epochs']):
        if cur_metric > best_metric:
            # rm old checkpoint
            for ckpt_path in glob.glob(working_dir + '/model.*'):
                os.system("rm %s" % ckpt_path)
            # replace with new checkpoint
            torch.save(model.state_dict(), working_dir + '/model.%s.ckpt' % epoch)
    
            best_metric = cur_metric
    
        for i in range(0, len(src['content']), config['data']['batch_size']):
            batch_idx = i // config['data']['batch_size']
            
            # generate current training data batch
            input_content, input_aux, output = data.minibatch(src, src, i, config['data']['batch_size'],
                                                              config['data']['max_len'], config['model']['model_type'])
            input_content_src, _, srclens, srcmask, _ = input_content
            input_ids_aux, _, auxlens, auxmask, _ = input_aux
            input_data_tgt, output_data_tgt, _, _, _ = output
            
            # train the model with current training data batch
            decoder_logit, decoder_probs = model(input_content_src, srcmask, srclens,
                                                 input_ids_aux, auxmask, auxlens, input_data_tgt)
            # setup the optimizer
            optimizer.zero_grad()
            loss = loss_criterion(decoder_logit.contiguous().view(-1, len(src['tok2id'])),
                                  output_data_tgt.view(-1))
            losses_since_last_report.append(loss.item())
            
            # perform backpropagation
            loss.backward()
            
            # clip gradients            
            _ = nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_norm'])
            
            # update model params
            optimizer.step()
            
            # print out the training information
            if batch_idx % config['training']['batches_per_report'] == 0:
                s = float(time.time() - start_since_last_report)
                wps = (config['data']['batch_size'] * config['training']['batches_per_report']) / s
                avg_loss = np.mean(losses_since_last_report)
                info = (epoch, batch_idx, num_batches, wps, avg_loss, dev_loss, cur_metric)
                logging.info('EPOCH: %s ITER: %s/%s WPS: %.2f LOSS: %.4f DEV_LOSS: %.4f DEV_PERFORM: %.4f' % info)
                start_since_last_report = time.time()
                losses_since_last_report = []

        # start evaluate the model on entire dev set
        logging.info('EPOCH %s COMPLETE. VALIDATING...' % epoch)
        model.eval()
        
        # compute validation loss
        start = time.time()
        logging.info('Computing dev_loss on validation data ...')
        dev_loss = evaluation.evaluate_lpp(model=model, src=src_dev, tgt=src_dev, config=config)
        dev_rouge, decoded_sents = evaluation.evaluate_rouge(model=model, src=tgt_dev, tgt=tgt_dev, config=config)
        logging.info('...done!')
    
        '''
        if args.bleu and epoch >= config['training'].get('bleu_start_epoch', 1):
            cur_metric, edit_distance, precision, recall, inputs, preds, golds, auxs = evaluation.inference_metrics(
                                                            model, src_truth, tgt_truth, config)
            # generate decode dataset
            with open(working_dir + '/auxs.%s' % epoch, 'w') as f:
                f.write('\n'.join(auxs) + '\n')
            with open(working_dir + '/inputs.%s' % epoch, 'w') as f:
                f.write('\n'.join(inputs) + '\n')
            with open(working_dir + '/preds.%s' % epoch, 'w') as f:
                f.write('\n'.join(preds) + '\n')
            with open(working_dir + '/golds.%s' % epoch, 'w') as f:
                f.write('\n'.join(golds) + '\n')
    
            logging.info('eval_precision: ', precision)
            logging.info('eval_recall: ', recall)
            logging.info('eval_edit_distance: ', edit_distance)
            logging.info('eval_bleu: ', cur_metric)
        else:
            # compute model performance on validation set
            logging.info('Computing model performance on validation data ...')
            cur_metric, decoded_results = evaluation.evaluate_lpp_perform(model=model, src=src_truth, 
                                                                          tgt=tgt_truth, config=config)
            # generate decode dataset
            with open(working_dir + '/preds.%s' % epoch, 'w') as f:
                for line in decoded_results:
                    f.write(' ||| '.join(line))
                    f.write('\n')
        '''
        
        
        logging.info('DEV_LOSS: %s. DEV_ROUGE: %s. TIME: %.2fs CHECKPOINTING...' 
                     % (dev_loss, dev_rouge, (time.time() - start)))
        
        # switch back to train mode
        model.train()
        

def predict_unaligned(epoch, model, src, tgt, config, working_dir):
    searcher = models.GreedySearchDecoder(model)
    
    logging.info('Generating alter styled sentences on validation data ...')
    decoded_results = []
    for j in range(0, len(src['data'])):
        # batch_size = 1
        input_content, _, _ = data.minibatch(src, tgt, j, 1, 
                                             config['data']['max_len'], 
                                             config['model']['model_type'], 
                                             is_test=True)
        input_content_src, _, srclens, srcmask, _ = input_content
        
        tgt_dist_measurer = tgt['dist_measurer']
        related_content_tgt = tgt_dist_measurer.most_similar(j)   # list of n seq_str
        # related_content_tgt = source_content_str, target_content_str, target_att_str, idx, score
        
        n_decoded_sents = []
        for i, single_data_tgt in enumerate(related_content_tgt):
            # retrieve related attributes
            input_ids_aux, auxlens, auxmask = word2id(single_data_tgt[2], None, tgt, config['data']['max_len'])
            input_ids_aux = Variable(torch.LongTensor(input_ids_aux))
            auxlens = Variable(torch.LongTensor(auxlens))
            auxmask = Variable(torch.LongTensor(auxmask))
            
            if CUDA:
                input_ids_aux = input_ids_aux.cuda()
                auxlens = auxlens.cuda()
                auxmask = auxmask.cuda()
            
            _, decoded_data_tgt = searcher(model, input_content_src, srcmask, srclens,
                                           input_ids_aux, auxmask, auxlens,
                                           20, tgt['tok2id']['<s>'])
            n_decoded_sents.append(id2word(decoded_data_tgt, tgt))
        decoded_results.append(n_decoded_sents)
#        print('Source content sentence:'+' '.join(related_content_tgt[0][1]))
#        print('Decoded data sentence:'+n_decoded_sents[0])
        
    with open(working_dir + '/transfered.%s' % epoch, 'w') as f:
        for line in decoded_results:
            f.write(' ||| '.join(line))
            f.write('\n')

    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config", required=True)
    parser.add_argument("--bleu", help="do BLEU eval", action='store_true')

    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))
    
    working_dir = config['data']['working_dir']
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    
    config_path = os.path.join(working_dir, 'config.json')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump(config, f)
    
    # set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='%s/train_log' % working_dir)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)
    
    # start training
    train(config, working_dir)
    
    
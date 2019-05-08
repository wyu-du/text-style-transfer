import json
import numpy as np
import logging
import argparse
import os

import torch
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
    if config['model']['model_type'] == 'pointer':
        model = models.PointerModel(vocab_size=len(src['tok2id']), pad_id=src['tok2id']['<pad>'], config=config)
    if config['model']['model_type'] == 'delete':
        model = models.DeleteModel(vocab_size=len(src['tok2id']), pad_id=src['tok2id']['<pad>'], config=config)
    return model


def test(config, working_dir):
    # load data
    src, tok_weights_dict = data.gen_train_data(src=config['data']['src'], tgt=config['data']['tgt'], config=config)
    src_truth, tgt_truth = data.gen_dev_data(src=config['data']['src_truth'], tgt=config['data']['tgt_truth'],
                                             tok_weights_dict=tok_weights_dict, config=config)
    logging.info('Reading data done!')
    
    # build model
    model = build_model(src, config)
    logging.info('MODEL HAS %s params' %  model.count_params())
    
    # load the most recent checkpoint
    model, epoch = attempt_load_model(model=model, checkpoint_dir=working_dir)
    
    if CUDA:
        model = model.cuda()
    
    # start evaluate the model on entire dev set
    model.eval()
    logging.info('Computing model performance on validation data ...')
    
    if args.bleu:
        cur_metric, edit_distance, precision, recall, inputs, preds, golds, auxs = evaluation.inference_bleu(
                                                        model, src_truth, tgt_truth, config)
        # output decode dataset
        with open(working_dir + '/auxs.%s' % epoch, 'w') as f:
            f.write('\n'.join(auxs) + '\n')
        with open(working_dir + '/inputs.%s' % epoch, 'w') as f:
            f.write('\n'.join(inputs) + '\n')
        with open(working_dir + '/preds.%s' % epoch, 'w') as f:
            f.write('\n'.join(preds) + '\n')
        with open(working_dir + '/golds.%s' % epoch, 'w') as f:
            f.write('\n'.join(golds) + '\n')

        logging.info('eval_precision: %f' % precision)
        logging.info('eval_recall: %f' % recall)
        logging.info('eval_edit_distance: %f' % edit_distance)
        logging.info('eval_bleu: %f' % cur_metric)
    else:
        # compute model performance on validation set
        cur_metric, edit_distance, precision, recall, inputs, preds, golds, auxs = evaluation.inference_rouge(
                                                        model, src_truth, tgt_truth, config)
        # output decode dataset
        with open(working_dir + '/auxs.%s' % epoch, 'w') as f:
            f.write('\n'.join(auxs) + '\n')
        with open(working_dir + '/inputs.%s' % epoch, 'w') as f:
            f.write('\n'.join(inputs) + '\n')
        with open(working_dir + '/preds.%s' % epoch, 'w') as f:
            f.write('\n'.join(preds) + '\n')
        with open(working_dir + '/golds.%s' % epoch, 'w') as f:
            f.write('\n'.join(golds) + '\n')
        
        logging.info('eval_precision: %f' % precision)
        logging.info('eval_recall: %f' % recall)
        logging.info('eval_edit_distance: %f' % edit_distance)
        logging.info('eval_rouge: %f' % cur_metric)

    
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
                        filename='%s/test_log' % working_dir)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger('').addHandler(console)
    
    # start training
    test(config, working_dir)
    
    
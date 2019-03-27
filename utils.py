import glob
import os
import torch


def get_latest_ckpt(ckpt_dir):
    ckpts = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    # nothing to load, continue with fresh params
    if len(ckpts) == 0:
        return -1, None
    ckpts = map(lambda ckpt: (
        int(ckpt.split('.')[1]),
        ckpt), ckpts)
    # get most recent checkpoint
    epoch, ckpt_path = sorted(ckpts)[-1]
    return epoch, ckpt_path


def attempt_load_model(model, checkpoint_dir=None, checkpoint_path=None):
    assert checkpoint_dir or checkpoint_path

    if checkpoint_dir:
        epoch, checkpoint_path = get_latest_ckpt(checkpoint_dir)
    else:
        epoch = int(checkpoint_path.split('.')[-2])

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print('Load from %s sucessful!' % checkpoint_path)
        return model, epoch + 1
    else:
        return model, 0


def nested_iter(d):
    for k, v in sorted(d.items()):
        if isinstance(v, dict):
            for ki, vi in nested_iter(v):
                yield k + '|' + ki, vi
        else:
            yield k, v


def config_val_string(config):
    config_items = [kv for kv in nested_iter(config)]
    config_vals = map(lambda x: str(x[1]), config_items)
    return ','.join(config_vals)


def config_key_string(config):
    config_items = [kv for kv in nested_iter(config)]
    config_keys = map(lambda x: str(x[0]), config_items)
    return ','.join(config_keys)

def id2word(decoded_tensor, tgt):
    decoded_array = decoded_tensor.cpu().numpy()
    sent = []
    for i in range(len(decoded_array[0])):
        word = tgt['id2tok'][decoded_array[0, i]]
        if word == '</s>' or word == '<pad>':
            break
        sent.append(word)
    if '<s>' in sent:
        sent.remove('<s>')
    return ' '.join(sent)


def word2id(seq_str, tag, tgt, max_len):
    wid_list = []
    seq_len = 0
    mask = []
    if tag == '<s>':
        wid_list.append(tgt['tok2id']['<s>'])
        words = seq_str.strip().split()
        for word in words:
            if word in tgt['tok2id'].keys():
                wid = tgt['tok2id'][word]
            else:
                wid = tgt['tok2id']['<unk>']
            wid_list.append(wid)
        if len(wid_list) < max_len:
            seq_len = len(wid_list)
            wid_list += (max_len-len(wid_list))*[tgt['tok2id']['<pad>']]
            mask = [0]*seq_len + [1]*(max_len-seq_len)
        else:
            seq_len = max_len
            wid_list = wid_list[:max_len]
            mask = [0]*seq_len
    if tag == '</s>':
#        words = seq_str.strip().split()
        for word in seq_str:
            if word in tgt['tok2id'].keys():
                wid = tgt['tok2id'][word]
            else:
                wid = tgt['tok2id']['<unk>']
            wid_list.append(wid)
        wid_list.append(tgt['tok2id']['</s>'])
        if len(wid_list) < max_len:
            seq_len = len(wid_list)
            wid_list += (max_len-len(wid_list))*[tgt['tok2id']['<pad>']]
            mask = [0]*seq_len + [1]*(max_len-seq_len)
        else:
            seq_len = max_len
            wid_list = wid_list[:max_len]
            mask = [0]*seq_len
    if tag == None:
        words = seq_str.strip().split()
        for word in words:
            if word in tgt['tok2id'].keys():
                wid = tgt['tok2id'][word]
            else:
                wid = 1
            wid_list.append(wid)
        if len(wid_list) < max_len:
            seq_len = len(wid_list)
            wid_list += (max_len-len(wid_list))*[tgt['tok2id']['<pad>']]
            mask = [0]*seq_len + [1]*(max_len-seq_len)
        else:
            seq_len = max_len
            wid_list = wid_list[:max_len]
            mask = [0]*seq_len
    return [wid_list], [seq_len], [mask]
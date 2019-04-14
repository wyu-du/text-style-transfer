import os
import re
from nltk.tokenize import word_tokenize

def load_data(file_path):
    outs = []
    for file in os.listdir(file_path):
        if file[0] != '.':
            with open(file_path+'/'+file, 'r') as f:
                doc = f.read().split('.')
                for line in doc:
                    line = line.strip("[ '\t]")
                    line = re.sub("[\!\/_$%^*(+\"\n]+|[ª+——！-，。？、~@#￥%……&*（）]+", "", line)
                    if len(line.split()) > 2:
                        tokens = word_tokenize(line)
                        out = ' '.join(tokens)
                        outs.append(out)
    return outs
        
data = load_data('data/hp')

train_data = data[:8*len(data)//10]
dev_data = data[8*len(data)//10 : 9*len(data)//10]
test_data = data[9*len(data)//10:]

with open('train.hp.txt', 'w') as f:
    for line in train_data:
        f.write(line+'\n')
        
with open('dev.hp.txt', 'w') as f:
    for line in dev_data:
        f.write(line+'\n')
        
with open('test.hp.txt', 'w') as f:
    for line in test_data:
        f.write(line+'\n')
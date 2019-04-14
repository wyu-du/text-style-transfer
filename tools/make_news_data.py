import re
import json
from nltk.tokenize import word_tokenize

def load_data(file_path):
    outs = []
    with open(file_path, 'r') as f:
        for line in f:
            doc = json.loads(line)
            line = doc["short_description"]
            line = re.sub("[\!\/_$%^*(+\"\n]+|[ª+——！-，。？、~@#￥%……&*（）]+", "", line)
            tokens = word_tokenize(line)
            out = ' '.join(tokens)
            outs.append(out)
    return outs
        
data = load_data('data/News_Category_Dataset_v2.json')

train_data = data[:8*len(data)//10]
dev_data = data[8*len(data)//10 : 9*len(data)//10]
test_data = data[9*len(data)//10:]

with open('train.new.txt', 'w') as f:
    for line in train_data:
        f.write(line+'\n')
        
with open('dev.new.txt', 'w') as f:
    for line in dev_data:
        f.write(line+'\n')
        
with open('test.new.txt', 'w') as f:
    for line in test_data:
        f.write(line+'\n')
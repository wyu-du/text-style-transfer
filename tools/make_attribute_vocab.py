"""
make_attribute_vocab.py 

Use Losgistic Regression to help find the attribute vocabulary

By running this file directly, you will get a .20k file containing all the selected attributes.
By calling the make_attribute method while training/testing the model, you will get a dictionary whose key is token and value is corresponding weight's square in the model.
"""
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import numpy as np

def make_attribute(train_corpus0, train_corpus1, test_corpus0=None, test_corpus1=None):

    # Stopword list (Reference NLTK)
    stopwords = set(['i', 'me', 'my', 'myself', 'us', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])
    
    # Build dataset
    def build_dataset(corpus0, corpus1):
        X = []
        Y = []
        for line in open(corpus0, encoding="utf8"):
            X.append(line)
            Y.append(0)
        for line in open(corpus1, encoding="utf8"):
            X.append(line)
            Y.append(1)
        return X, Y
    
    X_train, Y_train = build_dataset(train_corpus0, train_corpus1)
    
    # Tokenizing text
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    print("X_train.shape：" + str(X_train.shape))
    
    # Training a classifier
    clf = LogisticRegression(random_state=12, solver='lbfgs', multi_class='multinomial')
    clf.fit(X_train, Y_train)
    
    # Evaluating the performance on test set
    if(test_corpus0 != None and test_corpus1 != None):
        X_test, Y_test = build_dataset(test_corpus0, test_corpus1)
        X_test = vectorizer.transform(X_test)
        pred = clf.predict(X_test)
        print("Test accuracy: %.3f" % (np.mean(pred == Y_test)))
    
    # Get the classifier weights
    tok_weights_dict = {}
    for (i, (tok, weight)) in enumerate(zip(vectorizer.get_feature_names(), (clf.coef_**2)[0].tolist())):
        if(tok not in stopwords):
            tok_weights_dict[tok] = weight
    
    return tok_weights_dict
    
if __name__=='__main__':
    # Directory of corpus
    train_corpus0 = './data/yelp/sentiment.train.0'
    train_corpus1 = './data/yelp/sentiment.train.1'
    test_corpus0 = './data/yelp/sentiment.dev.0'
    test_corpus1 = './data/yelp/sentiment.dev.1'
    
    # Number of attributes to be selected
    top_attr_num = 1000
    
    # Directory of attribute file
    attr_file = './data/yelp/dict_attr_top' + str(top_attr_num) + '.20k'
    
    # Run method
    tok_weights_dict = make_attribute(train_corpus0, train_corpus1, test_corpus0=test_corpus0, test_corpus1=test_corpus1)
    
    # Save in file
    with open(attr_file, 'w', encoding='utf8') as f:
        for tok in sorted(tok_weights_dict, key=lambda k: tok_weights_dict[k], reverse=True)[:top_attr_num]:
            if(len(tok) == 0):
                continue
            f.write(tok+'\n')
            
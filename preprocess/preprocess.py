import argparse
import pandas as pd
import numpy as np
import re
import sys
from functools import reduce
import pickle

def ssplit(paragraph):
    '''
    String -> [String]
    '''
    sents = re.findall(u'[^!?。，\ \.\!\?]+[!?。\.\!\?]?', paragraph, flags=re.U)
    return [re.sub(r"[\n\t\s]*", "", s) for s in sents]

def filter_Chinese(text):
    '''
    filter out non-Chinese characters
    '''
    return ''.join(list(filter(lambda c: '\u4e00' <= c <= '\u9fa5',text)))


def padding(doc,max_num_sent,max_length_sent):
    '''
    input is the index form of a document
    [[int]] -> [[int]]
    '''
    padded_doc = []
    for sentence in doc:
        padded_sentence = sentence[:max_length_sent] if len(sentence) >= max_length_sent else sentence + [0] * (max_length_sent - len(sentence))
        padded_doc.append(padded_sentence)
    if len(doc) >= max_num_sent:
        padded_doc = padded_doc[:max_num_sent]
    else:
        padded_doc = padded_doc + [([0]*max_length_sent)] * (max_num_sent - len(padded_doc))
    return padded_doc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sent_length', type=int, default = 20,help="""the length of sentences truncated""")
    parser.add_argument('-sent_num', type=int, default = 15,help="""the number of sentences of a review""")
    parser.add_argument('-train_prop', type=float, default = 0.95)
    args = parser.parse_args()


    data_frame = pd.read_csv('data/train_first.csv')
    predict_frame = pd.read_csv('data/predict_first.csv')
    ########## Step 1 ###########
    #过滤掉非汉字字符,分句
    all_docs = []
    all_labels = []
    for index, row in data_frame.iterrows():
        row_values = row.values
        raw_text = row_values[1]
        label = row_values[2]
        sents = ssplit(raw_text)
        doc = [filter_Chinese(s) for s in sents if filter_Chinese(s)]
        all_docs.append(doc)
        all_labels.append(label)

    predict_sents = []
    predict_ids = []
    for index, row in predict_frame.iterrows():
        row_values = row.values
        ID = row_values[0]
        raw_text = row_values[1]
        sents = ssplit(raw_text)
        predict_sents.append(sents)
        predict_ids.append(ID)

    ########## Step 2 ###########
    #制作char2index和label2index字典，并存储为pickle。
    chars = set([])
    labels = set(all_labels)
    for doc in all_docs:
        for s in doc:
            chars = chars | set(s)

    indices = range(2,len(chars)+2)
    char2index = {char: cid for char, cid in zip(chars,indices)}
    char2index['unk'] = 1
    char2index['pad'] = 0

    indices = range(len(labels))
    label2index = {label: lid for label, lid in zip(labels,indices)}

    pickle.dump(char2index,open('char2index.pickle','wb'))
    pickle.dump(label2index,open('label2index.pickle','wb'))


    ########## Step 3 ###########
    #将doc和label变成index, 然后pad,存储
    docs = []
    labels = []
    for doc,label in zip(all_docs,all_labels):
        label = label2index[label]
        doc_idx = []
        for s in doc:
            idx = [char2index[c] if c in char2index else 1 for c in ' '.join(s).split()]
            doc_idx.append(idx)
        docs.append(padding(doc_idx,args.sent_num,args.sent_length))
        labels.append(label)

    length = len(labels)
    pickle.dump((docs[:int(length*args.train_prop)],labels[:int(length*args.train_prop)]),open('train_preprocessed.pickle','wb'))
    pickle.dump((docs[int(length*args.train_prop):],labels[int(length*args.train_prop):]),open('test_preprocessed.pickle','wb'))
    pickle.dump((docs[:200],labels[:200]),open('tiny_preprocessed.pickle','wb')) #小规模数据用于修改模型代码时测试可行性

    ###### Predict Data #######
    sents = []
    ids  = []
    for sent,ID in zip(predict_sents,predict_ids):
        doc_idx = []
        for s in doc:
            idx = [char2index[c] if c in char2index else 1 for c in ' '.join(s).split()]
            doc_idx.append(idx)
        sents.append(padding(doc_idx,args.sent_num, args.sent_length))
        ids.append(ID)
    pickle.dump((sents,ids),open('predict_preprocessed.pickle','wb'))
    ############## 存储参数，用于训练 ##########
    config = {}
    config['vocab_size'] = len(char2index)
    config['num_labels'] = len(label2index)
    config['sent_length'] = args.sent_length
    config['sent_num'] = args.sent_num
    pickle.dump(config,open('config_preprocess.pickle','wb'))

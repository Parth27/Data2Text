import pandas as pd
import numpy as np
import re
import os
from collections import Counter
import pickle
import codecs
import pickle
from sklearn.preprocessing import LabelEncoder
import spacy

ids = []
sentences = []
titles = []
link = []
with open("Datasets/wikipedia-biography-dataset/train/train.id") as f:
    ids.extend(f.read().split('\n'))
f1 = open("Datasets/wikipedia-biography-dataset/train/train.sent",encoding = 'utf-8')
for line in f1:
    sentences.append(line)
f1 = open("Datasets/wikipedia-biography-dataset/train/train.title",encoding = 'utf-8')
for line in f1:
    titles.append(line)
f1 = open("Datasets/wikipedia-biography-dataset/train/train.nb",encoding = 'utf-8')
for line in f1:
    link.append(line)
    
ids = ids[:-1]
f1.close()

i = 0
j = 0
text = []
for i in link:
    text.append(sentences[j:j+int(i)])
    j = j+int(i)

for i in range(len(text)):
    temp = ''
    for j in text[i]:
        temp += j.replace('\n',' ')
    
    text[i] = re.findall(r'[a-zA-Z1-9.,]+',temp)

def build_vocab(data,len_vocab):
    words=[]
    for i in range(len(data)):
        words.extend(data[i])
   
    words=Counter(words).most_common(len_vocab)
    
    vocab={}
    for i in range(len(words)):
        vocab[words[i][0]]=i
        
    vocab['UNK']=len(vocab)
    vocab['<start>'] = len(vocab)
    vocab['<stop>'] = len(vocab)

    reverse_vocab={}
    for i in range(len(words)):
        reverse_vocab[i]=words[i][0]
        
    reverse_vocab[len(reverse_vocab)]='UNK'
    reverse_vocab[len(reverse_vocab)]='<start>'
    reverse_vocab[len(reverse_vocab)]='<stop>'

    return vocab,reverse_vocab

vocab,reverse_vocab = build_vocab(text,50000)
with open("Vocabulary.p",'wb') as fp:
    pickle.dump(vocab,fp,protocol=pickle.HIGHEST_PROTOCOL)
with open("Reverse_vocab.p",'wb') as fp:
    pickle.dump(reverse_vocab,fp,protocol=pickle.HIGHEST_PROTOCOL)

tables = []
f1 = open("C:/Users/pdiwanj/Datasets/wikipedia-biography-dataset/test/test.box",encoding = 'utf-8')
for line in f1:
    tables.append(line.split('\t'))

fields = []
inputs = []
for i in range(len(tables)):
    fields.append([x.split(':')[0] for x in tables[i]])
    inputs.append([x.split(':')[1] for x in tables[i]])

with open("fields.p","wb") as fp:
    pickle.dump(fields,fp,protocol=pickle.HIGHEST_PROTOCOL)
with open("inputs.p","wb") as fp:
    pickle.dump(inputs,fp,protocol=pickle.HIGHEST_PROTOCOL)

del sentences
del ids
del tables
del link

for i in range(len(text)):
    text[i].insert(0,'<start>')
    text[i].insert(len(text[i]),'<stop>')

for i in range(len(inputs)):
    index = [x for x in range(len(inputs[i])) if inputs[i][x] in ('<none>',"","'","''"," ",",",", ","-","_")]
    
    remove_f = [fields[i][x] for x in index]
    for j in remove_f:
        fields[i].remove(j)
        
    inputs[i] = [x for x in inputs[i] if x not in ('<none>',"","'","''"," ",",",", ","-","_")]
    
del index
field = []
pos = []

for j in fields:
    temp2 = []
    for i in j:
        temp = i.split('_')
        temp1 = re.split(r'_+[0-9]+',i)
        temp2.append((temp1[0],int(temp[-1])))
    
    field.append(temp2)

all_fields = []
for i in range(len(field)):
    for j in field[i]:
        all_fields.append(j[0])
        
all_fields = list(set(all_fields))

le = LabelEncoder()

labels = le.fit_transform(all_fields)
labels = list(zip(all_fields,labels))
labels = dict(labels)

labeled_fields = []
temp = []
for j in range(len(field)):
    labeled_fields.append([(labels[x[0]],x[1]) for x in field[j]])

def prep_data(fields,inputs,outputs,labels):
    X_fields = []
    X_inputs = []
    y = []
    max_output_len = max([len(outputs[x]) for x in range(len(outputs))])
    
    max_input_len = max([len(fields[x]) for x in range(len(fields))])
    
    max_output_len = 6332
    max_input_len = 694
    
    for i in range(len(fields)):
        if len(fields[i])>max_input_len or len(outputs[i])>max_output_len:
            continue
            
        pads = [(len(labels),0) for x in range(max_input_len-len(fields[i]))]
        X_fields.append(pads+fields[i])
        
        pads = [vocab['<pad>'] for x in range(max_input_len-len(inputs[i]))]
        
        X_inputs.append(pads+[vocab[x] if x in vocab.keys() else vocab['UNK'] for x in inputs[i]])
        
        pads = [vocab['<pad>'] for x in range(max_output_len-len(outputs[i]))]
        
        y.append([vocab[x] if x in vocab.keys() else vocab['UNK'] for x in text[i]]+pads)
        
    X_fields = np.array(X_fields)
    X_inputs = np.array(X_inputs)
    y = np.array(y)
    return X_fields,X_inputs,y

X_fields,X_inputs,y = prep_data(labeled_fields,inputs,text,labels)

with open("train_inputs.p","wb") as fp:
    pickle.dump(X_inputs,fp,protocol=pickle.HIGHEST_PROTOCOL)
with open("train_outputs.p","wb") as fp:
    pickle.dump(y,fp,protocol=pickle.HIGHEST_PROTOCOL)
with open("train_fields.p","wb") as fp:
    pickle.dump(X_fields,fp,protocol=pickle.HIGHEST_PROTOCOL)
with open("final_labels.p","wb") as fp:
    pickle.dump(labels,fp,protocol=pickle.HIGHEST_PROTOCOL)
    
fp.close()

model = spacy.load('en_core_web_md')

embeddings = []
for i in vocab.keys():
    embeddings.append(model(i).vector)

np.save("Embeddings",embeddings,allow_pickle=False)

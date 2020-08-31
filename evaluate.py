import pandas as pd
import numpy as np
import re
import os
import tensorflow as tf
from collections import Counter
import pickle
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu

with open("Vocabulary.p",'rb') as fp:
    vocab = pickle.load(fp)
with open("Reverse_vocab.p",'rb') as fp:
    reverse_vocab = pickle.load(fp)

embeddings = np.load("Embeddings.npy")

with open("final_labels.p",'rb') as fp:
    labels = pickle.load(fp)
    
fp.close()

y_inputs = np.load('y_inputs.npy')
y_outputs = np.load('y_outputs.npy')
X_fields = np.load('X_fields.npy')
X_pos = np.load('X_pos.npy')
test_inputs = np.load('bio_test_inputs.npy')

y_inputs = y_inputs[:1000]
y_outputs = y_outputs[:1000]
X_fields = X_fields[:1000]
X_pos = X_pos[:1000]
test_inputs = test_inputs[:1000]

def bleu_score(X,y):
    return sentence_bleu(y,X)

def rouge_score(X,y,evaluator):
    target = " ".join(y)
    output = " ".join(X)

    return evaluator.get_scores(output,target)

model = tf.keras.models.load_model('model_6.h5')

bleu = []
rouge = []

evaluator = Rouge()
batch = 0
samples = []
while(batch+50<=test_inputs.shape[0]):
    out = model.predict([test_inputs[batch:batch+50].reshape(50,-1),X_fields[batch:batch+50].reshape(50,-1),
                     X_pos[batch:batch+50].reshape(50,-1),y_inputs[batch:batch+50].reshape(50,-1)])
    for i in range(len(out)):
        samples.append(np.argmax(out[i],axis=1).reshape(-1))
    batch += 50

for i in range(len(samples)):
    index = min([x for x in range(len(samples[i])) if samples[i][x]==vocab['<pad>']])

    prediction = [reverse_vocab[x] for x in samples[i][:index]]

    index = min([x for x in range(len(y_outputs[i])) if y_outputs[i][x]==vocab['<pad>']])

    target = [reverse_vocab[x] for x in y_outputs[i,:,0]]

    bleu.append(bleu_score(prediction,target))

    score = rouge_score(prediction,target,evaluator)
    rouge.append((score[0]['rouge-1']['r'] + score[0]['rouge-1']['f'] + score[0]['rouge-1']['p'])/3)

    if i%100==0:
        print(bleu[-1],rouge[-1])

final_score = [sum(bleu)/len(bleu),sum(rouge)/len(rouge)]
print("Final score: Bleu = "+str(final_score[0])+", Rouge = "+str(final_score[1]))

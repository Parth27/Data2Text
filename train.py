import pandas as pd
import numpy as np
import re
import os
import tensorflow as tf
from collections import Counter
import pickle

with open("Vocabulary.p",'rb') as fp:
    vocab = pickle.load(fp)
with open("Reverse_vocab.p",'rb') as fp:
    reverse_vocab = pickle.load(fp)

embeddings = np.load("Embeddings.npy")

with open("final_labels.p",'rb') as fp:
    labels = pickle.load(fp)
    
fp.close()
outputs = np.load('final_outputs.npz')
train_outputs = outputs['arr_0']

y_inputs = np.copy(train_outputs[:,:-1])
y_outputs = np.copy(train_outputs[:,1:]).reshape(train_outputs.shape[0],train_outputs.shape[1]-1,1)

print(y_outputs.shape,y_inputs.shape)
del train_outputs
del outputs

inputs = np.load('small_train_inputs.npz')
train_inputs = inputs['arr_0']

del inputs

fields = np.load('small_train_fields.npz')
train_fields = fields['arr_0']

X_fields = np.copy(train_fields[:,:,0]).reshape(train_fields.shape[0],train_fields.shape[1])
X_pos = np.copy(train_fields[:,:,1]).reshape(train_fields.shape[0],train_fields.shape[1])

del train_fields
print(X_fields.shape,X_pos.shape)

del fields

y_inputs = y_inputs[:100000]
y_outputs = y_outputs[:100000]
train_inputs = train_inputs[:100000]
X_fields = X_fields[:100000]
X_pos = X_pos[:100000]

for i in range(train_inputs.shape[0]):
  minimum = min([x for x in range(len(X_fields[i].reshape(X_fields[i].shape[0]))) if X_fields[i,x]!=6306])
  train_inputs[i][:minimum] = [vocab['<pad>'] for j in range(minimum)]

#bio model
n_hidden = 100
f_hidden = 50
input_shape = 694
embedding_size = 300
field_embedding_size = len(labels)+1
max_pos = 595
decoder_input_size = y_outputs.shape[1]

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#encoder_input = tf.keras.layers.Input(shape=(None,embedding_size))

encoder_input = tf.keras.layers.Input(shape=(input_shape,))
field_encoder_input = tf.keras.layers.Input(shape=(input_shape,))
positions_input = tf.keras.layers.Input(shape=(input_shape,))

field_embedding_layer = tf.keras.layers.Embedding(field_embedding_size,10)
position_embedding_layer = tf.keras.layers.Embedding(max_pos,3)

embedding_layer = tf.keras.layers.Embedding(len(vocab),embedding_size,embeddings_initializer=tf.keras.initializers.Constant(embeddings))
encoder_lstm = tf.keras.layers.LSTM(n_hidden,return_sequences = True,return_state = True,dropout = 0.35,recurrent_dropout = 0.35)

field_embeddings = field_embedding_layer(field_encoder_input)
position_embeddings = position_embedding_layer(positions_input)

field_lstm = tf.keras.layers.LSTM(f_hidden,return_sequences = True,return_state = True)

decoder_lstm = tf.keras.layers.LSTM(n_hidden,return_sequences = True,return_state = True,dropout = 0.3,recurrent_dropout = 0.3)

field_outputs,state_a,state_c = field_lstm(tf.keras.layers.concatenate([field_embeddings,position_embeddings]))

embedded_sequence = embedding_layer(encoder_input)
embedded_sequence = tf.keras.layers.Dropout(0.3)(embedded_sequence)
encoder_outputs,a,c = encoder_lstm(tf.keras.layers.concatenate([embedded_sequence,field_outputs]))

#out = dualAttention(n_hidden,encoder_outputs,field_embeddings,embedded_sequence)

attention = tf.keras.layers.Dense(1,activation='tanh')(encoder_outputs)
attention = tf.squeeze(attention,[2])
attention_vector = tf.keras.layers.Dense(n_hidden,activation='softmax')(attention)

encoder_states = [attention_vector,c]

decoder_input = tf.keras.layers.Input(shape=(None,))
decoder_embeddings = embedding_layer(decoder_input)
decoder_outputs,_,_ = decoder_lstm(decoder_embeddings,initial_state=encoder_states)

dense = tf.keras.layers.Dense(len(vocab),activation='softmax')
decoder_outputs = dense(decoder_outputs)

model = tf.keras.models.Model([encoder_input,field_encoder_input,positions_input,decoder_input],decoder_outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.006,beta_1 = 0.99)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',metrics=['accuracy'])

print(model.summary())

#train model
num_epochs = 24
batch_size = 100

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

tf.keras.backend.set_session(sess)

with tf.device('/gpu:0'):
  hist = model.fit([train_inputs,X_fields,X_pos,y_inputs],
            y_outputs,batch_size=batch_size,epochs=num_epochs,validation_split = 0.1)

model.save('model.h5')

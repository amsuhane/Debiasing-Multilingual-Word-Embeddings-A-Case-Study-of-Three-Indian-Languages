import sys
import gc
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix 
import random
import tensorflow as tf
import pickle
from keras.layers import Layer
import keras.backend as K
import pickle

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()

def train (num):
  X = []
  X.extend(gendered_split['M']['X'])
  X.extend(gendered_split['F']['X'])

  y = []
  y.extend(gendered_split['M']['y'])
  y.extend(gendered_split['F']['y'])

  tokenizer = Tokenizer(num_words= MAX_NB_WORDS,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',   lower=True,   split=" ",)  
  tokenizer.fit_on_texts(X)
  X = tokenizer.texts_to_sequences(X)
  X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
      
  embedding_matrix = np.zeros((MAX_NB_WORDS, EMBEDDING_DIM))
  c1 = 0
  c2 = 0
  for word,i in tokenizer.word_index.items():
    try:
      embedding_matrix[i]=embedding[word]
      c1 = c1 +1
    except:
      c2 = c2 +1
      #continue    
  print(c1,c2)
  embedding_matrix = np.nan_to_num(embedding_matrix)  
  mapin={}
  for i in range(len(s)):
    mapin[s[i]]=i
      
  print(mapin)
  for i in range(len(y)):
    y[i]=mapin[y[i]]

  y=np.array(y)
  y=np.reshape(y,(-1,1))   
  y=to_categorical(y)
  NUM_TITLES=len(s)    

  gender_list = []
  gender_list.extend(np.ones((len(gendered_split['M']['y']),)))
  gender_list.extend(-1*np.ones((len(gendered_split['F']['y']),)))
  print(len(gender_list))

  random.seed(num)
  np.random.seed(num)
  tf.random.set_seed(num)

  c = list(zip(X, y, gender_list))
  random.shuffle(c)
  X, y, gender_list = zip(*c)
  split = int(0.75*len(X))
  X_train = np.array(X[:split])
  X_test = np.array(X[split:])

  Y_train = np.array(y[:split])
  Y_test = np.array(y[split:])

  gen_train = np.array(gender_list[:split])
  gen_test = np.array(gender_list[split:])

  X = np.array(X)
  y = np.array(y)

  model = Sequential()
  model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1],weights=[embedding_matrix],trainable=False))
  model.add(Bidirectional(LSTM(50)))# ,return_sequences=True))
  # model.add(attention())
  model.add(Dense(NUM_TITLES, activation='softmax'))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  history = model.fit(X_train, Y_train, epochs=1, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])

  y_pred=model.predict(X_test)
  Y_test2 = np.argmax(Y_test, axis=1) # Convert one-hot to index
  y_pred2 = np.argmax(y_pred, axis=1) # Convert one-hot to index

  Y_test_male = []
  Y_test_female = []
  Y_pred_male = []
  Y_pred_female = []

  for i in range(len(gen_test)):
    if gen_test[i]==1: #Male
      Y_test_male.append(Y_test2[i])
      Y_pred_male.append(y_pred2[i])
    elif gen_test[i]==-1: #Female
      Y_test_female.append(Y_test2[i])
      Y_pred_female.append(y_pred2[i])

  Y_test_male = np.array(Y_test_male)
  Y_test_female = np.array(Y_test_female)

  Y_pred_male = np.array(Y_pred_male)
  Y_pred_female = np.array(Y_pred_female)

  acc = {'M': [], 'F':[]}
  print(classification_report(Y_test_male, Y_pred_male))   
  cm = confusion_matrix(Y_test_male, Y_pred_male)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  acc['M'] = cm.diagonal()

  print(classification_report(Y_test_female, Y_pred_female))   
  cm = confusion_matrix(Y_test_female, Y_pred_female)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  acc['F'] = cm.diagonal()
    
  m_arr = np.array(acc['M'])
  f_arr = np.array(acc['F'])
    
  print(m_arr.mean())
  print(f_arr.mean())
  diff = np.abs(m_arr-f_arr)
  print(diff.mean())
  return acc

MAX_NB_WORDS = 300000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 300
NUM_TITLES=0
epochs = 100
batch_size = 32

pre=sys.argv[1]
l1=sys.argv[2]
l2=sys.argv[3]

folder_dir = {
  'LID':'LID',
  'EQR':'EQR',
  'fin':'fasttext',
  'LDD':'LDD'
}

bio_f = f'data/BIOS/processed_bio_{l1}.pkl'
emb_f = f'data/embedding/{folder_dir[pre]}/{pre}_emb_bilingual_{l1}_{l2}'

with open(emb_f,"rb") as F:
  embedding=pickle.load(F)
with open(bio_f,"rb") as F:
  data=pickle.load(F)

s=set()
for sent in data:
    s.add(sent['title_en'])
s=list(s)
print(len(s),s)

gendered_split={'M':{},'F':{}}
for gender in ['M','F']:
    gendered_split[gender]['X']=[]
    gendered_split[gender]['y']=[]
for i,sent in enumerate(data):
    gendered_split[sent['gender']]['X'].append(sent['bio'])
    gendered_split[sent['gender']]['y'].append(sent['title_en'])

res = []
for i in range(5):
  res.append(train((200*i)%39))

op = []
for r in res:
  m_arr = np.array(r['M'])
  f_arr = np.array(r['F'])
  diff = np.abs(m_arr-f_arr)
  temp = np.array([m_arr.mean(),f_arr.mean(),diff.mean()])
  op.append(temp)

op = np.array(op)
print(op)
print(pre,l1,l2,op.mean(axis=0))

print("Saving Answers HERE")
male=op.mean(axis=0)[0]
female=op.mean(axis=0)[1]
diff=op.mean(axis=0)[2]
with open("ans_LID_EQR_Cross.csv","a") as F:
  F.write(pre+","+l1+","+l2+","+str(male)+","+str(female)+","+str(diff)+"\n")

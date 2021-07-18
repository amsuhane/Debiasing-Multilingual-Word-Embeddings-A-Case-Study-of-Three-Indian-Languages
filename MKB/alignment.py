import numpy as np
import pickle
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from gensim import utils
import gensim
import sys
import os
import gc
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.layers import LSTM,Input,concatenate,Dense,Dot
from keras.models import Model
# from keras.utils import to_categorical
from keras.layers import Embedding

pre1=sys.argv[1]
pre2=sys.argv[2]
l1=sys.argv[3]
l2=sys.argv[4]

class FastVector:
    def __init__(self, vector_file='', transform=None):
        self.word2id = {}
        self.id2word = []

        print('reading word vectors from %s' % vector_file)
        with open(vector_file, 'r') as f:
            (self.n_words, self.n_dim) = \
                (int(x) for x in f.readline().rstrip('\n').split(' '))
            self.embed = np.zeros((self.n_words, self.n_dim))
            print(self.n_words, self.n_dim)
            for i, line in enumerate(f):
              elems = line.rstrip('\n').split(' ')
              self.word2id[elems[0]] = i
              self.id2word.append(elems[0])
              try:
                  self.embed[i] = elems[1:self.n_dim+1]
              except:                  
                  print(i)

        if transform is not None:
            print('Applying transformation to embedding')
            self.apply_transform(transform)

    def apply_transform(self, transform):
        transmat = np.loadtxt(transform) if isinstance(transform, str) else transform
        self.embed = np.matmul(self.embed, transmat)

    def export(self, outpath):

        fout = open(outpath, "w")

        fout.write(str(self.n_words) + " " + str(self.n_dim) + "\n")
        for token in self.id2word:
            vector_components = ["%.6f" % number for number in self[token]]
            vector_as_string = " ".join(vector_components)

            out_line = token + " " + vector_as_string + "\n"
            fout.write(out_line)

        fout.close()


    @classmethod
    def cosine_similarity(cls, vec_a, vec_b):
        """Compute cosine similarity between vec_a and vec_b"""
        return np.dot(vec_a, vec_b) / \
            (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def __contains__(self, key):
        return key in self.word2id

    def __getitem__(self, key):
        return self.embed[self.word2id[key]]

def my_save_word2vec_format(fname, vocab, vectors, binary=True, total_vec=2):
  if not (vocab or vectors):
      raise RuntimeError("no input")
  if total_vec is None:
      total_vec = len(vocab)
  vector_size = vectors.shape[1]
  assert (len(vocab), vector_size) == vectors.shape
  with utils.smart_open(fname, 'wb') as fout:
      print(total_vec, vector_size)
      fout.write(utils.to_utf8("%s %s\n" % (total_vec, vector_size)))
      # store in sorted order: most frequent words at the top
      for word, row in vocab.items():
          if binary:
              row = row.astype(REAL)
              fout.write(utils.to_utf8(word) + b" " + row.tostring())
          else:
              fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join(repr(val) for val in row))))
def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []

    for (source, target) in bilingual_dictionary:
        if source in source_dictionary and target in target_dictionary:
            if np.unique(np.isnan(source_dictionary[source]))[0]==False and np.unique(np.isnan(target_dictionary[target]))[0]==False : 
              source_matrix.append(source_dictionary[source])
              target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)

folder_name = {pre1:pre1, pre2:pre2}
if pre1=='fin':
    folder_name[pre1]='fasttext'
if pre2=='fin':
    folder_name[pre2]='fasttext'

emb_l1 = pickle.load(open(f'../data/embedding/{folder_name[pre1]}/{pre1}_{l1}_fast_text_300.pkl','rb'))  
m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=50)
m.vocab = emb_l1
m.vectors = np.array(list(emb_l1.values()))
my_save_word2vec_format(binary=False, fname=f'../data/aligned_files/alligned_{l1}.txt', total_vec=len(emb_l1), vocab=m.vocab, vectors=m.vectors)              
en_dictionary = FastVector(vector_file=f'../data/aligned_files/alligned_{l1}.txt')

# fin_bn_fast_text_300.pkl  LDD_fin_bn_fast_text_300.pkl
emb_l2 = pickle.load(open(f'../data/embedding/{folder_name[pre2]}/{pre2}_{l2}_fast_text_300.pkl','rb')) 
m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=50)
m.vocab = emb_l2
m.vectors = np.array(list(emb_l2.values()))
my_save_word2vec_format(binary=False, fname=f'../data/aligned_files/alligned_{l2}.txt', total_vec=len(emb_l2), vocab=m.vocab, vectors=m.vectors)              
hi_dictionary = FastVector(vector_file=f'../data/aligned_files/alligned_{l2}.txt')

en_words = set(en_dictionary.word2id.keys())
hi_words = set(hi_dictionary.word2id.keys())
overlap = list(en_words & hi_words)
bilingual_dictionary = [(entry, entry) for entry in overlap]


with open('../data/MKB/biling_dict_%s_%s.pkl'%(l2,l1),'rb') as F:
    dd=pickle.load(F)

bilingual_dictionary_add = []
for key in dd.keys():
    bilingual_dictionary_add.append((dd[key], key))

bilingual_dictionary.extend(bilingual_dictionary_add)

# form the training matrices
source_matrix, target_matrix = make_training_matrices(hi_dictionary, en_dictionary, bilingual_dictionary)

# learn and apply the transformation
transform = learn_transformation(source_matrix, target_matrix)
hi_dictionary.apply_transform(transform)

dic = {}
for w in en_dictionary.id2word:
    dic[w] = en_dictionary[w]

for w in hi_dictionary.id2word:
    dic[w] = hi_dictionary[w]
gc.collect()

en_words = set(en_dictionary.word2id.keys())
hi_words = set(hi_dictionary.word2id.keys())
overlap = list(en_words & hi_words)
bilingual_dictionary = [(entry, entry) for entry in overlap]

en_sentences,hi_sentences=[],[]

print("../data/MKB/train_"+l1+"_"+l2+"_"+l2.upper()+"."+l2)
with open("../data/MKB/train_"+l1+"_"+l2+"_"+l2.upper()+"."+l2,"r") as F:
    for line in F:
        hi_sentences.append(line)
print("../data/MKB/train_"+l1+"_"+l2+"_"+l1.upper()+"."+l1)
with open("../data/MKB/train_"+l1+"_"+l2+"_"+l1.upper()+"."+l1,"r") as F:
    for line in F:
        en_sentences.append(line)

bilingual_dictionary_add = []

for key in dd.keys():
    bilingual_dictionary_add.append((dd[key], key))
l = []
for h,e in bilingual_dictionary:
    try:
        en_vector = en_dictionary[e]
        hi_vector = hi_dictionary[h]
        l.append(FastVector.cosine_similarity(en_vector, hi_vector))
    except:
        pass
sum2 = []
import math
for x in l:
    if (math.isnan(x)):
        continue
    else:
        sum2.append(x)

bilingual_dictionary.extend(bilingual_dictionary_add)
from keras.preprocessing.text import Tokenizer
t=Tokenizer()

t.fit_on_texts(en_sentences+hi_sentences)
embedding_matrix=np.zeros((len(t.word_index.items())+1,300))
cnt=0
for w,i in t.word_index.items():
    try:
        embedding_matrix[i]=en_dictionary[w]
    except:
        try:
            embedding_matrix[i]=hi_dictionary[w]
        except:
            cnt+=1
            embedding_matrix[i]=np.array([0]*300)
print("OOV:",cnt)

inp_hi=t.texts_to_sequences(hi_sentences)
inp_en=t.texts_to_sequences(en_sentences)
inp_hi=pad_sequences(inp_hi,maxlen=20,padding='post')
inp_en=pad_sequences(inp_en,maxlen=20,padding='post')
inp=[[inp_en[i],inp_hi[i],1.00] for i in range(len(en_sentences))]

inp2=[[inp_en[i],random.choice(inp_hi[:i-5]),0.00] for i in range(6,len(hi_sentences))]
inp.extend(inp2)
inp2=[[inp_en[i],random.choice(inp_hi[:i-5]),0.00] for i in range(6,len(hi_sentences))]
inp.extend(inp2)
random.shuffle(inp)

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

hidden_size = 300
input_size = 10
SPLIT=int(len(inp)*0.25)

X_test=inp[10000:10100]
X=inp[:10000]
y_test,y=[],[]
print(len(X))
for i in X:
    y.append(int(i[2]))
for i in X_test:
    y_test.append(int(i[2]))
y=np.array(y)
y_test=np.array(y_test)
X1,X2,X1_test,X2_test=[],[],[],[]

for i in range(len(X)):
    X1.append(np.array(X[i][0]))
    X2.append(np.array(X[i][1]))
for i in range(len(X_test)):
    X1_test.append(np.array((X_test[i][0])))
    X2_test.append(np.array(X_test[i][1]))

X1=np.asarray(X1)
X2=np.asarray(X2)
X1_test=np.asarray(X1_test)
X2_test=np.asarray(X2_test)

embedding_layer = Embedding(len(t.word_index.items())+1,
                            300,
                            weights=[embedding_matrix],
                            input_length=20,
                            trainable=False)
input1 = Input(shape=(20,))
input1_emb=embedding_layer(input1)
x1 = LSTM(50)(input1_emb)
input2 = Input(shape=(20,))
input2_emb=embedding_layer(input2)
x2 = LSTM(50)(input2_emb)
x = Dot(axes=1)([x1,x2])
# x = Dense(50)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[input1,input2], outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc',f1_m,precision_m, recall_m])

print(len(X1),len(X2))
model.fit([X1,X2],y,epochs=20,validation_split=0.2)
score=0
tot=0

for i in range(len(X1_test)):
    if y_test[i]==0:
        continue
    if tot==100:
        break
    tot+=1
    sims=[]
    for j in range(len(X1_test[:100])):
        sims.append( (model.predict([ X1_test[i:i+1],X2_test[j:j+1] ])[0][0],j) )
    sims.sort()
    sims.reverse()
    print(i,[i[1] for i in sims[:10]])
    if i in [i[1] for i in sims[:10]]:
        score+=1

if not os.path.isfile('Alignment_Runner.csv'):
    with open("Alignment_Runner.csv","w") as F:
        F.write('pre1'+","+'pre2'+","+'l1'+","+'l2'+","+'score'+'\n')
with open("Alignment_Runner.csv","a") as F:
    F.write(pre1+","+pre2+","+l1+","+l2+","+str(score/tot)+'\n')
print(l1+l2,score/tot)

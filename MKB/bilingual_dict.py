import pickle
import os
from googletrans import Translator
translator = Translator()

os.chdir('../data/MKB/')

# Load pairs
pairs_hi= [] 
with open('hi-en.txt', 'r') as f:
  for i, line in enumerate(f):
    elems = line.rstrip('\n').split(' ')[0].split()
    pairs_hi.append((elems[0], elems[1]))  
pairs_hi2 = []
with open('en-hi.txt', 'r') as f:
  for i, line in enumerate(f):
    elems = line.rstrip('\n').split(' ')[0].split()
    pairs_hi2.append((elems[1], elems[0]))  
pairs_bn = []
with open('bn-en.txt', 'r') as f:
  for i, line in enumerate(f):
    elems = line.rstrip('\n').split(' ')[0].split()
    pairs_bn.append((elems[0], elems[1]))  
pairs_bn2 = []
with open('en-bn.txt', 'r') as f:
  for i, line in enumerate(f):
    elems = line.rstrip('\n').split(' ')[0].split()
    pairs_bn2.append((elems[1], elems[0]))

dic = {}
for k in pairs_hi :
  if k[1] not in dic.keys():
    dic[k[1]]= {'hi': [], 'bn': [],'te':[]}
  dic[k[1]]['hi'].append(k[0])
for k in pairs_hi2:
  if k[1] not in dic.keys():
    dic[k[1]]= {'hi': [], 'bn': [],'te':[]}
  dic[k[1]]['hi'].append(k[0])
for k in pairs_bn2 :
  if k[1] not in dic.keys():
    dic[k[1]]= {'hi': [], 'bn': [],'te':[]}
  dic[k[1]]['bn'].append(k[0])
for k in pairs_bn :
  if k[1] not in dic.keys():
    dic[k[1]]= {'hi': [], 'bn': [],'te':[]}
  dic[k[1]]['bn'].append(k[0])

en_dic = {}
for k,v in dic.items():
  en_dic[k] = {'hi': list(set(v['hi'])), 'bn': list(set(v['bn']))}
en_hi = {}
for k,v in en_dic.items():
  if len(v['hi'])>0:
    if len([v['hi']])==1 and (v['hi'][0]==k):
      continue
    else:
      en_hi[k] = v['hi'][0]
pickle.dump(en_hi,open('biling_dict_en_hi.pkl','wb'))
hi_en = {}
for k,v in en_dic.items():    
  if len(v['hi'])>0:
    if len([v['hi']])==1 and (v['hi'][0]==k):
      continue
    else:
      for i in v['hi']:
        hi_en[i] = k
pickle.dump(hi_en,open('biling_dict_hi_en.pkl','wb'))

en_bn = {}
for k,v in en_dic.items():
  if len(v['bn'])>0:
    en_bn[k] = v['bn'][0]
print('en_bn',len(en_bn))
pickle.dump(en_bn,open('biling_dict_en_bn.pkl','wb'))
bn_en = {}
for k,v in en_dic.items():    
  if len(v['bn'])>0:
    for i in v['bn']:
      bn_en[i] = k
print('bn_en',len(bn_en))
pickle.dump(bn_en,open('biling_dict_bn_en.pkl','wb'))

en_te = {}
for k,v in en_dic.items():
  if len(v['hi'])>0:
    if len(v['hi'])==1 and (v['hi'][0]==k):
      continue
    else:
      print(k)
      en_te[k] = translator.translate(k, dest='te', src='en').text
  else:
    en_te[k] = translator.translate(k, dest='te', src='en').text
pickle.dump(en_te,open('biling_dict_en_te.pkl','wb'))
en_te = pickle.load(open('biling_dict_en_te.pkl','rb'))
te_en = {}
for k,v in en_te.items():
  te_en[v] = k
print('te_en',len(te_en))
pickle.dump(te_en,open('biling_dict_te_en.pkl','wb'))


hi_te, te_hi = {}, {}
for k,v in en_te.items():
  try:
    hi_te[en_hi[k]]= en_te[k]
  except:
    pass
pickle.dump(hi_te,open('biling_dict_hi_te.pkl','wb'))
for k,v in en_te.items():
  try:
    te_hi[en_te[k]]= en_hi[k]
  except:
    pass
pickle.dump(te_hi,open('biling_dict_te_hi.pkl','wb'))

bn_te = {}
for k,v in en_te.items():
  try:
    bn_te[en_bn[k]]= en_te[k]
  except:
    pass
pickle.dump(bn_te,open('biling_dict_bn_te.pkl','wb'))
te_bn = {}
for k,v in en_te.items():
  try:
    te_bn[en_te[k]]= en_bn[k]
  except:
    pass
pickle.dump(te_bn,open('biling_dict_te_bn.pkl','wb'))

hi_bn = {}
for k,v in en_hi.items():
  try:
    hi_bn[v]= en_bn[k]
  except:
    pass
pickle.dump(hi_bn,open('biling_dict_hi_bn.pkl','wb'))
bn_hi = {}
for k,v in en_hi.items():
  try:
    bn_hi[en_bn[k]]= v
  except:
    pass  
pickle.dump(bn_hi,open('biling_dict_bn_hi.pkl','wb'))


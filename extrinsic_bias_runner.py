import pickle
import os

langs=['en','hi','bn','te']

for i in range(len(langs)):
    for j in range(len(langs)):
        if langs[i]==langs[j]:
            continue
        if (langs[i],langs[j]) in [('en','hi'),('en','bn'),('hi','te')]:
            continue
        for pre in ["LID","EQR"]:
            l1=langs[i]
            l2=langs[j]
            a=os.system(f'python extrensic_bias.py {pre} {l1} {l2}')
            print("done:",pre,l1,l2,a)
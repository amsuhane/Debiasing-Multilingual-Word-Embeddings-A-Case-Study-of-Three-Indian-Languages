import pickle
import os

langs=['en','hi','bn','te']

for l1 in langs:
    for l2 in langs:
        if l1!=l2:
            a=os.system(f'python generate_cross_lingual_embeddings.py {l1} {l2}')
            print("done:",l1,l2,a)

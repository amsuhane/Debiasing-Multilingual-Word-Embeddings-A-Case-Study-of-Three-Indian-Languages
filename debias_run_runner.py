import pickle
import os

num_lang_pairs=10
langs=['en','hi','bn','te']
pre_list = ['LID', 'EQR']

for pre in pre_list:
    for l1 in langs:
        for l2 in langs:
            if l1!=l2:
                a=os.system(f'python  debias_run.py {l1} {l2} {pre} {num_lang_pairs}')
                print("done:",l1,l2, pre, a)

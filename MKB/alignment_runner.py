import os

for l1 in ["en"]:
  for l2 in ['hi','bn','te']:
    for pre1 in ["fin"]:
        for pre2 in ["fin"]:
            a=os.system(f'python alignment.py {pre1} {pre2} {l1} {l2}')
            print("done:",pre1,pre2,l1,l2,a)
            a=os.system(f'python alignment.py {pre1} {pre2} {l2} {l1}')
            print("done:",pre1,pre2,l2,l1,a)            
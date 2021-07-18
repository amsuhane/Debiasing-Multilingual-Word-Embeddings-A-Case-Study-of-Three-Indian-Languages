import os

for l1 in ['hi','en','bn','te']:
  for l2 in ['hi','en','bn','te']:
    if l1==l2:
      continue
    for pre1 in ["fin"]:
        for pre2 in ["fin"]:
            a=os.system(f'python alignment.py {pre1} {pre2} {l1} {l2}')
            print("done:",pre1,l1,l2,a)
import os

dic={'en':0, 'hi':1, 'te':2,'bn':3}

embeddings=["emb_bilingual_hi_en",
            'emb_bilingual_bn_en',
            'emb_bilingual_te_en',
            'emb_bilingual_en_hi',
            'emb_bilingual_en_bn',
            'emb_bilingual_en_te',
            'emb_bilingual_bn_hi',
            'emb_bilingual_te_hi',
            'emb_bilingual_hi_te',
            'emb_bilingual_te_bn',
            'emb_bilingual_bn_te',
            'emb_bilingual_hi_bn']
embeddings = ['../data/embedding/Bilingual/'+i for i in embeddings]


for emb in embeddings:
    for limit in [1,2,4,6,8,-1]:
        l1,l2=dic[emb.split("_")[-2]],dic[emb.split("_")[-1]]
        print(emb,os.system(f'python intrinsic_bias.py {emb} 2 {limit}'))
        break
    break
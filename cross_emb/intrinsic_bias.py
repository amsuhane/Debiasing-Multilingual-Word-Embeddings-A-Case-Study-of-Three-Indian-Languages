import sys
import pickle
import numpy as np
from utils import *
import math
from sklearn.decomposition import PCA
from projpursuit import projpursuit
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon

def de_nan(e):
  for i in range(len(e)):
    e[i]=np.nan_to_num(e[i])
  return e

def var (direction,axes):
    direction = np.asarray(direction)
    direction = direction/np.linalg.norm(direction)
    for i in range(len(axes)):
        axes[i] = axes[i] - np.dot(axes[i],direction)*direction
    return np.linalg.norm(np.std(axes, axis=0))

def top_k (axes,k=4):
    if(len(axes))==0:
        return []
    PCs = []
    axes=de_nan(axes)
    pca = PCA(n_components=len(axes))
    
    principalComponents = pca.fit_transform(axes)

    for i in range(len(axes)):
        PCs.append([pca.components_[i],var(pca.components_[i],axes)])
    PCs = sorted(PCs, key = lambda x: x[1],reverse=True)
    for i in range(len(PCs)):
        PCs[i]= PCs[i][0]
    if len(PCs)!=0:
      return (PCs[:k])
    return []
def top_k_PPA (axes,k=4):
    if(len(axes))==0:
        return []
    PCs = []
    X = np.random.rand(len(axes), len(axes[0]))
    X/=1000000000
    for i in range(len(axes)):
        for j in range(len(axes[i])):
            X[i][j]+=axes[i][j]
    X=X.T
    T, V, PPOUT = projpursuit(X, p=len(axes)-1)
    T=T.T
    PPs = []
    for i in range(len(axes)-1):
        PPs.append([T[i],var(T[i],axes)])
    PPs = sorted(PPs, key = lambda x: x[1],reverse=True)
    for i in range(len(PPs)):
        PPs[i]= PPs[i][0]
    return (PPs[:k])


def cosine1(x, y):
    if (np.linalg.norm(x)*np.linalg.norm(y))==0:
      return 0
    return np.dot(x,y.T)/(np.linalg.norm(x)*np.linalg.norm(y))

def dis(OG,X,emb):
    ans=0
    cnt=0
    for x in X:
      if x not in emb.keys():
          continue
      cnt+=1
      ans+=(1-cosine1(de_nan(emb[x]),OG)) 
    assert cnt>0
    ans/=cnt
    assert ans is not np.nan
    return ans

def InBias(O,Sm,Sf,emb):
    ans=0
    cnt=0
    for o in O:
      if o not in emb.keys():
          continue
      cnt+=1
      ans+=abs(dis(de_nan(emb[o]),Sm,emb)-dis(de_nan(emb[o]),Sf,emb))
    ans/=cnt
    return ans

def ECT(emb,E_plus,E_minus,neutral_list_words):
    vec_plus,vec_minus=np.array([0]*300,dtype=np.float),np.array([0]*300,dtype=np.float)
    for e in E_plus:
        vec_plus+=emb[e]     
    for e in E_minus:
        vec_minus+=emb[e]
    vec_plus /= len(E_plus)
    vec_minus /= len(E_minus)
    sim1= []
    sim2=[]
    for word in neutral_list_words:
        if word not in vec_plus.keys() or word not in vec_plus.keys():
            continue
        sim1.append(1-cosine1(vec_plus,emb[word]))
        sim2.append(1-cosine1(vec_minus,emb[word]))
        
    return spearmanr(sim1, sim2)


def debias(g, N, candidates):
    if g is None or len(g)==0 or len(N)==0 or len(candidates)==0:
      return list([])
    dic = {}
    for i in emb_full.keys():
        word = np.asarray(emb_full[i])/(np.linalg.norm(np.asarray(emb_full[i])))
        for g_i in g:
            g_i = g_i/np.linalg.norm(g_i)
            word = word - np.dot(word,g_i)*g_i
        dic[i]=word

    for (a,b) in candidates:
        if a not in emb_full.keys() or b not in emb_full.keys():
          continue
        dic[a] = emb_full[a]
        dic[b] = emb_full[b]
        vec=(dic[a] + dic[b]) / 2
        for g_i in g:
            g_i = g_i/np.linalg.norm(g_i)
            vec = vec - np.dot(word,g_i)*g_i
            
        z = np.sqrt(1 - np.linalg.norm(vec)**2)
        component,diff=0,dic[a]-dic[b]
        lis=[]
        for g_i in g:
            g_i = g_i/np.linalg.norm(g_i)
            component += np.dot(diff,g_i)
            lis.append(abs(np.dot(diff,g_i)))

        lis=[(i/sum(lis))**0.5 for i in lis]
        if component < 0:
            z = -z
        dic[a],dic[b]=vec,vec
        j_idx=0
        for g_i in g:
            dic[a] += lis[j_idx]* z * g_i
            dic[b] += -1*lis[j_idx]* z * g_i
            j_idx+=1
    if len(dic)!=0:
      return dic
    else:
      return list([])   

def cosine1(x, y):
    if (np.linalg.norm(x)*np.linalg.norm(y))==0:
      return 0
    return np.dot(x,y.T)/(np.linalg.norm(x)*np.linalg.norm(y))

def return_test_sets():
    Adj_en = ['fair', 'good', 'light', 'nervous', 'rich', 'modern', 'plain', 'important', 'special', 'powerful', 'bright', 'evil', 'alert', 'concerned', 'unusual', 'odd', 'new', 'little', 'low', 'big', 'high', 'difficult', 'easy', 'kind', 'famous', 'friendly', 'helpful', 'careful', 'nice', 'perfect', 'better', 'young', 'happy', 'dark', 'average', 'super', 'hard', 'small', 'wild', 'poor', 'confused', 'fat', 'thin', 'long', 'clean', 'strong', 'large', 'bad', 'strange', 'old']
    Adj_hi = ['गंभीर', 'महत्वाकांक्षी', 'आक्रामक', 'यथार्थवादी', 'उदारवादी', 'आशावादी', 'संकीर्ण', 'कम', 'आकृतियों', 'परिपत्र', 'वर्ग', 'त्रिकोणीय', 'स्वाद', 'नमकीन', 'मसालेदार', 'कठिन', 'स्वच्छ', 'आसान', 'उपवास', 'भारी', 'प्रकाश', 'स्थानीय', 'शक्तिशाली', 'शांत', 'सही', 'बहुत', 'कमज़ोर', 'मात्राएँ', 'अनेक', 'नारंगी', 'लाल', 'सफ़ेद', 'आकार', 'अनुशासित', 'सरल', 'दयालु', 'ईमानदार', 'मेहनती', 'आराध्य', 'स्मार्ट', 'परिष्कृत', 'पराक्रमी', 'प्रेरित', 'व्यापक']
    Adj_be = ['গম্ভীর', 'উচ্চাকাঙ্ক্ষী', 'আক্রমনাত্মক', 'বাস্তবানুগ', 'মধ্যপন্থী', 'আশাবাদী', 'কম', 'সঙ্কীর্ণ', 'আকার', 'বিজ্ঞপ্তি', 'বর্গাকার', 'স্বাদ', 'নোনতা', 'মশলাদার', 'কঠিন', 'পরিষ্কার', 'সহজ', 'উপোস', 'ভারী', 'প্রকাশ', 'স্থানীয়', 'শক্তিশালী', 'শান্ত', 'খুব', 'দুর্বল', 'অনেক', 'কমলা', 'লাল', 'সাদা', 'আকৃতি', 'আকার', 'নিয়মনিষ্ঠ', 'সহজ', 'সদয়', 'পরিশ্রমী', 'আরাধ্য', 'স্মার্ট', 'বাস্তববুদ্ধিসম্পন্ন', 'মহৎ', 'অনুপ্রাণিত', 'ব্যাপক', 'স্বচ্ছ', 'সত্যি', 'পরাক্রমী', 'পরিষ্কৃত']
    Adj_te = ['సీరియస్', 'ప్రతిష్టాత్మక', 'దూకుడు', 'యదార్థ', 'నియంత్రించు', 'ఆప్టిమిస్టిక్', 'నిశితం', 'తక్కువ', 'ఆకారాలు', 'సర్క్యులర్', 'చదరపు', 'రుచి', 'ఉప్పగా', 'ఉండే', 'తెలంగాణ', 'కష్టం', 'క్లీన్', 'సులువు', 'ఉపవాసం', 'భారీ', 'కాంతి', 'స్థానిక', 'శక్తివంతమైన', 'శాంతిగా', 'హక్కు', 'చాలా', 'బలహీనుడు', 'వాల్యూమ్స్', 'అనేక', 'నారింజ', 'ఎరుపు', 'వైట్', 'ఆకారం', 'క్రమశిక్షణ', 'సాధారణ', 'నిజాయితీ', 'కష్టపడి', 'పూజ్యమైన', 'స్మార్ట్', 'అధునాతన', 'ప్రేరణ', 'సమగ్ర', 'లిగా', 'వేడిగా', 'మబ్బుగా', 'వర్షంగా', 'మంచుగా', 'ఎండగా', 'గాలిగా', 'వేడిగా', 'శక్తివంతమైన', 'శాంతమైన', 'సరైన', 'మెల్లని', 'మెత్తని', 'ఎక్కువ', 'బలహీనమైన', 'తప్పు', 'చిన్నవయసు', 'పరిమాణములు', 'కొద్ది', 'చాలా', 'భాగం', 'కొన్ని', 'కొంచెం', 'మొత్తం']

    NE = ['professor', 'administrator', 'ambassador', 'architect', 'artist', 'dean', 'athlete', 'director', 'author', 'baker', 'boss', 'driver', 'captain', 'chancellor', 'clerk', 'coach', 'collector', 'commander', 'commissioner', 'composer', 'conductor', 'consultant', 'cop', 'critic', 'deputy', 'doctor', 'student', 'editor', 'farmer', 'pilot', 'writer', 'designer', 'guitarist', 'historian', 'journalist', 'judge', 'lieutenant', 'manager', 'mathematician', 'minister', 'musician', 'novelist', 'officer', 'philosopher', 'poet', 'politician', 'president', 'principal', 'saint', 'scholar', 'scientist', 'secretary', 'singer', 'songwriter', 'soldier', 'steward', 'substitute', 'teacher', 'warrior']
    NH = ['वास्तुकार', 'राजगीर', 'दर्जी', 'माली', 'पत्रकार', 'न्यायाधीश', 'वकील', 'व्यवस्थापक', 'राजदूत', 'कलाकार', 'आयुक्त', 'संगीतकार', 'सलाहकार', 'आलोचक', 'उप', 'छात्र', 'संपादक', 'इतिहासकार', 'प्रबंधक', 'गणितज्ञ', 'मंत्री', 'उपन्यासकार', 'अधिकारी', 'दार्शनिक', 'राजनीतिज्ञ', 'अध्यक्ष', 'संत', 'सचिव']
    NB = ['রাজমিস্ত্রি', 'দর্জি', 'মালী', 'সাংবাদিক', 'বিচারক', 'উকিল', 'অ্যাডমিন', 'রাষ্ট্রদূত', 'শিল্পী', 'কমিশনার', 'সুরকার', 'পরামর্শকারী', 'সমালোচক', 'উপাচার্য', 'শিক্ষার্থী', 'সম্পাদক', 'ইতিহাসবেত্তা', 'ম্যানেজার', 'গণিতবিদ', 'মন্ত্রী', 'ঔপন্যাসিক', 'অফিসার', 'দার্শনিক', 'রাজনীতিজ্ঞ', 'চেয়ারম্যান', 'সাধু', 'সচিব', 'নির্দেশক', 'গীতিকার']    
    NT = ['న్యాయమూర్తి', 'న్యాయవాది', 'నిర్వాహకుడు', 'రాయబారి', 'కళాకారుడు', 'కమిషనర్', 'సంగీతకారుడు', 'కన్సల్టెంట్', 'విమర్శకుడు', 'సబ్', 'విద్యార్థులు', 'ఎడిటర్', 'చరిత్రకారుడు', 'మేనేజర్', 'గణిత','శాస్త్రవేత్తలు', 'మంత్రి', 'నవలా', 'రచయిత', 'ఆఫీసర్', 'ఫిలాసఫర్స్', 'రాజకీయ', 'చైర్మన్', 'సెయింట్', 'కార్యదర్శి', 'విద్యార్థి', 'పాఠశాల']

    A = ['जज', 'लेफ्टिनेंट', 'प्रिंसिपल', 'कंडक्टर', 'नर्स', 'प्रोफेसर', 'पायलट', 'बेकर', 'ड्राइवर', 'कप्तान', 'चांसलर', 'क्लर्क', 'कलेक्टर', 'कमांडर', 'डेल्टा', 'एनीमे', 'स्पा', 'जेलीफ़िश', 'रजिस्ट्री', 'अलास्का', 'बायोमेडिकल', 'अल्ट्रासाउंड', 'एयरोस्पेस', 'क्रिकेट', 'डेल्टा']
    C = ['বিচারক', 'লেফটানেন্ট', 'অধ্যক্ষ', 'প্রিন্সিপাল', 'কন্ডাকটর', 'নার্স', 'অধ্যাপক', 'পাইলট', 'চালক', 'অধিনায়ক', 'আচার্য', 'কেরানী', 'সংগ্রাহক', 'সেনাপতি', 'সেনাধ্যক্ষ']

    E = NE + Adj_en
    H = NH + Adj_hi + A
    B = NB + Adj_be + C
    T = NT + Adj_te    
    All = E + H + B + T

    return E, H, B, T, All

def return_gender_pairs():
    G_hi = [['मर्द', 'औरत'], ['लड़का', 'लड़की'], ['आदमी', 'महिला'], ['पुरुष', 'स्त्री'], ['बेटा', 'बेटी'], ['पति', 'पत्नी'], ['मां', 'पिता'], ['चाचा', 'चाची'], ['मामा', 'मामी'], ['राजा', 'रानी'], ['नाना', 'नानी'], ['भाई', 'बहन'], ['भैया', 'भाभी'], ['देवर', 'देवरानी'], ['देव', 'देवी'], ['पुत्र', 'पुत्री'], ['श्रीमान', 'श्रीमति'], ['बालक', 'बालिका'], ['भतीजा', 'भतीजी'], ['नाना', 'नानी'], ['साधु', 'साध्वी']]
    G_be = [['পুরুষ', 'মহিলা'], ['ছেলে', 'মেয়ে'], ['পুত্র', 'কন্যা'], ['স্বামী', 'স্ত্রী'], ['মা', 'বাবা'], ['মামা', 'মামী'], ['রাজা', 'রানী'], ['দাদু', 'দিদা'], ['ভাই', 'বোন'], ['দাদা', 'বৌদি'], ['দেব', 'দেবী'], ['পুত্র', 'কন্যা'], ['শ্রীমান', 'শ্রীমতি'], ['ভাইপো', 'ভাইঝি'], ['বোনপো', 'বোনঝি'], ['ঠাকুরদা', 'ঠাকুমা'], ['সন্ন্যাসী', 'সন্ন্যাসিনী'], ['কাকা', 'কাকী'], ['দেওর', 'জা'], ['নন্দাই', 'ননদ'], ['শালা', 'শালী']]
    G_en = [['he', 'she'], ['boy', 'girl'], ['boys', 'girls'], ['brother', 'sister'], ['brothers', 'sisters'], ['king', 'queen'], ['kings', 'queens'], ['male', 'female'], ['man', 'woman'], ['men', 'women'], ['son', 'daughter'], ['his', 'her'], ['prince', 'princess'], ['gal', 'guy'], ['actor', 'actress'], ['husband', 'wife'], ['father', 'mother'], ['god', 'goddess']]
    G_te = [['మగ', 'ఆడ'], ['కొడుకు', 'కుమార్తె'], ['భర్త', 'భార్య'], ['తల్లి', 'తండ్రి'], ['అంకుల్', 'అత్త'], ['రాజు', 'రాణి'], ['సోదరుడు', 'సోదరి'], ['సోదరుడు', 'బావ'], ['బ్రదర్', 'సోదరి'], ['దేవుడు', 'దేవత'], ['కొడుకు', 'కుమార్తె'], ['మిస్టర్', 'శ్రీమతి'], ['అబ్బాయి', 'అమ్మాయి'], ['మేనల్లుడు', 'మేనకోడలు'], ['ఎద్దు', 'ఆవు']]
    limit = int(sys.argv[3])
    G_en = G_en[:limit]
    G_be = G_be[:limit]
    G_hi = G_hi[:limit]
    G_te = G_te[:limit]

    hi_pairs_train = [['मर्द', 'औरत'], ['लड़का', 'लड़की'], ['आदमी', 'महिला'], ['पुरुष', 'स्त्री'], ['बेटा', 'बेटी'], ['पति', 'पत्नी'], ['मां', 'पिता'], ['चाचा', 'चाची'], ['मामा', 'मामी'], ['राजा', 'रानी']]
    hi_pairs_test = [['भाई', 'बहन'], ['भैया', 'भाभी'], ['देवर', 'देवरानी'], ['देव', 'देवी'], ['पुत्र', 'पुत्री'], ['श्रीमान', 'श्रीमति'], ['बालक', 'बालिका'], ['भतीजा', 'भतीजी'], ['नाना', 'नानी'], ['साधु', 'साध्वी']]    
    
    en_pairs_train = [['he', 'she'], ['boy', 'girl'], ['brother', 'sister'], ['king', 'queen'], ['male', 'female'], ['man', 'woman'], ['son', 'daughter'], ['his', 'her'], ['prince', 'princess'], ['actor', 'actress']]
    en_pairs_test = [['boys', 'girls'], ['brothers', 'sisters'], ['kings', 'queens'], ['men', 'women'], ['gal', 'guy'], ['husband', 'wife'], ['father', 'mother'], ['god', 'goddess'], ['father-in-law', 'mother-in-law'], ['son-in-law', 'daughter-in-law']]

    be_pairs_train = [['পুরুষ', 'মহিলা'], ['ছেলে', 'মেয়ে'], ['পুত্র', 'কন্যা'], ['স্বামী', 'স্ত্রী'], ['মা', 'বাবা'], ['মামা', 'মামী'], ['রাজা', 'রানী'], ['দাদু', 'দিদা'], ['ভাই', 'বোন'], ['দাদা', 'বৌদি']]
    be_pairs_test = [['দেব', 'দেবী'], ['পুত্র', 'কন্যা'], ['শ্রীমান', 'শ্রীমতি'], ['ভাইপো', 'ভাইঝি'], ['বোনপো', 'বোনঝি'], ['ঠাকুরদা', 'ঠাকুমা'], ['সন্ন্যাসী', 'সন্ন্যাসিনী'], ['কাকা', 'কাকী'], ['দেওর', 'জা'], ['নন্দাই', 'ননদ'], ['শালা', 'শালী']]

    te_pairs_train = [['మగ', 'ఆడ'], ['కొడుకు', 'కుమార్తె'], ['భర్త', 'భార్య'], ['తల్లి', 'తండ్రి'], ['అంకుల్', 'అత్త'], ['రాజు', 'రాణి'], ['సోదరుడు', 'సోదరి'], ['సోదరుడు', 'బావ'], ['బ్రదర్', 'సోదరి'], ['దేవుడు', 'దేవత']]
    te_pairs_test = [['కొడుకు', 'కుమార్తె'], ['మిస్టర్', 'శ్రీమతి'], ['అబ్బాయి', 'అమ్మాయి'], ['మేనల్లుడు', 'మేనకోడలు'], ['ఎద్దు', 'ఆవు']]

    axes_hi = []
    axes_en = []
    axes_be = []
    axes_te = []

    for i in hi_pairs_train:
        try:
          temp = np.asarray(emb_full[i[1]])- np.asarray(emb_full[i[0]])
          axes_hi.append(temp/np.linalg.norm(temp))
        except:
          pass

    for i in en_pairs_train:
        try:
          temp = np.asarray(emb_full[i[1]])- np.asarray(emb_full[i[0]])
          axes_en.append(temp/np.linalg.norm(temp))
        except:
          pass
        
    for i in be_pairs_train:
        try:
          temp = np.asarray(emb_full[i[1]])- np.asarray(emb_full[i[0]])
          axes_be.append(temp/np.linalg.norm(temp))
        except:
          pass
        
    for i in te_pairs_train:
        try:
          temp = np.asarray(emb_full[i[1]])- np.asarray(emb_full[i[0]])
          axes_te.append(temp/np.linalg.norm(temp))
        except:
          pass

    axes_all = []
    axes_all.extend(axes_hi)
    axes_all.extend(axes_en)
    axes_all.extend(axes_be)
    axes_all.extend(axes_te)
    
    e_plusE,e_minusE = [],[]
    for i in en_pairs_test:
        e_minusE.append(i[0])
        e_plusE.append(i[1])
    e_plusH,e_minusH = [],[]
    for i in hi_pairs_test:
        e_minusH.append(i[1])
        e_plusH.append(i[0])
    e_plusB,e_minusB = [],[]
    for i in be_pairs_test:
        e_minusB.append(i[1])
        e_plusB.append(i[0])        
    e_plusT,e_minusT = [],[]
    for i in te_pairs_test:
        e_minusT.append(i[1])
        e_plusT.append(i[0])        
    
    e_plus = e_plusH + e_plusE + e_plusB + e_plusT
    e_minus = e_minusH + e_minusE + e_minusB + e_minusT
    return axes_en, axes_hi, axes_be, axes_te, axes_all, e_plusH, e_minusH, e_plusE, e_minusE, e_plusB, e_minusB, e_plusT, e_minusT, e_plus, e_minus, G_hi,G_be,G_en,G_te

def run_tests(embeddings_list, K_GENDER,lang,num_langs):
    
    NE, NH, NB, NT, N = return_test_sets()
    axis_en, axis_hi, axis_be, axis_te, axis_all, e_plusH, e_minusH, e_plusE, e_minusE, e_plusB, e_minusB, e_plusT, e_minusT, e_plus, e_minus,G_hi,G_be,G_en,G_te = return_gender_pairs()
    G_all=[]
    G_all.extend(G_hi)
    G_all.extend(G_be)
    G_all.extend(G_en)
    G_all.extend(G_te)

    g_hi_PPA = top_k(axis_hi,k=K_GENDER)
    g_en_PPA = top_k(axis_en,k=K_GENDER)    
    g_be_PPA = top_k(axis_be,k=K_GENDER)
    g_te_PPA = top_k(axis_te,k=K_GENDER)
    g_all_PPA = top_k(axis_all, k=K_GENDER)
    g_all_PPA_temp = top_k(axis_all, K_GENDER*3)    
    g_eq_rep = []

    temp = []
    if g_all_PPA_temp is None:
      g_all_PPA_temp=[]

    for i in g_all_PPA_temp:
        sum_hi = 0
        sum_en = 0
        sum_be = 0
        sum_te = 0

        d={'en':0,'hi':1,'te':2,'be':3}
        cnt=[0]*4
        for j in [i[0] for i in G_hi]+[i[1] for i in G_hi]+NH:
          try:
            sum_hi = sum_hi + cosine1(i,emb_full[j])
            cnt[d['hi']]+=1
          except:
            pass
        for j in [i[0] for i in G_en]+[i[1] for i in G_en]+NE:
          try:
            cnt[d['en']]+=1
            sum_en = sum_en + cosine1(i,emb_full[j])
          except:
            pass
        for j in [i[0] for i in G_be]+[i[1] for i in G_be]+NB:
          try:
            cnt[d['be']]+=1
            sum_be = sum_be + cosine1(i,emb_full[j])
          except:
            pass
        for j in [i[0] for i in G_te]+[i[1] for i in G_te]+NT:
          try:
            cnt[d['te']]+=1
            sum_te = sum_te + cosine1(i,emb_full[j])
          except:
            pass
        AA=[[sum_en,'en'],[sum_hi,'hi'],[sum_te,'te'],[sum_be,'be']]

        for i in range(len(AA)):
            if cnt[d[AA[i][1]]]!=0:
                AA[i][0]=AA[i][0]/cnt[d[AA[i][1]]]
        AA.sort()
        temp.append(AA[-1][1])

    count=int(K_GENDER/num_langs)
    for i in range(len(temp)):
        if temp[i]=='hi':
            g_eq_rep.append(g_all_PPA_temp[i])
            count-=1
        if count==0:
            break
    count=int(K_GENDER/num_langs)
    for i in range(len(temp)):
        if temp[i]=='en':
            g_eq_rep.append(g_all_PPA_temp[i])
            count-=1
        if count==0:
            break
    count=int(K_GENDER/num_langs)
    for i in range(len(temp)):
        if temp[i]=='be':
            g_eq_rep.append(g_all_PPA_temp[i])
            count-=1
        if count==0:
            break    
    count=int(K_GENDER/num_langs)
    for i in range(len(temp)):
        if temp[i]=='te':
            g_eq_rep.append(g_all_PPA_temp[i])
            count-=1
        if count==0:
            break    
    
    embed_g_hi_PPA = debias(g_hi_PPA, NH,G_hi)
    embed_g_en_PPA = debias(g_en_PPA, NE,G_en)
    embed_g_be_PPA = debias(g_be_PPA, NB,G_be)
    embed_g_te_PPA = debias(g_te_PPA, NT,G_te)
    embed_g_all_PPA = debias(g_all_PPA, N,G_all)
    embed_g_eq_rep_PPA = debias(g_eq_rep, N,G_all)
    
    for i in embed_g_all_PPA.keys():
      embed_g_all_PPA[i]=np.nan_to_num(embed_g_all_PPA[i])

    for i in embed_g_eq_rep_PPA.keys():
      embed_g_eq_rep_PPA[i]=np.nan_to_num(embed_g_eq_rep_PPA[i])

    with open("Result.csv","a") as F:
      F.write(embedding_name+","+sys.argv[3]+","+sys.argv[1]+","+str(InBias(N,e_plus,e_minus,embed_g_eq_rep_PPA))+"\n")   

if __name__=="__main__":
  global emb_full
  emb_full={}

  embedding_name=sys.argv[1]

  with open(embedding_name,"rb") as F:
    emb_full=pickle.load(F)
  for i in emb_full.keys():
    emb_full[i]=np.nan_to_num(emb_full[i])

  run_tests([emb_full], K_GENDER=4,lang='all',num_langs=int(sys.argv[2]))

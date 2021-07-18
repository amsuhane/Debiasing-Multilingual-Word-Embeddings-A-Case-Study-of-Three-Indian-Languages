import numpy as np
import yaml
import sys
import pickle

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

if __name__=="__main__":

    l1=sys.argv[1]
    l2=sys.argv[2]

    l1_dictionary = FastVector(vector_file=f'data/aligned_files/alligned_{l1}.txt')
    l2_dictionary = FastVector(vector_file=f'data/aligned_files/alligned_{l2}.txt')

    l1_words = set(l1_dictionary.word2id.keys())
    l2_words = set(l2_dictionary.word2id.keys())
    overlap = list(l1_words & l2_words)
    bilingual_dictionary = [(entry, entry) for entry in overlap]

    with open(f'data/MKB/biling_dict_{l2}_{l1}.pkl','rb') as F:
        dd=pickle.load(F)

    bilingual_dictionary_add = []
    for key in dd.keys():
        bilingual_dictionary_add.append((dd[key], key))

    bilingual_dictionary.extend(bilingual_dictionary_add)

    # form the training matrices
    source_matrix, target_matrix = make_training_matrices(l2_dictionary, l1_dictionary, bilingual_dictionary)

    # learn and apply the transformation
    transform = learn_transformation(source_matrix, target_matrix)
    l2_dictionary.apply_transform(transform)

    dic = {}
    for w in l2_dictionary.id2word:
        dic[w] = l2_dictionary[w]

    for w in l1_dictionary.id2word:
        dic[w] = l1_dictionary[w]

    pickle.dump(dic,open(f'data/embedding/Bilingual/emb_bilingual_{l2}_{l1}','wb'))  # l2->l1        
import re

import networkx as nx
import numpy as np
import pymorphy2
from scipy.spatial import distance


def cdist_min_n(XA, XB, topn, metric='euclidean', *args, **kwargs):
    """Take n elements from XB nearest to XA"""
    try:
        if not isinstance(XA, np.ndarray):
            XA = np.asarray(XA).reshape(1, -1)
        if not isinstance(XB, np.ndarray):
            XB = np.asarray(XB)
        n = min(topn, XB.shape[0])
        cd = distance.cdist(XA, XB, metric, *args, **kwargs)
        ind = np.unravel_index(cd.ravel().argpartition(n-1)[:n], cd.shape)
        return sorted([(cd[(ind[0][i], ind[1][i])], ind[0][i], ind[1][i]) for i in range(n)])
    except Exception as e:
        return str(e)


class MLemma:
    _pos_translator = {'ADJF': 'ADJ', 'ADJS': 'ADJ',
                       'ADVB': 'ADV', 'GRND': 'GRND',
                       'INFN': 'VERB', 'NOUN': 'NOUN', 'NPRO': 'NOUN',
                       'VERB': 'VERB'}

    def __init__(self, *args, **kwargs):
        self._stopwords = list(np.loadtxt('russian.txt', dtype=str, encoding='utf-8'))
        self._lemma = pymorphy2.MorphAnalyzer()

    def normal_form(self, s):
        return self._lemma.normal_forms(s)[0]


class Graph():
    _lemma = MLemma(language='ru')

    def __init__(self, *name):
        self.g = GVector()  # .from_pickle()
        self.load(*name)

        self.node_index = {}
        self.node_labels = {}
        self.node_vectors = []
        self.get_node_vectors()

        self.vocab_pure = {}
        pass

    def load(self, name):
        self.graph = nx.read_graphml(name)

    def get_edge_labels(self, node=None):
        labels = {}
        for n in self.graph.nodes:
            for x in g.graph[n]:
                try:
                    labels.update({(n, x): g.graph[n][x]['label']})
                except Exception as e:
                    pass

        return labels

    def get_index_pure(self, word):
        try:
            return self.vocab_pure[word]
        except:
            return 0

    def get_node_vectors(self, node=None):
        for i, (node, prop) in enumerate(self.graph.nodes.items()):
            self.node_index.update({i: node})
            try:
                label = prop['label']
                # self.node_labels.update({node: label})
                self.node_labels.update({node: label})
                label = self.prepare_string(label, remove_punctuation=True, lower=True)
                label = self.tokenize(label, normalize=True)
                self.node_vectors.append(self.g.vector(label))
                # self.node_vectors.append(self.predict(label))
            except Exception as e:
                self.node_labels.update({node: ''})

        return self.node_labels

    def children(self, node):
        return self.graph.succ[node]

    def parents(self, node):
        return self.graph.pred[node]

    def prepare_string(self, s, remove_punctuation=False, lower=True):
        if remove_punctuation:
            s = re.sub(r'[–!"#$%&\'()*+,./:;<=>?@\[\\\]^_`{|}~«»]+', ' ', s)
            s = re.sub(r'\d+', ' ', s)

        s = re.sub(r'\s+', ' ', s)

        if lower:
            s = s.lower()
        return s

    def similar(self, text, topn=10, threshold=0.6):
        ret = []
        if isinstance(text, str):
            text = self.tokenize(text, normalize=True)
            vec = self.g.vector(text)  # [0]
            # vec = self.predict(text)  # [0]
            if vec is not None:
                sim = cdist_min_n([vec], self.node_vectors, topn, 'cosine')
                if len(sim) > 0:
                    # Get list of the best (word, distance)
                    best_nodes = [(self.node_index[x[2]],
                                   self.node_labels[self.node_index[x[2]]], x[0]) for x in sim
                                  if x[0] <= threshold]

                    if len(best_nodes) > 0:
                        ret = best_nodes
        return ret

    def tokenize(self, text, **options):
        '''Return list for list, str - otherwise'''
        ret = []
        try:
            if 'lower' in options and options['lower']:
                text = text.lower()
            # res = nltk.word_tokenize(text)
            if 'not_separator' in options:
                text = re.sub(options['not_separator'], chr(2), text)
            ret = re.split(r'[ !"#$%&\'()*+,-./:;<=>?@\[\\\]_`{|}~^]', text)
            ret = [x for x in ret if len(x) > 0]
            if 'not_separator' in options:
                ret = [re.sub(chr(2), options['not_separator'], x) for x in ret]
            if 'normalize' in options and options['normalize']:
                for i, w in enumerate(ret):
                    ret[i] = self._lemma.normal_form(w)
        except Exception as e:
            print(e)
        return [x for x in ret if x not in self._lemma._stopwords]


class GVector:
    _lemma = MLemma()

    def __init__(self, filenames='tayga_upos_skipgram_300_2_2019.txt',
                 binary=False, encoding='utf-8', zipped=False,
                 load=None, limit=None):
        try:
            if filenames is not None:
                self.read_me(filenames, binary, encoding=encoding, zipped=zipped, limit=limit)
        except Exception as e:
            pass

    def get_index_pure(self, word):
        try:
            return self.vocab_pure[word]
        except:
            return 0

    def get_vector_pure(self, word):
        return self.vectors[self.get_index_pure(word)]

    def read_me(self, filenames, binary=True, encoding='utf-8', zipped=False, limit=None):
        '''Numeration starts from 1, because 0 reserved for UNKNOWN words'''
        index = 1
        # for filename in pp.to_arg_list(filenames):
        for filename in [filenames]:
            # with open(os.path.join(self._folder, filename),
            with open(filename,
                      'r',
                      buffering=1024 if binary else 1,
                      encoding=None if binary else encoding) as f:
                vector_num, vector_size = map(int, f.readline().split())
                vectors = np.ndarray((vector_num if limit is None else min(vector_num, limit), vector_size), dtype=float)
                vocab = {}
                vocab_pure = {}
                read = 0
                for line in f:
                    tokens = line.rstrip().split()
                    vocab.update({tokens[0]:index})
                    vocab_pure.update({tokens[0].split('_')[0]:index})
                    vectors[read] = np.asarray(list(map(float, tokens[1:])))
                    index += 1
                    read += 1
                    if limit is not None and read>=limit:
                        break
            if not hasattr(self, 'vectors'):
                self.vectors = np.concatenate((np.zeros((1, vector_size)), vectors))
                self.vocab = vocab
                self.vocab_pure = vocab_pure
            else:
                self.vectors = np.concatenate((self.vectors, vectors))
                self.vocab.update(vocab)
                self.vocab_pure.update(vocab_pure)
        return self

    def vector(self, words, weights: list = None, to_set=True, return_all=False):
        """Get vector for arbitrary text or list of words"""

        if to_set:
            words = set(words)
        if len(words) == 0:
            return None#,False #np.zeros(self.vector_size)

        vv = [] #np.empty((0, self.vector_size))
        for i, w in enumerate(words):  # .split():
            w = w.strip()
            if w != '':
                try:
                    vec = self.get_vector_pure(w)
                    if vec is not None:
                        vv.append(vec if weights is None else vec * weights[i])
                except Exception as e:
                    pass
        n = len(vv)
        if n == 1:
            v = vv[0]
        elif n > 1:
            v = np.average(np.asarray(vv), axis=0)
        elif n == 0:
            v = None
        return (v, vv) if return_all else v


if __name__ == '__main__':
    try:
        g = Graph('Polytech_total.graphml')

        sim = g.similar('физика')

        pass
    except Exception as e:
        print(e)

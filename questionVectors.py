import gensim
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import operator
from nltk import word_tokenize
from nltk.corpus import stopwords
# find a numerical representation for each question using
# google news pre trained word2vec model for each word in the
# question and then averaging the feature vectors of words
# of a question as a feature vector of the whole sentence


import word2vecReaderUtils as utils
from numpy import exp, dot, zeros, outer, random, dtype, float32 as REAL, \
    uint32, seterr, array, uint8, vstack, argsort, fromstring, sqrt, newaxis, \
    ndarray, empty, sum as np_sum, prod
from six import string_types
from gensim import matutils


class Vocab(object):
    """A single vocabulary item, used internally for constructing binary trees (incl. both word leaves and inner nodes)."""

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


class Word2Vec:
    """
    Class for training, using and evaluating neural networks described in https://code.google.com/p/word2vec/

    The model can be stored/loaded via its `save()` and `load()` methods, or stored/loaded in a format
    compatible with the original word2vec implementation via `save_word2vec_format()` and `load_word2vec_format()`.

    """

    def __init__(self, sentences=None, size=100, alpha=0.025, window=5, min_count=5,
                 sample=0, seed=1, workers=1, min_alpha=0.0001, sg=1, hs=1, negative=0, cbow_mean=0):
        """
        Initialize the model from an iterable of `sentences`. Each sentence is a
        list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
        this module for such examples.

        If you don't supply `sentences`, the model is left uninitialized -- use if
        you plan to initialize it in some other way.

        `sg` defines the training algorithm. By default (`sg=1`), skip-gram is used. Otherwise, `cbow` is employed.

        `size` is the dimensionality of the feature vectors.

        `window` is the maximum distance between the current and predicted word within a sentence.

        `alpha` is the initial learning rate (will linearly drop to zero as training progresses).

        `seed` = for the random number generator.

        `min_count` = ignore all words with total frequency lower than this.

        `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
            default is 0 (off), useful value is 1e-5.

        `workers` = use this many worker threads to train the model (=faster training with multicore machines).

        `hs` = if 1 (default), hierarchical sampling will be used for model training (else set to 0).

        `negative` = if > 0, negative sampling will be used, the int for negative
        specifies how many "noise words" should be drawn (usually between 5-20).

        `cbow_mean` = if 0 (default), use the sum of the context word vectors. If 1, use the mean.
        Only applies when cbow is used.

        """
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to word (string)
        self.sg = int(sg)
        self.table = None  # for negative sampling --> this needs a lot of RAM! consider setting back to None before saving
        self.layer1_size = int(size)
        # if size % 4 != 0:
        #    logger.warning("consider setting layer size to a multiple of 4 for greater performance")
        self.alpha = float(alpha)
        self.window = int(window)
        self.seed = seed
        self.min_count = min_count
        self.sample = sample
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)

    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, norm_only=True):
        """
        Load the input-hidden weight matrix from the original C word2vec-tool format.

        Note that the information stored in the file is incomplete (the binary tree is missing),
        so while you can query for word similarity etc., you cannot continue training
        with a model loaded this way.

        `binary` is a boolean indicating whether the data is in binary word2vec format.
        `norm_only` is a boolean indicating whether to only store normalised word2vec vectors in memory.
        Word counts are read from `fvocab` filename, if set (this is the file generated
        by `-save-vocab` flag of the original C tool).
        """
        counts = None
        if fvocab is not None:
            # logger.info("loading word counts from %s" % (fvocab))
            counts = {}
            with utils.smart_open(fvocab) as fin:
                for line in fin:
                    word, count = utils.to_unicode(line).strip().split()
                    counts[word] = int(count)

        # logger.info("loading projection weights from %s" % (fname))
        with utils.smart_open(fname) as fin:
            header = utils.to_unicode(fin.readline())
            vocab_size, layer1_size = map(int, header.split())  # throws for invalid file format
            result = Word2Vec(size=layer1_size)
            result.syn0 = zeros((vocab_size, layer1_size), dtype=REAL)
            if binary:
                binary_len = dtype(REAL).itemsize * layer1_size
                for line_no in range(vocab_size):
                    # mixed text and binary: read text first, then binary
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b' ':
                            break
                        if ch != b'\n':  # ignore newlines in front of words (some binary files have newline, some don't)
                            word.append(ch)
                    word = utils.to_unicode(b''.join(word), encoding='latin-1')

                    if counts is None:
                        result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    elif word in counts:
                        result.vocab[word] = Vocab(index=line_no, count=counts[word])
                    else:
                        # logger.warning("vocabulary file is incomplete")
                        result.vocab[word] = Vocab(index=line_no, count=None)
                    result.index2word.append(word)
                    result.syn0[line_no] = fromstring(fin.read(binary_len), dtype=REAL)
            else:
                for line_no, line in enumerate(fin):
                    parts = utils.to_unicode(line).split()
                    if len(parts) != layer1_size + 1:
                        raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                    word, weights = parts[0], map(REAL, parts[1:])
                    if counts is None:
                        result.vocab[word] = Vocab(index=line_no, count=vocab_size - line_no)
                    elif word in counts:
                        result.vocab[word] = Vocab(index=line_no, count=counts[word])
                    else:
                        # logger.warning("vocabulary file is incomplete")
                        result.vocab[word] = Vocab(index=line_no, count=None)
                    result.index2word.append(word)
                    result.syn0[line_no] = weights
        # logger.info("loaded %s matrix from %s" % (result.syn0.shape, fname))
        result.init_sims(norm_only)
        return result

    def init_sims(self, replace=False):
        """
        Precompute L2-normalized vectors.

        If `replace` is set, forget the original vectors and only keep the normalized
        ones = saves lots of memory!

        Note that you **cannot continue training** after doing a replace. The model becomes
        effectively read-only = you can call `most_similar`, `similarity` etc., but not `train`.

        """
        if getattr(self, 'syn0norm', None) is None or replace:
            # logger.info("precomputing L2-norms of word weight vectors")
            if replace:
                for i in range(self.syn0.shape[0]):
                    self.syn0[i, :] /= sqrt((self.syn0[i, :] ** 2).sum(-1))
                self.syn0norm = self.syn0
                if hasattr(self, 'syn1'):
                    del self.syn1
            else:
                self.syn0norm = (self.syn0 / sqrt((self.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)

    def __getitem__(self, word):

        return self.syn0[self.vocab[word].index]

    def __contains__(self, word):
        return word in self.vocab


def loadTwitterModel():
    model_path = "word2vec_twitter_model.bin"
    print("Loading the model, this can take some time...")
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    return model

# above this line is just to load word2vec model trained on Twitter (can be ignored)
#####################################################################################

stop_words = stopwords.words('english')
data = pd.read_csv('Data/train.csv')
data = data.drop(['id', 'qid1', 'qid2'], axis=1)

# calculate tf-idf weights
tfidfdata = data.dropna()
questions = list(tfidfdata['question1']) + list(tfidfdata['question2'])
tfidf = TfidfVectorizer(lowercase=False, )
tfidf.fit_transform(questions)
weights = dict(zip(tfidf.get_feature_names(), tfidf.idf_))


# returns sum of word vectors for each sentence
def SumSent2vec(s, model, num_features):
    words = str(s).lower()  # .decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    # featureVec = np.zeros(num_features, dtype="float32")
    for w in words:
        try:
            M.append(model[w])
            # featureVec = np.add(featureVec, model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())
    # return featureVec / np.sqrt((featureVec ** 2).sum())


# returns TFIDF weighted average of word vectors for each sentence
def AvgSent2vecTFIDF(s, model, num_features):
    words = str(s).lower()  # .decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            try:
                idf = weights[w]
            except:
                idf = 0
            M.append(model[w] * idf)
        except:
            continue
    M = np.array(M)
    v = M.mean(axis=0)
    return v / np.sqrt((v ** 2).sum())


# returns average of word vectors for each sentence
def AvgSent2vec(s, model, num_features):
    print(s)
    words = str(s).lower()  # .decode('utf-8')
    print(words)
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    wordNum = 0
    global counter
    for w in words:
        try:
            wordNum += 1
            M.append(model[w])
        except:
            continue
    if M:
        a = 2
    else:
        counter += 1
        print(counter)
    M = np.array(M)
    v = M.mean(axis=0)
    return v / np.sqrt((v ** 2).sum())


# returns sum of word vectors for each sentence
def svd(s, model, num_features):
    words = str(s).lower()  # .decode('utf-8')
    words = word_tokenize(words)
    # words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            print(w)
            continue
    if M:
        P1, D1, Q1 = np.linalg.svd(M, full_matrices=True)
        # Lindex, value = max(enumerate([D1]), key=operator.itemgetter(1))
        # print('Q1', Q1)

        Max = Q1[D1.argmax(axis=0)]
        # print("max", Max)
        return Max / np.sqrt((Max ** 2).sum())
    else:
        print(s)
        print("aaay baba")
        return [0] * num_features


# return mean of top 2 singular vectors of a matrix consists of representations of words of a sentence
def TwoSvd(s, model, num_features):
    words = str(s).lower()  # .decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    if M:
        temp = []
        P1, D1, Q1 = np.linalg.svd(M, full_matrices=True)
        if second_largest(D1):
            secondMax = Q1[list(D1).index(second_largest(D1))]
        else:
            print(D1)
            print("don't have second")
            secondMax = [0] * num_features
        Max = Q1[D1.argmax(axis=0)]
        temp.append(secondMax)
        temp.append(Max)
        temp = np.array(temp)
        meaan = temp.mean(axis=0)
        return meaan / np.sqrt((meaan ** 2).sum())
    else:
        print("dont have first ")
        return [0] * num_features


# return second lorgest singular value
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None


# returns dot product of word vectors for each sentence
def pairWiseProduct(s, model, num_features):
    words = str(s).lower()  # .decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    # featureVec = np.zeros(num_features, dtype="float32")
    for w in words:
        try:
            M.append(model[w])
            # featureVec = np.add(featureVec, model[w])
        except:
            continue
    M = np.array(M)
    v = [1] * num_features
    for word in M:
        for i in range(num_features):
            v[i] = v[i] * word[i]

    v = np.array(v)

    return v / np.sqrt((v ** 2).sum())


# returns subtract of question1 and question2 as feature aggregation method
def difff(q1, q2, num_features):
    return np.subtract(q1, q2)


# returns sum of question1 and question2 as feature aggregation method
def summ(q1, q2, num_features):
    return np.add(q1, q2)


# returns multiplication of question1 and question2 as feature aggregation method
def multt(q1, q2, num_features):
    return np.multiply(q1, q2)


q1FVecs = np.zeros((data.shape[0], 300))
q2FVecs = np.zeros((data.shape[0], 300))


def execute(embModel, aggMethod):
    if embModel == "googleNews":
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    elif embModel == "Twitter":
        model = loadTwitterModel()
    else:
        print("specified embegging model is not defined")
        return

    if aggMethod == "sum":
        for i, q in tqdm(enumerate(data.question1.values)):
            q1FVecs[i, :] = SumSent2vec(q, model, 300)
        print("question1 calculated")

        for i, q in tqdm(enumerate(data.question2.values)):
            q2FVecs[i, :] = SumSent2vec(q, model, 300)
        print("question2 calculated")

    elif aggMethod == "average":
        for i, q in tqdm(enumerate(data.question1.values)):
            q1FVecs[i, :] = AvgSent2vec(q, model, 300)
        print("question1 calculated")

        for i, q in tqdm(enumerate(data.question2.values)):
            q2FVecs[i, :] = AvgSent2vec(q, model, 300)
        print("question2 calculated")

    elif aggMethod == "svd":
        for i, q in tqdm(enumerate(data.question1.values)):
            q1FVecs[i, :] = svd(q, model, 300)
        print("question1 calculated")

        for i, q in tqdm(enumerate(data.question2.values)):
            q2FVecs[i, :] = svd(q, model, 300)
        print("question2 calculated")

    elif aggMethod == "pairWiseProduct":
        for i, q in tqdm(enumerate(data.question1.values)):
            q1FVecs[i, :] = pairWiseProduct(q, model, 300)
        print("question1 calculated")

        for i, q in tqdm(enumerate(data.question2.values)):
            q2FVecs[i, :] = pairWiseProduct(q, model, 300)
        print("question2 calculated")

    dfFeatures = np.concatenate((q1FVecs, q2FVecs), axis=1)
    FinalFeatures = dfFeatures

# un comment if you want to use subtract of question1 and question2 as feature aggregation method

    # difFeatures = np.zeros((data.shape[0], 300))
    # for i, q in tqdm(enumerate(dfFeatures)):
    #     difFeatures[i, :] = difff(q[:300], q[300:], 300)
    # FinalFeatures = difFeatures

# un comment if you want to use multiplication of question1 and question2 as feature aggregation method
    # mulFeatures = np.zeros((data.shape[0], 300))
    # for i, q in tqdm(enumerate(dfFeatures)):
    #     mulFeatures[i, :] = multt(q[:300], q[300:], 300)
    # FinalFeatures = mulFeatures

# un comment if you want to use sum of question1 and question2 as feature aggregation method
    # sumFeatures = np.zeros((data.shape[0], 300))
    # for i, q in tqdm(enumerate(dfFeatures)):
    #     sumFeatures[i, :] = summ(q[:300], q[300:], 300)
    # FinalFeatures = sumFeatures

    train_df = pd.DataFrame(FinalFeatures)

    train_df = pd.concat([train_df, data["is_duplicate"]], axis=1)
    train_df.to_csv('sum.csv', index=False)
    print("dfFeatures2")

    return train_df

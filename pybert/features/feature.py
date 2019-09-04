import numpy as np

class TextFeature(object):
    def __init__(self,n_dims):
        self.n_dims = n_dims

    def char_stat_feature(self, sentence):
        feature = {}
        feature['n_chars'] = len(sentence)
        feature['n_caps'] = sum(1 for char in sentence if char.isupper())
        feature['caps_rate'] = feature['n_caps'] / feature['n_chars']
        features = np.array(list(feature.values()))
        return features

    def word_stat_feature(self,sentence):
        feature = {}
        tokens = sentence.split()
        feature['n_words'] = len(tokens)
        feature['unique_words'] = len(set(tokens))
        feature['unique_rate'] = feature['unique_words'] / feature['n_words']
        features = np.array(list(feature.values()))
        return features

    def unk_word_feature(self,vocab):
        features = vocab.unk.astype('f')
        features[0] = 0
        features = features[:, None]
        return features

    def idf_word_feature(self,vocab):
        dfs = np.array(list(vocab.word_freq.values()))
        dfs[0] = vocab.n_documents
        features = np.log(vocab.n_documents / dfs)
        features = features[:, None]
        return features
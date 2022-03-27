# -*- coding:utf-8 -*-

"""



Author:

    Chengliang Zhao, bruce.e.zhao@gmail.com



Reference:

    [1] Jizhe Wang, Pipei Huang, Huan Zhao, Zhibo Zhang, Binqiang Zhao, and Dik Lun Lee. 2018. Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba. In <i>Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining</i> (<i>KDD '18</i>). Association for Computing Machinery, New York, NY, USA, 839–848. DOI:https://doi.org/10.1145/3219819.3219869


"""
from ..walker import RandomWalker
from gensim.models import Word2Vec
import pandas as pd
import numpy as np


class EGES:
    def __init__(self, graph, walk_length, num_walks, workers=1):

        self.graph = graph
        self.w2v_model = None
        self._embeddings = {}

        self.walker = RandomWalker(
            graph, p=1, q=1, )
        self.sentences = self.walker.simulate_walks(
            num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["epochs"] = iter

        # print("Learning embedding vectors...")
        # model = Word2Vec(**kwargs)
        # print("Learning embedding vectors done!")

        # self.w2v_model = model
        # return model


        

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

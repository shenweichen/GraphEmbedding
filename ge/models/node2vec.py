# -*- coding:utf-8 -*-
import pandas as pd
import networkx as nx
import csrgraph as cg

import gc
import numba
import time
import numpy as np
import pandas as pd
from gensim.models import word2vec




class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0,threads=1):
    
        if type(threads) is not int:
            raise ValueError('Threads must be int!')
        if walk_length<1:
            raise ValueError('Walk lengh must be >1')
        if num_walks<1:
            raise ValueError('num_walks must be >1')
        if type(walk_length) is not int or type(num_walks) is not int:
            raise ValueError('Walk length or num_walks must be int')
            
        self.walk_length=walk_length
        self.num_walks=num_walks
        self.p=p
        self.q=q
        self.threads=threads
        # todo numba-based use_rejection_samplling
        
        if not isinstance(graph, cg.csrgraph):
            self.graph = cg.csrgraph(graph, threads=self.threads)
        if self.graph.threads != self.threads:
            self.graph.set_threads(self.threads)
        self.node_names = self.graph.names
        if type(self.node_names[0]) not in [int, str, np.int32, np.uint32, 
                                       np.int64, np.uint64]:
            raise ValueError("Graph node names must be int or str!")
            
        
            
    def train(self, embed_size=128, window_size=5, workers=3, iters=5 **kwargs):
        print('Start making random walks...')
        start=time.time()
        self.sentences=self.graph.random_walks(walklen=self.walk_length,epochs=self.num_walks, \
                                               return_weight=self.p,neighbor_weight=self.q).astype(str).tolist() # It seems gensim word2vec only accept list and string types data
        end=time.time()
        print('Random walks uses '+str(end-start)+' seconds')
            
            

        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec don't need to use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iters
        
        print("Learning embedding vectors...")
        model = word2vec.Word2Vec(sentences=self.sentences,**kwargs) ##Avoid to copy self.sentences in order to save the memory
        print("Learning embedding vectors done!")

        self.w2v_model = model
        self.node_dict = dict(zip(np.arange(len(self.node_names)).astype(str),self.node_names)) # map the node_names to the original node names


    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.node_dict.keys():
            self._embeddings[self.node_dict[word]] = self.w2v_model.wv[self.node_dict[word]]

        return self._embeddings

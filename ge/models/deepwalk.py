# -*- coding:utf-8 -*-

"""



Author:

    Weichen Shen,wcshen1994@163.com



Reference:

    [1] Perozzi B, Al-Rfou R, Skiena S. Deepwalk: Online learning of social representations[C]//Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2014: 701-710.(http://www.perozzi.net/publications/14_kdd_deepwalk.pdf)



"""
from ..walker import RandomWalker
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
from csrgraph import csrgraph
class DeepWalk:
    def __init__(self, graph, walk_length, num_walks, workers=1,use_csrgraph=False):
        self.use_csrgraph=use_csrgraph
        self.w2v_model = None
        self._embeddings = {}
        
        if self.use_csrgraph:
            node_names=list(graph.nodes())
            self.graph=csrgraph(graph,nodenames=node_names,threads=workers)

            self.sentences = pd.DataFrame(self.graph.random_walks(
                epochs=num_walks, walklen=walk_length,  return_weight=1.,neighbor_weight=1.))
            # Map nodeId -> node name
            node_dict = dict(zip(np.arange(len(node_names)), node_names))
            
            for col in self.sentences.columns:
                self.sentences[col] = self.sentences[col].map(node_dict).astype(str)
            # Somehow gensim only trains on this list iterator
            # it silently mistrains on array input
            self.sentences = [list(x) for x in self.sentences.itertuples(False, None)]
            
        else:
            self.graph = graph
            self.walker = RandomWalker(
                graph, p=1, q=1, )
            self.sentences = self.walker.simulate_walks(
                num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)

    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        
        
        if self.use_csrgraph:
            kwargs["sentences"] = self.sentences
            kwargs["min_count"] = kwargs.get("min_count", 0)
            kwargs["vector_size"] = embed_size
            kwargs["sg"] = 1  # skip gram
            kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
            kwargs["workers"] = workers
            kwargs["window"] = window_size
            kwargs["epochs"] = iter
    
            print("Learning embedding vectors...")
            model = Word2Vec(**kwargs)
            print("Learning embedding vectors done!")
            self.w2v_model = model
            
        else:
            kwargs["sentences"] = self.sentences
            kwargs["min_count"] = kwargs.get("min_count", 0)
            kwargs["vector_size"] = embed_size
            kwargs["sg"] = 1  # skip gram
            kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
            kwargs["workers"] = workers
            kwargs["window"] = window_size
            kwargs["epochs"] = iter
    
            print("Learning embedding vectors...")
            model = Word2Vec(**kwargs)
            print("Learning embedding vectors done!")
    
            self.w2v_model = model
            
        return model

    def get_embeddings(self,):
        if self.w2v_model is None:
            print("model not train")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings

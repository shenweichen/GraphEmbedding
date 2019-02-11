from __future__ import print_function

import math
from joblib import Parallel, delayed
import random
import itertools

import numpy as np
import pandas as pd
from tqdm import trange

from .alias import alias_sample, create_alias_table
from .utils import partition_num


class RandomWalker:
    def __init__(self, G, p=1, q=1):
        """
        :param G:
        :param p: Return parameter,controls the likelihood of immediately revisiting a node in the walk.
        :param q: In-out parameter,allows the search to differentiate between “inward” and “outward” nodes
        """
        self.G = G
        self.p = p
        self.q = q

    def deepwalk_walk(self, walk_length, start_node):

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    pos = (prev, cur)
                    next = cur_nbrs[alias_sample(alias_edges[pos][0],
                                                 alias_edges[pos][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G

        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))


        #pd.to_pickle(walks, 'random_walks.pkl')
        return walks

    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1 and self.q == 1:
                    walks.append(self.deepwalk_walk(
                        walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(
                        walk_length=walk_length, start_node=v))
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in G.neighbors(dst):
            weight = G[dst][dst_nbr].get('weight', 1.0)
            if dst_nbr == src:
                unnormalized_probs.append(weight/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}
        triads = {}

        #look_up_dict = self.look_up_dict
        #node_size = self.node_size
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


class BiasedWalker:
    def __init__(self, idx2node, temp_path):

        self.idx2node = idx2node
        self.idx = list(range(len(self.idx2node)))
        self.temp_path = temp_path
        pass

    def simulate_walks(self, num_walks, walk_length):

        layers_adj = pd.read_pickle(self.temp_path+'layers_adj.pkl')
        #    #存储每一层的邻接点列表
        layers_alias = pd.read_pickle(self.temp_path+'layers_alias.pkl')
        layers_accept = pd.read_pickle(self.temp_path+'layers_accept.pkl')
        gamma = pd.read_pickle(self.temp_path+'gamma.pkl')
        # 大于平均权重的邻接点个数
        walks = []
        initialLayer = 0

        nodes = self.idx  # list(self.g.nodes())
        print('Walk iteration:')
        for walk_iter in trange(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self._exec_random_walk(layers_adj, layers_alias,
                                                    layers_accept, v, walk_length, gamma))
            # walk = self._exec_ramdom_walks_for_chunck(
            #     nodes, g, alias_method_j, alias_method_q, walk_length, gamma)
            # walks.extend(walk)
        pd.to_pickle(walks, self.temp_path + 'walks.pkl')
        return walks

    # def _exec_ramdom_walks_for_chunck(self, nodes, graphs, alias_method_j, alias_method_q, walk_length, gamma):
    #     walks = []
    #     for v in nodes:
    #         walks.append(self._exec_random_walk(graphs, alias_method_j,
    #                                             alias_method_q, v, walk_length, gamma))
    #     return walks

    def _exec_random_walk(self, graphs, layers_alias, layers_accept, v, walk_length, gamma):
        original_v = v
        initialLayer = 0
        layer = initialLayer

        path = []
        path.append(self.idx2node[v])

        while len(path) < walk_length:
            r = random.random()

            if(r < 0.3):  # 在同一层
                v = chooseNeighbor(v, graphs, layers_alias,
                                   layers_accept, layer)
                path.append(self.idx2node[v])

            else:  # 跳到不同的层
                r = random.random()
                try:
                    limiar_moveup = prob_moveup(gamma[layer][v])
                except:
                    print(layer, v)
                    raise ValueError()

                if(r > limiar_moveup):
                    if(layer > initialLayer):
                        layer = layer - 1
                else:
                    if((layer + 1) in graphs and v in graphs[layer + 1]):
                        layer = layer + 1

        return path


def chooseNeighbor(v, graphs, layers_alias, layers_accept, layer):

    v_list = graphs[layer][v]

    idx = alias_sample(layers_accept[layer][v], layers_alias[layer][v])
    v = v_list[idx]

    return v


def prob_moveup(gamma):
    x = math.log(gamma + math.e)
    p = (x / (x + 1))
    return p

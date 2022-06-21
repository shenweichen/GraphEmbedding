import networkx as nx
import pytest

from ge import Node2Vec


@pytest.mark.parametrize(
    'use_rejection_sampling',
    [True, False
     ]
)
def test_Node2Vec(use_rejection_sampling):
    G = nx.read_edgelist('./tests/Wiki_edgelist.txt',
                         create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Node2Vec(G, walk_length=10, num_walks=80,
                     p=0.25, q=4, workers=1, use_rejection_sampling=use_rejection_sampling)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()


if __name__ == "__main__":
    pass
